# Coupled-Newton helpers and driver for moving diffusion problems

function overwrite_future_volume!(capacity::Capacity, new_diag::AbstractVector{T}) where {T}
    cap_index = length(capacity.A)
    block = capacity.A[cap_index]
    half = size(block, 1) ÷ 2
    length(new_diag) == half || throw(DimensionMismatch("Expected $(half) entries, got $(length(new_diag))"))
    @inbounds for (idx, value) in enumerate(new_diag)
        block[idx, idx] = value
    end
    return capacity
end

function coupled_newton_step!(phase::Phase, bc::AbstractBoundary, scheme::String,
    φω::AbstractVector{T}, φγ::AbstractVector{T}, rhsω::AbstractVector{T}, rhsγ::AbstractVector{T}) where {T<:Real}

    operator = phase.operator
    capacity = phase.capacity
    dims = operator.size
    len_dims = length(dims)
    cap_index = len_dims
    half = length(φω)

    height_block = capacity.A[cap_index]
    size(height_block, 1) == 2 * half || throw(DimensionMismatch("Incompatible operator and state dimensions."))

    V_future = height_block[1:half, 1:half]
    V_past = height_block[half+1:end, half+1:end]
    V_future_diag = LinearAlgebra.diag(V_future)
    V_past_diag = LinearAlgebra.diag(V_past)

    Id = build_I_D(operator, phase.Diffusion_coeff, capacity)
    W! = operator.Wꜝ[1:half, 1:half]
    G = operator.G[1:half, 1:half]
    H = operator.H[1:half, 1:half]
    Id = Id[1:half, 1:half]

    Iα, Iβ = build_I_bc(operator, bc)
    Iα = Iα[1:half, 1:half]
    Iβ = Iβ[1:half, 1:half]
    Iγ = capacity.Γ[1:half, 1:half]

    psip = scheme == "CN" ? psip_cn : psip_be
    θ = scheme == "CN" ? 0.5 : 1.0
    Ψ = SparseArrays.spdiagm(0 => psip.(V_past_diag, V_future_diag))

    Lωω = Id * G' * W! * G * Ψ
    Lωγ = Id * G' * W! * H * Ψ
    Lγω = Iβ * H' * W! * G
    Lγγ = Iβ * H' * W! * H + Iα * Iγ

    Fω = V_future_diag .* φω + θ * (Lωω * φω) + (Lωγ * φγ) + (V_past_diag .- V_future_diag) .* φγ - rhsω
    R = V_future_diag .- V_past_diag + θ * (Lγω * φω) + Lγγ * φγ
    Gres = φγ - rhsγ
    residual_vec = vcat(Fω, R, Gres)

    J11 = SparseArrays.spdiagm(0 => V_future_diag) + θ * Lωω
    J12 = Lωγ + SparseArrays.spdiagm(0 => V_past_diag .- V_future_diag)
    J13 = SparseArrays.spdiagm(0 => φω .- φγ)

    J21 = θ * Lγω
    J22 = Lγγ
    J23 = SparseArrays.spdiagm(0 => ones(T, half))

    zero_block = SparseArrays.spzeros(T, half, half)
    I_half = SparseArrays.spdiagm(0 => ones(T, half))

    J_top = hcat(J11, J12, J13)
    J_mid = hcat(J21, J22, J23)
    J_bot = hcat(zero_block, I_half, zero_block)
    J = vcat(J_top, J_mid, J_bot)


    # Solve for the increments
    δ = IterativeSolvers.gmres(J,-residual_vec)
    δΦω = δ[1:half]
    δΦγ = δ[half+1:2*half]
    δV = δ[2*half+1:3*half]

    return δΦω, δΦγ, δV, residual_vec
end

function solve_MovingLiquidDiffusionUnsteadyMono_coupledNewton!(s::Solver, phase::Phase, xf, Δt::Float64, Tₛ::Float64, Tₑ::Float64,
    bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh::AbstractMesh, scheme::String;
    Newton_params=(100, 1e-10, 1e-10, 1.0))

    s.A === nothing && error("Solver is not initialized. Call a solver constructor first.")

    println("Solving the problem (coupled Newton):")
    println("- Moving problem")
    println("- Non prescibed motion")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")

    dims = phase.operator.size
    len_dims = length(dims)
    len_dims == 2 || error("Coupled Newton routine currently supports 1D problems.")
    n = dims[1]
    cap_index = len_dims

    max_iter, tol, reltol, damping = Newton_params

    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Float64[]
    timestep_history = Tuple{Float64, Float64}[]

    t = Tₛ
    push!(timestep_history, (t, Δt))

    φω = s.x === nothing ? zeros(n) : copy(s.x[1:n])
    φγ = s.x === nothing ? zeros(n) : copy(s.x[n+1:2n])
    if s.x === nothing
        s.x = vcat(φω, φγ)
    end

    V_future_diag = LinearAlgebra.diag(phase.capacity.A[cap_index][1:n, 1:n])
    current_xf = xf
    step_id = 1

    while t < Tₑ + eps()
        rhsω = s.b[1:n]
        rhsγ = s.b[n+1:2n]

        iter = 0
        residual_norm = Inf
        new_xf = current_xf

        while iter < max_iter && residual_norm > tol && residual_norm > reltol * max(1.0, abs(current_xf))
            iter += 1

            δΦω, δΦγ, δV, residual_vec = coupled_newton_step!(phase, bc, scheme, φω, φγ, rhsω, rhsγ)

            φω .+= damping .* δΦω
            φγ .+= damping .* δΦγ
            V_future_diag .+= damping .* δV
            overwrite_future_volume!(phase.capacity, V_future_diag)

            s.x = vcat(φω, φγ)

            residual_norm = LinearAlgebra.norm(residual_vec)

            if !haskey(residuals, step_id)
                residuals[step_id] = Float64[]
            end
            push!(residuals[step_id], residual_norm)

            δxf = damping * Statistics.mean(δV)
            new_xf = current_xf + δxf

            println("Iteration $(iter) | xf = $(new_xf) | residual = $(residual_norm)")

            if residual_norm <= tol || residual_norm <= reltol * max(1.0, abs(new_xf))
                break
            end

            tn = t
            tn1 = min(t + Δt, Tₑ)
            body = (xx, tt, _=0) -> (xx - (current_xf * (tn1 - tt) / Δt + new_xf * (tt - tn) / Δt))
            STmesh = SpaceTimeMesh(mesh, [tn, tn1], tag=mesh.tag)
            capacity = Capacity(body, STmesh)
            operator = DiffusionOps(capacity)
            phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

            s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, vcat(φω, φγ), Δt, tn, scheme)
            BC_border_mono!(s.A, s.b, bc_b, mesh; t=tn1)

            rhsω = s.b[1:n]
            rhsγ = s.b[n+1:2n]
            V_future_diag = LinearAlgebra.diag(phase.capacity.A[cap_index][1:n, 1:n])
            current_xf = new_xf
        end

        current_xf = new_xf
        push!(xf_log, current_xf)

        println("Converged after $(iter) iterations with xf = $(current_xf), residual = $(residual_norm)")

        Tᵢ = vcat(φω, φγ)
        s.x = Tᵢ
        push!(s.states, s.x)

        if t >= Tₑ
            break
        end

        t += Δt
        push!(timestep_history, (t, Δt))

        body = (xx, tt, _=0) -> (xx - current_xf)
        STmesh = SpaceTimeMesh(mesh, [Δt, 2Δt], tag=mesh.tag)
        capacity = Capacity(body, STmesh)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, 0.0, scheme)
        BC_border_mono!(s.A, s.b, bc_b, mesh; t=t)

        rhsω = s.b[1:n]
        rhsγ = s.b[n+1:2n]
        V_future_diag = LinearAlgebra.diag(phase.capacity.A[cap_index][1:n, 1:n])

        step_id += 1
    end

    return s, residuals, xf_log, timestep_history
end
