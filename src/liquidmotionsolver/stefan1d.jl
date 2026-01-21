"""
    StefanMono1D(phase, bc_b, bc_i, Δt, Tᵢ, mesh, scheme)

Set up a 1D Stefan solver (monophasic) on a moving interface. The geometry is
handled through the moving-capacity machinery as in the diffusion solver.
"""
function StefanMono1D(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::Penguin.Mesh{1}, scheme::String)
    println("Solver Creation:")
    println("- Stefan problem (1D)")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady diffusion")

    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, [], [])

    if scheme == "CN"
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "CN")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "CN")
    else
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "BE")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "BE")
    end
    BC_border_mono!(s.A, s.b, bc_b, mesh; t=0.0)
    s.x = Tᵢ
    return s
end

"""
    stefan_residual_vector_1d(phase, T, ic, scheme)

Return the Stefan residual vector per cell:
    r = ρL (H_{n+1} - H_n) - q
where q is the interface heat flux contribution. Also returns the height
profiles and flux vector for diagnostics.
"""
function stefan_residual_vector_1d(phase::Phase, T::AbstractVector, ic::InterfaceConditions, scheme::String)
    cap_index = length(phase.operator.size)
    Vn_1 = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
    Vn   = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]

    psip = scheme == "CN" ? psip_cn : psip_be
    Ψn1 = Diagonal(psip.(diag(Vn), diag(Vn_1)))

    W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
    G  = phase.operator.G[1:end÷2, 1:end÷2]
    H  = phase.operator.H[1:end÷2, 1:end÷2]
    Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
    Id = Id[1:end÷2, 1:end÷2]

    Tₒ, Tᵧ = T[1:end÷2], T[end÷2+1:end]
    flux_vec = Id * H' * W! * G * Ψn1 * Tₒ + Id * H' * W! * H * Ψn1 * Tᵧ

    Hₙ_profile, Hₙ₊₁_profile = extract_height_profiles(phase.capacity, phase.operator.size)
    ρL = ic.flux.value
    residual_vec = ρL .* (Hₙ₊₁_profile .- Hₙ_profile) .- flux_vec

    return residual_vec, Hₙ_profile, Hₙ₊₁_profile, flux_vec
end

"""
    solve_StefanMono1D!(s, phase, xf, Δt, Tₛ, Tₑ, bc_b, bc, ic, mesh, scheme; kwargs...)

Solve the 1D Stefan problem using a Gauss–Newton/least-squares update on the
interface position. At each time slab we:
1. Solve diffusion with the current interface guess.
2. Build the Stefan residual vector r = ρL (H_{n+1}-H_n) - q.
3. Approximate ∂r/∂xf with a finite difference on the geometry.
4. Take a damped least-squares step Δxf = -(Jᵀ r)/(Jᵀ J), clamped to keep the
   interface inside the mesh.
"""
function solve_StefanMono1D!(s::Solver, phase::Phase, xf::Float64, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh::Penguin.Mesh{1}, scheme::String; 
    method=Base.:\,
    algorithm=nothing,
    Newton_params=(10, 1e-8, 1e-8, 0.5),
    fd_eps::Float64=1e-3,
    step_max::Union{Nothing, Float64}=nothing,
    kwargs...)

    s.A === nothing && error("Solver is not initialized. Call StefanMono1D first.")

    println("Solving Stefan problem (1D, least-squares interface update):")

    max_iter, tol, reltol, damping = Newton_params
    x_nodes = mesh.nodes[1]
    Δx = length(x_nodes) > 1 ? x_nodes[2] - x_nodes[1] : 1.0
    x_min = x_nodes[1] - Δx/2
    x_max = x_nodes[end] + Δx/2
    step_cap = step_max === nothing ? Δx : step_max

    # Logs
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Float64[]
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (Tₛ, Δt))

    t = Tₛ
    Tᵢ = s.x
    push!(s.states, s.x)

    step_id = 1
    xf_prev = xf

    while t < Tₑ
        if t + Δt > Tₑ
            Δt = Tₑ - t
        end

        converged = false
        current_xf = xf_prev

        for iter in 1:max_iter
            # Geometry for this trial interface (linear in time on the slab)
            tn = t
            tn1 = t + Δt
            body = (xx, tt, _=0)->(xx - (xf_prev*((tn1 - tt)/Δt) + current_xf*((tt - tn)/Δt)))
            STmesh = SpaceTimeMesh(mesh, [tn, tn1], tag=mesh.tag)
            capacity = Capacity(body, STmesh)
            operator = DiffusionOps(capacity)
            phase_current = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

            s.A = A_mono_unstead_diff_moving(phase_current.operator, phase_current.capacity, phase_current.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase_current.operator, phase_current.capacity, phase_current.Diffusion_coeff, phase_current.source, bc, Tᵢ, Δt, tn, scheme)
            BC_border_mono!(s.A, s.b, bc_b, mesh; t=tn1)

            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            Tᵢ = s.x

            res_vec, _, _, _ = stefan_residual_vector_1d(phase_current, Tᵢ, ic, scheme)
            err = norm(res_vec, Inf)
            push!(get!(residuals, step_id, Float64[]), err)

            if (err <= tol) || (err <= reltol * max(abs(current_xf), 1.0))
                converged = true
                xf_prev = current_xf
                push!(xf_log, current_xf)
                break
            end

            # Finite-difference Jacobian of the residual w.r.t interface position (geometry only)
            fd_shift = fd_eps * Δx
            trial_fd = clamp(current_xf + fd_shift, x_min, x_max)
            body_fd = (xx, tt, _=0)->(xx - (xf_prev*((tn1 - tt)/Δt) + trial_fd*((tt - tn)/Δt)))
            capacity_fd = Capacity(body_fd, STmesh)
            operator_fd = DiffusionOps(capacity_fd)
            phase_fd = Phase(capacity_fd, operator_fd, phase.source, phase.Diffusion_coeff)
            res_fd, _, _, _ = stefan_residual_vector_1d(phase_fd, Tᵢ, ic, scheme)

            denom_shift = trial_fd - current_xf
            J = denom_shift != 0.0 ? (res_fd .- res_vec) ./ denom_shift : zero(res_vec)
            ls_num = dot(J, res_vec)
            ls_den = dot(J, J) + eps()
            Δxf = -damping * ls_num / ls_den
            Δxf = clamp(Δxf, -step_cap, step_cap)

            new_xf = clamp(current_xf + Δxf, x_min, x_max)
            current_xf = new_xf
        end

        if !converged
            println("Warning: time step $step_id did not meet tolerance (residual=$(residuals[step_id][end]))")
            xf_prev = current_xf
            push!(xf_log, current_xf)
        end

        t += Δt
        push!(timestep_history, (t, Δt))
        push!(s.states, Tᵢ)
        step_id += 1
    end

    return s, residuals, xf_log, timestep_history
end

