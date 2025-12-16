# Moving - Navier-Stokes - Unsteady - Monophasic

mutable struct MovingNavierStokesUnsteadyMono{N}
    fluid::Fluid{N}
    bc_u::NTuple{N, BorderConditions}
    pressure_gauge::AbstractPressureGauge
    bc_cut::NTuple{N, AbstractBoundary}
    convection::Union{Nothing, NavierStokesConvection{N}}

    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    x::Vector{Float64}
    states::Vector{Vector{Float64}}
    times::Vector{Float64}
    prev_conv::Union{Nothing, NTuple{N, Vector{Float64}}}
end

function MovingNavierStokesUnsteadyMono(fluid::Fluid{N},
                                        bc_u::NTuple{N, BorderConditions},
                                        pressure_gauge::AbstractPressureGauge,
                                        bc_cut::Union{AbstractBoundary, NTuple{N, AbstractBoundary}};
                                        x0=zeros(0)) where {N}
    println("Solver Creation:")
    println("- Moving problem")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Navier-Stokes problem")

    nu_components = ntuple(i -> prod(fluid.operator_u[i].size), N)
    np = prod(fluid.operator_p.size)
    Ntot = 2 * sum(nu_components) + np

    x_init = length(x0) == Ntot ? copy(x0) : zeros(Ntot)

    A = spzeros(Float64, Ntot, Ntot)
    b = zeros(Ntot)

    cut_bc = normalize_cut_bc(bc_cut, N)

    return MovingNavierStokesUnsteadyMono{N}(fluid, bc_u, pressure_gauge, cut_bc,
                                             nothing, A, b, x_init, Vector{Float64}[], Float64[], nothing)
end

function MovingNavierStokesUnsteadyMono(fluid::Fluid{2},
                                        bc_ux::BorderConditions,
                                        bc_uy::BorderConditions,
                                        pressure_gauge::AbstractPressureGauge,
                                        bc_cut::Union{AbstractBoundary, NTuple{2, AbstractBoundary}};
                                        x0=zeros(0))
    return MovingNavierStokesUnsteadyMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut; x0=x0)
end


# Convection helper (space–time aware)
function compute_moving_convection_vectors(convection::NavierStokesConvection{N},
                                           data,
                                           advecting_state::AbstractVector{<:Real},
                                           advected_state::AbstractVector{<:Real}=advecting_state) where {N}
    nu_components = data.nu_components

    uω_adv = Vector{Vector{Float64}}(undef, N)
    uγ_adv = Vector{Vector{Float64}}(undef, N)
    offset = 0
    for i in 1:N
        n = nu_components[i]
        uω_adv[i] = Vector{Float64}(view(advecting_state, offset + 1:offset + n))
        offset += n
        uγ_adv[i] = Vector{Float64}(view(advecting_state, offset + 1:offset + n))
        offset += n
    end
    uω_adv_tuple = Tuple(uω_adv)
    uγ_adv_tuple = Tuple(uγ_adv)

    same_state = advected_state === advecting_state
    qω_tuple = nothing
    qγ_tuple = nothing
    if same_state
        qω_tuple = uω_adv_tuple
        qγ_tuple = uγ_adv_tuple
    else
        uω_advected = Vector{Vector{Float64}}(undef, N)
        uγ_advected = Vector{Vector{Float64}}(undef, N)
        offset = 0
        for i in 1:N
            n = nu_components[i]
            uω_advected[i] = Vector{Float64}(view(advected_state, offset + 1:offset + n))
            offset += n
            uγ_advected[i] = Vector{Float64}(view(advected_state, offset + 1:offset + n))
            offset += n
        end
        qω_tuple = Tuple(uω_advected)
        qγ_tuple = Tuple(uγ_advected)
    end

    bulk = ntuple(Val(N)) do i
        idx = Int(i)
        build_convection_matrix(convection.stencils[idx], uω_adv_tuple)
    end

    K_adv = ntuple(Val(N)) do i
        idx = Int(i)
        build_K_matrix(convection.stencils[idx], rotated_interfaces(uγ_adv_tuple, idx))
    end

    K_advected = same_state ? K_adv : ntuple(Val(N)) do i
        idx = Int(i)
        build_K_matrix(convection.stencils[idx], rotated_interfaces(qγ_tuple, idx))
    end

    conv_vectors = ntuple(Val(N)) do i
        idx = Int(i)
        bulk[idx] * qω_tuple[idx] .- 0.5 .* (K_adv[idx] * qω_tuple[idx] .+ K_advected[idx] * uω_adv_tuple[idx])
    end

    return conv_vectors
end


# Block builder (2D moving)
function navierstokes2D_moving_blocks(fluid::Fluid{2},
                                      ops_u::Tuple{DiffusionOps, DiffusionOps},
                                      caps_u::Tuple{Capacity, Capacity},
                                      op_p::DiffusionOps,
                                      cap_p::Capacity,
                                      scheme::Symbol,
                                      convection::NavierStokesConvection{2})
    base = stokes2D_moving_blocks(fluid, ops_u, caps_u, op_p, cap_p, scheme)

    return (; base..., nu_components=(base.nu_x, base.nu_y), convection=convection)
end


function assemble_navierstokes2D_moving!(s::MovingNavierStokesUnsteadyMono{2}, data, Δt::Float64,
                                         x_prev::AbstractVector{<:Real},
                                         t_prev::Float64, t_next::Float64,
                                         θ::Float64,
                                         conv_prev::Union{Nothing, NTuple{2, Vector{Float64}}})
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    θc = 1.0 - θ

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_con = 2 * sum_nu

    # Momentum blocks
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = data.Vn_1_ux + θ * data.visc_x_ω * data.Ψn1_ux
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = -(data.Vn_1_ux - data.Vn_ux) + θ * data.visc_x_γ * data.Ψn1_ux
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = data.Vn_1_uy + θ * data.visc_y_ω * data.Ψn1_uy
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = -(data.Vn_1_uy - data.Vn_uy) + θ * data.visc_y_γ * data.Ψn1_uy
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    # Tie rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    # Continuity rows
    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ

    uωx_prev = view(x_prev, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev, off_uγy+1:off_uγy+nu_y)

    f_prev_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_ux, t_prev)
    f_next_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_ux, t_next)
    f_prev_x = f_prev_x[1:end÷2]
    f_next_x = f_next_x[1:end÷2]
    load_x = data.Vx * (θ .* f_next_x .+ θc .* f_prev_x)

    f_prev_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_uy, t_prev)
    f_next_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_uy, t_next)
    f_prev_y = f_prev_y[1:end÷2]
    f_next_y = f_next_y[1:end÷2]
    load_y = data.Vy * (θ .* f_next_y .+ θc .* f_prev_y)

    rhs_mom_x = (data.Vn_ux - θc * data.visc_x_ω * data.Ψn_ux) * uωx_prev
    rhs_mom_x .-= θc * data.visc_x_γ * data.Ψn_ux * uγx_prev
    rhs_mom_x .+= load_x

    rhs_mom_y = (data.Vn_uy - θc * data.visc_y_ω * data.Ψn_uy) * uωy_prev
    rhs_mom_y .-= θc * data.visc_y_γ * data.Ψn_uy * uγy_prev
    rhs_mom_y .+= load_y

    conv_curr = compute_moving_convection_vectors(data.convection, data, x_prev)
    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    if conv_prev === nothing
        rhs_mom_x .-= ρ_val .* conv_curr[1]
        rhs_mom_y .-= ρ_val .* conv_curr[2]
    else
        rhs_mom_x .-= ρ_val .* (1.5 .* conv_curr[1] .- 0.5 .* conv_prev[1])
        rhs_mom_y .-= ρ_val .* (1.5 .* conv_curr[2] .- 0.5 .* conv_prev[2])
    end

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut[1], data.cap_ux, t_next)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut[2], data.cap_uy, t_next)
    g_cut_x = g_cut_x[1:end÷2]
    g_cut_y = g_cut_y[1:end÷2]

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y, zeros(np))

    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 t=t_next)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    s.convection = data.convection
    return conv_curr
end


function solve_MovingNavierStokesUnsteadyMono!(s::MovingNavierStokesUnsteadyMono{2},
                                               body::Function,
                                               mesh::AbstractMesh,
                                               Δt::Float64, Tₛ::Float64, Tₑ::Float64,
                                               bc_b_u::Tuple{BorderConditions, BorderConditions},
                                               bc_cut::Union{AbstractBoundary, NTuple{2, AbstractBoundary}};
                                               scheme::Symbol=:BE,
                                               method=Base.:\,
                                               algorithm=nothing,
                                               geometry_method::String="VOFI",
                                               kwargs...)
    println("Solving the problem:")
    println("- Moving problem")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Navier-Stokes problem")

    θ = scheme == :CN ? 0.5 : 1.0

    s.bc_u = bc_b_u
    s.bc_cut = normalize_cut_bc(bc_cut, 2)
    convection_data = s.convection === nothing ? build_convection_data(s.fluid) : s.convection
    s.convection = convection_data

    dx = mesh.nodes[1][2] - mesh.nodes[1][1]
    dy = mesh.nodes[2][2] - mesh.nodes[2][1]
    mesh_ux = Penguin.Mesh((length(mesh.nodes[1])-1, length(mesh.nodes[2])-1),
                            (mesh.nodes[1][end] - mesh.nodes[1][1], mesh.nodes[2][end] - mesh.nodes[2][1]),
                            (mesh.nodes[1][1] - 0.5*dx, mesh.nodes[2][1]))
    mesh_uy = Penguin.Mesh((length(mesh.nodes[1])-1, length(mesh.nodes[2])-1),
                            (mesh.nodes[1][end] - mesh.nodes[1][1], mesh.nodes[2][end] - mesh.nodes[2][1]),
                            (mesh.nodes[1][1], mesh.nodes[2][1] - 0.5*dy))

    t = Tₛ
    push!(s.times, t)
    push!(s.states, copy(s.x))

    conv_prev = s.prev_conv

    println("Time : $(t)")
    while t < Tₑ - 1e-12 * max(1.0, Tₑ)
        dt_step = min(Δt, Tₑ - t)
        t_next = t + dt_step
        println("Time : $(t_next)")

        STmesh_ux = Penguin.SpaceTimeMesh(mesh_ux, [t, t_next], tag=mesh.tag)
        STmesh_uy = Penguin.SpaceTimeMesh(mesh_uy, [t, t_next], tag=mesh.tag)
        STmesh_p = Penguin.SpaceTimeMesh(mesh, [t, t_next], tag=mesh.tag)

        capacity_ux = Capacity(body, STmesh_ux; method=geometry_method, kwargs...)
        capacity_uy = Capacity(body, STmesh_uy; method=geometry_method, kwargs...)
        capacity_p = Capacity(body, STmesh_p; method=geometry_method, kwargs...)

        operator_ux = DiffusionOps(capacity_ux)
        operator_uy = DiffusionOps(capacity_uy)
        operator_p = DiffusionOps(capacity_p)

        data = navierstokes2D_moving_blocks(s.fluid,
                                            (operator_ux, operator_uy),
                                            (capacity_ux, capacity_uy),
                                            operator_p, capacity_p,
                                            scheme,
                                            convection_data)

        x_prev = s.x
        conv_curr = assemble_navierstokes2D_moving!(s, data, dt_step, x_prev, t, t_next, θ, conv_prev)

        solve_moving_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        push!(s.times, t_next)
        push!(s.states, copy(s.x))
        println("Solver Extremum : ", maximum(abs.(s.x)))

        conv_prev = (copy(conv_curr[1]), copy(conv_curr[2]))
        t = t_next
    end

    s.prev_conv = conv_prev
    return s.times, s.states
end


function solve_moving_navierstokes_linear_system!(s::MovingNavierStokesUnsteadyMono; method=Base.:\, algorithm=nothing, kwargs...)
    Ared, bred, keep_idx_rows, keep_idx_cols = remove_zero_rows_cols!(s.A, s.b)

    xred = nothing
    if algorithm !== nothing
        prob = LinearSolve.LinearProblem(Ared, bred)
        sol = LinearSolve.solve(prob, algorithm)
        xred = sol.u
    elseif method === Base.:\
        try
            xred = Ared \ bred
        catch e
            if e isa SingularException
                @warn "Direct solver hit SingularException; falling back to bicgstabl" sizeA=size(Ared)
                xred = IterativeSolvers.bicgstabl(Ared, bred)
            else
                rethrow(e)
            end
        end
    else
        xred = method(Ared, bred)
    end

    N = size(s.A, 2)
    s.x = zeros(N)
    s.x[keep_idx_cols] = xred
    return s
end


# Force diagnostics (useful for FSI)
function compute_moving_navierstokes_force_2D(s::MovingNavierStokesUnsteadyMono{2}, data)
    nu_x = data.nu_x
    nu_y = data.nu_y
    np = data.np
    total_velocity_dofs = 2 * (nu_x + nu_y)
    length(s.x) == total_velocity_dofs + np || error("State length mismatch for force computation.")

    pω = Vector{Float64}(view(s.x, total_velocity_dofs + 1:total_velocity_dofs + np))

    pressure_vec_x = -Vector{Float64}(data.grad_x * pω)
    pressure_vec_y = -Vector{Float64}(data.grad_y * pω)

    function viscous_part(op::DiffusionOps, cap::Capacity, μ, uω, uγ)
        G = op.G[1:end÷2, 1:end÷2]
        H = op.H[1:end÷2, 1:end÷2]
        W = op.Wꜝ[1:end÷2, 1:end÷2]
        Iμ = build_I_D(op, μ, cap)
        Iμ = Iμ[1:end÷2, 1:end÷2]
        grad_vec = G * uω
        if size(H, 2) != 0
            grad_vec .+= H * uγ
        end
        mixed = W * grad_vec
        return Vector{Float64}(Iμ * (G' * mixed))
    end

    uωx = Vector{Float64}(view(s.x, 1:nu_x))
    uγx = Vector{Float64}(view(s.x, nu_x + 1:2nu_x))
    uωy = Vector{Float64}(view(s.x, 2nu_x + 1:2nu_x + nu_y))
    uγy = Vector{Float64}(view(s.x, 2nu_x + nu_y + 1:2nu_x + 2nu_y))

    visc_x = viscous_part(data.op_ux, data.cap_ux, s.fluid.μ, uωx, uγx)
    visc_y = viscous_part(data.op_uy, data.cap_uy, s.fluid.μ, uωy, uγy)

    force_density_x = pressure_vec_x .+ visc_x
    force_density_y = pressure_vec_y .+ visc_y
    integrated_force = SVector(sum(force_density_x), sum(force_density_y))

    return (; pressure=(pressure_vec_x, pressure_vec_y),
            viscous=(visc_x, visc_y),
            force_density=(force_density_x, force_density_y),
            integrated_force=integrated_force)
end


# Moving - Navier-Stokes - Unsteady - Monophasic - Rigid FSI (2D)
mutable struct MovingNavierStokesFSI2D
    navier::MovingNavierStokesUnsteadyMono{2}
    body_shape::Function
    mass::Float64
    center::SVector{2,Float64}
    velocity::SVector{2,Float64}
    external_force::Function
    centers::Vector{SVector{2,Float64}}
    velocities::Vector{SVector{2,Float64}}
    forces::Vector{SVector{2,Float64}}
end

function MovingNavierStokesFSI2D(navier::MovingNavierStokesUnsteadyMono{2},
                                 body_shape::Function,
                                 mass::Real,
                                 center0::AbstractVector{<:Real},
                                 velocity0::AbstractVector{<:Real};
                                 external_force::Function=(t, c, v) -> SVector(0.0, 0.0))
    length(center0) == 2 || error("center0 must have length 2")
    length(velocity0) == 2 || error("velocity0 must have length 2")
    mass_val = Float64(mass)
    c0 = SVector{2,Float64}(center0...)
    v0 = SVector{2,Float64}(velocity0...)
    return MovingNavierStokesFSI2D(navier, body_shape, mass_val, c0, v0, external_force,
                                   SVector{2,Float64}[], SVector{2,Float64}[], SVector{2,Float64}[])
end

function solve_MovingNavierStokesFSI2D!(fsi::MovingNavierStokesFSI2D,
                                        mesh::AbstractMesh,
                                        Δt::Float64, Tₛ::Float64, Tₑ::Float64,
                                        bc_b_u::Tuple{BorderConditions, BorderConditions};
                                        scheme::Symbol=:BE,
                                        method=Base.:\,
                                        algorithm=nothing,
                                        geometry_method::String="VOFI",
                                        external_force::Function=fsi.external_force,
                                        kwargs...)
    println("Solving the problem:")
    println("- Moving problem (rigid FSI)")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Navier-Stokes problem")

    θ = scheme == :CN ? 0.5 : 1.0

    s = fsi.navier
    s.bc_u = bc_b_u

    dx = mesh.nodes[1][2] - mesh.nodes[1][1]
    dy = mesh.nodes[2][2] - mesh.nodes[2][1]
    mesh_ux = Penguin.Mesh((length(mesh.nodes[1])-1, length(mesh.nodes[2])-1),
                            (mesh.nodes[1][end] - mesh.nodes[1][1], mesh.nodes[2][end] - mesh.nodes[2][1]),
                            (mesh.nodes[1][1] - 0.5*dx, mesh.nodes[2][1]))
    mesh_uy = Penguin.Mesh((length(mesh.nodes[1])-1, length(mesh.nodes[2])-1),
                            (mesh.nodes[1][end] - mesh.nodes[1][1], mesh.nodes[2][end] - mesh.nodes[2][1]),
                            (mesh.nodes[1][1], mesh.nodes[2][1] - 0.5*dy))

    empty!(s.times); empty!(s.states)
    empty!(fsi.centers); empty!(fsi.velocities); empty!(fsi.forces)

    t = Tₛ
    push!(s.times, t)
    push!(s.states, copy(s.x))
    push!(fsi.centers, fsi.center)
    push!(fsi.velocities, fsi.velocity)

    conv_prev = s.prev_conv

    println("Time : $(t)")
    while t < Tₑ - 1e-12 * max(1.0, Tₑ)
        dt_step = min(Δt, Tₑ - t)
        t_next = t + dt_step
        println("Time : $(t_next)")

        c0 = fsi.center
        c1 = c0 + dt_step * fsi.velocity

        body_fn = rigid_body_path(fsi.body_shape, c0, c1, t, t_next)

        v_body = fsi.velocity
        bc_cut = (Dirichlet((x, y, t_local) -> v_body[1]),
                  Dirichlet((x, y, t_local) -> v_body[2]))
        s.bc_cut = normalize_cut_bc(bc_cut, 2)

        STmesh_ux = Penguin.SpaceTimeMesh(mesh_ux, [t, t_next], tag=mesh.tag)
        STmesh_uy = Penguin.SpaceTimeMesh(mesh_uy, [t, t_next], tag=mesh.tag)
        STmesh_p = Penguin.SpaceTimeMesh(mesh, [t, t_next], tag=mesh.tag)

        capacity_ux = Capacity(body_fn, STmesh_ux; method=geometry_method, kwargs...)
        capacity_uy = Capacity(body_fn, STmesh_uy; method=geometry_method, kwargs...)
        capacity_p = Capacity(body_fn, STmesh_p; method=geometry_method, kwargs...)

        operator_ux = DiffusionOps(capacity_ux)
        operator_uy = DiffusionOps(capacity_uy)
        operator_p = DiffusionOps(capacity_p)

        # Convection stencils use instantaneous geometry at t_next (spatial mesh) to avoid
        # the extra time dimension of the space-time operators.
        body_slice = (x, y) -> body_fn(x, y, t_next)
        conv_cap_ux = Capacity(body_slice, mesh_ux; method=geometry_method, kwargs...)
        conv_cap_uy = Capacity(body_slice, mesh_uy; method=geometry_method, kwargs...)
        conv_op_ux = DiffusionOps(conv_cap_ux)
        conv_op_uy = DiffusionOps(conv_cap_uy)
        convection = NavierStokesConvection{2}((build_convective_stencil(conv_cap_ux, conv_op_ux, 1),
                                               build_convective_stencil(conv_cap_uy, conv_op_uy, 2)))

        data = navierstokes2D_moving_blocks(s.fluid,
                                            (operator_ux, operator_uy),
                                            (capacity_ux, capacity_uy),
                                            operator_p, capacity_p,
                                            scheme,
                                            convection_data)

        x_prev = s.x
        conv_curr = assemble_navierstokes2D_moving!(s, data, dt_step, x_prev, t, t_next, θ, conv_prev)

        solve_moving_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        force_data = compute_moving_navierstokes_force_2D(s, data)
        F_fluid = force_data.integrated_force
        F_body = -F_fluid
        F_ext = SVector{2,Float64}(external_force(t_next, c0, v_body))
        a = (F_body + F_ext) / fsi.mass

        v_new = v_body + dt_step * a
        c_new = c1

        push!(s.times, t_next)
        push!(s.states, copy(s.x))
        push!(fsi.centers, c_new)
        push!(fsi.velocities, v_new)
        push!(fsi.forces, F_body)
        println("Solver Extremum : ", maximum(abs.(s.x)))
        println("Rigid center    : ", c_new, " | velocity: ", v_new, " | force: ", F_body)

        fsi.center = c_new
        fsi.velocity = v_new
        conv_prev = (copy(conv_curr[1]), copy(conv_curr[2]))
        t = t_next
    end

    s.prev_conv = conv_prev
    return s.times, s.states, fsi.centers, fsi.velocities, fsi.forces
end
