# Moving - Stokes - Prescribed Motion

@inline _normalize_bc_tuple(bc::BorderConditions, ::Val{1}) = (bc,)
@inline _normalize_bc_tuple(bc::NTuple{N,BorderConditions}, ::Val{N}) where {N} = bc
@inline function _normalize_bc_tuple(bc_tuple::Tuple, ::Val{N}) where {N}
    length(bc_tuple) == N || error("Expected $(N) velocity BC entries, got $(length(bc_tuple)).")
    return bc_tuple
end
@inline function _normalize_bc_tuple(bc::BorderConditions, ::Val{N}) where {N}
    error("Provide $(N) velocity boundary-condition sets; received a single BorderConditions.")
end

@inline _normalize_mesh_tuple(mesh::AbstractMesh, ::Val{1}) = (mesh,)
@inline _normalize_mesh_tuple(meshes::NTuple{N,AbstractMesh}, ::Val{N}) where {N} = meshes
@inline function _normalize_mesh_tuple(meshes::Tuple, ::Val{N}) where {N}
    length(meshes) == N || error("Expected $(N) velocity meshes, got $(length(meshes)).")
    return meshes
end
@inline function _normalize_mesh_tuple(mesh::AbstractMesh, ::Val{N}) where {N}
    error("Provide $(N) velocity meshes; received a single mesh.")
end

@inline _normalize_cut_tuple(bc::NTuple{N,AbstractBoundary}, ::Val{N}) where {N} = bc
@inline function _normalize_cut_tuple(bc_tuple::Tuple, ::Val{N}) where {N}
    length(bc_tuple) == N || error("Expected $(N) interface BC entries, got $(length(bc_tuple)).")
    return bc_tuple
end
@inline function _normalize_cut_tuple(bc::AbstractBoundary, ::Val{N}) where {N}
    return ntuple(_ -> bc, N)
end

@inline function theta_from_scheme(scheme::String)
    s = lowercase(strip(scheme))
    if s in ("cn", "crank_nicolson", "cranknicolson")
        return 0.5
    elseif s in ("be", "backward_euler", "implicit_euler")
        return 1.0
    else
        error("Unsupported time scheme $(scheme). Use \"CN\" or \"BE\".")
    end
end

function spacetime_timestep(capacity::AbstractCapacity)
    mesh = capacity.mesh
    mesh isa SpaceTimeMesh || error("Space-time capacities required for moving Stokes.")
    t_nodes = mesh.nodes[end]
    length(t_nodes) ≥ 2 || error("Space-time mesh must contain at least two time nodes.")
    return t_nodes[end] - t_nodes[1], t_nodes[1], t_nodes[end]
end

function normalize_prev_state(x_prev::AbstractVector{<:Real}, N::Int)
    if length(x_prev) == N
        return collect(x_prev)
    else
        y = zeros(N)
        copyto!(y, 1, x_prev, 1, min(N, length(x_prev)))
        return y
    end
end
"""
    MovingStokesMono(fluid::Fluid{N}, bc_u::NTuple{N,BorderConditions}, bc_cut::AbstractBoundary;
                     pressure_gauge::AbstractPressureGauge=DEFAULT_PRESSURE_GAUGE,
                     scheme::String="BE", t_eval::Union{Nothing,Float64}=nothing,
                     x0::Vector{Float64}=Float64[])

Create a solver for a monophasic Stokes system assembled on a space–time slab.
`fluid` must be built using `SpaceTimeMesh` objects for the velocity and pressure
fields over the time interval of interest (`[t, t+Δt]`). The returned solver
stores the assembled matrix and RHS for that slab but does not solve it.
"""
function MovingStokesMono(fluid::Fluid{N},
                          bc_u::NTuple{N,BorderConditions},
                          bc_cut;
                          pressure_gauge::AbstractPressureGauge=DEFAULT_PRESSURE_GAUGE,
                          scheme::String="BE",
                          t_eval::Union{Nothing,Float64}=nothing,
                          x0::Vector{Float64}=Float64[]) where {N}
    println("Solver Creation:")
    println("- Moving problem")
    println("- Monophasic problem")
    println("- Stokes problem")

    cut_tuple = _normalize_cut_tuple(bc_cut, Val(N))

    dt, t_prev_cap, t_next_cap = spacetime_timestep(fluid.capacity_u[1])
    t_next = t_eval === nothing ? t_next_cap : t_eval
    t_prev = t_next - dt

    A, b = assemble_moving_stokes(fluid, bc_u, cut_tuple, pressure_gauge;
                                  scheme=scheme, t_prev=t_prev, t_next=t_next,
                                  Δt=dt, x_prev=x0)
    Ndofs = size(A, 2)
    init = length(x0) == Ndofs ? copy(x0) : zeros(Ndofs)
    s = Solver(Unsteady, Monophasic, Stokes, A, b, init, [], [])
    isempty(s.states) && push!(s.states, copy(init))
    return s
end

MovingStokesMono(fluid::Fluid{1},
                 bc_u::BorderConditions,
                 bc_cut; kwargs...) =
    MovingStokesMono(fluid, (bc_u,), bc_cut; kwargs...)

function MovingStokesMono(fluid::Fluid{N},
                          bc_u_args::Vararg{BorderConditions,N};
                          bc_cut,
                          pressure_gauge::AbstractPressureGauge=DEFAULT_PRESSURE_GAUGE,
                          scheme::String="BE",
                          t_eval::Union{Nothing,Float64}=nothing,
                          x0::Vector{Float64}=Float64[]) where {N}
    return MovingStokesMono(fluid, Tuple(bc_u_args), bc_cut;
                            pressure_gauge=pressure_gauge,
                            scheme=scheme, t_eval=t_eval, x0=x0)
end

"""
    solve_MovingStokesMono!(s::Solver, fluid::Fluid{1}, body::Function,
                            mesh_u::AbstractMesh, mesh_p::AbstractMesh,
                            bc_u::BorderConditions, bc_cut::AbstractBoundary,
                            Δt::Float64, Tₛ::Float64, Tₑ::Float64;
                            pressure_gauge::AbstractPressureGauge=DEFAULT_PRESSURE_GAUGE,
                            scheme::String="BE", geometry_method::String="VOFI",
                            method=IterativeSolvers.gmres, algorithm=nothing, kwargs...)

Solve successive space–time slabs for a prescribed body motion. `mesh_u` and
`mesh_p` are the *spatial* meshes used to build the staggered velocity and
pressure grids; the routine constructs `SpaceTimeMesh` objects internally.
"""
function solve_MovingStokesMono!(s::Solver,
                                 fluid::Fluid{N},
                                 body::Function,
                                 mesh_u,
                                 mesh_p::AbstractMesh,
                                 bc_u,
                                 bc_cut,
                                 Δt::Float64,
                                 Tₛ::Float64,
                                 Tₑ::Float64;
                                 pressure_gauge::AbstractPressureGauge=DEFAULT_PRESSURE_GAUGE,
                                 scheme::String="BE",
                                 geometry_method::String="VOFI",
                                 method=IterativeSolvers.gmres,
                                 algorithm=nothing,
                                 kwargs...) where {N}
    s.A === nothing && error("Solver is not initialized. Call MovingStokesMono first.")
    println("Solving the problem:")
    println("- Moving problem")
    println("- Monophasic problem")
    println("- Stokes problem")

    bc_tuple = _normalize_bc_tuple(bc_u, Val(N))
    cut_tuple = _normalize_cut_tuple(bc_cut, Val(N))
    mesh_tuple = _normalize_mesh_tuple(mesh_u, Val(N))

    times = Float64[]
    t = Tₛ
    println("Time : $(t)")
    solve_system!(s; method=method, algorithm=algorithm, kwargs...)
    push!(s.states, copy(s.x))
    push!(times, t + Δt)

    while t < Tₑ
        t += Δt
        println("Time : $(t)")
        t_next = t + Δt

        STmesh_u = ntuple(i -> SpaceTimeMesh(mesh_tuple[i], [t, t_next], tag=mesh_tuple[i].tag), N)
        STmesh_p = SpaceTimeMesh(mesh_p, [t, t_next], tag=mesh_p.tag)
        capacity_u = ntuple(i -> Capacity(body, STmesh_u[i]; compute_centroids=true, method=geometry_method), N)
        capacity_p = Capacity(body, STmesh_p; compute_centroids=true, method=geometry_method)
        operator_u = ntuple(i -> DiffusionOps(capacity_u[i]), N)
        operator_p = DiffusionOps(capacity_p)
        moving_fluid = Fluid(STmesh_u, capacity_u, operator_u,
                             STmesh_p, capacity_p, operator_p,
                             fluid.μ, fluid.ρ, fluid.fᵤ, fluid.fₚ)

        x_prev = copy(s.x)
        s.A, s.b = assemble_moving_stokes(moving_fluid, bc_tuple, cut_tuple, pressure_gauge;
                                          scheme=scheme, t_prev=t, t_next=t_next,
                                          Δt=Δt, x_prev=x_prev)
        solve_system!(s; method=method, algorithm=algorithm, kwargs...)
        push!(s.states, copy(s.x))
        push!(times, t_next)
        println("Solver Extremum : ", maximum(abs.(s.x)))
    end

    return times
end

#########################
# Assembly helpers
#########################

@inline function moving_stokes_weights(scheme::String)
    sw = uppercase(scheme)
    if sw == "CN"
        return psip_cn, psim_cn
    elseif sw == "BE"
        return psip_be, psim_be
    else
        error("Unknown moving Stokes scheme $(scheme). Supported: \"CN\" or \"BE\".")
    end
end

@inline function spatial_rows(mat::AbstractMatrix)
    rows = size(mat, 1)
    if rows % 2 == 0 && rows > 0
        return mat[1:rows÷2, :]
    else
        return mat
    end
end

@inline function spatial_cols(mat::AbstractMatrix)
    cols = size(mat, 2)
    if cols % 2 == 0 && cols > 0
        return mat[:, 1:cols÷2]
    else
        return mat
    end
end

@inline spatial_block(mat::AbstractMatrix) = spatial_rows(spatial_cols(mat))

@inline function spatial_vector(vec::AbstractVector, n::Int)
    len = length(vec)
    cutoff = len % 2 == 0 ? len ÷ 2 : len
    return vec[1:min(n, cutoff)]
end

function spacetime_volume_blocks(Acap::SparseMatrixCSC)
    rows = size(Acap, 1)
    rows % 2 == 0 || error("Space-time capacity must have even size, got $(rows).")
    half = rows ÷ 2
    return Acap[1:half, 1:half], Acap[half+1:end, half+1:end]
end

function spacetime_weight_mats(capacity::AbstractCapacity,
                               nu::Int,
                               psip::Function,
                               psim::Function)
    cap_index = length(capacity.A)
    Vn_1, Vn = spacetime_volume_blocks(capacity.A[cap_index])
    diag_Vn_1_full = diag(Vn_1)
    diag_Vn_full = diag(Vn)
    length(diag_Vn_1_full) >= nu || error("Velocity DOFs ($(nu)) exceed space-time entries ($(length(diag_Vn_1_full))).")
    length(diag_Vn_full)  >= nu || error("Velocity DOFs ($(nu)) exceed space-time entries ($(length(diag_Vn_full))).")
    diag_Vn_1 = diag_Vn_1_full[1:nu]
    diag_Vn   = diag_Vn_full[1:nu]
    Ψn1 = spdiagm(0 => psip.(diag_Vn, diag_Vn_1))
    Ψn  = spdiagm(0 => psim.(diag_Vn, diag_Vn_1))
    ΔV  = spdiagm(0 => diag_Vn_1 .- diag_Vn)
    return Ψn1, Ψn, ΔV
end

function assemble_moving_stokes(fluid::Fluid{N},
                                bc_u::NTuple{N,BorderConditions},
                                bc_cut::NTuple{N,AbstractBoundary},
                                pressure_gauge::AbstractPressureGauge;
                                scheme::String="BE",
                                t_prev::Float64,
                                t_next::Float64,
                                Δt::Float64,
                                x_prev::AbstractVector{<:Real}) where {N}
    θ = theta_from_scheme(scheme)
    if N == 1
        return assemble_moving_stokes1D(fluid, bc_u[1], bc_cut[1], pressure_gauge;
                                        scheme=scheme, θ=θ, t_prev=t_prev, t_next=t_next,
                                        Δt=Δt, x_prev=x_prev)
    elseif N == 2
        return assemble_moving_stokes2D(fluid, bc_u, bc_cut, pressure_gauge;
                                        scheme=scheme, θ=θ, t_prev=t_prev, t_next=t_next,
                                        Δt=Δt, x_prev=x_prev)
    else
        error("MovingStokesMono assembly is implemented for 1D and 2D velocities (got N=$(N)).")
    end
end

function moving_stokes1D_blocks(fluid::Fluid{1}, scheme::String)
    op_u = fluid.operator_u[1]
    cap_u = fluid.capacity_u[1]
    op_p = fluid.operator_p
    cap_p = fluid.capacity_p

    G = spatial_block(op_u.G)
    H = spatial_block(op_u.H)
    W = spatial_block(op_u.Wꜝ)
    V = spatial_block(op_u.V)
    nu = size(G, 2)

    psip, psim = moving_stokes_weights(scheme)
    Ψn1, Ψn, ΔV = spacetime_weight_mats(cap_u, nu, psip, psim)

    Iμ = spatial_block(build_I_D(op_u, fluid.μ, cap_u))
    Iρ = spatial_block(build_I_D(op_u, fluid.ρ, cap_u))
    WG_uG = W * G
    WG_uH = W * H
    visc_uω = Iμ * (G' * WG_uG) * Ψn1
    visc_uγ = Iμ * (G' * WG_uH) * Ψn1

    tie = Ψn1
    Vp = spatial_block(cap_p.V)
    np = size(Vp, 1)

    grad_full = op_p.G + op_p.H
    size(grad_full, 1) >= nu || error("Gradient rows $(size(grad_full,1)) < nu $(nu)")
    size(grad_full, 2) >= np || error("Gradient cols $(size(grad_full,2)) < np $(np)")
    grad = -grad_full[1:nu, 1:np]

    Gp_full = op_p.G
    Hp_full = op_p.H
    size(Gp_full, 1) >= nu || error("Gp rows $(size(Gp_full,1)) < nu $(nu)")
    size(Gp_full, 2) >= np || error("Gp cols $(size(Gp_full,2)) < np $(np)")
    size(Hp_full, 1) >= nu || error("Hp rows $(size(Hp_full,1)) < nu $(nu)")
    size(Hp_full, 2) >= np || error("Hp cols $(size(Hp_full,2)) < np $(np)")
    Gp = Gp_full[1:nu, 1:np]
    Hp = Hp_full[1:nu, 1:np]
    div_uω = -(Gp' + Hp')
    div_uγ = Hp'

    mass = Iρ * V

    return (; nu, np, visc_uω, visc_uγ, grad, div_uω, div_uγ,
            tie, V, Ψn, Ψn1, ΔV, op_u, cap_u, cap_p, Vp, mass)
end

function moving_stokes2D_blocks(fluid::Fluid{2}, scheme::String)
    ops_u = fluid.operator_u
    caps_u = fluid.capacity_u
    op_p = fluid.operator_p
    cap_p = fluid.capacity_p

    psip, psim = moving_stokes_weights(scheme)

    function component_data(op, cap)
        G = spatial_block(op.G)
        H = spatial_block(op.H)
        W = spatial_block(op.Wꜝ)
        V = spatial_block(op.V)
        nu = size(G, 2)
        Ψn1, Ψn, ΔV = spacetime_weight_mats(cap, nu, psip, psim)
        Iμ = spatial_block(build_I_D(op, fluid.μ, cap))
        Iρ = spatial_block(build_I_D(op, fluid.ρ, cap))
        visc_ω = Iμ * (G' * (W * G)) * Ψn1
        visc_γ = Iμ * (G' * (W * H)) * Ψn1
        mass = Iρ * V
        return (; G, H, W, V, nu, Ψn1, Ψn, ΔV, visc_ω, visc_γ, mass)
    end

    data_x = component_data(ops_u[1], caps_u[1])
    data_y = component_data(ops_u[2], caps_u[2])

    grad_full = spatial_rows(spatial_cols(op_p.G + op_p.H))
    Vp = spatial_block(cap_p.V)
    np = size(Vp, 1)
    total_rows = size(grad_full, 1)
    expected_rows = data_x.nu + data_y.nu
    total_rows >= expected_rows || error("Gradient rows $(total_rows) < velocity DOFs $(expected_rows)")
    size(grad_full, 2) >= np || error("Gradient cols $(size(grad_full, 2)) < np $(np)")

    x_rows = 1:data_x.nu
    y_rows = data_x.nu+1:data_x.nu+data_y.nu
    grad_x = -grad_full[x_rows, 1:np]
    grad_y = -grad_full[y_rows, 1:np]

    Gp_full = spatial_rows(spatial_cols(op_p.G))
    Hp_full = spatial_rows(spatial_cols(op_p.H))
    size(Gp_full, 1) >= expected_rows || error("Gp rows insufficient for velocity DOFs.")
    size(Hp_full, 1) >= expected_rows || error("Hp rows insufficient for velocity DOFs.")
    size(Gp_full, 2) >= np || error("Gp cols < np.")
    size(Hp_full, 2) >= np || error("Hp cols < np.")

    Gp_x = Gp_full[x_rows, 1:np]
    Hp_x = Hp_full[x_rows, 1:np]
    Gp_y = Gp_full[y_rows, 1:np]
    Hp_y = Hp_full[y_rows, 1:np]

    div_x_ω = -(Gp_x' + Hp_x')
    div_x_γ = Hp_x'
    div_y_ω = -(Gp_y' + Hp_y')
    div_y_γ = Hp_y'

    return (; nu_x = data_x.nu, nu_y = data_y.nu, np,
            visc_x_ω = data_x.visc_ω, visc_x_γ = data_x.visc_γ,
            visc_y_ω = data_y.visc_ω, visc_y_γ = data_y.visc_γ,
            grad_x, grad_y,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ,
            tie_x = data_x.Ψn1, tie_y = data_y.Ψn1,
            Ψn_x = data_x.Ψn, Ψn_y = data_y.Ψn,
            ΔV_x = data_x.ΔV, ΔV_y = data_y.ΔV,
            Vx = data_x.V, Vy = data_y.V,
            mass_x = data_x.mass, mass_y = data_y.mass,
            op_ux = ops_u[1], op_uy = ops_u[2],
            cap_ux = caps_u[1], cap_uy = caps_u[2],
            op_p, cap_p, Vp)
end

function assemble_moving_stokes1D(fluid::Fluid{1},
                                  bc_u::BorderConditions,
                                  bc_cut::AbstractBoundary,
                                  pressure_gauge::AbstractPressureGauge;
                                  scheme::String,
                                  θ::Float64,
                                  t_prev::Float64,
                                  t_next::Float64,
                                  Δt::Float64,
                                  x_prev::AbstractVector{<:Real})
    data = moving_stokes1D_blocks(fluid, scheme)
    nu, np = data.nu, data.np
    rows = 3 * nu
    cols = 2 * nu + np
    A = spzeros(Float64, rows, cols)

    mass_dt = (1.0 / Δt) * data.mass
    θc = 1.0 - θ

    Ndofs = 2 * nu + np
    x_prev_vec = normalize_prev_state(x_prev, Ndofs)
    u_prev_ω = view(x_prev_vec, 1:nu)
    u_prev_γ = view(x_prev_vec, nu+1:2nu)

    A[1:nu, 1:nu] = mass_dt + θ * data.visc_uω
    A[1:nu, nu+1:2nu] = θ * data.visc_uγ - data.ΔV
    A[1:nu, 2nu+1:2nu+np] = data.grad

    A[nu+1:2nu, nu+1:2nu] = data.tie
    A[2nu+1:3nu, 1:nu] = data.div_uω
    A[2nu+1:3nu, nu+1:2nu] = data.div_uγ

    f_prev_full = safe_build_source(data.op_u, fluid.fᵤ, data.cap_u, t_prev)
    f_next_full = safe_build_source(data.op_u, fluid.fᵤ, data.cap_u, t_next)
    f_prev = spatial_vector(f_prev_full, nu)
    f_next = spatial_vector(f_next_full, nu)
    weighted_f = θ .* f_next .+ θc .* f_prev
    load = data.V * (data.Ψn * weighted_f)

    rhs_mom = mass_dt * u_prev_ω
    rhs_mom .-= θc * (data.visc_uω * u_prev_ω + data.visc_uγ * u_prev_γ)
    rhs_mom .+= load

    g_full = safe_build_g(data.op_u, bc_cut, data.cap_u, t_next)
    g_vec = spatial_vector(g_full, nu)
    g_cut = data.tie * g_vec

    b = vcat(rhs_mom, g_cut, zeros(np))

    apply_velocity_dirichlet!(A, b, bc_u, fluid.mesh_u[1];
                              nu=nu, uω_offset=0, uγ_offset=nu, t=t_next)

    apply_pressure_gauge_moving!(A, b, pressure_gauge, data.cap_p;
                                 p_offset=2nu, np=np, row_start=2nu+1, Vp=data.Vp)

    return A, b
end

function assemble_moving_stokes2D(fluid::Fluid{2},
                                  bc_u::NTuple{2,BorderConditions},
                                  bc_cut::NTuple{2,AbstractBoundary},
                                  pressure_gauge::AbstractPressureGauge;
                                  scheme::String,
                                  θ::Float64,
                                  t_prev::Float64,
                                  t_next::Float64,
                                  Δt::Float64,
                                  x_prev::AbstractVector{<:Real})
    data = moving_stokes2D_blocks(fluid, scheme)
    nu_x = data.nu_x
    nu_y = data.nu_y
    np = data.np
    sum_nu = nu_x + nu_y

    rows = 2 * sum_nu + np
    cols = rows
    A = spzeros(Float64, rows, cols)

    mass_x_dt = (1.0 / Δt) * data.mass_x
    mass_y_dt = (1.0 / Δt) * data.mass_y
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

    Ndofs = 2 * sum_nu + np
    x_prev_vec = normalize_prev_state(x_prev, Ndofs)
    uωx_prev = view(x_prev_vec, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev_vec, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev_vec, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev_vec, off_uγy+1:off_uγy+nu_y)

    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = mass_x_dt + θ * data.visc_x_ω
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = θ * data.visc_x_γ - data.ΔV_x
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = mass_y_dt + θ * data.visc_y_ω
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = θ * data.visc_y_γ - data.ΔV_y
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ

    f_prev_x_full = safe_build_source(data.op_ux, fluid.fᵤ, data.cap_ux, t_prev)
    f_next_x_full = safe_build_source(data.op_ux, fluid.fᵤ, data.cap_ux, t_next)
    f_prev_x = spatial_vector(f_prev_x_full, nu_x)
    f_next_x = spatial_vector(f_next_x_full, nu_x)
    weighted_fx = θ .* f_next_x .+ θc .* f_prev_x
    load_x = data.Vx * (data.Ψn_x * weighted_fx)

    f_prev_y_full = safe_build_source(data.op_uy, fluid.fᵤ, data.cap_uy, t_prev)
    f_next_y_full = safe_build_source(data.op_uy, fluid.fᵤ, data.cap_uy, t_next)
    f_prev_y = spatial_vector(f_prev_y_full, nu_y)
    f_next_y = spatial_vector(f_next_y_full, nu_y)
    weighted_fy = θ .* f_next_y .+ θc .* f_prev_y
    load_y = data.Vy * (data.Ψn_y * weighted_fy)

    rhs_mom_x = mass_x_dt * uωx_prev
    rhs_mom_x .-= θc * (data.visc_x_ω * uωx_prev + data.visc_x_γ * uγx_prev)
    rhs_mom_x .+= load_x

    rhs_mom_y = mass_y_dt * uωy_prev
    rhs_mom_y .-= θc * (data.visc_y_ω * uωy_prev + data.visc_y_γ * uγy_prev)
    rhs_mom_y .+= load_y

    g_x_full = safe_build_g(data.op_ux, bc_cut[1], data.cap_ux, t_next)
    g_y_full = safe_build_g(data.op_uy, bc_cut[2], data.cap_uy, t_next)
    g_x = spatial_vector(g_x_full, nu_x)
    g_y = spatial_vector(g_y_full, nu_y)
    g_cut_x = data.tie_x * g_x
    g_cut_y = data.tie_y * g_y

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y, zeros(np))

    apply_velocity_dirichlet_2D!(A, b, bc_u[1], bc_u[2], fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 t=t_next)

    apply_pressure_gauge_moving!(A, b, pressure_gauge, data.cap_p;
                                 p_offset=off_p, np=np, row_start=row_con+1, Vp=data.Vp)

    return A, b
end

function apply_pressure_gauge_moving!(A::SparseMatrixCSC{Float64, Int}, b,
                                      gauge::AbstractPressureGauge,
                                      capacity_p::AbstractCapacity;
                                      p_offset::Int, np::Int, row_start::Int,
                                      Vp::SparseMatrixCSC{Float64, Int})
    diagV = diag(Vp)
    if gauge isa PinPressureGauge
        idx = gauge.index
        if idx === nothing
            tol = 1e-12
            idx = findfirst(x -> x > tol, diagV)
            idx === nothing && (idx = 1)
        end
        1 ≤ idx ≤ np || error("PinPressureGauge index $(idx) outside valid range 1:$(np)")
        row = row_start + idx - 1
        col = p_offset + idx
        enforce_dirichlet!(A, b, row, col, 0.0)
    elseif gauge isa MeanPressureGauge
        weights = copy(diagV)
        if isempty(weights)
            error("MeanPressureGauge requires at least one pressure DOF")
        end
        if all(isapprox.(weights, 0.0; atol=1e-12))
            weights .= 1.0
        end
        total = sum(weights)
        total == 0.0 && (weights .= 1.0; total = sum(weights))
        weights ./= total
        row = row_start
        A[row, :] .= 0.0
        for j in 1:np
            A[row, p_offset + j] = weights[j]
        end
        b[row] = 0.0
    else
        error("Unknown pressure gauge type $(typeof(gauge))")
    end
    return nothing
end
