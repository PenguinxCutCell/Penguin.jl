# Moving - Stokes - Unsteady - Monophasic
"""
    MovingStokesUnsteadyMono

Solver for unsteady Stokes equations with prescribed moving geometry.
Based on StokesMono and MovingDiffusionUnsteadyMono patterns.
Uses SpaceTimeMesh with body as (x, y, t) for 2D problems.

Each velocity grid (u, v) and pressure grid (p) will have their own SpaceTimeMesh.
The matrix implementation accounts for V^{n+1} and V^{n} capacities and
the -(Vn_1 - Vn) term for each velocity component.
"""
mutable struct MovingStokesUnsteadyMono{N}
    fluid::Fluid{N}
    bc_u::NTuple{N, BorderConditions}
    pressure_gauge::AbstractPressureGauge
    bc_cut::NTuple{N, AbstractBoundary}

    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    x::Vector{Float64}
    states::Vector{Vector{Float64}}
    times::Vector{Float64}
end

# Normalize cut-cell boundary conditions to one per velocity component
normalize_cut_bc(bc_cut::AbstractBoundary, N::Int) = ntuple(_ -> bc_cut, N)
normalize_cut_bc(bc_cut::NTuple{N, AbstractBoundary}, ::Int) where {N} = bc_cut
function normalize_cut_bc(bc_cut, N::Int)
    throw(ArgumentError("bc_cut must be an AbstractBoundary or NTuple{$N,AbstractBoundary}; got $(typeof(bc_cut))"))
end

# Helper functions for time integration weighting (same as diffusion solver)
function psip_stokes_cn(args::Vararg{T,2}) where {T<:Real}
    if all(iszero, args)
        0.0
    elseif all(!iszero, args)
        0.5
    elseif iszero(args[1]) && !iszero(args[2])
        0.5
    elseif !iszero(args[1]) && iszero(args[2])
        1.0
    else
        0.0
    end
end

function psim_stokes_cn(args::Vararg{T,2}) where {T<:Real}
    if all(iszero, args)
        0.0
    elseif all(!iszero, args)
        0.5
    elseif iszero(args[1]) && !iszero(args[2])  # Fresh
        0.5
    elseif !iszero(args[1]) && iszero(args[2])  # Dead
        0.0
    else
        0.0
    end
end

function psip_stokes_be(args::Vararg{T,2}) where {T<:Real}
    if all(iszero, args)
        0.0
    elseif all(!iszero, args)
        1.0
    else
        1.0
    end
end

function psim_stokes_be(args::Vararg{T,2}) where {T<:Real}
    0.0
end

@inline function spacetime_spatial_size(dims::NTuple{N,Int}) where {N}
    nt = dims[end]
    total = prod(dims)
    spatial = div(total, nt)
    return spatial, nt, total
end

@inline function time_block_indices(spatial::Int, nt::Int, time_idx::Int=1)
    (1 <= time_idx <= nt) || throw(ArgumentError("time_idx must be in 1:$nt, got $time_idx"))
    start = (time_idx - 1) * spatial + 1
    return start:start + spatial - 1
end

"""
    MovingStokesUnsteadyMono(fluid, bc_u, pressure_gauge, bc_cut; scheme=:BE, x0=zeros(0))

Create a solver for the unsteady Stokes problem with prescribed moving geometry.

# Arguments
- `fluid::Fluid{N}`: The fluid object containing capacities and operators.
- `bc_u::NTuple{N, BorderConditions}`: The border conditions for velocity components.
- `pressure_gauge::AbstractPressureGauge`: The pressure gauge (pin or mean).
- `bc_cut`: The interface/cut-cell boundary conditions for velocity, either a single `AbstractBoundary` (applied to all components) or an `NTuple{N,AbstractBoundary}` for component-wise values.
- `scheme::Symbol`: Time integration scheme (:CN or :BE).
- `x0`: Initial state vector (optional).
"""
function MovingStokesUnsteadyMono(fluid::Fluid{N},
                                   bc_u::NTuple{N, BorderConditions},
                                   pressure_gauge::AbstractPressureGauge,
                                   bc_cut::Union{AbstractBoundary, NTuple{N, AbstractBoundary}};
                                   scheme::Symbol=:BE,
                                   x0=zeros(0)) where {N}
    println("Solver Creation:")
    println("- Moving problem")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Stokes problem")

    # Calculate total DOFs
    nu_components = ntuple(i -> prod(fluid.operator_u[i].size), N)
    np = prod(fluid.operator_p.size)
    Ntot = 2 * sum(nu_components) + np

    x_init = length(x0) == Ntot ? x0 : zeros(Ntot)

    # Initialize with empty matrices (will be assembled in solve)
    A = spzeros(Float64, Ntot, Ntot)
    b = zeros(Ntot)

    cut_bc = normalize_cut_bc(bc_cut, N)

    s = MovingStokesUnsteadyMono{N}(fluid, bc_u, pressure_gauge, cut_bc,
                                     A, b, x_init, Vector{Float64}[], Float64[])
    return s
end

# Constructor for 2D with separate bc arguments
function MovingStokesUnsteadyMono(fluid::Fluid{2},
                                   bc_ux::BorderConditions,
                                   bc_uy::BorderConditions,
                                   pressure_gauge::AbstractPressureGauge,
                                   bc_cut::Union{AbstractBoundary, NTuple{2, AbstractBoundary}};
                                   scheme::Symbol=:BE,
                                   x0=zeros(0))
    return MovingStokesUnsteadyMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut;
                                     scheme=scheme, x0=x0)
end



"""
    stokes2D_moving_blocks(fluid, ops_u, caps_u, op_p, cap_p)

Build operator blocks for moving 2D Stokes problem using SpaceTime operators.
Similar to stokes2D_blocks but extracts V^{n+1} and V^{n} from capacity.A.
"""
function stokes2D_moving_blocks(fluid::Fluid{2}, 
                                 ops_u::Tuple{DiffusionOps, DiffusionOps},
                                 caps_u::Tuple{Capacity, Capacity},
                                 op_p::DiffusionOps,
                                 cap_p::Capacity,
                                 scheme::Symbol)
    # Operator sizes (for space-time mesh, dims = (nx, ny, nt))
    dims_ux = ops_u[1].size
    dims_uy = ops_u[2].size
    dims_p = op_p.size

    len_dims = length(dims_ux)
    cap_index = len_dims  # 3 for 2D (nx, ny, nt)

    # Extract spatial sizes
    if len_dims == 3
        nu_x, nt_ux, total_ux = spacetime_spatial_size(dims_ux)
        nu_y, nt_uy, total_uy = spacetime_spatial_size(dims_uy)
        np, nt_p, total_p = spacetime_spatial_size(dims_p)
    else
        error("Moving Stokes 2D requires 3D space-time mesh (nx, ny, nt)")
    end

    # Extract V^{n+1} and V^{n} for each velocity component
    Vn_1_ux = caps_u[1].A[cap_index][1:end÷2, 1:end÷2]
    Vn_ux = caps_u[1].A[cap_index][end÷2+1:end, end÷2+1:end]
    Vn_1_uy = caps_u[2].A[cap_index][1:end÷2, 1:end÷2]
    Vn_uy = caps_u[2].A[cap_index][end÷2+1:end, end÷2+1:end]
    Vn_1_p = cap_p.A[cap_index][1:end÷2, 1:end÷2]
    Vn_p = cap_p.A[cap_index][end÷2+1:end, end÷2+1:end]

    # Time integration weighting
    if scheme == :CN
        psip, psim = psip_stokes_cn, psim_stokes_cn
    else
        psip, psim = psip_stokes_be, psim_stokes_be
    end

    Ψn1_ux = Diagonal(psip.(Vn_ux, Vn_1_ux))
    Ψn1_uy = Diagonal(psip.(Vn_uy, Vn_1_uy))
    Ψn1_p = Diagonal(psip.(Vn_p, Vn_1_p))
    Ψn_ux = Diagonal(psim.(Vn_ux, Vn_1_ux))
    Ψn_uy = Diagonal(psim.(Vn_uy, Vn_1_uy))
    Ψn_p = Diagonal(psim.(Vn_p, Vn_1_p))

    # Extract operator sub-blocks at t^{n+1} (drop temporal derivative block)
    cols_ux = time_block_indices(nu_x, nt_ux)
    rows_ux_x = cols_ux
    rows_ux_y = total_ux .+ cols_ux

    cols_uy = time_block_indices(nu_y, nt_uy)
    rows_uy_x = cols_uy
    rows_uy_y = total_uy .+ cols_uy

    cols_p = time_block_indices(np, nt_p)
    total_grad_rows = size(op_p.G, 1)
    total_grad_needed = 2 * total_p  # x- and y-gradient blocks; drop temporal block
    total_grad_rows >= total_grad_needed || error("Pressure gradient rows ($total_grad_rows) must include x and y blocks ($total_grad_needed) in 2D moving Stokes.")
    rows_per_time = div(total_p, nt_p)
    rows_per_time == np || error("Pressure gradient rows per time ($rows_per_time) must equal pressure spatial DOFs ($np) in 2D moving Stokes.")
    rows_block = time_block_indices(rows_per_time, nt_p)  # pick the t^{n+1} slice
    rows_p_x = rows_block[1:nu_x]
    rows_p_y = total_p .+ rows_block[1:nu_y]

    W_ux = blockdiag(ops_u[1].Wꜝ[rows_ux_x, rows_ux_x], ops_u[1].Wꜝ[rows_ux_y, rows_ux_y])
    G_ux = vcat(ops_u[1].G[rows_ux_x, cols_ux],
                ops_u[1].G[rows_ux_y, cols_ux])
    H_ux = vcat(ops_u[1].H[rows_ux_x, cols_ux],
                ops_u[1].H[rows_ux_y, cols_ux])
    V_ux = ops_u[1].V[cols_ux, cols_ux]

    W_uy = blockdiag(ops_u[2].Wꜝ[rows_uy_x, rows_uy_x], ops_u[2].Wꜝ[rows_uy_y, rows_uy_y])
    G_uy = vcat(ops_u[2].G[rows_uy_x, cols_uy],
                ops_u[2].G[rows_uy_y, cols_uy])
    H_uy = vcat(ops_u[2].H[rows_uy_x, cols_uy],
                ops_u[2].H[rows_uy_y, cols_uy])
    V_uy = ops_u[2].V[cols_uy, cols_uy]

    # Pressure operators (space-time) on the t^{n+1} slice
    G_p_full_x = op_p.G[rows_p_x, cols_p]
    H_p_full_x = op_p.H[rows_p_x, cols_p]
    G_p_full_y = op_p.G[rows_p_y, cols_p]
    H_p_full_y = op_p.H[rows_p_y, cols_p]

    # Build viscosity operators
    μ = fluid.μ
    Iμ_ux = build_I_D(ops_u[1], μ, caps_u[1])
    Iμ_uy = build_I_D(ops_u[2], μ, caps_u[2])
    Iμ_ux = Iμ_ux[cols_ux, cols_ux]
    Iμ_uy = Iμ_uy[cols_uy, cols_uy]

    # Viscous blocks
    visc_x_ω = Iμ_ux * G_ux' * W_ux * G_ux
    visc_x_γ = Iμ_ux * G_ux' * W_ux * H_ux
    visc_y_ω = Iμ_uy * G_uy' * W_uy * G_uy
    visc_y_γ = Iμ_uy * G_uy' * W_uy * H_uy

    # Pressure gradient (need to extract x and y rows)
    grad_x = -(G_p_full_x + H_p_full_x)
    grad_y = -(G_p_full_y + H_p_full_y)

    # Divergence operators
    Gp_x = G_p_full_x
    Hp_x = H_p_full_x
    Gp_y = G_p_full_y
    Hp_y = H_p_full_y

    div_x_ω = -(Gp_x' + Hp_x')
    div_x_γ = Hp_x'
    div_y_ω = -(Gp_y' + Hp_y')
    div_y_γ = Hp_y'

    # Mass matrices for density
    ρ = fluid.ρ
    mass_x = build_I_D(ops_u[1], ρ, caps_u[1])
    mass_y = build_I_D(ops_u[2], ρ, caps_u[2])
    mass_x = mass_x[cols_ux, cols_ux] * V_ux
    mass_y = mass_y[cols_uy, cols_uy] * V_uy

    return (; nu_x, nu_y, np,
            op_ux=ops_u[1], op_uy=ops_u[2], op_p,
            cap_ux=caps_u[1], cap_uy=caps_u[2], cap_p,
            visc_x_ω, visc_x_γ, visc_y_ω, visc_y_γ,
            grad_x, grad_y,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ,
            tie_x=I(nu_x), tie_y=I(nu_y),
            mass_x, mass_y,
            Vx=V_ux, Vy=V_uy,
            Vn_1_ux, Vn_ux, Vn_1_uy, Vn_uy,
            Ψn1_ux, Ψn1_uy, Ψn_ux, Ψn_uy, Ψn1_p, Ψn_p)
end



"""
    assemble_stokes2D_moving!(s, data, Δt, x_prev, t_prev, t_next, θ, mesh)

Assemble the system matrix and RHS for moving 2D Stokes.
Accounts for V^{n+1}, V^{n}, and -(Vn_1 - Vn) terms.

For the moving case, similar to MovingDiffusionUnsteadyMono:
- LHS block (ω,ω): Vn_1 + θ * visc_ω * Ψn1
- LHS block (ω,γ): -(Vn_1 - Vn) + θ * visc_γ * Ψn1  
- RHS: Vn * u_prev + source terms
"""
function assemble_stokes2D_moving!(s::MovingStokesUnsteadyMono{2}, data, Δt::Float64,
                                    x_prev::AbstractVector{<:Real},
                                    t_prev::Float64, t_next::Float64,
                                    θ::Float64, mesh::AbstractMesh)
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    θc = 1.0 - θ

    # Column offsets
    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_p = 2 * sum_nu

    # Row offsets
    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_con = 2 * sum_nu

    # Following the pattern from MovingDiffusionUnsteadyMono:
    # LHS block (ω,ω): Vn_1 + θ * visc_ω * Ψn1
    # LHS block (ω,γ): -(Vn_1 - Vn) + θ * visc_γ * Ψn1
    # Note: visc terms are already built as G' W G (positive definite form)
    
    # Momentum x-component rows
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = data.Vn_1_ux + θ * data.visc_x_ω * data.Ψn1_ux
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = -(data.Vn_1_ux - data.Vn_ux) + θ * data.visc_x_γ * data.Ψn1_ux
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np] = data.grad_x

    # Tie x rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x

    # Momentum y-component rows
    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = data.Vn_1_uy + θ * data.visc_y_ω * data.Ψn1_uy
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = -(data.Vn_1_uy - data.Vn_uy) + θ * data.visc_y_γ * data.Ψn1_uy
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np] = data.grad_y

    # Tie y rows
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    # Continuity rows
    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ

    # Extract previous state
    uωx_prev = view(x_prev, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev, off_uγy+1:off_uγy+nu_y)

    # Build source terms
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

    # RHS momentum with previous time level contribution
    # Following the diffusion pattern: b1 = (Vn - θc * visc_ω * Ψn) * u_prev - θc * visc_γ * Ψn * uγ_prev + source
    rhs_mom_x = (data.Vn_ux - θc * data.visc_x_ω * data.Ψn_ux) * uωx_prev
    rhs_mom_x .-= θc * data.visc_x_γ * data.Ψn_ux * uγx_prev
    rhs_mom_x .+= load_x

    rhs_mom_y = (data.Vn_uy - θc * data.visc_y_ω * data.Ψn_uy) * uωy_prev
    rhs_mom_y .-= θc * data.visc_y_γ * data.Ψn_uy * uγy_prev
    rhs_mom_y .+= load_y

    # Interface conditions (cut-cell BC at t_next)
    g_cut_x = safe_build_g(data.op_ux, s.bc_cut[1], data.cap_ux, t_next)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut[2], data.cap_uy, t_next)
    g_cut_x = g_cut_x[1:end÷2]
    g_cut_y = g_cut_y[1:end÷2]

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y, zeros(np))

    # Apply velocity Dirichlet BCs
    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                  nu_x=nu_x, nu_y=nu_y,
                                  uωx_off=off_uωx, uγx_off=off_uγx,
                                  uωy_off=off_uωy, uγy_off=off_uγy,
                                  row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                  row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                  t=t_next)

    # Apply pressure gauge
    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end




"""
    solve_MovingStokesUnsteadyMono!(s, body, mesh, Δt, Tₛ, Tₑ, bc_b_u, bc_cut; scheme=:BE, kwargs...)

Solve the unsteady Stokes problem with prescribed moving geometry.

# Arguments
- `s::MovingStokesUnsteadyMono`: The solver object.
- `body::Function`: Body function (x, y, t) defining the geometry.
- `mesh::AbstractMesh`: The base spatial mesh.
- `Δt::Float64`: Time step.
- `Tₛ::Float64`: Start time.
- `Tₑ::Float64`: End time.
- `bc_b_u`: Border conditions for velocity (tuple for 2D).
- `bc_cut`: Cut-cell/interface boundary condition (single value or tuple per component).
- `scheme::Symbol`: Time scheme (:CN or :BE).
- `kwargs...`: Additional arguments (e.g., geometry_method).
"""
function solve_MovingStokesUnsteadyMono!(s::MovingStokesUnsteadyMono{2},
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
    println("- Stokes problem")

    θ = scheme == :CN ? 0.5 : 1.0

    # Allow boundary conditions to be provided at solve-time
    s.bc_u = bc_b_u
    s.bc_cut = normalize_cut_bc(bc_cut, 2)

    # Create mesh_u tuple (staggered grids)
    dx = mesh.nodes[1][2] - mesh.nodes[1][1]
    dy = mesh.nodes[2][2] - mesh.nodes[2][1]
    mesh_ux = Penguin.Mesh((length(mesh.nodes[1])-1, length(mesh.nodes[2])-1),
                            (mesh.nodes[1][end] - mesh.nodes[1][1], mesh.nodes[2][end] - mesh.nodes[2][1]),
                            (mesh.nodes[1][1] - 0.5*dx, mesh.nodes[2][1]))
    mesh_uy = Penguin.Mesh((length(mesh.nodes[1])-1, length(mesh.nodes[2])-1),
                            (mesh.nodes[1][end] - mesh.nodes[1][1], mesh.nodes[2][end] - mesh.nodes[2][1]),
                            (mesh.nodes[1][1], mesh.nodes[2][1] - 0.5*dy))

    # Store times and initial state
    t = Tₛ
    push!(s.times, t)
    push!(s.states, copy(s.x))

    println("Time : $(t)")

    # Time loop
    while t < Tₑ - 1e-12 * max(1.0, Tₑ)
        dt_step = min(Δt, Tₑ - t)
        t_next = t + dt_step

        println("Time : $(t_next)")

        # Create SpaceTime meshes for this time interval
        STmesh_ux = Penguin.SpaceTimeMesh(mesh_ux, [t, t_next], tag=mesh.tag)
        STmesh_uy = Penguin.SpaceTimeMesh(mesh_uy, [t, t_next], tag=mesh.tag)
        STmesh_p = Penguin.SpaceTimeMesh(mesh, [t, t_next], tag=mesh.tag)

        # Build capacities with moving body
        capacity_ux = Capacity(body, STmesh_ux; method=geometry_method, kwargs...)
        capacity_uy = Capacity(body, STmesh_uy; method=geometry_method, kwargs...)
        capacity_p = Capacity(body, STmesh_p; method=geometry_method, kwargs...)

        # Build operators
        operator_ux = DiffusionOps(capacity_ux)
        operator_uy = DiffusionOps(capacity_uy)
        operator_p = DiffusionOps(capacity_p)

        # Build blocks for this time step
        data = stokes2D_moving_blocks(s.fluid,
                                       (operator_ux, operator_uy),
                                       (capacity_ux, capacity_uy),
                                       operator_p, capacity_p,
                                       scheme)

        # Assemble system
        x_prev = s.x
        assemble_stokes2D_moving!(s, data, dt_step, x_prev, t, t_next, θ, mesh)

        # Solve system
        solve_moving_stokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        # Store result
        push!(s.times, t_next)
        push!(s.states, copy(s.x))
        println("Solver Extremum : ", maximum(abs.(s.x)))

        t = t_next
    end

    return s.times, s.states
end


"""
    solve_moving_stokes_linear_system!(s; method, algorithm, kwargs...)

Solve the linear system for moving Stokes.
"""
function solve_moving_stokes_linear_system!(s::MovingStokesUnsteadyMono; method=Base.:\, algorithm=nothing, kwargs...)
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

"""
    compute_navierstokes_force_diagnostics(s::MovingStokesUnsteadyMono, data)

Volume-integrated force diagnostics for the moving Stokes solver on a single
time slab. `data` must be the output of `stokes2D_moving_blocks` built with the same
space–time capacities used to assemble the step.
"""
function compute_navierstokes_force_diagnostics(s::MovingStokesUnsteadyMono{N}, data) where {N}
    nu_components = N == 1 ? (data.nu,) :
                    N == 2 ? (data.nu_x, data.nu_y) :
                    N == 3 ? (data.nu_x, data.nu_y, data.nu_z) :
                    error("Force diagnostics currently implemented for 2D (got N=$(N)).")

    total_velocity_dofs = 2 * sum(nu_components)
    np = data.np
    length(s.x) == total_velocity_dofs + np || error("State vector length mismatch: expected $(total_velocity_dofs + np), got $(length(s.x)).")
    pω = Vector{Float64}(view(s.x, total_velocity_dofs + 1:total_velocity_dofs + np))

    grads = Vector{SparseMatrixCSC{Float64,Int}}(undef, N)
    ops = Vector{DiffusionOps}(undef, N)
    caps = Vector{Capacity}(undef, N)
    if N == 1
        grads[1] = data.grad_x
        ops[1] = data.op_ux
        caps[1] = data.cap_ux
    elseif N == 2
        grads[1] = data.grad_x
        grads[2] = data.grad_y
        ops[1] = data.op_ux
        ops[2] = data.op_uy
        caps[1] = data.cap_ux
        caps[2] = data.cap_uy
    elseif N == 3
        grads[1] = data.grad_x
        grads[2] = data.grad_y
        grads[3] = data.grad_z
        ops[1] = data.op_ux
        ops[2] = data.op_uy
        ops[3] = data.op_uz
        caps[1] = data.cap_ux
        caps[2] = data.cap_uy
        caps[3] = data.cap_uz
    end

    g_p = Vector{Vector{Float64}}(undef, N)
    L_u = Vector{Vector{Float64}}(undef, N)
    pressure_part = Vector{Vector{Float64}}(undef, N)
    viscous_part = Vector{Vector{Float64}}(undef, N)
    force_density = Vector{Vector{Float64}}(undef, N)

    integrated_pressure = zeros(Float64, N)
    integrated_viscous = zeros(Float64, N)
    integrated_force = zeros(Float64, N)

    offset = 0
    for α in 1:N
        nu = nu_components[α]
        uω = Vector{Float64}(view(s.x, offset + 1:offset + nu))
        uγ = Vector{Float64}(view(s.x, offset + nu + 1:offset + 2nu))
        offset += 2nu

        grad = grads[α]
        gp_vec = -Vector{Float64}(grad * pω)
        pressure_vec = -gp_vec

        op = ops[α]
        cap = caps[α]
        G = op.G[1:end÷2, 1:end÷2]
        H = op.H[1:end÷2, 1:end÷2]
        W = op.Wꜝ[1:end÷2, 1:end÷2]
        Iμ = build_I_D(op, s.fluid.μ, cap)
        Iμ = Iμ[1:end÷2, 1:end÷2]

        G_u = Vector{Float64}(G * uω)
        H_u = if size(H, 2) == 0
            zeros(Float64, size(G_u))
        else
            Vector{Float64}(H * uγ)
        end
        mixed = Vector{Float64}(W * (G_u + H_u))
        Lu_vec = Vector{Float64}(G' * mixed)
        visc_vec = Vector{Float64}(Iμ * Lu_vec)

        force_vec = pressure_vec .+ visc_vec

        g_p[α] = gp_vec
        L_u[α] = Lu_vec
        pressure_part[α] = pressure_vec
        viscous_part[α] = visc_vec
        force_density[α] = force_vec

        integrated_pressure[α] = sum(pressure_vec)
        integrated_viscous[α] = sum(visc_vec)
        integrated_force[α] = sum(force_vec)
    end

    return (; g_p=Tuple(g_p),
            L_u=Tuple(L_u),
            pressure=Tuple(pressure_part),
            viscous=Tuple(viscous_part),
            force_density=Tuple(force_density),
            integrated_pressure=integrated_pressure,
            integrated_viscous=integrated_viscous,
            integrated_force=integrated_force)
end
