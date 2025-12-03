abstract type CouplingStrategy end

"""
    PassiveCoupling()

One-way coupling: advance Navier–Stokes first, then transport the scalar with
the new velocity field. No feedback from the scalar to the momentum equations.
"""
struct PassiveCoupling <: CouplingStrategy end

"""
    PicardCoupling(; tol_T=1e-6, tol_U=1e-6, maxiter=5, relaxation=1.0)

Two-way coupling via fixed-point (Picard) iterations within each time step.
Velocity is solved with buoyancy from the previous scalar iterate, then the
scalar is updated with the new velocity field. Iterates repeat until the scalar
and optionally the velocity changes drop below the prescribed tolerances.
"""
struct PicardCoupling <: CouplingStrategy
    tol_T::Float64
    tol_U::Float64
    maxiter::Int
    relaxation::Float64
end

PicardCoupling(; tol_T::Real=1e-6, tol_U::Real=1e-6, maxiter::Int=5, relaxation::Real=1.0) =
    PicardCoupling(float(tol_T), float(tol_U), maxiter, float(relaxation))

"""
    MonolithicCoupling(; tol=1e-8, maxiter=12, damping=1.0, verbose=false)

Fully coupled Newton solve where velocity, pressure, and scalar unknowns are
solved simultaneously at every time step. The Jacobian includes buoyancy
feedback and the sensitivity of the scalar equation to the velocity field.
"""
struct MonolithicCoupling <: CouplingStrategy
    tol::Float64
    maxiter::Int
    damping::Float64
    verbose::Bool
end

MonolithicCoupling(; tol::Real=1e-8, maxiter::Int=12, damping::Real=1.0, verbose::Bool=false) =
    MonolithicCoupling(float(tol), maxiter, float(damping), verbose)

@inline function _nearest_index(vec::AbstractVector{<:Real}, val::Real)
    idx = searchsortedfirst(vec, val)
    if idx <= 1
        return 1
    elseif idx > length(vec)
        return length(vec)
    else
        prev_val = vec[idx - 1]
        curr_val = vec[idx]
        return abs(val - prev_val) <= abs(curr_val - val) ? idx - 1 : idx
    end
end

function _build_projection_to_scalar(cap_scalar::Capacity{2}, cap_velocity::Capacity{2})
    xs_scalar = cap_scalar.mesh.nodes[1]
    ys_scalar = cap_scalar.mesh.nodes[2]
    xs_velocity = cap_velocity.mesh.nodes[1]
    ys_velocity = cap_velocity.mesh.nodes[2]

    Nx_s, Ny_s = length(xs_scalar), length(ys_scalar)
    Nx_v, Ny_v = length(xs_velocity), length(ys_velocity)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for j in 1:Ny_s
        y = ys_scalar[j]
        iy = _nearest_index(ys_velocity, y)
        for i in 1:Nx_s
            x = xs_scalar[i]
            ix = _nearest_index(xs_velocity, x)
            row = i + (j - 1) * Nx_s
            col = ix + (iy - 1) * Nx_v
            push!(rows, row)
            push!(cols, col)
            push!(vals, 1.0)
        end
    end

    return sparse(rows, cols, vals, Nx_s * Ny_s, Nx_v * Ny_v)
end

@inline function _select_bulk_block(n::Int)
    return sparse(collect(1:n), collect(1:n), ones(Float64, n), n, 2n)
end

function _collect_dirichlet_rows(bc::BorderConditions, mesh::AbstractMesh)
    rows = Int[]
    for (ci, _) in mesh.tag.border_cells
        key = classify_boundary_cell_fast(ci, mesh)
        cond = get(bc.borders, key, nothing)
        cond isa Dirichlet || continue
        push!(rows, cell_to_index(mesh, ci))
    end
    return sort(unique(rows))
end

@inline function _scheme_string_pretty(scheme::Symbol)
    s = lowercase(String(scheme))
    if s in ("cn", "crank_nicolson", "cranknicolson")
        return "CN"
    elseif s in ("be", "backward_euler", "implicit_euler")
        return "BE"
    else
        error("Unsupported time scheme $(scheme). Use :CN or :BE.")
    end
end


function build_temperature_interpolation(temp_cap::Capacity{2},
                                         vel_cap::Capacity{2})
    coords_temp = temp_cap.C_ω
    coords_vel = vel_cap.C_ω

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (i, pos) in enumerate(coords_vel)
        best_idx = 1
        best_dist = Inf
        for (j, center) in enumerate(coords_temp)
            dx = pos[1] - center[1]
            dy = pos[2] - center[2]
            dist = dx * dx + dy * dy
            if dist < best_dist
                best_dist = dist
                best_idx = j
            end
        end
        push!(rows, i)
        push!(cols, best_idx)
        push!(vals, 1.0)
    end

    return sparse(rows, cols, vals, length(coords_vel), length(coords_temp))
end


mutable struct NavierStokesScalarCoupler{Mode<:CouplingStrategy}
    momentum::NavierStokesMono{2}
    scalar_capacity::Capacity{2}
    diffusivity::Union{Float64,Function}
    scalar_source::Function
    bc_scalar::BorderConditions
    bc_scalar_cut::AbstractBoundary
    strategy::Mode
    β::Float64
    gravity::SVector{2,Float64}
    T_ref::Float64
    scalar_state::Vector{Float64}
    prev_scalar_state::Vector{Float64}
    velocity_state::Vector{Float64}
    prev_velocity_state::Vector{Float64}
    p_half::Vector{Float64}
    conv_prev::Union{Nothing,NTuple{2,Vector{Float64}}}
    navier_data::NamedTuple
    scalar_nodes::NTuple{2,Vector{Float64}}
    select_Tω::SparseMatrixCSC{Float64,Int}
    dirichlet_rows::Vector{Int}
    proj_uω_to_scalar::NTuple{2,SparseMatrixCSC{Float64,Int}}
    interp_T_to_velocity::NTuple{2,SparseMatrixCSC{Float64,Int}}
    scalar_D_p::NTuple{2,SparseMatrixCSC{Float64,Int}}
    scalar_S_m::NTuple{2,SparseMatrixCSC{Float64,Int}}
    scalar_SmA::NTuple{2,SparseMatrixCSC{Float64,Int}}
    buoyancy_matrix::SparseMatrixCSC{Float64,Int}
    buoyancy_offset::Vector{Float64}
    times::Vector{Float64}
    velocity_states::Vector{Vector{Float64}}
    scalar_states::Vector{Vector{Float64}}
    store_states::Bool
end

function NavierStokesScalarCoupler(momentum::NavierStokesMono{2},
                                   scalar_capacity::Capacity{2},
                                   κ::Union{Float64,Function},
                                   scalar_source::Function,
                                   bc_scalar::BorderConditions,
                                   bc_scalar_cut::AbstractBoundary;
                                   strategy::CouplingStrategy=PassiveCoupling(),
                                   β::Real=0.0,
                                   gravity::NTuple{2,Real}=(0.0, -1.0),
                                   T_ref::Real=0.0,
                                   T0::Union{Nothing,Vector{Float64}}=nothing,
                                   store_states::Bool=true)
    data = navierstokes2D_blocks(momentum)
    nu_x, nu_y, np = data.nu_x, data.nu_y, data.np
    N_vel = 2 * (nu_x + nu_y) + np

    velocity_state = length(momentum.x) == N_vel ? copy(momentum.x) : zeros(Float64, N_vel)
    prev_velocity_state = copy(velocity_state)

    p_half = zeros(Float64, np)
    if length(momentum.x) == N_vel && !isempty(momentum.x)
        p_half .= momentum.x[2 * (nu_x + nu_y) + 1:end]
    end

    conv_prev = momentum.prev_conv
    if conv_prev !== nothing && length(conv_prev) != 2
        conv_prev = nothing
    end

    nodes_scalar = ntuple(i -> copy(scalar_capacity.mesh.nodes[i]), 2)
    Nx_T = length(nodes_scalar[1])
    Ny_T = length(nodes_scalar[2])
    N_scalar = Nx_T * Ny_T

    scalar_state = if T0 === nothing
        zeros(Float64, 2 * N_scalar)
    else
        length(T0) == 2 * N_scalar || error("Initial scalar vector must have length $(2 * N_scalar).")
        copy(T0)
    end
    prev_scalar_state = copy(scalar_state)

    proj_x = _build_projection_to_scalar(scalar_capacity, momentum.fluid.capacity_u[1])
    proj_y = _build_projection_to_scalar(scalar_capacity, momentum.fluid.capacity_u[2])
    interp_T_to_ux = build_temperature_interpolation(scalar_capacity, momentum.fluid.capacity_u[1])
    interp_T_to_uy = build_temperature_interpolation(scalar_capacity, momentum.fluid.capacity_u[2])

    _, _, _, _, _, D_p, S_m, _ = compute_base_operators(scalar_capacity)
    SmA = ntuple(i -> S_m[Int(i)] * scalar_capacity.A[Int(i)], 2)

    select_Tω = _select_bulk_block(N_scalar)
    dirichlet_rows = _collect_dirichlet_rows(bc_scalar, scalar_capacity.mesh)

    ρ = momentum.fluid.ρ
    ρ isa Function && error("Spatially varying density is not supported yet for the coupled solver.")
    ρ_val = float(ρ)
    β_val = float(β)
    gravity_vec = SVector{2,Float64}(gravity)

    total_rows = 2 * (nu_x + nu_y) + np
    total_scalar_cols = 2 * N_scalar
    row_uωx = 0
    row_uωy = 2 * nu_x

    B = spzeros(Float64, total_rows, total_scalar_cols)
    offset = zeros(Float64, total_rows)
    ones_scalar = ones(Float64, N_scalar)

    if β_val != 0.0 && (gravity_vec[1] != 0.0 || gravity_vec[2] != 0.0)
        if gravity_vec[1] != 0.0
            block_x = (-ρ_val * β_val * gravity_vec[1]) * data.Vx * interp_T_to_ux * select_Tω
            B[row_uωx+1:row_uωx+nu_x, :] = block_x
            offset[row_uωx+1:row_uωx+nu_x] .= ρ_val * β_val * gravity_vec[1] *
                                              data.Vx * (interp_T_to_ux * (float(T_ref) .* ones_scalar))
        end
        if gravity_vec[2] != 0.0
            block_y = (-ρ_val * β_val * gravity_vec[2]) * data.Vy * interp_T_to_uy * select_Tω
            B[row_uωy+1:row_uωy+nu_y, :] = block_y
            offset[row_uωy+1:row_uωy+nu_y] .= ρ_val * β_val * gravity_vec[2] *
                                              data.Vy * (interp_T_to_uy * (float(T_ref) .* ones_scalar))
        end
    end

    times = Float64[0.0]
    velocity_hist = store_states ? Vector{Vector{Float64}}([copy(velocity_state)]) : Vector{Vector{Float64}}()
    scalar_hist = store_states ? Vector{Vector{Float64}}([copy(scalar_state)]) : Vector{Vector{Float64}}()

    return NavierStokesScalarCoupler(momentum,
                                     scalar_capacity,
                                     κ,
                                     scalar_source,
                                     bc_scalar,
                                     bc_scalar_cut,
                                     strategy,
                                     β_val,
                                     gravity_vec,
                                     float(T_ref),
                                     scalar_state,
                                     prev_scalar_state,
                                     velocity_state,
                                     prev_velocity_state,
                                     p_half,
                                     conv_prev,
                                     data,
                                     nodes_scalar,
                                     select_Tω,
                                     dirichlet_rows,
                                     (proj_x, proj_y),
                                     (interp_T_to_ux, interp_T_to_uy),
                                     (D_p[1], D_p[2]),
                                     (S_m[1], S_m[2]),
                                     (SmA[1], SmA[2]),
                                     B,
                                     offset,
                                     times,
                                     velocity_hist,
                                     scalar_hist,
                                     store_states)
end

@inline function _split_velocity_components(data, state::AbstractVector{<:Real})
    nu_x = data.nu_x
    nu_y = data.nu_y
    uωx = @view state[1:nu_x]
    uγx = @view state[nu_x+1:2nu_x]
    uωy = @view state[2nu_x+1:2nu_x+nu_y]
    uγy = @view state[2nu_x+nu_y+1:2*(nu_x+nu_y)]
    return uωx, uγx, uωy, uγy
end

function _build_scalar_system(c::NavierStokesScalarCoupler,
                              velocity_state::AbstractVector{<:Real},
                              Δt::Float64,
                              t_prev::Float64,
                              scheme::Symbol,
                              T_prev::Vector{Float64})
    scheme_str = _scheme_string_pretty(scheme)
    data = c.navier_data
    uωx, _, uωy, _ = _split_velocity_components(data, velocity_state)

    u_bulk_x = c.proj_uω_to_scalar[1] * Vector{Float64}(uωx)
    u_bulk_y = c.proj_uω_to_scalar[2] * Vector{Float64}(uωy)

    N_scalar = length(c.scalar_nodes[1]) * length(c.scalar_nodes[2])
    u_interface = zeros(Float64, 2 * N_scalar)
    operator = ConvectionOps(c.scalar_capacity, (u_bulk_x, u_bulk_y), u_interface)

    A_T = A_mono_unstead_advdiff(operator, c.scalar_capacity, c.diffusivity,
                                 c.bc_scalar_cut, Δt, scheme_str)
    b_T = b_mono_unstead_advdiff(operator, c.scalar_source, c.scalar_capacity, c.diffusivity,
                                 c.bc_scalar_cut, T_prev, Δt, t_prev, scheme_str)

    BC_border_mono!(A_T, b_T, c.bc_scalar, c.scalar_capacity.mesh)
    return A_T, b_T, operator
end

function _build_scalar_steady_system(c::NavierStokesScalarCoupler,
                                     velocity_state::AbstractVector{<:Real})
    data = c.navier_data
    uωx, _, uωy, _ = _split_velocity_components(data, velocity_state)

    u_bulk_x = c.proj_uω_to_scalar[1] * Vector{Float64}(uωx)
    u_bulk_y = c.proj_uω_to_scalar[2] * Vector{Float64}(uωy)

    N_scalar = length(c.scalar_nodes[1]) * length(c.scalar_nodes[2])
    operator = ConvectionOps(c.scalar_capacity, (u_bulk_x, u_bulk_y), zeros(2 * N_scalar))

    A_T = A_mono_stead_advdiff(operator, c.scalar_capacity, c.diffusivity, c.bc_scalar_cut)
    b_T = b_mono_stead_advdiff(operator, c.scalar_source, c.scalar_capacity, c.bc_scalar_cut)

    BC_border_mono!(A_T, b_T, c.bc_scalar, c.scalar_capacity.mesh)
    return A_T, b_T
end

@inline function _apply_dirichlet_filter!(C::SparseMatrixCSC{Float64,Int}, rows::Vector{Int})
    isempty(rows) && return C
    for r in rows
        C[r, :] .= 0.0
    end
    return C
end

function _temperature_velocity_jacobian(c::NavierStokesScalarCoupler,
                                        T_state::Vector{Float64},
                                        T_prev::Vector{Float64},
                                        Δt::Float64,
                                        scheme::Symbol)
    scheme_str = _scheme_string_pretty(scheme)
    data = c.navier_data
    nu_x, nu_y = data.nu_x, data.nu_y
    total_ns = 2 * (nu_x + nu_y) + data.np
    N_scalar = size(c.select_Tω, 1)

    select_Tω = c.select_Tω
    Tω_curr = select_Tω * T_state
    Tω_prev = select_Tω * T_prev
    base_T = scheme_str == "BE" ? Tω_curr : Tω_prev

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    if nu_x > 0
        diag_vec = c.scalar_S_m[1] * base_T
        block = Δt * (c.scalar_D_p[1] * spdiagm(0 => Vector{Float64}(diag_vec)) * c.scalar_SmA[1] * c.proj_uω_to_scalar[1])
        rx, cx, vx = findnz(block)
        for k in eachindex(rx)
            push!(rows, rx[k])
            push!(cols, cx[k])
            push!(vals, vx[k])
        end
    end

    if nu_y > 0
        diag_vec = c.scalar_S_m[2] * base_T
        block = Δt * (c.scalar_D_p[2] * spdiagm(0 => Vector{Float64}(diag_vec)) * c.scalar_SmA[2] * c.proj_uω_to_scalar[2])
        ry, cy, vy = findnz(block)
        col_offset = 2 * nu_x
        for k in eachindex(ry)
            push!(rows, ry[k])
            push!(cols, cy[k] + col_offset)
            push!(vals, vy[k])
        end
    end

    Cmat = sparse(rows, cols, vals, 2 * N_scalar, total_ns)
    return _apply_dirichlet_filter!(Cmat, c.dirichlet_rows)
end

function _temperature_velocity_jacobian_steady(c::NavierStokesScalarCoupler,
                                               T_state::Vector{Float64})
    data = c.navier_data
    nu_x, nu_y = data.nu_x, data.nu_y
    total_ns = 2 * (nu_x + nu_y) + data.np
    N_scalar = length(c.scalar_nodes[1]) * length(c.scalar_nodes[2])

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    if nu_x > 0
        diag_vec = c.scalar_S_m[1] * T_state
        block = c.scalar_D_p[1] * spdiagm(0 => Vector{Float64}(diag_vec)) * c.scalar_SmA[1] * c.proj_uω_to_scalar[1]
        rx, cx, vx = findnz(block)
        for k in eachindex(rx)
            push!(rows, rx[k])
            push!(cols, cx[k])
            push!(vals, vx[k])
        end
    end

    if nu_y > 0
        diag_vec = c.scalar_S_m[2] * T_state
        block = c.scalar_D_p[2] * spdiagm(0 => Vector{Float64}(diag_vec)) * c.scalar_SmA[2] * c.proj_uω_to_scalar[2]
        ry, cy, vy = findnz(block)
        col_offset = 2 * nu_x
        for k in eachindex(ry)
            push!(rows, ry[k])
            push!(cols, cy[k] + col_offset)
            push!(vals, vy[k])
        end
    end

    Cmat = sparse(rows, cols, vals, 2 * N_scalar, total_ns)
    return _apply_dirichlet_filter!(Cmat, c.dirichlet_rows)
end

function _buoyancy_forces(c::NavierStokesScalarCoupler, data, T_state::Vector{Float64})
    T_bulk = c.select_Tω * T_state
    ΔT_bulk = T_bulk .- float(c.T_ref)

    T_on_ux = c.interp_T_to_velocity[1] * ΔT_bulk
    T_on_uy = c.interp_T_to_velocity[2] * ΔT_bulk

    ρ = c.momentum.fluid.ρ
    ρ isa Function && error("Spatially varying density not supported for buoyancy.")
    ρ_val = float(ρ)

    β = c.β
   g = c.gravity

    buoyancy_x = -ρ_val * β * g[1] .* T_on_ux
    buoyancy_y = -ρ_val * β * g[2] .* T_on_uy

    return data.Vx * buoyancy_x, data.Vy * buoyancy_y
end

function _apply_buoyancy_rhs!(c::NavierStokesScalarCoupler, data, T_state::Vector{Float64})
    nu_x = data.nu_x
    nu_y = data.nu_y

    row_uωx = 0
    row_uωy = 2 * nu_x

    force_x, force_y = _buoyancy_forces(c, data, T_state)

    c.momentum.b[row_uωx+1:row_uωx+nu_x] .+= force_x
    c.momentum.b[row_uωy+1:row_uωy+nu_y] .+= force_y
end


@inline function nearest_index(vec::AbstractVector{<:Real}, val::Real)
    idx = searchsortedfirst(vec, val)
    if idx <= 1
        return 1
    elseif idx > length(vec)
        return length(vec)
    else
        prev_val = vec[idx - 1]
        curr_val = vec[idx]
        return abs(val - prev_val) <= abs(curr_val - val) ? idx - 1 : idx
    end
end

function project_field_between_nodes(values::AbstractVector{<:Real},
                                     src_nodes::NTuple{2,Vector{Float64}},
                                     dst_nodes::NTuple{2,Vector{Float64}})
    Nx_src = length(src_nodes[1])
    Ny_src = length(src_nodes[2])
    field = reshape(Vector{Float64}(values), (Nx_src, Ny_src))

    Nx_dst = length(dst_nodes[1])
    Ny_dst = length(dst_nodes[2])
    projected = Vector{Float64}(undef, Nx_dst * Ny_dst)

    for j in 1:Ny_dst
        y = dst_nodes[2][j]
        iy = nearest_index(src_nodes[2], y)
        for i in 1:Nx_dst
            x = dst_nodes[1][i]
            ix = nearest_index(src_nodes[1], x)
            projected[i + (j - 1) * Nx_dst] = field[ix, iy]
        end
    end

    return projected
end


function _build_scalar_steady_system(c::NavierStokesScalarCoupler,
                                     data,
                                     velocity_state::AbstractVector{<:Real})
    nodes_scalar = c.scalar_nodes
    mesh_ux = c.momentum.fluid.mesh_u[1]
    mesh_uy = c.momentum.fluid.mesh_u[2]

    uωx, _, uωy, _ = _split_velocity_components(data, velocity_state)
    u_bulk_x = project_field_between_nodes(uωx, (mesh_ux.nodes[1], mesh_ux.nodes[2]), nodes_scalar)
    u_bulk_y = project_field_between_nodes(uωy, (mesh_uy.nodes[1], mesh_uy.nodes[2]), nodes_scalar)

    N = length(nodes_scalar[1]) * length(nodes_scalar[2])
    operator = ConvectionOps(c.scalar_capacity, (u_bulk_x, u_bulk_y), zeros(2N))

    A_T = A_mono_stead_advdiff(operator, c.scalar_capacity, c.diffusivity, c.bc_scalar_cut)
    b_T = b_mono_stead_advdiff(operator, c.scalar_source, c.scalar_capacity, c.bc_scalar_cut)

    BC_border_mono!(A_T, b_T, c.bc_scalar, c.scalar_capacity.mesh)

    return A_T, b_T
end

@inline function _momentum_residual(c::NavierStokesScalarCoupler,
                                    A_ns::SparseMatrixCSC{Float64,Int},
                                    b_ns::Vector{Float64},
                                    T_state::Vector{Float64})
    return A_ns * c.velocity_state .- (b_ns .+ c.buoyancy_matrix * T_state .+ c.buoyancy_offset)
end

@inline function _picard_residuals(T_new::Vector{Float64}, T_old::Vector{Float64},
                                   U_new::Vector{Float64}, U_old::Vector{Float64},
                                   tol_T::Float64, tol_U::Float64)
    res_T = maximum(abs, T_new .- T_old)
    res_U = maximum(abs, U_new .- U_old)
    return res_T < tol_T, res_U < tol_U, res_T, res_U
end

function _update_state_history!(c::NavierStokesScalarCoupler,
                                velocity_state::Vector{Float64},
                                scalar_state::Vector{Float64},
                                time::Float64)
    push!(c.times, time)
    if c.store_states
        push!(c.velocity_states, copy(velocity_state))
        push!(c.scalar_states, copy(scalar_state))
    end
end


function solve_linear_system(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64};
                             method=Base.:\, algorithm=nothing, kwargs...)
    Ared, bred, keep_idx_rows, keep_idx_cols = remove_zero_rows_cols!(A, b)

    # If reduced system empty, return zero vector of appropriate size
    nfull = size(A, 2)
    if isempty(bred)
        return zeros(Float64, nfull)
    end

    # Solve reduced system
    sol_red = nothing
    if algorithm !== nothing
        prob = LinearSolve.LinearProblem(Ared, bred)
        sol = LinearSolve.solve(prob, algorithm; kwargs...)
        sol_red = Vector{Float64}(sol.u)
    elseif method === Base.:\
        try
            sol_red = Ared \ bred
        catch err
            if err isa SingularException
                @warn "Direct solve failed with SingularException on reduced system; falling back to bicgstabl" sizeA=size(Ared)
                sol_red = IterativeSolvers.bicgstabl(Ared, bred)
            else
                rethrow(err)
            end
        end
    else
        sol_red = method(Ared, bred; kwargs...)
    end

    # Expand reduced solution back to full size (zeros at removed columns)
    N = size(A, 2)
    x_full = zeros(N)
    x_full[keep_idx_cols] = sol_red

    return x_full
end

function _solve_scalar_system(A::SparseMatrixCSC{Float64,Int},
                              b::Vector{Float64};
                              method=Base.:\,
                              algorithm=nothing,
                              kwargs...)
    return solve_linear_system(A, b; method=method, algorithm=algorithm, kwargs...)
end

function _solve_block_system(J::SparseMatrixCSC{Float64,Int},
                             rhs::Vector{Float64};
                             method=Base.:\,
                             algorithm=nothing,
                             kwargs...)
    if algorithm !== nothing
        prob = LinearSolve.LinearProblem(J, rhs)
        sol = LinearSolve.solve(prob, algorithm; kwargs...)
        return Vector{Float64}(sol.u)
    elseif method === Base.:\
        return J \ rhs
    else
        return method(J, rhs; kwargs...)
    end
end

function _advance_passive!(c::NavierStokesScalarCoupler,
                           Δt::Float64,
                           t_prev::Float64,
                           t_next::Float64,
                           scheme::Symbol;
                           method=Base.:\,
                           algorithm=nothing,
                           kwargs...)
    θ = scheme_to_theta(scheme)
    conv_curr = assemble_navierstokes2D_unsteady!(c.momentum,
                                                  c.navier_data,
                                                  Δt,
                                                  c.prev_velocity_state,
                                                  c.p_half,
                                                  t_prev,
                                                  t_next,
                                                  θ,
                                                  c.conv_prev)
    solve_navierstokes_linear_system!(c.momentum; method=method, algorithm=algorithm, kwargs...)
    c.velocity_state .= c.momentum.x

    p_offset = 2 * (c.navier_data.nu_x + c.navier_data.nu_y)
    c.p_half .= c.velocity_state[p_offset+1:p_offset+c.navier_data.np]

    A_T, b_T, _ = _build_scalar_system(c, c.velocity_state, Δt, t_prev, scheme, c.prev_scalar_state)
    T_next = _solve_scalar_system(A_T, b_T; method=method, algorithm=algorithm, kwargs...)
    c.scalar_state .= T_next

    c.conv_prev = ntuple(Val(2)) do i
        copy(conv_curr[Int(i)])
    end
    c.momentum.prev_conv = c.conv_prev
end

function _advance_picard!(c::NavierStokesScalarCoupler,
                          Δt::Float64,
                          t_prev::Float64,
                          t_next::Float64,
                          strategy::PicardCoupling,
                          scheme::Symbol;
                          method=Base.:\,
                          algorithm=nothing,
                          kwargs...)
    θ = scheme_to_theta(scheme)
    conv_curr = assemble_navierstokes2D_unsteady!(c.momentum,
                                                  c.navier_data,
                                                  Δt,
                                                  c.prev_velocity_state,
                                                  c.p_half,
                                                  t_prev,
                                                  t_next,
                                                  θ,
                                                  c.conv_prev)
    A_ns = copy(c.momentum.A)
    b_ns_base = copy(c.momentum.b)

    T_iter = copy(c.scalar_state)
    U_iter = copy(c.velocity_state)

    converged = false
    res_T = Inf
    res_U = Inf

    for k in 1:strategy.maxiter
        rhs = b_ns_base .+ c.buoyancy_matrix * T_iter .+ c.buoyancy_offset
        c.momentum.A = A_ns
        c.momentum.b = rhs
        solve_navierstokes_linear_system!(c.momentum; method=method, algorithm=algorithm, kwargs...)
        U_new = copy(c.momentum.x)

        A_T, b_T, _ = _build_scalar_system(c, U_new, Δt, t_prev, scheme, c.prev_scalar_state)
        T_raw = _solve_scalar_system(A_T, b_T; method=method, algorithm=algorithm, kwargs...)
        T_new = strategy.relaxation * T_raw .+ (1.0 - strategy.relaxation) .* T_iter

        ok_T, ok_U, res_T, res_U = _picard_residuals(T_new, T_iter, U_new, U_iter,
                                                     strategy.tol_T, strategy.tol_U)
        T_iter .= T_new
        U_iter .= U_new

        if ok_T
            converged = true
            break
        end
    end

    converged || @warn "Picard coupling did not converge: residuals (T=$(res_T), U=$(res_U))"

    c.velocity_state .= U_iter
    c.scalar_state .= T_iter
    c.momentum.x .= U_iter

    p_offset = 2 * (c.navier_data.nu_x + c.navier_data.nu_y)
    c.p_half .= c.velocity_state[p_offset+1:p_offset+c.navier_data.np]

    c.conv_prev = ntuple(Val(2)) do i
        copy(conv_curr[Int(i)])
    end
    c.momentum.prev_conv = c.conv_prev
end

function _advance_monolithic!(c::NavierStokesScalarCoupler,
                              Δt::Float64,
                              t_prev::Float64,
                              t_next::Float64,
                              strategy::MonolithicCoupling,
                              scheme::Symbol;
                              method=Base.:\,
                              algorithm=nothing,
                              kwargs...)
    θ = scheme_to_theta(scheme)
    conv_curr = assemble_navierstokes2D_unsteady!(c.momentum,
                                                  c.navier_data,
                                                  Δt,
                                                  c.prev_velocity_state,
                                                  c.p_half,
                                                  t_prev,
                                                  t_next,
                                                  θ,
                                                  c.conv_prev)

    A_ns = copy(c.momentum.A)
    b_ns_base = copy(c.momentum.b)

    total_ns = size(A_ns, 2)
    N_scalar = length(c.scalar_state)
    total_unknowns = total_ns + N_scalar

    state = vcat(c.velocity_state, c.scalar_state)

    solve_kwargs = (; kwargs...)

    converged = false
    res_norm = Inf

    for iter in 1:strategy.maxiter
        U = view(state, 1:total_ns)
        T = view(state, total_ns+1:total_unknowns)

        R_u = A_ns * U .- (b_ns_base .+ c.buoyancy_matrix * Vector{Float64}(T) .+ c.buoyancy_offset)

        c.velocity_state .= U
        A_T, b_T, _ = _build_scalar_system(c, U, Δt, t_prev, scheme, c.prev_scalar_state)
        R_T = A_T * T .- b_T

        residual = vcat(R_u, R_T)
        res_norm = maximum(abs, residual)

        if strategy.verbose
            println("[Monolithic] Iteration $(iter): residual = $(res_norm)")
        end

        if res_norm < strategy.tol * (1.0 + maximum(abs, state))
            converged = true
            break
        end

        J_uT = -c.buoyancy_matrix
        J_Tu = _temperature_velocity_jacobian(c, Vector{Float64}(T), c.prev_scalar_state, Δt, scheme)

        J_top = hcat(A_ns, J_uT)
        J_bottom = hcat(J_Tu, A_T)
        J = vcat(J_top, J_bottom)

        J_reduced, residual_reduced, row_idx, col_idx = remove_zero_rows_cols_separate!(J, residual)
        δ_reduced = _solve_block_system(J_reduced, -residual_reduced; method=method, algorithm=algorithm, solve_kwargs...)
        δ_full = zeros(size(J, 2))
        δ_full[col_idx] = δ_reduced
        state .= state .+ strategy.damping .* δ_full
    end

    converged || @warn "Monolithic coupling did not converge: residual $(res_norm)"

    c.velocity_state .= view(state, 1:total_ns)
    c.scalar_state .= view(state, total_ns+1:total_unknowns)
    c.momentum.x .= c.velocity_state

    p_offset = 2 * (c.navier_data.nu_x + c.navier_data.nu_y)
    c.p_half .= c.velocity_state[p_offset+1:p_offset+c.navier_data.np]

    c.conv_prev = ntuple(Val(2)) do i
        copy(conv_curr[Int(i)])
    end
    c.momentum.prev_conv = c.conv_prev
end

function solve_NavierStokesScalarCoupling_steady!(c::NavierStokesScalarCoupler;
                                                  tol::Float64=1e-6,
                                                  maxiter::Int=25,
                                                  method=Base.:\,
                                                  algorithm=nothing,
                                                  kwargs...)
    data = navierstokes2D_blocks(c.momentum)
    U_prev = copy(c.velocity_state)
    T_prev = copy(c.scalar_state)

    relaxation = 1.0
    outer_max = maxiter
    verbose = false

    if c.strategy isa PicardCoupling
        relaxation = clamp(c.strategy.relaxation, 0.0, 1.0)
        outer_max = c.strategy.maxiter
    elseif c.strategy isa MonolithicCoupling
        relaxation = clamp(c.strategy.damping, 0.0, 1.0)
        outer_max = c.strategy.maxiter
        verbose = c.strategy.verbose
    end

    residual = Inf
    iter = 0

    while iter < outer_max && residual > tol
        assemble_navierstokes2D_steady_picard!(c.momentum, data, U_prev)
        _apply_buoyancy_rhs!(c, data, T_prev)
        solve_navierstokes_linear_system!(c.momentum; method=method, algorithm=algorithm, kwargs...)
        U_new = copy(c.momentum.x)

        A_T, b_T = _build_scalar_steady_system(c, data, U_new)
        T_new = solve_linear_system(A_T, b_T; method=method, algorithm=algorithm, kwargs...)

        U_relaxed = relaxation .* U_new .+ (1.0 - relaxation) .* U_prev
        T_relaxed = relaxation .* T_new .+ (1.0 - relaxation) .* T_prev

        res_u = maximum(abs, U_relaxed .- U_prev)
        res_t = maximum(abs, T_relaxed .- T_prev)
        residual = max(res_u, res_t)

        iter += 1
        verbose && println("[Coupled steady] iter=$(iter) maxΔu=$(res_u) maxΔT=$(res_t)")

        U_prev .= U_relaxed
        T_prev .= T_relaxed
    end

    if residual > tol
        @warn "Coupled steady solver did not converge" final_residual=residual iterations=iter tol=tol
    end

    c.velocity_state .= U_prev
    c.scalar_state .= T_prev
    c.prev_velocity_state .= U_prev
    c.prev_scalar_state .= T_prev
    c.momentum.x .= U_prev
    c.momentum.prev_conv = nothing

    c.times = Float64[]
    if c.store_states
        c.velocity_states = [copy(U_prev)]
        c.scalar_states = [copy(T_prev)]
    else
        c.velocity_states = Vector{Vector{Float64}}()
        c.scalar_states = Vector{Vector{Float64}}()
    end

    return U_prev, T_prev, residual
end

function step!(c::NavierStokesScalarCoupler;
               Δt::Float64,
               t_prev::Float64,
               scheme::Symbol=:CN,
               method=Base.:\,
               algorithm=nothing,
               kwargs...)
    t_next = t_prev + Δt
    if c.strategy isa PassiveCoupling
        _advance_passive!(c, Δt, t_prev, t_next, scheme; method=method, algorithm=algorithm, kwargs...)
    elseif c.strategy isa PicardCoupling
        _advance_picard!(c, Δt, t_prev, t_next, c.strategy, scheme; method=method, algorithm=algorithm, kwargs...)
    elseif c.strategy isa MonolithicCoupling
        _advance_monolithic!(c, Δt, t_prev, t_next, c.strategy, scheme; method=method, algorithm=algorithm, kwargs...)
    else
        error("Unsupported coupling strategy $(typeof(c.strategy))")
    end
    return t_next
end

function solve_NavierStokesScalarCoupling!(c::NavierStokesScalarCoupler;
                                           Δt::Float64,
                                           T_end::Float64,
                                           scheme::Symbol=:CN,
                                           method=Base.:\,
                                           algorithm=nothing,
                                           kwargs...)
    t = 0.0
    while t < T_end - 1e-12 * max(1.0, T_end)
        println("Time = $(t), advancing by Δt = $(Δt)")
        dt_step = min(Δt, T_end - t)
        t_next = step!(c; Δt=dt_step, t_prev=t, scheme=scheme,
                       method=method, algorithm=algorithm, kwargs...)
        _update_state_history!(c, c.velocity_state, c.scalar_state, t_next)
        c.prev_velocity_state .= c.velocity_state
        c.prev_scalar_state .= c.scalar_state
        t = t_next
    end
    return c.times, c.velocity_states, c.scalar_states
end
