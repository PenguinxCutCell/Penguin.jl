"""
    NavierStokesHeat2D

Two-way coupled time integrator for a two-dimensional Navier–Stokes / heat system
under the Boussinesq approximation. The velocity/pressure are solved first with
buoyancy forcing from the previous temperature field, then the temperature is
advected by the new velocity field. This is a fractional step (operator splitting)
approach that is easier to debug and extend than monolithic coupling.
"""
mutable struct NavierStokesHeat2D
    momentum::NavierStokesMono{2}
    temperature_capacity::Capacity{2}
    thermal_diffusivity::Union{Float64,Function}
    heat_source::Function
    bc_temperature::BorderConditions
    bc_temperature_cut::AbstractBoundary
    β::Float64
    gravity::SVector{2,Float64}
    T_ref::Float64
    temperature::Vector{Float64}
    velocity_states::Vector{Vector{Float64}}
    temperature_states::Vector{Vector{Float64}}
    times::Vector{Float64}
    scalar_nodes::NTuple{2,Vector{Float64}}
    interp_ux::SparseMatrixCSC{Float64,Int}
    interp_uy::SparseMatrixCSC{Float64,Int}
end

function NavierStokesHeat2D(momentum::NavierStokesMono{2},
                            temperature_capacity::Capacity{2},
                            κ::Union{Float64,Function},
                            heat_source::Function,
                            bc_temperature::BorderConditions,
                            bc_temperature_cut::AbstractBoundary;
                            β::Float64=1.0,
                            gravity::NTuple{2,Float64}=(0.0, -1.0),
                            T_ref::Float64=0.0,
                            T0::Union{Nothing,Vector{Float64}}=nothing)
    node_counts = ntuple(i -> length(temperature_capacity.mesh.nodes[i]), 2)
    N = prod(node_counts)
    T_init = if T0 === nothing
        zeros(2N)
    else
        length(T0) == 2N || error("Initial temperature vector must have length $(2N).")
        copy(T0)
    end

    gravity_vec = SVector{2,Float64}(gravity)
    scalar_nodes = ntuple(i -> copy(temperature_capacity.mesh.nodes[i]), 2)

    interp_ux = build_temperature_interpolation(temperature_capacity, momentum.fluid.capacity_u[1])
    interp_uy = build_temperature_interpolation(temperature_capacity, momentum.fluid.capacity_u[2])

    return NavierStokesHeat2D(momentum,
                              temperature_capacity,
                              κ,
                              heat_source,
                              bc_temperature,
                              bc_temperature_cut,
                              β,
                              gravity_vec,
                              T_ref,
                              T_init,
                              Vector{Vector{Float64}}(),
                              Vector{Vector{Float64}}(),
                              Float64[],
                              scalar_nodes,
                              interp_ux,
                              interp_uy)
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


function _scheme_string(scheme::Symbol)
    s = lowercase(String(scheme))
    if s in ("cn", "crank_nicolson", "cranknicolson")
        return "CN"
    elseif s in ("be", "backward_euler", "implicit_euler")
        return "BE"
    else
        error("Unsupported time scheme $(scheme). Use :CN or :BE.")
    end
end


function build_temperature_system(s::NavierStokesHeat2D,
                                  data,
                                  velocity_state::AbstractVector{<:Real},
                                  Δt::Float64,
                                  t_prev::Float64,
                                  scheme::Symbol,
                                  T_prev::Vector{Float64})
    scheme_str = _scheme_string(scheme)

    nodes_scalar = s.scalar_nodes
    mesh_ux = s.momentum.fluid.mesh_u[1]
    mesh_uy = s.momentum.fluid.mesh_u[2]

    uωx, _, uωy, _ = _split_velocity_components(data, velocity_state)
    u_bulk_x = project_field_between_nodes(uωx, (mesh_ux.nodes[1], mesh_ux.nodes[2]), nodes_scalar)
    u_bulk_y = project_field_between_nodes(uωy, (mesh_uy.nodes[1], mesh_uy.nodes[2]), nodes_scalar)

    N = length(nodes_scalar[1]) * length(nodes_scalar[2])
    u_interface = zeros(Float64, 2 * N)
    operator = ConvectionOps(s.temperature_capacity, (u_bulk_x, u_bulk_y), u_interface)

    A_T = A_mono_unstead_advdiff(operator, s.temperature_capacity, s.thermal_diffusivity,
                                 s.bc_temperature_cut, Δt, scheme_str)
    b_T = b_mono_unstead_advdiff(operator, s.heat_source, s.temperature_capacity,
                                 s.bc_temperature_cut, T_prev, Δt, t_prev, scheme_str)

    BC_border_mono!(A_T, b_T, s.bc_temperature, s.temperature_capacity.mesh)

    return A_T, b_T
end

function solve_NavierStokesHeat2D_unsteady!(s::NavierStokesHeat2D;
                                            Δt::Float64,
                                            T_end::Float64,
                                            scheme::Symbol=:CN,
                                            method=Base.:\,
                                            algorithm=nothing,
                                            store_states::Bool=true,
                                            kwargs...)
    θ = scheme_to_theta(scheme)
    scheme_str = _scheme_string(scheme)
    ns = s.momentum
    data = navierstokes2D_blocks(ns)
    p_offset = 2 * (data.nu_x + data.nu_y)
    np = data.np

    # Initialize velocity state
    x_prev = length(ns.x) == p_offset + np ? copy(ns.x) : zeros(p_offset + np)
    p_half_prev = zeros(Float64, np)
    if length(ns.x) == p_offset + np && !isempty(ns.x)
        p_half_prev .= ns.x[p_offset+1:p_offset+np]
    end

    # Initialize temperature state
    T_prev = copy(s.temperature)

    # Store initial states
    s.velocity_states = store_states ? Vector{Vector{Float64}}([copy(x_prev)]) : Vector{Vector{Float64}}()
    s.temperature_states = store_states ? Vector{Vector{Float64}}([copy(T_prev)]) : Vector{Vector{Float64}}()
    s.times = Float64[0.0]

    conv_prev = ns.prev_conv
    if conv_prev !== nothing && length(conv_prev) != 2
        conv_prev = nothing
    end

    kwargs_nt = (; kwargs...)

    t = 0.0
    while t < T_end - 1e-12 * max(1.0, T_end)
        dt_step = min(Δt, T_end - t)
        t_next = t + dt_step

        println("="^60)
        println("[NavierStokesHeat2D] Time step: t = $(round(t; digits=6)) → $(round(t_next; digits=6))")
        
        # ============================================================
        # STEP 1: Solve momentum with buoyancy forcing from T_prev
        # ============================================================
        
        # Assemble standard Navier-Stokes system
        conv_curr = assemble_navierstokes2D_unsteady!(ns, data, dt_step,
                                                      x_prev, p_half_prev,
                                                      t, t_next, θ, conv_prev)
        
        # Add Boussinesq buoyancy forcing: f_buoyancy = ρ * β * g * (T - T_ref)
        ρ = ns.fluid.ρ
        ρ_val = ρ isa Function ? 1.0 : ρ
        β = s.β
        g = s.gravity
        
        # Extract bulk temperatures and interpolate to velocity grids
        N_temp = length(s.scalar_nodes[1]) * length(s.scalar_nodes[2])
        Tω_prev = T_prev[1:N_temp]
        T_deviation = Tω_prev .- s.T_ref
        
        # Interpolate temperature to velocity grid and compute buoyancy force
        T_on_ux = s.interp_ux * T_deviation
        T_on_uy = s.interp_uy * T_deviation
        
        buoyancy_x = -ρ_val * β * g[1] .* T_on_ux
        buoyancy_y = -ρ_val * β * g[2] .* T_on_uy
        
        # Add to RHS (momentum equations)
        row_uωx = 0
        row_uωy = 2 * data.nu_x
        ns.b[row_uωx+1:row_uωx+data.nu_x] .+= data.Vx * buoyancy_x
        ns.b[row_uωy+1:row_uωy+data.nu_y] .+= data.Vy * buoyancy_y
        
        # Solve momentum system
        x_new = solve_linear_system(ns.A, ns.b; method=method, algorithm=algorithm, kwargs_nt...)
        ns.x .= x_new
        
        println("  → Momentum solved: max|u| = $(round(maximum(abs, x_new); digits=6))")
        
        # ============================================================
        # STEP 2: Solve temperature advected by new velocity field
        # ============================================================
        
        # Build convection operators with updated velocity
        A_T, b_T = build_temperature_system(s, data, x_new, dt_step, t, scheme, T_prev)
        
        # Solve temperature system
        T_next = solve_linear_system(A_T, b_T; method=method, algorithm=algorithm, kwargs_nt...)
        s.temperature .= T_next
        
        println("  → Temperature solved: T ∈ [$(round(minimum(T_next); digits=4)), $(round(maximum(T_next); digits=4))]")
        
        # ============================================================
        # STEP 3: Update state for next iteration
        # ============================================================
        
        x_prev = copy(x_new)
        p_half_prev .= x_new[p_offset+1:p_offset+np]
        conv_prev = ntuple(Val(2)) do i
            copy(conv_curr[Int(i)])
        end
        T_prev = copy(T_next)
        ns.prev_conv = conv_prev

        t = t_next
        push!(s.times, t)
        if store_states
            push!(s.velocity_states, copy(x_new))
            push!(s.temperature_states, copy(T_next))
        end
    end

    println("="^60)
    println("Simulation complete: $(length(s.times)) time steps")
    return s.times, s.velocity_states, s.temperature_states
end
