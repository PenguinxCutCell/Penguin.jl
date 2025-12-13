# Moving Stokes - Scalar Coupling
"""
    MovingStokesScalarCoupler{N}

Solver for coupled moving Stokes and scalar transport (e.g., temperature/heat).
Combines MovingStokesUnsteadyMono with advection-diffusion of a scalar field.

The scalar field (e.g., temperature) is advected by the velocity field from Stokes
and can optionally feed back to the momentum equation through buoyancy forces.

# Fields
- `stokes::MovingStokesUnsteadyMono{N}`: The moving Stokes solver
- `scalar_capacity::Capacity{N}`: Capacity for the scalar field
- `diffusivity::Union{Float64,Function}`: Thermal diffusivity (or scalar diffusion coefficient)
- `scalar_source::Function`: Source term for scalar equation
- `bc_scalar::BorderConditions`: Border conditions for scalar
- `bc_scalar_cut::AbstractBoundary`: Cut-cell boundary for scalar
- `β::Float64`: Thermal expansion coefficient (for buoyancy)
- `gravity::SVector{N,Float64}`: Gravity vector
- `T_ref::Float64`: Reference temperature
- `scalar_state::Vector{Float64}`: Current scalar state (Tω, Tγ)
- `prev_scalar_state::Vector{Float64}`: Previous scalar state
- `times::Vector{Float64}`: Time history
- `velocity_states::Vector{Vector{Float64}}`: Velocity state history
- `scalar_states::Vector{Vector{Float64}}`: Scalar state history
"""
mutable struct MovingStokesScalarCoupler{N}
    stokes::MovingStokesUnsteadyMono{N}
    scalar_capacity::Capacity{N}
    diffusivity::Union{Float64,Function}
    scalar_source::Function
    bc_scalar::BorderConditions
    bc_scalar_cut::AbstractBoundary
    β::Float64
    gravity::SVector{N,Float64}
    T_ref::Float64
    scalar_state::Vector{Float64}
    prev_scalar_state::Vector{Float64}
    times::Vector{Float64}
    velocity_states::Vector{Vector{Float64}}
    scalar_states::Vector{Vector{Float64}}
end

"""
    MovingStokesScalarCoupler(stokes, scalar_capacity, κ, scalar_source, bc_scalar, bc_scalar_cut; kwargs...)

Create a coupled moving Stokes-scalar solver.

# Arguments
- `stokes::MovingStokesUnsteadyMono{N}`: The moving Stokes solver
- `scalar_capacity::Capacity{N}`: Capacity for scalar field mesh
- `κ::Union{Float64,Function}`: Diffusivity coefficient
- `scalar_source::Function`: Source term for scalar equation
- `bc_scalar::BorderConditions`: Border boundary conditions for scalar
- `bc_scalar_cut::AbstractBoundary`: Cut-cell boundary condition for scalar

# Keyword Arguments
- `β::Real=0.0`: Thermal expansion coefficient for buoyancy
- `gravity::NTuple{N,Real}=...`: Gravity vector (default: zeros except last component = -1)
- `T_ref::Real=0.0`: Reference temperature for buoyancy
- `T0::Union{Nothing,Vector{Float64}}=nothing`: Initial scalar field
"""
function MovingStokesScalarCoupler(stokes::MovingStokesUnsteadyMono{N},
                                   scalar_capacity::Capacity{N},
                                   κ::Union{Float64,Function},
                                   scalar_source::Function,
                                   bc_scalar::BorderConditions,
                                   bc_scalar_cut::AbstractBoundary;
                                   β::Real=0.0,
                                   gravity::Union{Nothing,NTuple{N,Real}}=nothing,
                                   T_ref::Real=0.0,
                                   T0::Union{Nothing,Vector{Float64}}=nothing) where {N}
    
    # Default gravity: downward in last dimension
    if gravity === nothing
        g_tuple = ntuple(i -> i == N ? -1.0 : 0.0, N)
    else
        g_tuple = gravity
    end
    gravity_vec = SVector{N,Float64}(g_tuple)
    
    # Initialize scalar state
    N_scalar_nodes = prod(size(scalar_capacity.mesh))
    N_scalar = 2 * N_scalar_nodes  # (ω, γ) components
    
    scalar_state = if T0 === nothing
        zeros(Float64, N_scalar)
    else
        length(T0) == N_scalar || error("Initial scalar vector must have length $(N_scalar).")
        copy(T0)
    end
    prev_scalar_state = copy(scalar_state)
    
    times = Float64[0.0]
    velocity_states = Vector{Vector{Float64}}([copy(stokes.x)])
    scalar_states = Vector{Vector{Float64}}([copy(scalar_state)])
    
    return MovingStokesScalarCoupler{N}(
        stokes,
        scalar_capacity,
        κ,
        scalar_source,
        bc_scalar,
        bc_scalar_cut,
        float(β),
        gravity_vec,
        float(T_ref),
        scalar_state,
        prev_scalar_state,
        times,
        velocity_states,
        scalar_states
    )
end

"""
    _build_scalar_system_moving!(operator, capacity, κ, scalar_source, bc_scalar, bc_scalar_cut,
                                  scalar_prev, Δt, t_prev, t_next, scheme)

Build the system matrix and RHS for scalar transport with a given velocity field.
This is for moving geometry using SpaceTimeMesh operators.
"""
function _build_scalar_system_moving(operator::ConvectionOps,
                                     capacity::Capacity,
                                     κ::Union{Float64,Function},
                                     scalar_source::Function,
                                     bc_scalar_cut::AbstractBoundary,
                                     scalar_prev::Vector{Float64},
                                     Δt::Float64,
                                     t_prev::Float64,
                                     t_next::Float64,
                                     scheme::Symbol)
    
    scheme_str = scheme == :CN ? "CN" : "BE"
    
    # Build matrix and RHS using existing advection-diffusion functions
    A_T = A_mono_unstead_advdiff(operator, capacity, κ, bc_scalar_cut, Δt, scheme_str)
    # Signature: b_mono_unstead_advdiff(operator, f, capacite, D, bc, Tᵢ, Δt, t, scheme)
    b_T = b_mono_unstead_advdiff(operator, scalar_source, capacity, κ, bc_scalar_cut,
                                  scalar_prev, Δt, t_prev, scheme_str)
    
    return A_T, b_T
end

"""
    _extract_velocity_bulk(c, data, velocity_state)

Extract bulk velocity components for scalar transport.
For moving geometry, we need to project velocity from staggered grids to scalar mesh.
"""
function _extract_velocity_bulk_2D(c::MovingStokesScalarCoupler{2},
                                   stokes_state::Vector{Float64},
                                   mesh_ux::AbstractMesh,
                                   mesh_uy::AbstractMesh,
                                   nu_x::Int,
                                   nu_y::Int)
    # Extract ω components of velocity
    uωx = stokes_state[1:nu_x]
    uωy = stokes_state[2*nu_x+1:2*nu_x+nu_y]
    
    # Simple projection: interpolate velocity to scalar mesh
    # For now, use nearest neighbor or simple averaging
    scalar_nodes = c.scalar_capacity.mesh.nodes
    Nx_T = length(scalar_nodes[1])
    Ny_T = length(scalar_nodes[2])
    N_scalar = Nx_T * Ny_T
    
    # Simple approach: reshape velocities and interpolate
    # This is a simplified version - in practice might want more sophisticated interpolation
    u_bulk_x = zeros(Float64, N_scalar)
    u_bulk_y = zeros(Float64, N_scalar)
    
    ux_nodes = mesh_ux.nodes
    uy_nodes = mesh_uy.nodes
    Nx_ux = length(ux_nodes[1])
    Ny_ux = length(ux_nodes[2])
    Nx_uy = length(uy_nodes[1])
    Ny_uy = length(uy_nodes[2])
    
    # Reshape velocity fields
    Ux = reshape(uωx, (Nx_ux, Ny_ux))
    Uy = reshape(uωy, (Nx_uy, Ny_uy))
    
    # Project to scalar mesh (nearest neighbor for simplicity)
    # TODO: For better accuracy, consider pre-computing index mappings or using
    # direct index calculations for uniform grids instead of searchsortedfirst
    for j in 1:Ny_T
        y = scalar_nodes[2][j]
        jx = clamp(searchsortedfirst(ux_nodes[2], y), 1, Ny_ux)
        jy = clamp(searchsortedfirst(uy_nodes[2], y), 1, Ny_uy)
        
        for i in 1:Nx_T
            x = scalar_nodes[1][i]
            ix = clamp(searchsortedfirst(ux_nodes[1], x), 1, Nx_ux)
            iy = clamp(searchsortedfirst(uy_nodes[1], x), 1, Nx_uy)
            
            idx = i + (j - 1) * Nx_T
            u_bulk_x[idx] = Ux[ix, jx]
            u_bulk_y[idx] = Uy[iy, jy]
        end
    end
    
    return u_bulk_x, u_bulk_y
end

"""
    _apply_buoyancy_force_2D!(c, stokes_data, scalar_state, row_uωx, row_uωy)

Apply buoyancy force to the Stokes momentum equation RHS.
"""
function _apply_buoyancy_force_2D!(c::MovingStokesScalarCoupler{2},
                                  stokes_data,
                                  scalar_state::Vector{Float64})
    if c.β == 0.0
        return  # No buoyancy
    end
    
    # Extract scalar bulk values
    N_scalar_nodes = length(scalar_state) ÷ 2
    T_bulk = scalar_state[1:N_scalar_nodes]
    
    # Calculate temperature deviation
    ΔT = T_bulk .- c.T_ref
    
    # Interpolate temperature to velocity points (simplified)
    # In practice, might want more sophisticated interpolation
    nu_x = stokes_data.nu_x
    nu_y = stokes_data.nu_y
    
    # Simple approach: assume uniform temperature for each velocity component
    # TODO: Full implementation should use proper spatial interpolation from scalar mesh to velocity mesh
    T_mean = mean(ΔT)
    
    # Buoyancy force: F = -ρ * β * g * ΔT
    ρ = c.stokes.fluid.ρ
    ρ isa Function && error("Spatially varying density not supported for buoyancy.")
    ρ_val = ρ isa Number ? Float64(ρ) : float(ρ)
    
    # Apply to momentum equations
    row_uωx = 0
    row_uωy = 2 * nu_x
    
    force_x = -ρ_val * c.β * c.gravity[1] * T_mean
    force_y = -ρ_val * c.β * c.gravity[2] * T_mean
    
    # Add buoyancy to RHS (simplified - uniform force)
    if stokes_data.Vx !== nothing
        c.stokes.b[row_uωx+1:row_uωx+nu_x] .+= force_x
    end
    if stokes_data.Vy !== nothing
        c.stokes.b[row_uωy+1:row_uωy+nu_y] .+= force_y
    end
end

"""
    solve_MovingStokesScalarCoupler!(c, body, mesh, Δt, Tₛ, Tₑ, bc_b_u, bc_cut; kwargs...)

Solve the coupled moving Stokes-scalar problem.

# Arguments
- `c::MovingStokesScalarCoupler{N}`: The coupled solver
- `body::Function`: Body function defining the moving geometry
- `mesh::AbstractMesh`: Base spatial mesh (for pressure)
- `Δt::Float64`: Time step
- `Tₛ::Float64`: Start time
- `Tₑ::Float64`: End time
- `bc_b_u`: Border conditions for velocity
- `bc_cut`: Cut-cell boundary condition for velocity

# Keyword Arguments
- `scheme::Symbol=:BE`: Time integration scheme (:BE or :CN)
- `method=Base.:\\`: Linear solver method
- `algorithm=nothing`: LinearSolve algorithm
- `geometry_method::String="VOFI"`: Geometry integration method
- `coupling::Symbol=:passive`: Coupling strategy (:passive for one-way coupling)
"""
function solve_MovingStokesScalarCoupler!(c::MovingStokesScalarCoupler{2},
                                         body::Function,
                                         mesh::AbstractMesh,
                                         Δt::Float64,
                                         Tₛ::Float64,
                                         Tₑ::Float64,
                                         bc_b_u::Tuple{BorderConditions, BorderConditions},
                                         bc_cut::Union{AbstractBoundary, NTuple{2, AbstractBoundary}};
                                         scheme::Symbol=:BE,
                                         method=Base.:\,
                                         algorithm=nothing,
                                         geometry_method::String="VOFI",
                                         coupling::Symbol=:passive,
                                         kwargs...)
    
    println("Solving coupled moving Stokes-scalar problem:")
    println("- Moving problem")
    println("- Stokes + Scalar transport")
    println("- Coupling: $(coupling)")
    println("- Scheme: $(scheme)")
    
    # Reset state history
    c.times = Float64[]
    c.velocity_states = Vector{Vector{Float64}}[]
    c.scalar_states = Vector{Vector{Float64}}[]
    
    t = Tₛ
    push!(c.times, t)
    push!(c.velocity_states, copy(c.stokes.x))
    push!(c.scalar_states, copy(c.scalar_state))
    
    println("Time : $(t)")
    
    # Create staggered velocity meshes
    dx = mesh.nodes[1][2] - mesh.nodes[1][1]
    dy = mesh.nodes[2][2] - mesh.nodes[2][1]
    mesh_ux = Penguin.Mesh((length(mesh.nodes[1])-1, length(mesh.nodes[2])-1),
                           (mesh.nodes[1][end] - mesh.nodes[1][1], mesh.nodes[2][end] - mesh.nodes[2][1]),
                           (mesh.nodes[1][1] - 0.5*dx, mesh.nodes[2][1]))
    mesh_uy = Penguin.Mesh((length(mesh.nodes[1])-1, length(mesh.nodes[2])-1),
                           (mesh.nodes[1][end] - mesh.nodes[1][1], mesh.nodes[2][end] - mesh.nodes[2][1]),
                           (mesh.nodes[1][1], mesh.nodes[2][1] - 0.5*dy))
    
    # Time loop
    while t < Tₑ - 1e-12 * max(1.0, Tₑ)
        dt_step = min(Δt, Tₑ - t)
        t_next = t + dt_step
        
        println("Time : $(t_next)")
        
        # Create SpaceTime meshes for this time interval
        STmesh_ux = Penguin.SpaceTimeMesh(mesh_ux, [t, t_next], tag=mesh.tag)
        STmesh_uy = Penguin.SpaceTimeMesh(mesh_uy, [t, t_next], tag=mesh.tag)
        STmesh_p = Penguin.SpaceTimeMesh(mesh, [t, t_next], tag=mesh.tag)
        STmesh_scalar = Penguin.SpaceTimeMesh(c.scalar_capacity.mesh, [t, t_next], tag=mesh.tag)
        
        # Build capacities with moving body
        capacity_ux = Capacity(body, STmesh_ux; method=geometry_method, kwargs...)
        capacity_uy = Capacity(body, STmesh_uy; method=geometry_method, kwargs...)
        capacity_p = Capacity(body, STmesh_p; method=geometry_method, kwargs...)
        capacity_scalar = Capacity(body, STmesh_scalar; method=geometry_method, kwargs...)
        
        # Build operators
        operator_ux = DiffusionOps(capacity_ux)
        operator_uy = DiffusionOps(capacity_uy)
        operator_p = DiffusionOps(capacity_p)
        
        # Update fluid capacities and operators (for Stokes assembly)
        c.stokes.fluid.capacity_u = (capacity_ux, capacity_uy)
        c.stokes.fluid.operator_u = (operator_ux, operator_uy)
        c.stokes.fluid.capacity_p = capacity_p
        c.stokes.fluid.operator_p = operator_p
        
        # Build Stokes blocks for this time step
        θ = scheme == :CN ? 0.5 : 1.0
        stokes_data = stokes2D_moving_blocks(c.stokes.fluid,
                                            (operator_ux, operator_uy),
                                            (capacity_ux, capacity_uy),
                                            operator_p, capacity_p,
                                            scheme)
        
        # Assemble and solve Stokes system
        x_prev_stokes = c.stokes.x
        assemble_stokes2D_moving!(c.stokes, stokes_data, dt_step,
                                 x_prev_stokes, t, t_next, θ, mesh)
        
        # Apply buoyancy if needed
        if coupling != :passive && c.β != 0.0
            _apply_buoyancy_force_2D!(c, stokes_data, c.prev_scalar_state)
        end
        
        # Solve Stokes
        solve_moving_stokes_linear_system!(c.stokes; method=method, algorithm=algorithm, kwargs...)
        
        # Extract velocity for scalar transport
        nu_x = stokes_data.nu_x
        nu_y = stokes_data.nu_y
        u_bulk_x, u_bulk_y = _extract_velocity_bulk_2D(c, c.stokes.x, mesh_ux, mesh_uy, nu_x, nu_y)
        
        # Build scalar transport operators with velocity field
        N_scalar_nodes = prod(size(c.scalar_capacity.mesh))
        u_interface = zeros(Float64, 2 * N_scalar_nodes)  # Interface velocity
        convection_operator = ConvectionOps(capacity_scalar, (u_bulk_x, u_bulk_y), u_interface)
        
        # Build and solve scalar system
        A_scalar, b_scalar = _build_scalar_system_moving(convection_operator,
                                                         capacity_scalar,
                                                         c.diffusivity,
                                                         c.scalar_source,
                                                         c.bc_scalar_cut,
                                                         c.prev_scalar_state,
                                                         dt_step,
                                                         t,
                                                         t_next,
                                                         scheme)
        
        # Apply border boundary conditions for scalar
        BC_border_mono!(A_scalar, b_scalar, c.bc_scalar, c.scalar_capacity.mesh)
        
        # Solve scalar system
        Ared, bred, keep_idx_rows, keep_idx_cols = remove_zero_rows_cols!(A_scalar, b_scalar)
        
        T_next_red = if algorithm !== nothing
            prob = LinearSolve.LinearProblem(Ared, bred)
            sol = LinearSolve.solve(prob, algorithm)
            sol.u
        elseif method === Base.:\
            try
                Ared \ bred
            catch e
                if e isa SingularException
                    @warn "Direct solver hit SingularException; falling back to bicgstabl"
                    IterativeSolvers.bicgstabl(Ared, bred)
                else
                    rethrow(e)
                end
            end
        else
            method(Ared, bred)
        end
        
        N_scalar = 2 * N_scalar_nodes
        T_next = zeros(N_scalar)
        T_next[keep_idx_cols] = T_next_red
        
        # Update states
        c.scalar_state .= T_next
        c.prev_scalar_state .= T_next
        
        # Store results
        push!(c.times, t_next)
        push!(c.velocity_states, copy(c.stokes.x))
        push!(c.scalar_states, copy(c.scalar_state))
        
        println("  Max velocity: $(maximum(abs.(c.stokes.x)))")
        println("  Max scalar: $(maximum(abs.(c.scalar_state)))")
        
        t = t_next
    end
    
    return c.times, c.velocity_states, c.scalar_states
end
