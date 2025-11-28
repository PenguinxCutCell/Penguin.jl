# 3D Stefan Problem Solver with Front Tracking
# Based on the 2D routine StefanMono2D, solve_StefanMono2D!

"""
    compute_volume_jacobian_3D(mesh::Penguin.Mesh{3}, front, epsilon::Float64=1e-6)

Compute the volume Jacobian for 3D front tracking using finite differences.
Returns a dictionary mapping cell indices (i,j,k) to lists of (marker_idx, jacobian_value) pairs.

# Arguments
- `mesh::Penguin.Mesh{3}`: The 3D mesh
- `front`: The 3D front tracker object (FrontTracker3D)
- `epsilon::Float64`: Epsilon for finite difference computation
"""
function compute_volume_jacobian_3D(mesh::Penguin.Mesh{3}, front, epsilon::Float64=1e-6)
    # Extract mesh data
    x_faces = vcat(mesh.nodes[1][1], mesh.nodes[1][2:end])
    y_faces = vcat(mesh.nodes[2][1], mesh.nodes[2][2:end])
    z_faces = vcat(mesh.nodes[3][1], mesh.nodes[3][2:end])
    
    # Call FrontCutTracking function for 3D volume jacobian
    return FrontCutTracking.compute_volume_jacobian_3d(front, x_faces, y_faces, z_faces, epsilon)
end

"""
    smooth_displacements_3D!(displacements::Vector{Float64}, 
                             markers::Vector{Tuple{Float64,Float64,Float64}}, 
                             is_closed::Bool=true,
                             smoothing_factor::Float64=0.5,
                             window_size::Int=2)

Smooth marker displacements using weighted averaging of neighbors in 3D.
This helps maintain interface regularity and stability.
"""
function smooth_displacements_3D!(displacements::Vector{Float64}, 
                                  markers::Vector{Tuple{Float64,Float64,Float64}}, 
                                  is_closed::Bool=true,
                                  smoothing_factor::Float64=0.5,
                                  window_size::Int=2)
    n = length(displacements)
    if n <= 1
        return displacements
    end
    
    # Create a copy of original displacements
    original_displacements = copy(displacements)
    
    for i in 1:n
        # Calculate weighted sum of neighbors
        neighbor_sum = 0.0
        weight_sum = 0.0
        
        for j in -window_size:window_size
            if j == 0
                continue  # Skip the marker itself
            end
            
            # Handle wrapping for closed surfaces
            idx = i + j
            if is_closed
                idx = mod1(idx, n)
            else
                if idx < 1 || idx > n
                    continue
                end
            end
            
            # Calculate weight based on distance
            distance = sqrt((markers[i][1] - markers[idx][1])^2 + 
                           (markers[i][2] - markers[idx][2])^2 +
                           (markers[i][3] - markers[idx][3])^2)
            weight = 1.0 / (distance + 1e-10)
            
            neighbor_sum += weight * original_displacements[idx]
            weight_sum += weight
        end
        
        # Calculate weighted average
        if weight_sum > 0
            neighbor_avg = neighbor_sum / weight_sum
            displacements[i] = (1.0 - smoothing_factor) * original_displacements[i] + 
                               smoothing_factor * neighbor_avg
        end
    end
    
    return displacements
end

"""
    StefanMono3D(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, 
                 Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)

Create a 3D Stefan problem solver for monophasic problems with moving interface.

# Arguments
- `phase::Phase`: The phase object containing capacity and operator
- `bc_b::BorderConditions`: Border conditions for the domain boundaries
- `bc_i::AbstractBoundary`: Boundary condition at the interface
- `Δt::Float64`: Time step size
- `Tᵢ::Vector{Float64}`: Initial temperature field
- `mesh::AbstractMesh`: 3D spatial mesh
- `scheme::String`: Time integration scheme ("BE" for Backward Euler or "CN" for Crank-Nicolson)

# Returns
- `Solver`: Initialized solver object for the 3D Stefan problem
"""
function StefanMono3D(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- 3D Stefan problem")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, [], [])    
    if scheme == "CN"
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "CN")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "CN")
    else # BE
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "BE")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "BE")
    end
    
    BC_border_mono!(s.A, s.b, bc_b, mesh; t=0.0)
    
    s.x = Tᵢ
    return s
end

"""
    solve_StefanMono3D!(s::Solver, phase::Phase, front, Δt::Float64, 
                        Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, 
                        bc::AbstractBoundary, ic::InterfaceConditions, 
                        mesh::Penguin.Mesh{3}, scheme::String; kwargs...)

Solve the 3D Stefan problem with front tracking for a spherical interface.

# Arguments
- `s::Solver`: The solver object (must be initialized with StefanMono3D)
- `phase::Phase`: The phase object containing capacity and operator
- `front`: The 3D front tracker object representing the interface (sphere)
- `Δt::Float64`: Time step size
- `Tₛ::Float64`: Start time
- `Tₑ::Float64`: End time
- `bc_b::BorderConditions`: Border conditions for the domain boundaries
- `bc::AbstractBoundary`: Boundary condition at the interface
- `ic::InterfaceConditions`: Interface conditions (contains flux jump condition)
- `mesh::Penguin.Mesh{3}`: 3D spatial mesh
- `scheme::String`: Time integration scheme ("BE" or "CN")

# Keyword Arguments
- `method`: Linear solver method (default: Base.:\\)
- `Newton_params`: Newton iteration parameters (max_iter, tol, reltol, α)
- `jacobian_epsilon`: Epsilon for finite difference Jacobian computation
- `smooth_factor`: Smoothing factor for displacement averaging
- `window_size`: Window size for displacement smoothing
- `algorithm`: Algorithm for linear solver
- `plot_results::Bool`: Whether to generate plots (default: false)

# Returns
- `s::Solver`: Updated solver object with states history
- `residuals::Dict`: Residual history for each time step
- `xf_log::Dict`: Interface marker positions at each time step
- `timestep_history::Vector`: Time and timestep size history
- `phase_3d::Phase`: Final phase object
- `position_increments::Dict`: Position increment history for each time step
"""
function solve_StefanMono3D!(s::Solver, phase::Phase, front, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh::Penguin.Mesh{3}, scheme::String; 
                            method=Base.:\,
                            Newton_params=(100, 1e-6, 1e-6, 1.0),
                            jacobian_epsilon=1e-6, smooth_factor=0.5, window_size=10,
                            algorithm=nothing,
                            plot_results::Bool=false,
                            kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call StefanMono3D first.")
    end

    println("Solving 3D Stefan problem with Front Tracking:")
    println("- Monophasic problem")
    println("- Phase change with moving interface (sphere)")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Unpack parameters
    max_iter = Newton_params[1]
    tol = Newton_params[2]
    reltol = Newton_params[3]
    α = Newton_params[4]
    
    # Extract interface flux parameter (latent heat)
    ρL = ic.flux.value
    
    # Initialize tracking variables
    t = Tₛ
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Dict{Int, Vector{Tuple{Float64, Float64, Float64}}}()
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))
    position_increments = Dict{Int, Vector{Float64}}()

    # Determine dimensions
    dims = phase.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Create the 3D indices
    nx, ny, nz, _ = dims
    n = nx * ny * nz  # Total number of nodes in the mesh
    
    # Store initial state
    Tᵢ = s.x
    push!(s.states, s.x)
    
    # Get initial interface markers
    markers = get_markers_3d(front)
    
    # Store initial interface position
    xf_log[1] = markers
    
    # Main time stepping loop
    timestep = 1
    phase_3d = phase

    while t < Tₑ
        # Display current time step
        if timestep == 1
            println("\nFirst time step: t = $(round(t, digits=6))")
        else
            println("\nTime step $(timestep), t = $(round(t, digits=6))")
        end

        # Get current markers and calculate normals
        markers = get_markers_3d(front)
        normals = compute_marker_normals_3d(front, markers)
        
        # Update time for this step
        t += Δt
        tₙ = t - Δt
        tₙ₊₁ = t
        time_interval = [tₙ, tₙ₊₁]
        
        # Calculate total number of markers
        n_markers = length(markers)
        
        # Initialize displacement vector and residual vector
        displacements = zeros(n_markers)
        residual_norm_history = Float64[]
        position_increment_history = Float64[]

        # Newton iterations
        for iter in 1:max_iter
            # 1. Solve temperature field with current interface position
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            Tᵢ = s.x
            
            # Get capacity matrices
            V_matrices = phase.capacity.A[cap_index]
            Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]
            Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]
            Vₙ₊₁_matrix = diag(Vₙ₊₁_matrix)
            Vₙ_matrix = diag(Vₙ_matrix)
            Vₙ₊₁_matrix = reshape(Vₙ₊₁_matrix, (nx, ny, nz))
            Vₙ_matrix = reshape(Vₙ_matrix, (nx, ny, nz))
            
            # 2. Calculate the interface flux
            W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
            G = phase.operator.G[1:end÷2, 1:end÷2]
            H = phase.operator.H[1:end÷2, 1:end÷2]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:end÷2, 1:end÷2]
            
            Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
            interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            
            # Reshape to get flux per cell
            interface_flux_3d = reshape(interface_flux, (nx, ny, nz))

            # Compute volume Jacobian for the 3D mesh
            volume_jacobian = compute_volume_jacobian_3D(mesh, front, jacobian_epsilon)
            
            # 3. Build least squares system
            cells_idx = Tuple{Int, Int, Int}[]
            
            # Precompute affected cells
            for i in 1:nx
                for j in 1:ny
                    for k in 1:nz
                        if haskey(volume_jacobian, (i,j,k)) && !isempty(volume_jacobian[(i,j,k)])
                            push!(cells_idx, (i, j, k))
                        end
                    end
                end
            end

            # Build Jacobian matrix
            row_indices = Int[]
            col_indices = Int[]
            values = Float64[]
            
            m = length(cells_idx)
            
            for (eq_idx, (i, j, k)) in enumerate(cells_idx)
                for (marker_idx, jac_value) in volume_jacobian[(i,j,k)]
                    # FrontCutTracking returns 0-based marker indices
                    # Convert to Julia's 1-based indexing
                    if 0 <= marker_idx < n_markers
                        push!(row_indices, eq_idx)
                        push!(col_indices, marker_idx + 1)
                        push!(values, ρL * jac_value)
                    end
                end
            end
            
            # Create Jacobian matrix J for the system
            J = sparse(row_indices, col_indices, values, m, n_markers)
            
            # 4. Calculate current residual vector F
            F = zeros(m)
            
            for (eq_idx, (i, j, k)) in enumerate(cells_idx)
                volume_change = Vₙ₊₁_matrix[i,j,k] - Vₙ_matrix[i,j,k]
                flux = interface_flux_3d[i,j,k]
                F[eq_idx] = ρL * volume_change - flux
            end
            
            # 5. Solve the least squares system with regularization
            JTJ = J' * J
            
            # Add regularization
            diag_JTJ = diag(JTJ)
            min_diag = 1e-10 * maximum(diag_JTJ)
            for i in 1:length(diag_JTJ)
                if diag_JTJ[i] < min_diag
                    diag_JTJ[i] = min_diag
                end
            end
            
            reg_param = 1e-6
            reg_JTJ = JTJ + reg_param * Diagonal(diag_JTJ)
            
            # Solve system
            newton_step = zeros(n_markers)
            try
                newton_step = reg_JTJ \ (J' * F)
            catch e
                println("Matrix solver failed, using SVD as backup")
                F_svd = svd(Matrix(reg_JTJ))
                svd_tol = eps(Float64) * max(size(reg_JTJ)...) * maximum(F_svd.S)
                S_inv = [s > svd_tol ? 1/s : 0.0 for s in F_svd.S]
                JTF = J' * F
                newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
            end
            
            # Calculate position increment norm
            position_increment_norm = α * norm(newton_step)
            push!(position_increment_history, position_increment_norm)
            
            # Apply the step
            displacements -= α * newton_step
            
            # Smooth the displacements for stability
            smooth_displacements_3D!(displacements, markers, true, smooth_factor, window_size)
            
            # Print maximum displacement for diagnostics
            max_disp = maximum(abs.(displacements))
            println("Maximum displacement (after smoothing): $max_disp")
            
            # Calculate residual norm for convergence check
            residual_norm = norm(F)
            push!(residual_norm_history, residual_norm)
            
            # Report progress
            println("Iteration $iter | Residual = $residual_norm")
            
            # Check convergence
            if residual_norm < tol
                println("Converged after $iter iterations with residual $residual_norm")
                break
            end
            
            # 6. Update marker positions
            new_markers = copy(markers)
            for i in 1:n_markers
                normal = normals[i]
                new_markers[i] = (
                    markers[i][1] + displacements[i] * normal[1],
                    markers[i][2] + displacements[i] * normal[2],
                    markers[i][3] + displacements[i] * normal[3]
                )
            end
            
            # Print mean radius for diagnostic (assuming spherical interface)
            center_x = sum(m[1] for m in new_markers) / length(new_markers)
            center_y = sum(m[2] for m in new_markers) / length(new_markers)
            center_z = sum(m[3] for m in new_markers) / length(new_markers)
            
            mean_radius = mean([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2 + (m[3] - center_z)^2) for m in new_markers])
            println("Mean radius: $(round(mean_radius, digits=6))")
            
            # 7. Create updated front tracking object
            updated_front = FrontTracker3D(new_markers)
            
            # 8. Create space-time level set for capacity calculation
            body = (x, y, z, t_local, _=0) -> begin
                τ = (t_local - tₙ) / Δt
                sdf1 = -sdf_3d(front, x, y, z)
                sdf2 = -sdf_3d(updated_front, x, y, z)
                return (1 - τ) * sdf1 + τ * sdf2
            end
            
            # 9. Update space-time mesh and capacity
            STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
            capacity = Capacity(body, STmesh; compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase_updated = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)
            
            # 10. Rebuild the matrix system
            s.A = A_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, phase_updated.source, 
                                            bc, Tᵢ, Δt, tₙ, scheme)
            
            BC_border_mono!(s.A, s.b, bc_b, mesh; t=tₙ₊₁)
            
            # 11. Update phase for next iteration
            phase = phase_updated

            body_3d = (x, y, z, _=0) -> body(x, y, z, tₙ₊₁)
            capacity_3d = Capacity(body_3d, mesh; compute_centroids=false)
            operator_3d = DiffusionOps(capacity_3d)
            phase_3d = Phase(capacity_3d, operator_3d, phase.source, phase.Diffusion_coeff)
        end
        
        # Store residuals from this time step
        residuals[timestep] = residual_norm_history
        position_increments[timestep] = position_increment_history

        # Update front with new marker positions
        new_markers = copy(markers)
        for i in 1:n_markers
            normal = normals[i]
            new_markers[i] = (
                markers[i][1] + displacements[i] * normal[1],
                markers[i][2] + displacements[i] * normal[2],
                markers[i][3] + displacements[i] * normal[3]
            )
        end
        
        # Update front with new markers
        set_markers_3d!(front, new_markers)
        
        # Store updated interface position
        xf_log[timestep+1] = new_markers
        
        # Store solution
        push!(s.states, s.x)
        
        println("Time: $(round(t, digits=6))")
        println("Max temperature: $(maximum(abs.(s.x)))")
    
        # Increment timestep counter
        timestep += 1
    end
    
    return s, residuals, xf_log, timestep_history, phase_3d, position_increments
end
