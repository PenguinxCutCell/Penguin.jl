# Full Moving 3D - Diffusion - Unsteady - Monophasic

"""
    solve_MovingLiquidDiffusionUnsteadyMono3D!(s, phase, Interface_position, Hₙ⁰, sₙ, Δt, Tₑ, bc_b, bc, ic, mesh, scheme; kwargs...)

Solve the unsteady diffusion problem with moving interface in a monophasic 3D problem.
The interface is tracked using a height function approach.

# Arguments
- `s::Solver`: The solver object
- `phase::Phase`: The phase object containing capacity and operator
- `Interface_position`: Initial interface position (2D array for 3D problem)
- `Hₙ⁰`: Initial height profile (2D array for 3D problem)
- `sₙ`: Initial interface function sₙ(y, z)
- `Δt::Float64`: Time step
- `Tₑ::Float64`: End time
- `bc_b::BorderConditions`: Border conditions
- `bc::AbstractBoundary`: Interface boundary condition
- `ic::InterfaceConditions`: Interface conditions
- `mesh`: Spatial mesh
- `scheme::String`: Time integration scheme ("BE" or "CN")

# Keyword Arguments
- `interpo="linear"`: Interpolation method ("linear", "quad", or "cubic")
- `Newton_params=(1000, 1e-10, 1e-10, 1.0)`: Newton solver parameters (max_iter, tol, reltol, α)
- `cfl_target=0.5`: Target CFL number for adaptive timestep
- `Δt_min=1e-4`: Minimum timestep
- `Δt_max=1.0`: Maximum timestep
- `adaptive_timestep=true`: Whether to use adaptive timestepping
- `method=Base.:\\`: Linear solver method
- `algorithm=nothing`: Algorithm for linear solver
"""
function solve_MovingLiquidDiffusionUnsteadyMono3D!(s::Solver, phase::Phase, Interface_position, Hₙ⁰, sₙ, Δt::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh, scheme::String; interpo="linear", Newton_params=(1000, 1e-10, 1e-10, 1.0), cfl_target=0.5,
    Δt_min=1e-4,
    Δt_max=1.0,
    adaptive_timestep=true, method=Base.:\, 
    algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the problem:")
    println("- Moving problem")
    println("- Non prescribed motion")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- 3D Diffusion problem")

    # Solve system for the initial condition
    t=0.0
    println("Time : $(t)")

    # Params
    ρL = ic.flux.value
    max_iter = Newton_params[1]
    tol      = Newton_params[2]
    reltol   = Newton_params[3]
    α        = Newton_params[4]

    # Log residuals and interface positions for each time step:
    nt = Int(round(Tₑ/Δt))
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = []
    reconstruct = []
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))

    # Determine dimensions (should be 4D for 3D space + time)
    dims = phase.operator.size
    len_dims = length(dims)
    spatial_shape = spatial_shape_from_dims(dims)
    
    if len_dims != 4
        error("3D solver requires 4-dimensional operator size (nx, ny, nz, nt), got $len_dims dimensions.")
    end
    
    nx = spatial_shape[1]
    ny = spatial_shape[2]
    nz = spatial_shape[3]
    n = nx * ny * nz

    # Initialize newton variables
    err = Inf
    err_rel = Inf
    iter = 0

    # Initialize newton height variables - Hₙ⁰ should be (ny x nz) matrix
    current_Hₙ = Hₙ⁰
    new_Hₙ = copy(current_Hₙ)

    # Initialize newton interface position variables
    current_xf = Interface_position
    new_xf = copy(current_xf)
    xf = copy(current_xf)
    
    # First time step : Newton to compute the interface position xf1
    while (iter < max_iter) && (err > tol) && (err_rel > reltol)
        iter += 1

        # 1) Solve the linear system
        solve_system!(s; method=method, algorithm=algorithm, kwargs...)
        Tᵢ = s.x

        # 2) Recompute heights
        Hₙ, Hₙ₊₁ = extract_height_profiles(phase.capacity, dims)

        # 3) Compute the interface flux term
        W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
        G  = phase.operator.G[1:end÷2, 1:end÷2]
        H  = phase.operator.H[1:end÷2, 1:end÷2]
        V  = phase.operator.V[1:end÷2, 1:end÷2]
        Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id = Id[1:end÷2, 1:end÷2]
        Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
        Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ

        # Reshape to 3D spatial array and sum along x-direction (dim 1)
        Interface_term = reshape(Interface_term, (nx, ny, nz))
        Interface_term = 1/(ρL) * dropdims(sum(Interface_term, dims=1), dims=1)  # (ny x nz)

        # 4) Update the height function
        res = Hₙ₊₁ - Hₙ - Interface_term
        new_Hₙ = current_Hₙ .+ α .* res  # Elementwise update

        # Compute error using a representative point
        mid_y, mid_z = max(1, size(current_Hₙ, 1) ÷ 2), max(1, size(current_Hₙ, 2) ÷ 2)
        err = abs(new_Hₙ[mid_y, mid_z] - current_Hₙ[mid_y, mid_z])
        err_rel = err / max(abs(current_xf[mid_y, mid_z]), eps())
        println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

        # Store residuals
        if !haskey(residuals, 1)
            residuals[1] = Float64[]
        end
        push!(residuals[1], err)

        # 5) Update geometry if not converged
        if (err <= tol) || (err_rel <= reltol)
            push!(xf_log, copy(new_xf))
            break
        end

        # Store tn+1 and tn
        tₙ₊₁ = t + Δt
        tₙ  = t

        # 6) Compute the new interface position table
        new_xf = interface_positions_from_heights(new_Hₙ, mesh)

        # 7) Construct interpolation function for new interface position: sₙ₊₁(y, z)
        y_centroids = mesh.centers[2]
        z_centroids = mesh.centers[3]
        
        sₙ₊₁ = bilinear_interpolation_3d(y_centroids, z_centroids, new_xf)

        # 8) Rebuild the domain with time interpolation
        body = (xx, yy, zz, tt) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0
            b = -4.0 * t_norm^2 + 4.0 * t_norm
            c = 2.0 * t_norm^2 - t_norm
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy, zz)
            pos_mid = 0.5 * (sₙ(yy, zz) + sₙ₊₁(yy, zz))
            pos_end = sₙ₊₁(yy, zz)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return xx - x_interp
        end
        STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
        capacity = Capacity(body, STmesh; compute_centroids=false)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        # 9) Rebuild the matrix A and the vector b
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, t, scheme)

        BC_border_mono!(s.A, s.b, bc_b, mesh; t=tₙ₊₁)

        # 10) Update variables
        current_Hₙ = copy(new_Hₙ)
        current_xf = copy(new_xf)
    end

    if (err <= tol) || (err_rel <= reltol)
        println("Converged after $iter iterations with Hₙ = $(maximum(new_Hₙ)), error = $err")
    else
        println("Reached max_iter = $max_iter with Hₙ = $(maximum(new_Hₙ)), error = $err")
    end

    Tᵢ = s.x
    push!(s.states, s.x)
    println("Time : $(t)")
    println("Max value : $(maximum(abs.(s.x)))")
    

    # Time loop
    k=2
    cfl_current = 0.0  # Track CFL for reporting
    while t<Tₑ
        # Compute interface velocity from fluxes
        W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
        G  = phase.operator.G[1:end÷2, 1:end÷2]
        H  = phase.operator.H[1:end÷2, 1:end÷2]
        V  = phase.operator.V[1:end÷2, 1:end÷2]
        Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id = Id[1:end÷2, 1:end÷2]
        Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
        Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        velocity_field = 1/(ρL) * abs.(Interface_term)
        
        # Adaptive timestep if requested
        if adaptive_timestep
            # Limit time left to not exceed Tₑ
            time_left = Tₑ - t
            Δt_max_current = min(Δt_max, time_left)
            
            Δt, cfl_current = adapt_timestep(velocity_field, mesh, cfl_target, Δt, Δt_min, Δt_max_current; 
                                  growth_factor=1.1, shrink_factor=0.8, safety_factor=0.9)
            
            push!(timestep_history, (t, Δt))
            println("Adaptive timestep: Δt = $(round(Δt, digits=6)), CFL = $(round(cfl_current, digits=3))")
        end

        # Update the time
        t+=Δt
        tₙ = t
        tₙ₊₁ = t + Δt
        println("Time : $(round(t, digits=6))")

        # 1) Construct an interpolation function for the interface position
        y_centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        z_centroids = range(mesh.nodes[3][1], mesh.nodes[3][end], length=length(mesh.nodes[3]))
        
        sₙ = bilinear_interpolation_3d(y_centroids, z_centroids, current_xf)
        sₙ₊₁ = bilinear_interpolation_3d(y_centroids, z_centroids, new_xf)

        # 1) Reconstruct
        STmesh = SpaceTimeMesh(mesh, [Δt, 2Δt], tag=mesh.tag)
        body = (xx, yy, zz, tt) -> begin
            # Normalized time parameter
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0
            b = -4.0 * t_norm^2 + 4.0 * t_norm
            c = 2.0 * t_norm^2 - t_norm
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy, zz)
            pos_mid = 0.5 * (sₙ(yy, zz) + sₙ₊₁(yy, zz))
            pos_end = sₙ₊₁(yy, zz)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return xx - x_interp
        end
        capacity = Capacity(body, STmesh; compute_centroids=false)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, 0.0, scheme)

        BC_border_mono!(s.A, s.b, bc_b, mesh; t=tₙ₊₁)

        push!(reconstruct, sₙ₊₁)

        # Initialize newton variables
        err = Inf
        err_rel = Inf
        iter = 0

        # Initialize newton height variables
        current_Hₙ = copy(new_Hₙ)
        new_Hₙ = copy(current_Hₙ)

        # Initialize newton interface position variables
        current_xf = copy(new_xf)
        new_xf = copy(current_xf)
        xf = copy(current_xf)

        # Newton to compute the interface position xf1
        while (iter < max_iter) && (err > tol) && (err_rel > reltol)
            iter += 1

            # 1) Solve the linear system
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            Tᵢ = s.x

            # 2) Recompute heights
            Hₙ, Hₙ₊₁ = extract_height_profiles(phase.capacity, dims)

            # 3) Compute the interface flux term
            W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
            G  = phase.operator.G[1:end÷2, 1:end÷2]
            H  = phase.operator.H[1:end÷2, 1:end÷2]
            V  = phase.operator.V[1:end÷2, 1:end÷2]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:end÷2, 1:end÷2]
            Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
            Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ

            # Reshape to 3D spatial array and sum along x-direction
            Interface_term = reshape(Interface_term, (nx, ny, nz))
            Interface_term = 1/(ρL) * dropdims(sum(Interface_term, dims=1), dims=1)

            # 4) Update the height function
            res = Hₙ₊₁ - Hₙ - Interface_term
            new_Hₙ = current_Hₙ .+ α .* res

            mid_y, mid_z = max(1, size(current_Hₙ, 1) ÷ 2), max(1, size(current_Hₙ, 2) ÷ 2)
            err = abs(new_Hₙ[mid_y, mid_z] - current_Hₙ[mid_y, mid_z])
            err_rel = err / max(abs(current_xf[mid_y, mid_z]), eps())
            println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

            # Store residuals
            if !haskey(residuals, k)
                residuals[k] = Float64[]
            end
            push!(residuals[k], err)

            # 5) Update geometry if not converged
            if (err <= tol) || (err_rel <= reltol)
                push!(xf_log, copy(new_xf))
                break
            end

            # Store tn+1 and tn
            tₙ₊₁ = t + Δt
            tₙ  = t

            # 6) Compute the new interface position table
            new_xf = interface_positions_from_heights(new_Hₙ, mesh)
            ensure_periodic_3d!(new_xf)

            # 7) Construct interpolation function for new interface position
            y_centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
            z_centroids = range(mesh.nodes[3][1], mesh.nodes[3][end], length=length(mesh.nodes[3]))
            
            sₙ₊₁ = bilinear_interpolation_3d(y_centroids, z_centroids, new_xf)

            # 8) Rebuild the domain
            body = (xx, yy, zz, tt) -> begin
                t_norm = (tt - tₙ) / Δt
                
                a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0
                b = -4.0 * t_norm^2 + 4.0 * t_norm
                c = 2.0 * t_norm^2 - t_norm
                
                pos_start = sₙ(yy, zz)
                pos_mid = 0.5 * (sₙ(yy, zz) + sₙ₊₁(yy, zz))
                pos_end = sₙ₊₁(yy, zz)
                
                x_interp = a * pos_start + b * pos_mid + c * pos_end
                
                return xx - x_interp
            end
            STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
            capacity = Capacity(body, STmesh; compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

            # 9) Rebuild the matrix A and the vector b
            s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, 0.0, scheme)

            BC_border_mono!(s.A, s.b, bc_b, mesh; t=tₙ₊₁)

            # 10) Update variables
            current_Hₙ = copy(new_Hₙ)
            current_xf = copy(new_xf)

        end

        if (err <= tol) || (err_rel <= reltol)
            println("Converged after $iter iterations with xf = $(maximum(new_Hₙ)), error = $err")
        else
            println("Reached max_iter = $max_iter with xf = $(maximum(new_Hₙ)), error = $err")
        end

        # Display timestep info
        if adaptive_timestep
            println("Time step info: Δt = $(round(Δt, digits=6)), CFL = $(round(cfl_current, digits=3))")
        end

        push!(s.states, s.x)
        println("Time : $(t)")
        println("Max value : $(maximum(abs.(s.x)))")
        k+=1
    end
    return s, residuals, xf_log, reconstruct, timestep_history
end 


"""
    bilinear_interpolation_3d(y_coords, z_coords, values)

Create a bilinear interpolation function for 3D interface position data.
Returns a function sₙ(y, z) that interpolates the interface position.

# Arguments
- `y_coords`: Y-coordinates of the grid points
- `z_coords`: Z-coordinates of the grid points
- `values`: 2D array of interface positions (ny x nz)
"""
function bilinear_interpolation_3d(y_coords, z_coords, values)
    ny = length(y_coords)
    nz = length(z_coords)
    
    # Ensure values is a proper 2D array
    vals = collect(values)
    if ndims(vals) == 1
        # If values is 1D, reshape to 2D
        vals = reshape(vals, (ny, nz))
    end
    
    function interpolate(y, z)
        # Find indices for y
        iy = 1
        while iy < ny && y_coords[iy+1] <= y
            iy += 1
        end
        iy = clamp(iy, 1, ny-1)
        
        # Find indices for z
        iz = 1
        while iz < nz && z_coords[iz+1] <= z
            iz += 1
        end
        iz = clamp(iz, 1, nz-1)
        
        # Get coordinates of corners
        y1, y2 = y_coords[iy], y_coords[min(iy+1, ny)]
        z1, z2 = z_coords[iz], z_coords[min(iz+1, nz)]
        
        # Get values at corners
        Q11 = vals[iy, iz]
        Q12 = vals[iy, min(iz+1, nz)]
        Q21 = vals[min(iy+1, ny), iz]
        Q22 = vals[min(iy+1, ny), min(iz+1, nz)]
        
        # Compute bilinear interpolation weights
        if abs(y2 - y1) < eps() || abs(z2 - z1) < eps()
            # Degenerate case: return average
            return (Q11 + Q12 + Q21 + Q22) / 4
        end
        
        ty = (y - y1) / (y2 - y1)
        tz = (z - z1) / (z2 - z1)
        
        # Clamp weights to [0, 1]
        ty = clamp(ty, 0.0, 1.0)
        tz = clamp(tz, 0.0, 1.0)
        
        # Bilinear interpolation
        return (1 - ty) * (1 - tz) * Q11 + 
               ty * (1 - tz) * Q21 + 
               (1 - ty) * tz * Q12 + 
               ty * tz * Q22
    end
    
    return interpolate
end


"""
    ensure_periodic_3d!(positions)

For 2D matrix-valued interface positions (3D problem), ensure periodicity
in both y and z directions by matching the edges.
"""
function ensure_periodic_3d!(positions::AbstractMatrix)
    ny, nz = size(positions)
    
    # Match last row to first row (y periodicity)
    if ny > 1
        positions[end, :] = positions[1, :]
    end
    
    # Match last column to first column (z periodicity)
    if nz > 1
        positions[:, end] = positions[:, 1]
    end
    
    return positions
end
