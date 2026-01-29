# Full Moving 2D - Diffusion - Unsteady - Monophasic

function solve_MovingLiquidDiffusionUnsteadyMono2D!(s::Solver, phase::Phase, Interface_position, Hₙ⁰, sₙ, Δt::Float64, Tstart::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh, scheme::String; interpo="linear", Newton_params=(1000, 1e-10, 1e-10, 1.0), cfl_target=0.5,
    Δt_min=1e-4,
    Δt_max=1.0,
    adaptive_timestep=true, method=IterativeSolvers.gmres, 
    algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the problem:")
    println("- Moving problem")
    println("- Non prescibed motion")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Solve system for the initial condition
    t=Tstart
    println("Time : $(t)")

    # Params
    ρL = ic.flux.value
    max_iter = Newton_params[1]
    tol      = Newton_params[2]
    reltol   = Newton_params[3]
    α        = Newton_params[4]

    # Log residuals and interface positions for each time step:
    nt = Int(round((Tₑ - Tstart)/Δt))
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = []
    reconstruct = []
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))

    # Determine how many dimensions
    dims = phase.operator.size
    len_dims = length(dims)
    spatial_shape = spatial_shape_from_dims(dims)
    nx = spatial_shape[1]
    ny = length(spatial_shape) >= 2 ? spatial_shape[2] : 1

    # Create the 1D or 2D indices
    if len_dims == 2
        # 1D case
        nt = dims[2]
        n = nx
    elseif len_dims == 3
        # 2D case
        nt = dims[3]
        n = nx * ny
    else
        error("Only 1D and 2D problems are supported.")
    end

    # Initialize newton variables
    err = Inf
    err_rel = Inf
    iter = 0

    # Initialize newton height variables
    current_Hₙ = Hₙ⁰
    new_Hₙ = current_Hₙ

    # Initialize newton interface position variables
    current_xf = Interface_position
    new_xf = current_xf
    xf = current_xf
    T_prev = s.x
    T_prev === nothing && error("Initial temperature state is not set (s.x is nothing). Set s.x before solving.")
    Tᵢ = T_prev
    
    # First time step : Newton to compute the interface position xf1
    while (iter < max_iter) && (err > tol) && (err_rel > reltol)
        iter += 1

        # 1) Solve the linear system
        solve_system!(s; method=method, algorithm=algorithm, kwargs...)
        T_trial = s.x

        # 2) Recompute heights
        Hₙ, Hₙ₊₁ = extract_height_profiles(phase.capacity, dims)

        # 3) Compute the interface flux term
        W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
        G  = phase.operator.G[1:end÷2, 1:end÷2]
        H  = phase.operator.H[1:end÷2, 1:end÷2]
        V  = phase.operator.V[1:end÷2, 1:end÷2]
        Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id = Id[1:end÷2, 1:end÷2]
        Tₒ, Tᵧ = T_trial[1:end÷2], T_trial[end÷2+1:end]
        Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        
        # Check if bc is a Gibbs-Thomson condition
        if bc isa GibbsThomson
            velocity = 1/(ρL) * abs.(Interface_term) / Δt # Still need to find the interface velocity. Right now i've got the lower veloc
            @show velocity
            np = prod(phase.operator.size)
            # Complete the velocity vector with zeros
            velocity = vcat(velocity, zeros(np-length(velocity)))
            bc.vᵞ = velocity
        end

        Interface_term = reshape(Interface_term, (nx, ny))
        Interface_term = 1/(ρL) * vec(sum(Interface_term, dims=1))
        #println("Interface term: ", Interface_term)

        # 4) Update the height function
        res = Hₙ₊₁ - Hₙ - Interface_term
        #println("res: ", res)
        new_Hₙ = current_Hₙ .+ α .* res            # Elementwise update for each column
        #println("new_Hₙ: ", new_Hₙ)
        err = abs.(new_Hₙ[5] .- current_Hₙ[5])
        err_rel = err/maximum(abs.(current_xf[5]))
        println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

        # Store residuals (if desired, you could store the full vector or simply the norm)
        if !haskey(residuals, 1)
            residuals[1] = Float64[]
        end
        push!(residuals[1], err)

        # 5) Update geometry if not converged
        if (err <= tol) || (err_rel <= reltol) || (iter == max_iter)
            push!(xf_log, new_xf)
            break
        end

        # Store tn+1 and tn
        tₙ₊₁ = t + Δt
        tₙ  = t

        # 6) Compute the new interface position table
        new_xf = interface_positions_from_heights(new_Hₙ, mesh)
        if length(new_xf) > 1
            new_xf = new_xf[1:end-1]
        end
        # 7) Construct a interpolation function for the new interface position : sn and sn+1
        centroids = mesh.centers[1]
        if interpo == "linear"
            #sₙ₊₁ = height_interpolation_linear(centroids, new_xf)
            #sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
            sₙ₊₁ = lin_interpol(centroids, new_xf)
        elseif interpo == "quad"
            #sₙ₊₁ = height_interpolation_quadratic(centroids, new_xf)
            #sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
            sₙ₊₁ = quad_interpol(centroids, new_xf)
        elseif interpo == "cubic"
            #sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
            sₙ₊₁ = cubic_interpol(centroids, new_xf)
        else
            println("Interpolation method not supported")
        end

        # 8) Rebuild the domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
        body = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return xx - x_interp
        end
        STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
        capacity = Capacity(body, STmesh; compute_centroids=false, method="VOFI", integration_method=:vofijul)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        # 9) Rebuild the matrix A and the vector b
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, T_prev, Δt, t, scheme)

        BC_border_mono!(s.A, s.b, bc_b, mesh; t=tₙ₊₁)

        # 10) Update variables
        current_Hₙ = new_Hₙ
        current_xf = new_xf
    end

    if (err <= tol) || (err_rel <= reltol)
        println("Converged after $iter iterations with Hₙ = $new_Hₙ, error = $err")
    else
        println("Reached max_iter = $max_iter with Hₙ = $new_Hₙ, error = $err")
    end

    Tᵢ = s.x
    push!(s.states, s.x)
    println("Time : $(t[1])")
    println("Max value : $(maximum(abs.(s.x)))")
    

    # Time loop
    k=2
    while t<Tₑ
        # Calcul de la vitesse d'interface à partir des flux
        W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
        G  = phase.operator.G[1:end÷2, 1:end÷2]
        H  = phase.operator.H[1:end÷2, 1:end÷2]
        V  = phase.operator.V[1:end÷2, 1:end÷2]
        Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id = Id[1:end÷2, 1:end÷2]
        Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
        Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        velocity_field = 1/(ρL) * abs.(Interface_term)
        
        # Adaptation du pas de temps si demandée
        if adaptive_timestep
            # Limiter le temps restant pour ne pas dépasser Tₑ
            time_left = Tₑ - t
            Δt_max_current = min(Δt_max, time_left)
            
            # Remarquez les paramètres nommés pour les facteurs
            Δt, cfl = adapt_timestep(velocity_field, mesh, cfl_target, Δt, Δt_min, Δt_max_current; 
                                  growth_factor=1.1, shrink_factor=0.8, safety_factor=0.9)
            
            push!(timestep_history, (t, Δt))
            println("Adaptive timestep: Δt = $(round(Δt, digits=6)), CFL = $(round(cfl, digits=3))")
        end

        # Update the time
        t+=Δt
        tₙ = t
        tₙ₊₁ = t + Δt
        println("Time : $(round(t, digits=6))")

        # 1) Construct an interpolation function for the interface position
        centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        if interpo == "linear"
            #sₙ = linear_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
            #sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
            sₙ = lin_interpol(centroids, current_xf)
            sₙ₊₁ = lin_interpol(centroids, new_xf)
        elseif interpo == "quad"
            #sₙ = extrapolate(scale(interpolate(current_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
            #sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
            sₙ = quad_interpol(centroids, current_xf)
            sₙ₊₁ = quad_interpol(centroids, new_xf)
        elseif interpo == "cubic"
            #sₙ = cubic_spline_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
            #sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
            sₙ = cubic_interpol(centroids, current_xf)
            sₙ₊₁ = cubic_interpol(centroids, new_xf)
        else
            println("Interpolation method not supported")
        end

        # 1) Reconstruct
        STmesh = SpaceTimeMesh(mesh, [Δt, 2Δt], tag=mesh.tag)
        body = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return xx - x_interp
        end
        capacity = Capacity(body, STmesh; compute_centroids=false, method="VOFI", integration_method=:vofijul)
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
        current_Hₙ = new_Hₙ
        new_Hₙ = current_Hₙ

        # Initialize newton interface position variables
        current_xf = new_xf
        new_xf = current_xf
        xf = current_xf
        T_prev = Tᵢ

        # Newton to compute the interface position xf1
        while (iter < max_iter) && (err > tol) && (err_rel > reltol)
            iter += 1

            # 1) Solve the linear system
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            T_trial = s.x

            # 2) Recompute heights
            Hₙ, Hₙ₊₁ = extract_height_profiles(phase.capacity, dims)

            # 3) Compute the interface flux term
            W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
            G  = phase.operator.G[1:end÷2, 1:end÷2]
            H  = phase.operator.H[1:end÷2, 1:end÷2]
            V  = phase.operator.V[1:end÷2, 1:end÷2]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:end÷2, 1:end÷2]
            Tₒ, Tᵧ = T_trial[1:end÷2], T_trial[end÷2+1:end]
            Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
                    
            # Check if bc is a Gibbs-Thomson condition
            if bc isa GibbsThomson
                velocity = 1/(ρL) * abs.(Interface_term)/ Δt # Still need to find the interface velocity
                @show velocity
                np = prod(phase.operator.size)
                # Complete the velocity vector with zeros
                velocity = vcat(velocity, zeros(np-length(velocity)))
                bc.vᵞ = velocity
            end

            Interface_term = reshape(Interface_term, (nx, ny))
            Interface_term = 1/(ρL) * vec(sum(Interface_term, dims=1))
            #println("Interface term: ", Interface_term)

            # 4) Update the height function
            res = Hₙ₊₁ - Hₙ - Interface_term
            #println("res: ", res)
            new_Hₙ = current_Hₙ .+ α .* res            # Elementwise update for each column
            #println("new_Hₙ: ", new_Hₙ)
            err = abs.(new_Hₙ[5] .- current_Hₙ[5])
            err_rel = err/maximum(abs.(current_xf[5]))
            println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

            # Store residuals (if desired, you could store the full vector or simply the norm)
            if !haskey(residuals, k)
                residuals[k] = Float64[]
            end
            push!(residuals[k], err)

            # 5) Update geometry if not converged
            if (err <= tol) || (err_rel <= reltol) || (iter == max_iter)
                push!(xf_log, new_xf)
                break
            end

            # Store tn+1 and tn
            tₙ₊₁ = t + Δt
            tₙ  = t

            # 6) Compute the new interface position table
            new_xf = interface_positions_from_heights(new_Hₙ, mesh)
            ensure_periodic!(new_xf)

            # 7) Construct a interpolation function for the new interface position :
            centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
            if interpo == "linear"
                #sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
                sₙ₊₁ = lin_interpol(centroids, new_xf)
            elseif interpo == "quad"
                #sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
                sₙ₊₁ = quad_interpol(centroids, new_xf)
            elseif interpo == "cubic"
                #sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Periodic())
                sₙ₊₁ = cubic_interpol(centroids, new_xf)
            else
                println("Interpolation method not supported")
            end

            # 8) Rebuild the domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
            body = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return xx - x_interp
            end
            STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
            capacity = Capacity(body, STmesh; compute_centroids=false, method="VOFI", integration_method=:vofijul)
            operator = DiffusionOps(capacity)
            phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

            # 9) Rebuild the matrix A and the vector b
            s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, T_prev, Δt, 0.0, scheme)

            BC_border_mono!(s.A, s.b, bc_b, mesh; t=tₙ₊₁)

            # 10) Update variables
            current_Hₙ = new_Hₙ
            current_xf = new_xf

        end

        if (err <= tol) || (err_rel <= reltol)
            println("Converged after $iter iterations with xf = $new_Hₙ, error = $err")
        else
            println("Reached max_iter = $max_iter with xf = $new_Hₙ, error = $err")
        end

        # Afficher les informations du pas de temps
        if adaptive_timestep
            println("Time step info: Δt = $(round(Δt, digits=6)), CFL = $(round(timestep_history[end][2], digits=3))")
        end

        Tᵢ = s.x
        push!(s.states, s.x)
        println("Time : $(t[1])")
        println("Max value : $(maximum(abs.(s.x)))")
        k+=1
    end
return s, residuals, xf_log, reconstruct, timestep_history
end 



# Moving - Diffusion - Unsteady - Diphasic
function A_diph_unstead_diff_moving_stef2(operator1::DiffusionOps, operator2::DiffusionOps, capacity1::Capacity, capacity2::Capacity, D1, D2, ic::InterfaceConditions, scheme::String)
    # Determine dimensionality from operator1
    dims1 = operator1.size
    dims2 = operator2.size
    len_dims1 = length(dims1)
    len_dims2 = length(dims2)

    # For both phases, define n1 and n2 as total dof
    n1 = prod(dims1)
    n2 = prod(dims2)

    # If 1D => n = nx; if 2D => n = nx*ny
    # (We use the same dimension logic for each operator.)
    if len_dims1 == 2
        # 1D problem
        nx1, _ = dims1
        nx2, _ = dims2
        n = nx1  # used for sub-block sizing
    elseif len_dims1 == 3
        # 2D problem
        nx1, ny1, _ = dims1
        nx2, ny2, _ = dims2
        n = nx1 * ny1
    else
        error("Only 1D or 2D supported, got dimension: $len_dims1")
    end

    # Retrieve jump & flux from the interface conditions
    jump, flux = ic.scalar, ic.flux


    # Build diffusion operators
    Id1 = build_I_D(operator1, D1, capacity1)
    Id2 = build_I_D(operator2, D2, capacity2)

    # Capacity indexing (2 for 1D, 3 for 2D)
    cap_index1 = len_dims1
    cap_index2 = len_dims2

    # Extract Vr−1 and Vr
    Vn1_1 = capacity1.A[cap_index1][1:end÷2, 1:end÷2]
    Vn1   = capacity1.A[cap_index1][end÷2+1:end, end÷2+1:end]
    Vn2_1 = capacity2.A[cap_index2][1:end÷2, 1:end÷2]
    Vn2   = capacity2.A[cap_index2][end÷2+1:end, end÷2+1:end]

    # Time integration weighting
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else
        psip, psim = psip_be, psim_be
    end

    Ψn1 = Diagonal(psip.(Vn1, Vn1_1))
    Ψn2 = Diagonal(psip.(Vn2, Vn2_1))

    Iₐ1, Iₐ2 = jump.α₁ * Penguin.I(size(Ψn1)[1]), jump.α₂ * Penguin.I(size(Ψn2)[1])
    Iᵦ1, Iᵦ2 = flux.β₁ * Penguin.I(size(Ψn1)[1]), flux.β₂ * Penguin.I(size(Ψn2)[1])


    # Operator sub-blocks for each phase
    W!1 = operator1.Wꜝ[1:end÷2, 1:end÷2]
    G1  = operator1.G[1:end÷2, 1:end÷2]
    H1  = operator1.H[1:end÷2, 1:end÷2]

    W!2 = operator2.Wꜝ[1:end÷2, 1:end÷2]
    G2  = operator2.G[1:end÷2, 1:end÷2]
    H2  = operator2.H[1:end÷2, 1:end÷2]

    Iᵦ1 = Iᵦ1
    Iᵦ2 = Iᵦ2
    Iₐ1 = Iₐ1
    Iₐ2 = Iₐ2
    Id1 = Id1[1:end÷2, 1:end÷2]
    Id2 = Id2[1:end÷2, 1:end÷2]

    # Construct blocks
    block1 = Vn1_1 + Id1 * G1' * W!1 * G1 * Ψn1
    block2 = -(Vn1_1 - Vn1) + Id1 * G1' * W!1 * H1 * Ψn1
    block3 = Vn2_1 + Id2 * G2' * W!2 * G2 * Ψn2
    block4 = -(Vn2_1 - Vn2) + Id2 * G2' * W!2 * H2 * Ψn2

    block5 = Iᵦ1 * H1' * W!1 * G1 * Ψn1
    block6 = Iᵦ1 * H1' * W!1 * H1 * Ψn1
    block7 = Iᵦ2 * H2' * W!2 * G2 * Ψn2
    block8 = Iᵦ2 * H2' * W!2 * H2 * Ψn2

    # Build the 4n×4n matrix
    A = spzeros(Float64, 4n, 4n)

    # Assign sub-blocks

    A1 = [block1 block2 spzeros(size(block2)) spzeros(size(block2))]

    A2 = [spzeros(size(block1)) Iₐ1 spzeros(size(block2)) -Iₐ2]


    A3 = [spzeros(size(block1)) spzeros(size(block2)) block3 block4]

    A4 = [spzeros(size(block1)) spzeros(size(block2)) spzeros(size(block3)) Iₐ2]

    A = [A1; A2; A3; A4]

    return A
end

function b_diph_unstead_diff_moving_stef2(operator1::DiffusionOps, operator2::DiffusionOps, capacity1::Capacity, capacity2::Capacity, D1, D2, f1::Function, f2::Function, ic::InterfaceConditions, Tᵢ::Vector{Float64}, Δt::Float64, t::Float64, scheme::String)
    # 1) Determine total degrees of freedom for each operator
    dims1 = operator1.size
    dims2 = operator2.size
    len_dims1 = length(dims1)
    len_dims2 = length(dims2)

    n1 = prod(dims1)  # total cells in phase 1
    n2 = prod(dims2)  # total cells in phase 2

    # 2) Identify which capacity index to read (2 for 1D, 3 for 2D)
    cap_index1 = len_dims1
    cap_index2 = len_dims2

    # 3) Build the source terms
    f1ₒn  = build_source(operator1, f1, t,      capacity1)
    f1ₒn1 = build_source(operator1, f1, t+Δt,  capacity1)
    f2ₒn  = build_source(operator2, f2, t,      capacity2)
    f2ₒn1 = build_source(operator2, f2, t+Δt,  capacity2)

    # 4) Build interface data
    jump, flux = ic.scalar, ic.flux
    Iᵧ1, Iᵧ2   = capacity1.Γ, capacity2.Γ
    gᵧ  = build_g_g(operator1, jump, capacity1)
    hᵧ  = build_g_g(operator2, flux, capacity2)
    Id1, Id2 = build_I_D(operator1, D1, capacity1), build_I_D(operator2, D2, capacity2)

    # 5) Extract Vr (current) & Vr−1 (previous) from each capacity
    Vn1_1 = capacity1.A[cap_index1][1:end÷2, 1:end÷2]
    Vn1   = capacity1.A[cap_index1][end÷2+1:end, end÷2+1:end]
    Vn2_1 = capacity2.A[cap_index2][1:end÷2, 1:end÷2]
    Vn2   = capacity2.A[cap_index2][end÷2+1:end, end÷2+1:end]

    # 6) Time-integration weighting
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else
        psip, psim = psip_be, psim_be
    end
    Ψn1 = Diagonal(psim.(Vn1, Vn1_1))
    Ψn2 = Diagonal(psim.(Vn2, Vn2_1))

    # 7) Determine whether 1D or 2D from dims1, and form local n for sub-blocks
    if len_dims1 == 2
        # 1D
        nx1, _ = dims1
        nx2, _ = dims2
        n = nx1
    else
        # 2D
        nx1, ny1, _ = dims1
        nx2, ny2, _ = dims2
        n = nx1 * ny1   # local block size for each operator
    end

    # 8) Build the bulk terms for each phase
    Tₒ1 = Tᵢ[1:end÷4]
    Tᵧ1 = Tᵢ[end÷4 + 1 : end÷2]

    Tₒ2 = Tᵢ[end÷2 + 1 : 3end÷4]
    Tᵧ2 = Tᵢ[3end÷4 + 1 : end]

    f1ₒn  = f1ₒn[1:end÷2]
    f1ₒn1 = f1ₒn1[1:end÷2]
    f2ₒn  = f2ₒn[1:end÷2]
    f2ₒn1 = f2ₒn1[1:end÷2]

    gᵧ = gᵧ[1:end÷2]
    hᵧ = hᵧ[1:end÷2]
    Iᵧ1 = Iᵧ1[1:end÷2, 1:end÷2]
    Iᵧ2 = Iᵧ2[1:end÷2, 1:end÷2]
    Id1 = Id1[1:end÷2, 1:end÷2]
    Id2 = Id2[1:end÷2, 1:end÷2]

    W!1 = operator1.Wꜝ[1:end÷2, 1:end÷2]
    G1  = operator1.G[1:end÷2, 1:end÷2]
    H1  = operator1.H[1:end÷2, 1:end÷2]
    V1  = operator1.V[1:end÷2, 1:end÷2]

    W!2 = operator2.Wꜝ[1:end÷2, 1:end÷2]
    G2  = operator2.G[1:end÷2, 1:end÷2]
    H2  = operator2.H[1:end÷2, 1:end÷2]
    V2  = operator2.V[1:end÷2, 1:end÷2]

    # 9) Build the right-hand side
    if scheme == "CN"
        b1 = (Vn1 - Id1 * G1' * W!1 * G1 * Ψn1) * Tₒ1 - 0.5 * Id1 * G1' * W!1 * H1 * Tᵧ1 + 0.5 * V1 * (f1ₒn + f1ₒn1)
        b3 = (Vn2 - Id2 * G2' * W!2 * G2 * Ψn2) * Tₒ2 - 0.5 * Id2 * G2' * W!2 * H2 * Tᵧ2 + 0.5 * V2 * (f2ₒn + f2ₒn1)
    else
        b1 = Vn1 * Tₒ1 + V1 * f1ₒn1
        b3 = Vn2 * Tₒ2 + V2 * f2ₒn1
    end

    # 10) Build boundary terms
    b2 = gᵧ
    b4 = gᵧ

    # Final right-hand side
    return vcat(b1, b2, b3, b4)
end


# Main solver function for the diphasic Stefan problem in 2D
function solve_MovingLiquidDiffusionUnsteadyDiph2D!(s::Solver, phase1::Phase, phase2::Phase, Interface_position, Hₙ⁰,sₙ, Δt::Float64, Tstart, Tₑ::Float64, bc_b::BorderConditions, ic::InterfaceConditions, mesh, scheme::String; interpo="quad", Newton_params=(1000, 1e-10, 1e-10, 1.0), method=IterativeSolvers.gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the problem:")
    println("- Moving problem")
    println("- Non prescibed motion")
    println("- Diphasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Solve system for the initial condition
    t = Tstart
    println("Time : $(t)")

    # Params
    ρL = ic.flux.value
    max_iter = Newton_params[1]
    tol      = Newton_params[2]
    reltol   = Newton_params[3]
    α        = Newton_params[4]

    # Log residuals and interface positions for each time step:
    nt = Int(round((Tₑ - Tstart)/Δt))
    residuals = [[] for _ in 1:2nt]
    xf_log = []
    reconstruct = []

    # Determine how many dimensions
    dims = phase1.operator.size
    len_dims = length(dims)
    spatial_shape = spatial_shape_from_dims(dims)
    nx = spatial_shape[1]
    ny = length(spatial_shape) >= 2 ? spatial_shape[2] : 1

    # Create the 1D or 2D indices
    if len_dims == 2
        # 1D case
        nt = dims[2]
        n = nx
    elseif len_dims == 3
        # 2D case
        nt = dims[3]
        n = nx * ny
    else
        error("Only 1D and 2D problems are supported.")
    end

    # Initialize newton variables
    err = Inf
    err_rel = Inf
    iter = 0

    # Initialize newton height variables
    current_Hₙ = Hₙ⁰
    new_Hₙ = current_Hₙ

    # Initialize newton interface position variables
    current_xf = Interface_position
    new_xf = current_xf
    xf = current_xf
    T_prev = s.x
    T_prev === nothing && error("Initial temperature state is not set (s.x is nothing). Set s.x before solving.")
    Tᵢ = T_prev

    # First time step : Newton to compute the interface position xf1
    while (iter < max_iter) && (err > tol) && (err_rel > reltol)
        iter += 1

        # 1) Solve the linear system
        solve_system!(s; method=method, algorithm=algorithm, kwargs...)
        T_trial = s.x

        # 2) Recompute heights for phase 1
        Hₙ_1, Hₙ₊₁_1 = extract_height_profiles(phase1.capacity, dims)

        # 3) Compute the interface flux term for phase 1
        W!1 = phase1.operator.Wꜝ[1:end÷2, 1:end÷2]
        G1  = phase1.operator.G[1:end÷2, 1:end÷2]
        H1  = phase1.operator.H[1:end÷2, 1:end÷2]
        V1  = phase1.operator.V[1:end÷2, 1:end÷2]
        Id1 = build_I_D(phase1.operator, phase1.Diffusion_coeff, phase1.capacity)
        Id1 = Id1[1:end÷2, 1:end÷2]
        Tₒ1, Tᵧ1 = T_trial[1:end÷4], T_trial[end÷4 + 1 : end÷2]
        Interface_term_1 = Id1 * H1' * W!1 * G1 * Tₒ1 + Id1 * H1' * W!1 * H1 * Tᵧ1

        # 4) Compute flux term for phase 2
        W!2 = phase2.operator.Wꜝ[1:end÷2, 1:end÷2]
        G2  = phase2.operator.G[1:end÷2, 1:end÷2]
        H2  = phase2.operator.H[1:end÷2, 1:end÷2]
        V2  = phase2.operator.V[1:end÷2, 1:end÷2]
        Id2 = build_I_D(phase2.operator, phase2.Diffusion_coeff, phase2.capacity)
        Id2 = Id2[1:end÷2, 1:end÷2]
        Tₒ2, Tᵧ2 =  T_trial[end÷2 + 1 : 3end÷4], T_trial[3end÷4 + 1 : end]
        Interface_term_2 = Id2 * H2' * W!2 * G2 * Tₒ2 + Id2 * H2' * W!2 * H2 * Tᵧ2

        # Combine interface terms and reshape to match the columns
        Interface_term = 1/(ρL) * (Interface_term_1 + Interface_term_2)
        Interface_term = reshape(Interface_term, (nx, ny))
        Interface_term = vec(sum(Interface_term, dims=1))
        
        # 4) Update the height function
        res = Hₙ₊₁_1 - Hₙ_1 - Interface_term
        new_Hₙ = current_Hₙ .+ α .* res            # Elementwise update for each column
        
        # Calculate error
        err = maximum(abs.(new_Hₙ .- current_Hₙ))
        err_rel = err / maximum(abs.(current_Hₙ))
        println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

        # Store residuals
        push!(residuals[1], err)

        # 5) Update geometry if not converged
        if (err <= tol) || (err_rel <= reltol)
            push!(xf_log, new_xf)
            break
        end

        # 6) Compute the new interface position table
        new_xf = interface_positions_from_heights(new_Hₙ, mesh)
        ensure_periodic!(new_xf)  # Ensure periodic BC in y-direction

        # 7) Construct interpolation functions for new interface position
        centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        if interpo == "linear"
            sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
        elseif interpo == "quad"
            sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Line())
        elseif interpo == "cubic"
            sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
        else
            println("Interpolation method not supported")
        end

        # 8) Rebuild the domains with linear time interpolation
        tₙ₊₁ = t + Δt
        tₙ = t
        body1 = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return xx - x_interp
        end
        body2 = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return -(xx - x_interp)
        end        
        STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
        capacity1 = Capacity(body1, STmesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
        capacity2 = Capacity(body2, STmesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
        operator1 = DiffusionOps(capacity1)
        operator2 = DiffusionOps(capacity2)
        phase1 = Phase(capacity1, operator1, phase1.source, phase1.Diffusion_coeff)
        phase2 = Phase(capacity2, operator2, phase2.source, phase2.Diffusion_coeff)

        # 9) Rebuild the matrix A and the vector b
        s.A = A_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, scheme)
        s.b = b_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, T_prev, Δt, t, scheme)

        BC_border_diph!(s.A, s.b, bc_b, mesh)

        # 10) Update variables
        current_Hₙ = new_Hₙ
        current_xf = new_xf
    end

    if (err <= tol) || (err_rel <= reltol)
        println("Converged after $iter iterations with Hₙ = $new_Hₙ, error = $err")
    else
        println("Reached max_iter = $max_iter with Hₙ = $new_Hₙ, error = $err")
    end

    # Save state after first time step
    Tᵢ = s.x
    push!(s.states, s.x)
    println("Time : $(t)")
    println("Max value : $(maximum(abs.(s.x)))")

    # Time loop for remaining steps
    k = 2
    while t < Tₑ
        t += Δt
        tₙ = t
        tₙ₊₁ = t + Δt
        println("Time : $(t)")

        # 1) Construct interpolation functions for interface position
        centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        if interpo == "linear"
            sₙ = linear_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Line())
            sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
        elseif interpo == "quad"
            sₙ = extrapolate(scale(interpolate(current_xf, BSpline(Quadratic())), centroids), Interpolations.Line())
            sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Line())
        elseif interpo == "cubic"
            sₙ = cubic_spline_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Line())# filepath: /home/libat/github/Penguin.jl/examples/2D/LiquidMoving/stefan_2d_2ph.jl
            sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
        else
            println("Interpolation method not supported")
        end

        # 1) Reconstruct
        STmesh = SpaceTimeMesh(mesh, [t-Δt, t], tag=mesh.tag)
        body1 = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return xx - x_interp
        end
        body2 = (xx, yy, tt, _=0) -> begin
            # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
            t_norm = (tt - tₙ) / Δt
            
            # Quadratic interpolation coefficients
            a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
            b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
            c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
            
            # Position at start, middle, and end points
            pos_start = sₙ(yy)
            pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
            pos_end = sₙ₊₁(yy)
            
            # Compute interpolated position
            x_interp = a * pos_start + b * pos_mid + c * pos_end
            
            # Return signed distance
            return -(xx - x_interp)
        end
        capacity1 = Capacity(body1, STmesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
        capacity2 = Capacity(body2, STmesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
        operator1 = DiffusionOps(capacity1)
        operator2 = DiffusionOps(capacity2)
        phase1 = Phase(capacity1, operator1, phase1.source, phase1.Diffusion_coeff)
        phase2 = Phase(capacity2, operator2, phase2.source, phase2.Diffusion_coeff)

        s.A = A_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, scheme)
        s.b = b_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, t, scheme)

        BC_border_diph!(s.A, s.b, bc_b, mesh)

        push!(reconstruct, sₙ₊₁)

        # Initialize newton variables
        err = Inf
        err_rel = Inf
        iter = 0

        # Initialize newton height variables
        current_Hₙ = new_Hₙ
        new_Hₙ = current_Hₙ

        # Initialize newton interface position variables
        current_xf = new_xf
        new_xf = current_xf
        xf = current_xf
        T_prev = Tᵢ

        # Newton to compute the interface position xf1
        while (iter < max_iter) && (err > tol) && (err_rel > reltol)
            iter += 1

            # 1) Solve the linear system
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            T_trial = s.x

            # 2) Recompute heights for phase 1
            Hₙ_1, Hₙ₊₁_1 = extract_height_profiles(phase1.capacity, dims)

            # 3) Compute the interface flux term for phase 1
            W!1 = phase1.operator.Wꜝ[1:end÷2, 1:end÷2]
            G1  = phase1.operator.G[1:end÷2, 1:end÷2]
            H1  = phase1.operator.H[1:end÷2, 1:end÷2]
            V1  = phase1.operator.V[1:end÷2, 1:end÷2]
            Id1 = build_I_D(phase1.operator, phase1.Diffusion_coeff, phase1.capacity)
            Id1 = Id1[1:end÷2, 1:end÷2]
            Tₒ1, Tᵧ1 = T_trial[1:end÷4], T_trial[end÷4 + 1 : end÷2]
            Interface_term_1 = Id1 * H1' * W!1 * G1 * Tₒ1 + Id1 * H1' * W!1 * H1 * Tᵧ1

            # 4) Compute flux term for phase 2
            W!2 = phase2.operator.Wꜝ[1:end÷2, 1:end÷2]
            G2  = phase2.operator.G[1:end÷2, 1:end÷2]
            H2  = phase2.operator.H[1:end÷2, 1:end÷2]
            V2  = phase2.operator.V[1:end÷2, 1:end÷2]
            Id2 = build_I_D(phase2.operator, phase2.Diffusion_coeff, phase2.capacity)
            Id2 = Id2[1:end÷2, 1:end÷2]
            Tₒ2, Tᵧ2 = T_trial[end÷2 + 1 : 3end÷4], T_trial[3end÷4 + 1 : end]
            Interface_term_2 = Id2 * H2' * W!2 * G2 * Tₒ2 + Id2 * H2' * W!2 * H2 * Tᵧ2

            # Combine interface terms and reshape to match the columns
            Interface_term = 1/(ρL) * (Interface_term_1 + Interface_term_2)
            Interface_term = reshape(Interface_term, (nx, ny))
            Interface_term = vec(sum(Interface_term, dims=1))

            # 4) Update the height function
            res = Hₙ₊₁_1 - Hₙ_1 - Interface_term
            new_Hₙ = current_Hₙ .+ α .* res            # Elementwise update for each column

            # Calculate error
            err = maximum(abs.(new_Hₙ .- current_Hₙ))
            err_rel = err / maximum(abs.(current_Hₙ))
            println("Iteration $iter | Hₙ (max) = $(maximum(new_Hₙ)) | err = $err | err_rel = $err_rel")

            # Store residuals
            push!(residuals[k], err)

            # 5) Update geometry if not converged
            if (err <= tol) || (err_rel <= reltol)
                push!(xf_log, new_xf)
                break
            end

            # 6) Compute the new interface position table
            new_xf = interface_positions_from_heights(new_Hₙ, mesh)
            ensure_periodic!(new_xf)  # Ensure Line BC in y-direction

            # 7) Construct interpolation functions for new interface position
            centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
            if interpo == "linear"
                sₙ₊₁ = linear_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
            elseif interpo == "quad"
                sₙ₊₁ = extrapolate(scale(interpolate(new_xf, BSpline(Quadratic())), centroids), Interpolations.Line())
            elseif interpo == "cubic"
                sₙ₊₁ = cubic_spline_interpolation(centroids, new_xf, extrapolation_bc=Interpolations.Line())
            else
                println("Interpolation method not supported")
            end

            # 8) Rebuild the domains with linear time interpolation
            tₙ₊₁ = t + Δt
            tₙ = t

            body1 = (xx, yy, tt, _=0) -> begin
                # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
                t_norm = (tt - tₙ) / Δt
                
                # Quadratic interpolation coefficients
                a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
                b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
                c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
                
                # Position at start, middle, and end points
                pos_start = sₙ(yy)
                pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
                pos_end = sₙ₊₁(yy)
                
                # Compute interpolated position
                x_interp = a * pos_start + b * pos_mid + c * pos_end
                
                # Return signed distance
                return xx - x_interp
            end
            body2 = (xx, yy, tt, _=0) -> begin
                # Normalized time parameter (0 to 1 over the interval [tₙ, tₙ₊₁])
                t_norm = (tt - tₙ) / Δt
                
                # Quadratic interpolation coefficients
                a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0  # = (1-t)² * (2t+1)
                b = -4.0 * t_norm^2 + 4.0 * t_norm      # = 4t(1-t)
                c = 2.0 * t_norm^2 - t_norm            # = t²(2t-1)
                
                # Position at start, middle, and end points
                pos_start = sₙ(yy)
                pos_mid = 0.5 * (sₙ(yy) + sₙ₊₁(yy))  # on pourrait utiliser une autre valeur intermédiaire
                pos_end = sₙ₊₁(yy)
                
                # Compute interpolated position
                x_interp = a * pos_start + b * pos_mid + c * pos_end
                
                # Return signed distance
                return -(xx - x_interp)
            end      
            STmesh = SpaceTimeMesh(mesh, [tₙ, tₙ₊₁], tag=mesh.tag)
            capacity1 = Capacity(body1, STmesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
            capacity2 = Capacity(body2, STmesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
            operator1 = DiffusionOps(capacity1)
            operator2 = DiffusionOps(capacity2)
            phase1 = Phase(capacity1, operator1, phase1.source, phase1.Diffusion_coeff)
            phase2 = Phase(capacity2, operator2, phase2.source, phase2.Diffusion_coeff)

            s.A = A_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, scheme)
            s.b = b_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, T_prev, Δt, t, scheme)

            BC_border_diph!(s.A, s.b, bc_b, mesh)

            # 9) Update variables
            current_Hₙ = new_Hₙ
            current_xf = new_xf
        end

        if (err <= tol) || (err_rel <= reltol)
            println("Converged after $iter iterations with Hₙ = $new_Hₙ, error = $err")
        else
            println("Reached max_iter = $max_iter with Hₙ = $new_Hₙ, error = $err")
        end

        # Save state after first time step
        Tᵢ = s.x
        push!(s.states, s.x)
        println("Max value : $(maximum(abs.(s.x)))")
        
        # Update variables
        k += 1
    end

    return s, residuals, xf_log, reconstruct
end
