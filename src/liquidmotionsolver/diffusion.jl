# Full Moving - Diffusion - Unsteady - Monophasic

mutable struct LearningRateState
    strategy::Symbol
    base_lr::Float64
    eps::Float64
    beta1::Float64
    beta2::Float64
    decay::Float64
    grad_accum::Float64
    mean_grad_sq::Float64
    m::Float64
    v::Float64
    t::Int
    prev_xf::Union{Nothing, Float64}
    prev_grad::Union{Nothing, Float64}
    min_lr::Float64
    max_lr::Float64
    last_lr::Float64
end

function normalize_lr_strategy(strategy::Union{Symbol, String})
    normalized = lowercase(String(strategy))
    normalized in ("constant", "none") && return :fixed
    normalized in ("bb", "barzilaiborwein", "barzilai-borwein") && return :barzilai_borwein
    normalized in ("rms_prop",) && return :rmsprop
    normalized in ("secant",) && return :secant
    return Symbol(normalized)
end

function init_learning_rate_state(strategy::Union{Symbol, String}, base_lr::Float64;
        eps::Float64=1e-8,
        beta1::Float64=0.9,
        beta2::Float64=0.999,
        decay::Float64=0.0,
        min_lr::Float64=0.0,
        max_lr::Float64=Inf)
    strat = normalize_lr_strategy(strategy)
    max_lr = max(max_lr, min_lr)
    return LearningRateState(strat, base_lr, eps, beta1, beta2, decay, 0.0, 0.0, 0.0, 0.0, 0, nothing, nothing, min_lr, max_lr, base_lr)
end

function apply_learning_rate_step!(state::LearningRateState, current_xf::Float64, grad::Float64)
    state.t += 1
    base_lr = state.decay > 0 ? state.base_lr / (1 + state.decay * (state.t - 1)) : state.base_lr
    lr = base_lr
    direction = grad
    custom_step = nothing

    if state.strategy === :adagrad
        state.grad_accum += grad * grad
        denom = sqrt(state.grad_accum) + state.eps
        lr = base_lr / denom
    elseif state.strategy === :rmsprop
        state.mean_grad_sq = state.beta2 * state.mean_grad_sq + (1 - state.beta2) * grad * grad
        denom = sqrt(state.mean_grad_sq) + state.eps
        lr = base_lr / denom
    elseif state.strategy === :nadam
        state.m = state.beta1 * state.m + (1 - state.beta1) * grad
        state.v = state.beta2 * state.v + (1 - state.beta2) * grad * grad
        bias_correction1 = max(1 - state.beta1^state.t, state.eps)
        bias_correction2 = max(1 - state.beta2^state.t, state.eps)
        m_hat = state.m / bias_correction1
        v_hat = state.v / bias_correction2
        denom = sqrt(v_hat) + state.eps
        lr = base_lr / denom
        direction = state.beta1 * m_hat + (1 - state.beta1) * grad / bias_correction1
    elseif state.strategy === :barzilai_borwein
        if !(state.prev_xf === nothing || state.prev_grad === nothing)
            prev_xf = state.prev_xf::Float64
            prev_grad = state.prev_grad::Float64
            Δx = current_xf - prev_xf
            Δg = grad - prev_grad
            denom = abs(Δg) > state.eps ? Δg * Δg : 0.0
            if denom > 0
                lr = abs(Δx * Δg) / denom
            end
        end
    elseif state.strategy === :secant
        if !(state.prev_xf === nothing || state.prev_grad === nothing)
            prev_xf = state.prev_xf::Float64
            prev_grad = state.prev_grad::Float64
            Δx = current_xf - prev_xf
            Δg = grad - prev_grad
            if abs(Δg) > state.eps
                proposed_step = -grad * (Δx / Δg)
                if grad == 0.0
                    custom_step = proposed_step
                    lr = state.base_lr
                else
                    max_step = state.max_lr * abs(grad)
                    min_step = state.min_lr * abs(grad)
                    if isfinite(max_step)
                        proposed_step = clamp(proposed_step, -max_step, max_step)
                    end
                    if min_step > 0
                        if abs(proposed_step) < min_step
                            proposed_step = sign(proposed_step) * min_step
                        end
                    end
                    custom_step = proposed_step
                    lr = abs(proposed_step) / max(abs(grad), state.eps)
                end
            end
        end
    end

    if custom_step === nothing
        lr = clamp(lr, state.min_lr, state.max_lr)
        step = lr * direction
    else
        step = custom_step
    end
    if !isfinite(step)
        step = 0.0
    end

    state.prev_xf = current_xf
    state.prev_grad = grad
    state.last_lr = lr
    return step
end

function normalize_lr_options(options)
    if options === nothing
        return (;)
    elseif options isa NamedTuple
        return options
    elseif options isa AbstractDict
        return (; options...)
    elseif options isa AbstractVector
        return (; options...)
    else
        error("learning_rate_options must be provided as a NamedTuple, Dict, vector of Pairs, or nothing.")
    end
end

"""
    MovingLiquidDiffusionUnsteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)

Create a solver for the unsteady diffusion problem with moving interface in a monophasic problem.

# Arguments
- phase::Phase: The phase object containing the capacity and the operator.
- bc_b::BorderConditions: The border conditions.
- bc_i::AbstractBoundary: The interface condition.
- Δt::Float64: The time step.
- Tᵢ::Vector{Float64}: The initial temperature field.
- mesh::AbstractMesh: The mesh.
- scheme::String: The scheme to use for the time discretization ("BE" or "CN").
"""
function MovingLiquidDiffusionUnsteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Moving problem")
    println("- Non prescibed motion")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, [], [])
    
    if scheme == "CN"
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "CN")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "CN")
    else # BE
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "BE")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "BE")
    end
    BC_border_mono!(s.A, s.b, bc_b, mesh; t=0.0)
    return s
end

function solve_MovingLiquidDiffusionUnsteadyMono!(s::Solver, phase::Phase, xf, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh::AbstractMesh, scheme::String; Newton_params=(1000, 1e-10, 1e-10, 1.0), cfl_target=0.5,
    Δt_min=1e-4,
    Δt_max=1.0,
    adaptive_timestep=true, method=IterativeSolvers.gmres,
    algorithm=nothing, learning_rate_strategy::Union{Symbol, String}=:fixed,
    learning_rate_options=nothing, kwargs...)
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
    t=Tₛ
    println("Time : $(t)")

    # Params
    ρL = ic.flux.value
    max_iter = Newton_params[1]
    tol      = Newton_params[2]
    reltol   = Newton_params[3]
    α        = Newton_params[4]
    lr_opts = normalize_lr_options(learning_rate_options)

    # Log residuals and interface positions for each time step:
    nt = Int(round(Tₑ/Δt))
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Float64[]
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))

    # Determine how many dimensions
    dims = phase.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Create the 1D or 2D indices
    if len_dims == 2
        # 1D case
        nx, nt = dims
        n = nx
    elseif len_dims == 3
        # 2D case
        nx, ny, nt = dims
        n = nx*ny
    else
        error("Only 1D and 2D problems are supported.")
    end

    err = Inf
    iter = 0
    current_xf = xf
    new_xf = current_xf
    xf = current_xf
    lr_state = init_learning_rate_state(learning_rate_strategy, α; lr_opts...)
    # First time step : Newton to compute the interface position xf1
    while (iter < max_iter) && (err > tol) && (err > reltol * abs(current_xf))
        iter += 1

        # 1) Solve the linear system
        solve_system!(s; method=method, algorithm=algorithm, kwargs...)
        Tᵢ = s.x

        # 2) Update volumes / compute new interface
        Hₙ_profile, Hₙ₊₁_profile = extract_height_profiles(phase.capacity, phase.operator.size)
        Hₙ   = sum(Hₙ_profile)
        Hₙ₊₁ = sum(Hₙ₊₁_profile)

        # Compute flux with temporal weighting
        Vn_1 = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
        Vn   = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
        if scheme == "CN"
            psip = psip_cn
        else
            psip = psip_be
        end
        Ψn1 = Diagonal(psip.(diag(Vn), diag(Vn_1)))
        
        W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
        G = phase.operator.G[1:end÷2, 1:end÷2]
        H = phase.operator.H[1:end÷2, 1:end÷2]
        V = phase.operator.V[1:end÷2, 1:end÷2]
        Id   = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id  = Id[1:end÷2, 1:end÷2]
        Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
        Interface_term = Id * H' * W! * G * Ψn1 * Tₒ + Id * H' * W! * H * Ψn1 * Tᵧ
        Interface_term = 1/(ρL) * sum(Interface_term)

        # New interface position: enforce H_{n+1} - H_n = q/(ρL)
        res = Hₙ₊₁ - Hₙ - Interface_term
        step = apply_learning_rate_step!(lr_state, current_xf, res)
        new_xf = current_xf + step
        err = abs(res)
        println("Iteration $iter | xf = $new_xf | error = $err | res = $res | α = $(lr_state.last_lr)")
        
        # Store residuals
        if !haskey(residuals, 1)
            residuals[1] = Float64[]
        end
        push!(residuals[1], err)

        # 3) Update geometry if not converged
        if (err <= tol) || (err <= reltol * abs(current_xf)) || (iter == max_iter)
            push!(xf_log, new_xf)
            break
        end

        # Store tn+1 and tn
        tn1 = t + Δt
        tn  = t

        # 4) Rebuild domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
        body = (xx,tt, _=0)->(xx - (xf*(tn1 - tt)/Δt + new_xf*(tt - tn)/Δt))
        STmesh = SpaceTimeMesh(mesh, [tn, tn1], tag=mesh.tag)
        capacity = Capacity(body, STmesh)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, t, scheme)

        BC_border_mono!(s.A, s.b, bc_b, mesh; t=tn1)

        # 5) Update variables
        current_xf = new_xf
    end

    if (err <= tol) || (err <= reltol * abs(current_xf))
    println("Converged after $iter iterations with xf = $new_xf, error = $err")
    else
        println("Reached max_iter = $max_iter with xf = $new_xf, error = $err")
    end
    
    Tᵢ = s.x
    push!(s.states, s.x)
    t += Δt                        # advance to the time level of the computed state
    println("Time : $(t)")
    println("Max value : $(maximum(abs.(s.x)))")

    # Time loop
    k=2
    while t < Tₑ
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

        # 1) Reconstruct geometry for next step on the correct time slab
        STmesh = SpaceTimeMesh(mesh, [t, t+Δt], tag=mesh.tag)
        #v_guess = (new_xf - xf)/Δt
        #body = (xx, tt, _=0) -> xx - ( new_xf - v_guess * (tt - t) )
        body = (xx,tt, _=0)->(xx - new_xf) 
        capacity = Capacity(body, STmesh)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, 0.0, scheme)

        BC_border_mono!(s.A, s.b, bc_b, mesh; t=t)

        err = Inf
        iter = 0
        current_xf = new_xf
        new_xf = current_xf
        xf = current_xf
        lr_state = init_learning_rate_state(learning_rate_strategy, α; lr_opts...)
        # Newton to compute the interface position xf1
        while (iter < max_iter) && (err > tol) && (err > reltol * abs(current_xf))
            iter += 1

            # 1) Solve the linear system
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            Tᵢ = s.x

            # 2) Update volumes / compute new interface
            Hₙ_profile, Hₙ₊₁_profile = extract_height_profiles(phase.capacity, phase.operator.size)
            Hₙ   = sum(Hₙ_profile)
            Hₙ₊₁ = sum(Hₙ₊₁_profile)
            
            # Compute flux with temporal weighting
            Vn_1 = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
            Vn   = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
            if scheme == "CN"
                psip = psip_cn
            else
                psip = psip_be
            end
            Ψn1 = Diagonal(psip.(diag(Vn), diag(Vn_1)))
            
            W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
            G = phase.operator.G[1:end÷2, 1:end÷2]
            H = phase.operator.H[1:end÷2, 1:end÷2]
            V = phase.operator.V[1:end÷2, 1:end÷2]
            Id   = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id  = Id[1:end÷2, 1:end÷2]
            Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
            Interface_term = Id * H' * W! * G * Ψn1 * Tₒ + Id * H' * W! * H * Ψn1 * Tᵧ
            Interface_term = 1/(ρL) * sum(Interface_term)

            # New interface position
            res = Hₙ₊₁ - Hₙ - Interface_term
            step = apply_learning_rate_step!(lr_state, current_xf, res)
            new_xf = current_xf + step
            err = abs(res)
            println("Iteration $iter | xf = $new_xf | error = $err | res = $res | α = $(lr_state.last_lr)")
            # Store residuals
            if !haskey(residuals, k)
                residuals[k] = Float64[]
            end
            push!(residuals[k], err)

            # 3) Update geometry if not converged
            if (err <= tol) || (err <= reltol * abs(current_xf)) || (iter == max_iter)
                push!(xf_log, new_xf)
                break
            end

            # Store tn+1 and tn
            tn1 = t + Δt
            tn  = t

            # 4) Rebuild domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
            body = (xx,tt, _=0)->(xx - (xf*(tn1 - tt)/Δt + new_xf*(tt - tn)/Δt))
            STmesh = SpaceTimeMesh(mesh, [tn, tn1], tag=mesh.tag)
            capacity = Capacity(body, STmesh)
            operator = DiffusionOps(capacity)
            phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

            s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt, t, scheme)

            BC_border_mono!(s.A, s.b, bc_b, mesh; t=tn1) # use mesh instead of phase.capacity.mesh for BC

            # 5) Update variables
            current_xf = new_xf

        end

        if (err <= tol) || (err <= reltol * abs(current_xf))
            println("Converged after $iter iterations with xf = $new_xf, error = $err")
        else
            println("Reached max_iter = $max_iter with xf = $new_xf, error = $err")
        end

        # Afficher les informations du pas de temps
        if adaptive_timestep
            println("Time step info: Δt = $(round(Δt, digits=6)), CFL = $(round(timestep_history[end][2], digits=3))")
        end

        push!(s.states, s.x)
        t += Δt
        println("Time : $(t)")
        println("Max value : $(maximum(abs.(s.x)))")
        k += 1
    end

    return s, residuals, xf_log, timestep_history
end 



# Moving - Diffusion - Unsteady - Diphasic
function A_diph_unstead_diff_moving_stef(operator1::DiffusionOps, operator2::DiffusionOps, capacite1::Capacity, capacite2::Capacity, D1, D2, ic::InterfaceConditions, scheme::String)
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
    Id1 = build_I_D(operator1, D1, capacite1)
    Id2 = build_I_D(operator2, D2, capacite2)

    # Capacity indexing (2 for 1D, 3 for 2D)
    cap_index1 = len_dims1
    cap_index2 = len_dims2

    # Extract Vr−1 and Vr
    Vn1_1 = capacite1.A[cap_index1][1:end÷2, 1:end÷2]
    Vn1   = capacite1.A[cap_index1][end÷2+1:end, end÷2+1:end]
    Vn2_1 = capacite2.A[cap_index2][1:end÷2, 1:end÷2]
    Vn2   = capacite2.A[cap_index2][end÷2+1:end, end÷2+1:end]

    # Time integration weighting
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else
        psip, psim = psip_be, psim_be
    end

    Ψn1 = Diagonal(psip.(Vn1, Vn1_1))
    Ψn2 = Diagonal(psip.(Vn2, Vn2_1))

    Iₐ1, Iₐ2 = jump.α₁ * I(size(Ψn1)[1]), jump.α₂ * I(size(Ψn2)[1])
    Iᵦ1, Iᵦ2 = flux.β₁ * I(size(Ψn1)[1]), flux.β₂ * I(size(Ψn2)[1])

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
    Id1  = Id1[1:n, 1:n]
    Id2  = Id2[1:n, 1:n]

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


function b_diph_unstead_diff_moving_stef(operator1::DiffusionOps, operator2::DiffusionOps, capacity1::Capacity, capacity2::Capacity, D1, D2, f1::Function, f2::Function, ic::InterfaceConditions, Tᵢ::Vector{Float64}, Δt::Float64, t::Float64, scheme::String)
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
        n1 = nx1
        n2 = nx2
    else
        # 2D
        nx1, ny1, _ = dims1
        nx2, ny2, _ = dims2
        n1   = nx1 * ny1   # local block size for each operator
        n2   = nx2 * ny2
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

function MovingLiquidDiffusionUnsteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Moving problem")
    println("- Non prescibed motion")
    println("- Diphasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Diphasic, Diffusion, nothing, nothing, nothing, [], [])
    
    if scheme == "CN"
        s.A = A_diph_unstead_diff_moving_stef(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, "CN")
        s.b = b_diph_unstead_diff_moving_stef(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, 0.0, "CN")
    else 
        s.A = A_diph_unstead_diff_moving_stef(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, "BE")
        s.b = b_diph_unstead_diff_moving_stef(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, 0.0, "BE")
    end
    BC_border_diph!(s.A, s.b, bc_b, mesh)
    return s
end


function solve_MovingLiquidDiffusionUnsteadyDiph!(s::Solver, phase1::Phase, phase2::Phase, xf, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, ic::InterfaceConditions, mesh::AbstractMesh, scheme::String; Newton_params=(1000, 1e-10, 1e-10, 1.0), method = IterativeSolvers.gmres, algorithm=nothing,
    learning_rate_strategy::Union{Symbol, String}=:fixed, learning_rate_options=nothing, kwargs...)
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
    t=Tₛ
    println("Time : $(t)")

    # Params
    ρL = ic.flux.value
    max_iter = Newton_params[1]
    tol      = Newton_params[2]
    reltol   = Newton_params[3]
    α        = Newton_params[4]
    lr_opts = normalize_lr_options(learning_rate_options)

    # Log residuals and interface positions for each time step:
    nt = Int(round(Tₑ/Δt))
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Float64[]

    # Determine how many dimensions
    dims = phase1.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Create the 1D or 2D indices
    if len_dims == 2
        # 1D case
        nx, nt = dims
        n = nx
    elseif len_dims == 3
        # 2D case
        nx, ny, nt = dims
        n = nx*ny
    else
        error("Only 1D and 2D problems are supported.")
    end

    err = Inf
    iter = 0
    current_xf = xf
    new_xf = current_xf
    xf = current_xf
    lr_state = init_learning_rate_state(learning_rate_strategy, α; lr_opts...)
    # First time step : Newton to compute the interface position xf1
    while (iter < max_iter) && (err > tol) && (err > reltol * abs(current_xf))
        iter += 1

        # 1) Solve the linear system
        solve_system!(s; method=method, algorithm=algorithm, kwargs...)
        Tᵢ = s.x

        # 2) Update volumes / compute new interface
        Vn_1 = phase1.capacity.A[cap_index][1:end÷2, 1:end÷2]
        Vn   = phase1.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
        Hₙ   = sum(diag(Vn))
        Hₙ₊₁ = sum(diag(Vn_1))

        # Compute flux for phase 1
        W!1 = phase1.operator.Wꜝ[1:end÷2, 1:end÷2]
        G1 = phase1.operator.G[1:end÷2, 1:end÷2]
        H1 = phase1.operator.H[1:end÷2, 1:end÷2]
        V1 = phase1.operator.V[1:end÷2, 1:end÷2]
        Id1   = build_I_D(phase1.operator, phase1.Diffusion_coeff, phase1.capacity)
        Id1  = Id1[1:end÷2, 1:end÷2]
        Tₒ1, Tᵧ1 = Tᵢ[1:end÷4], Tᵢ[end÷4 + 1 : end÷2]
        Interface_term_1 = Id1 * H1' * W!1 * G1 * Tₒ1 + Id1 * H1' * W!1 * H1 * Tᵧ1
        Interface_term_1 = 1/(ρL) * sum(Interface_term_1)

        # Compute flux for phase 2
        W!2 = phase2.operator.Wꜝ[1:end÷2, 1:end÷2]
        G2 = phase2.operator.G[1:end÷2, 1:end÷2]
        H2 = phase2.operator.H[1:end÷2, 1:end÷2]
        V2 = phase2.operator.V[1:end÷2, 1:end÷2]
        Id2   = build_I_D(phase2.operator, phase2.Diffusion_coeff, phase2.capacity)
        Id2  = Id2[1:end÷2, 1:end÷2]
        Tₒ2, Tᵧ2 = Tᵢ[end÷2 + 1 : 3end÷4], Tᵢ[3end÷4 + 1 : end]
        Interface_term_2 = Id2 * H2' * W!2 * G2 * Tₒ2 + Id2 * H2' * W!2 * H2 * Tᵧ2
        Interface_term_2 = 1/(ρL) * sum(Interface_term_2)

        # Compute Interface term
        Interface_term = Interface_term_1 + Interface_term_2

        # New interface position
        res = Hₙ₊₁ - Hₙ - Interface_term
        step = apply_learning_rate_step!(lr_state, current_xf, res)
        new_xf = current_xf + step
        err = abs(res)
        println("Iteration $iter | xf = $new_xf | error = $err | res = $res | α = $(lr_state.last_lr)")
        # Store residuals
        if !haskey(residuals, 1)
            residuals[1] = Float64[]
        end
        push!(residuals[1], err)

        # 3) Update geometry if not converged
        if (err <= tol) || (err <= reltol * abs(current_xf))
            push!(xf_log, new_xf)
            break
        end

        # Store tn+1 and tn
        tn1 = t + Δt
        tn  = t

        # 4) Rebuild domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
        body = (xx,tt, _=0)->(xx - (xf*(tn1 - tt)/Δt + new_xf*(tt - tn)/Δt))
        body_c = (xx,tt, _=0)->-(xx - (xf*(tn1 - tt)/Δt + new_xf*(tt - tn)/Δt))
        STmesh = SpaceTimeMesh(mesh, [tn, tn1], tag=mesh.tag)
        capacity = Capacity(body, STmesh)
        capacity_c = Capacity(body_c, STmesh)
        operator = DiffusionOps(capacity)
        operator_c = DiffusionOps(capacity_c)
        phase1 = Phase(capacity, operator, phase1.source, phase1.Diffusion_coeff)
        phase2 = Phase(capacity_c, operator_c, phase2.source, phase2.Diffusion_coeff)

        s.A = A_diph_unstead_diff_moving_stef(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, scheme)
        s.b = b_diph_unstead_diff_moving_stef(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, t, scheme)

        BC_border_diph!(s.A, s.b, bc_b, mesh)

        # 5) Update variables
        current_xf = new_xf
    end

    if (err <= tol) || (err <= reltol * abs(current_xf))
        println("Converged after $iter iterations with xf = $new_xf, error = $err")
    else
        println("Reached max_iter = $max_iter with xf = $new_xf, error = $err")
    end
    
    Tᵢ = s.x
    push!(s.states, s.x)
    println("Time : $(t[1])")
    println("Max value : $(maximum(abs.(s.x)))")

    # Time loop
    k=2
    while t < Tₑ
        t += Δt
        println("Time : $(t)")

        # 1) Reconstruct
        STmesh = SpaceTimeMesh(mesh, [Δt, 2Δt], tag=mesh.tag)
        #v_guess = (new_xf - xf)/Δt
        #body = (xx, tt, _=0) -> xx - ( new_xf - v_guess * (tt - t) )
        body = (xx,tt, _=0)->(xx - new_xf) 
        body_c = (xx,tt, _=0)->-(xx - new_xf)
        capacity = Capacity(body, STmesh)
        capacity_c = Capacity(body_c, STmesh)
        operator = DiffusionOps(capacity)
        operator_c = DiffusionOps(capacity_c)
        phase1 = Phase(capacity, operator, phase1.source, phase1.Diffusion_coeff)
        phase2 = Phase(capacity_c, operator_c, phase2.source, phase2.Diffusion_coeff)

        s.A = A_diph_unstead_diff_moving_stef(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, scheme)
        s.b = b_diph_unstead_diff_moving_stef(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, 0.0, scheme)

        BC_border_diph!(s.A, s.b, bc_b, mesh)

        err = Inf
        iter = 0
        current_xf = new_xf
        new_xf = current_xf
        xf = current_xf
        lr_state = init_learning_rate_state(learning_rate_strategy, α; lr_opts...)
        # Newton to compute the interface position xf1
        while (iter < max_iter) && (err > tol) && (err > reltol * abs(current_xf))
            iter += 1

            # 1) Solve the linear system
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            Tᵢ = s.x

            # 2) Update volumes / compute new interface
            Vn_1 = phase1.capacity.A[cap_index][1:end÷2, 1:end÷2]
            Vn   = phase1.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
            Hₙ   = sum(diag(Vn))
            Hₙ₊₁ = sum(diag(Vn_1))
            
            # Compute flux for phase 1
            W!1 = phase1.operator.Wꜝ[1:end÷2, 1:end÷2]
            G1 = phase1.operator.G[1:end÷2, 1:end÷2]
            H1 = phase1.operator.H[1:end÷2, 1:end÷2]
            V1 = phase1.operator.V[1:end÷2, 1:end÷2]
            Id1= build_I_D(phase1.operator, phase1.Diffusion_coeff, phase1.capacity)
            Id1  = Id1[1:end÷2, 1:end÷2]
            Tₒ1, Tᵧ1 = Tᵢ[1:end÷4], Tᵢ[end÷4 + 1 : end÷2]
            Interface_term_1 = Id1 * H1' * W!1 * G1 * Tₒ1 + Id1 * H1' * W!1 * H1 * Tᵧ1
            Interface_term_1 = 1/(ρL) * sum(Interface_term_1)
    
            # Compute flux for phase 2
            W!2 = phase2.operator.Wꜝ[1:end÷2, 1:end÷2]
            G2 = phase2.operator.G[1:end÷2, 1:end÷2]
            H2 = phase2.operator.H[1:end÷2, 1:end÷2]
            V2 = phase2.operator.V[1:end÷2, 1:end÷2]
            Id2   = build_I_D(phase2.operator, phase2.Diffusion_coeff, phase2.capacity)
            Id2  = Id2[1:end÷2, 1:end÷2]
            Tₒ2, Tᵧ2 = Tᵢ[end÷2 + 1 : 3end÷4], Tᵢ[3end÷4 + 1 : end]
            Interface_term_2 = Id2 * H2' * W!2 * G2 * Tₒ2 + Id2 * H2' * W!2 * H2 * Tᵧ2
            Interface_term_2 = 1/(ρL) * sum(Interface_term_2)
    
            # Compute Interface term
            Interface_term = Interface_term_1 + Interface_term_2

            # New interface position
            res = Hₙ₊₁ - Hₙ - Interface_term
            step = apply_learning_rate_step!(lr_state, current_xf, res)
            new_xf = current_xf + step
            err = abs(step)
            println("Iteration $iter | xf = $new_xf | error = $err | res = $res | α = $(lr_state.last_lr)")
            # Store residuals
            if !haskey(residuals, k)
                residuals[k] = Float64[]
            end
            push!(residuals[k], err)

            # 3) Update geometry if not converged
            if (err <= tol) || (err <= reltol * abs(current_xf))
                push!(xf_log, new_xf)
                break
            end

            # Store tn+1 and tn
            tn1 = t + Δt
            tn  = t

            # 4) Rebuild domain : # Add t interpolation : x - (xf*(tn1 - t)/(\Delta t) + xff*(t - tn)/(\Delta t))
            body = (xx,tt, _=0)->(xx - (xf*(tn1 - tt)/Δt + new_xf*(tt - tn)/Δt))
            body_c = (xx,tt, _=0)->-(xx - (xf*(tn1 - tt)/Δt + new_xf*(tt - tn)/Δt))
            STmesh = SpaceTimeMesh(mesh, [tn, tn1], tag=mesh.tag)
            capacity = Capacity(body, STmesh)
            capacity_c = Capacity(body_c, STmesh)
            operator = DiffusionOps(capacity)
            operator_c = DiffusionOps(capacity_c)
            phase1 = Phase(capacity, operator, phase1.source, phase1.Diffusion_coeff)
            phase2 = Phase(capacity_c, operator_c, phase2.source, phase2.Diffusion_coeff)

            s.A = A_diph_unstead_diff_moving_stef(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, scheme)
            s.b = b_diph_unstead_diff_moving_stef(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, t, scheme)

            BC_border_diph!(s.A, s.b, bc_b, mesh) # use mesh instead of phase.capacity.mesh for BC

            # 5) Update variables
            current_xf = new_xf

        end

        if (err <= tol) || (err <= reltol * abs(current_xf))
            println("Converged after $iter iterations with xf = $new_xf, error = $err")
        else
            println("Reached max_iter = $max_iter with xf = $new_xf, error = $err")
        end

        push!(s.states, s.x)
        println("Time : $(t[1])")
        println("Max value : $(maximum(abs.(s.x)))")
        k += 1
    end

    return s, residuals, xf_log
end 
