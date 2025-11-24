"""
    A_concentration_unsteady_diph(operator1, operator2, capacity1, capacity2, D1, D2, ic, scheme)

Construct the system matrix for concentration diffusion two-phase problem.

# Arguments
- operator1, operator2: Diffusion operators for concentration in phases 1 and 2
- capacity1, capacity2: Capacity for concentration in phases 1 and 2
- D1, D2: Mass diffusivity coefficients in phases 1 and 2
- ic: Interface condition for concentration
- scheme: Time integration scheme ("BE" or "CN")
"""
function A_concentration_unsteady_diph(
    operator1, operator2, capacity1, capacity2, 
    D1, D2, ic, scheme
)
    # Determine dimensions from operator
    dims = operator1.size
    len_dims = length(dims)
    
    if len_dims == 2
        # 1D problem
        nx, _ = dims
        n = nx
    elseif len_dims == 3
        # 2D problem
        nx, ny, _ = dims
        n = nx * ny
    else
        error("Only 1D or 2D supported, got dimension: $len_dims")
    end

    # Get capacity indices
    cap_index = len_dims
    
    # Retrieve interface conditions
    jump, flux = ic.scalar, ic.flux
    
    # Interface condition weights
    Iₐ1, Iₐ2 = jump.α₁ * I(n), jump.α₂ * I(n)
    Iᵦ1, Iᵦ2 = flux.β₁ * I(n), flux.β₂ * I(n)
    
    # Build diffusion operators
    Id1 = build_I_D(operator1, D1, capacity1)
    Id2 = build_I_D(operator2, D2, capacity2)
    
    # Extract capacity matrices
    Vn1_1 = capacity1.A[cap_index][1:end÷2, 1:end÷2]
    Vn1   = capacity1.A[cap_index][end÷2+1:end, end÷2+1:end]
    Vn2_1 = capacity2.A[cap_index][1:end÷2, 1:end÷2]
    Vn2   = capacity2.A[cap_index][end÷2+1:end, end÷2+1:end]
    
    # Time integration weights
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else # BE
        psip, psim = psip_be, psim_be
    end
    
    Ψn1 = Diagonal(psip.(Vn1, Vn1_1))
    Ψn2 = Diagonal(psip.(Vn2, Vn2_1))
    
    # Operator sub-blocks
    # Concentration phase 1
    W!1 = operator1.Wꜝ[1:n, 1:n]
    G1 = operator1.G[1:n, 1:n]
    H1 = operator1.H[1:n, 1:n]
    
    # Concentration phase 2
    W!2 = operator2.Wꜝ[1:n, 1:n]
    G2 = operator2.G[1:n, 1:n]
    H2 = operator2.H[1:n, 1:n]
    
    # Pre-slice matrices to right dimensions
    Id1 = Id1[1:n, 1:n]
    Id2 = Id2[1:n, 1:n]
    
    Iᵦ1 = Iᵦ1[1:n, 1:n]
    Iᵦ2 = Iᵦ2[1:n, 1:n]
    
    Iₐ1 = Iₐ1[1:n, 1:n]
    Iₐ2 = Iₐ2[1:n, 1:n]
    
    # Construct blocks for concentration
    block1 = Vn1_1 + Id1 * G1' * W!1 * G1 * Ψn1
    block2 = -(Vn1_1 - Vn1) + Id1 * G1' * W!1 * H1 * Ψn1
    block3 = Vn2_1 + Id2 * G2' * W!2 * G2 * Ψn2
    block4 = -(Vn2_1 - Vn2) + Id2 * G2' * W!2 * H2 * Ψn2
    
    block5 = Iᵦ1 * H1' * W!1 * G1 * Ψn1
    block6 = Iᵦ1 * H1' * W!1 * H1 * Ψn1
    block7 = Iᵦ2 * H2' * W!2 * G2 * Ψn2
    block8 = Iᵦ2 * H2' * W!2 * H2 * Ψn2
    
    # Build the 4n×4n matrix for C1, C1γ, C2, C2γ
    A = spzeros(Float64, 4n, 4n)
    
    # Concentration phase 1 (C1)
    A[1:n, 1:n] = block1
    A[1:n, n+1:2n] = block2
    
    # Interface concentration (C1γ)
    # The interface values should be equal: C1γ = Cm
    A[n+1:2n, n+1:2n] = I(n)

    # Concentration phase 2 (C2)
    A[2n+1:3n, 2n+1:3n] = block3
    A[2n+1:3n, 3n+1:4n] = block4
    
    # Interface concentration (C2γ)
    # The interface values should be equal: C2γ = Cm
    A[3n+1:4n, 3n+1:4n] = I(n)

    return A
end

"""
    b_concentration_unsteady_diph(operator1, operator2, capacity1, capacity2,
                               D1, D2, f1, f2, ic, Cᵢ, Δt, t, scheme)

Construct the right-hand side vector for concentration diffusion two-phase problem.
"""
function b_concentration_unsteady_diph(
    operator1, operator2, capacity1, capacity2,
    D1, D2, f1, f2, ic, Cᵢ, Δt, t, scheme
)
    # Determine problem dimensions
    dims = operator1.size
    len_dims = length(dims)
    
    # Get capacity index
    cap_index = len_dims
    
    # Determine problem size
    if len_dims == 2
        # 1D problem
        nx, _ = dims
        n = nx
    elseif len_dims == 3
        # 2D problem
        nx, ny, _ = dims
        n = nx * ny
    else
        error("Only 1D or 2D problems supported")
    end
    
    # Build source terms
    f1ₒn  = build_source(operator1, f1, t, capacity1)
    f1ₒn1 = build_source(operator1, f1, t+Δt, capacity1)
    f2ₒn  = build_source(operator2, f2, t, capacity2)
    f2ₒn1 = build_source(operator2, f2, t+Δt, capacity2)
    
    # Get interface data
    jump, flux = ic.scalar, ic.flux
    
    # Get the fixed concentration at the interface
    Cm = jump.value  # Fixed interface concentration
    
    # Extract interface data
    Iᵧ1, Iᵧ2 = capacity1.Γ, capacity2.Γ
    
    gᵧ = build_g_g(operator1, jump, capacity1)
    hᵧ = build_g_g(operator2, flux, capacity2)
    
    # Extract capacity matrices
    Vn1_1 = capacity1.A[cap_index][1:end÷2, 1:end÷2]
    Vn1   = capacity1.A[cap_index][end÷2+1:end, end÷2+1:end]
    Vn2_1 = capacity2.A[cap_index][1:end÷2, 1:end÷2]
    Vn2   = capacity2.A[cap_index][end÷2+1:end, end÷2+1:end]
    
    # Time integration weights
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else # BE
        psip, psim = psip_be, psim_be
    end
    
    Ψn1 = Diagonal(psim.(Vn1, Vn1_1))
    Ψn2 = Diagonal(psim.(Vn2, Vn2_1))
    
    # Slice initial conditions
    Cₒ1 = Cᵢ[1:n]
    Cᵧ1 = Cᵢ[n+1:2n]
    Cₒ2 = Cᵢ[2n+1:3n]
    Cᵧ2 = Cᵢ[3n+1:4n]
    
    # Slice and extract operators
    f1ₒn  = f1ₒn[1:n]
    f1ₒn1 = f1ₒn1[1:n]
    f2ₒn  = f2ₒn[1:n]
    f2ₒn1 = f2ₒn1[1:n]
    
    gᵧ = gᵧ[1:n]
    hᵧ = hᵧ[1:n]
    
    Iᵧ1 = Iᵧ1[1:n, 1:n]
    Iᵧ2 = Iᵧ2[1:n, 1:n]
    
    # Build ID operators
    Id1 = build_I_D(operator1, D1, capacity1)
    Id2 = build_I_D(operator2, D2, capacity2)
    
    Id1 = Id1[1:n, 1:n]
    Id2 = Id2[1:n, 1:n]
    
    # Get remaining operators
    W!1 = operator1.Wꜝ[1:n, 1:n]
    G1 = operator1.G[1:n, 1:n]
    H1 = operator1.H[1:n, 1:n]
    V1 = operator1.V[1:n, 1:n]
    
    W!2 = operator2.Wꜝ[1:n, 1:n]
    G2 = operator2.G[1:n, 1:n]
    H2 = operator2.H[1:n, 1:n]
    V2 = operator2.V[1:n, 1:n]
    
    # Build RHS for concentration
    if scheme == "CN"
        bC1 = (Vn1 - Id1 * G1' * W!1 * G1 * Ψn1) * Cₒ1 - 0.5 * Id1 * G1' * W!1 * H1 * Cᵧ1 + 0.5 * V1 * (f1ₒn + f1ₒn1)
        bC3 = (Vn2 - Id2 * G2' * W!2 * G2 * Ψn2) * Cₒ2 - 0.5 * Id2 * G2' * W!2 * H2 * Cᵧ2 + 0.5 * V2 * (f2ₒn + f2ₒn1)
    else  # BE
        bC1 = Vn1 * Cₒ1 + V1 * f1ₒn1
        bC3 = Vn2 * Cₒ2 + V2 * f2ₒn1
    end
    
    # C1γ = Cm
    bC2 = gᵧ
    
    # C1γ = Cm
    bC4 = gᵧ
    
    # Combine all components
    return vcat(bC1, bC2, bC3, bC4)
end

"""
    DiffusionUnsteadyConcentration(phase1, phase2, bc_b, ic, Δt, u0, scheme, Newton_params)

Create a solver for concentration diffusion with two phases.
"""
function DiffusionUnsteadyConcentration(
    phase1::Phase, phase2::Phase,
    bc_b::BorderConditions, 
    ic::InterfaceConditions, 
    Δt::Float64, u0::Vector{Float64}, 
    scheme::String,
    Newton_params::Tuple{Int64, Float64, Float64, Float64}
)
    println("Solver Creation:")
    println("- Two-phase Concentration Diffusion")
    println("- Unsteady problem")
    println("- Interface concentration balance")
    
    s = Solver(Unsteady, Diphasic, Diffusion, nothing, nothing, nothing, [], [])
    
    # Get dimensions and arrange initial conditions
    dims = phase1.operator.size
    len_dims = length(dims)
    
    if len_dims == 2
        # 1D problem
        nx, _ = dims
        n = nx
    else
        # 2D problem
        nx, ny, _ = dims
        n = nx * ny
    end
    
    s.A = A_concentration_unsteady_diph(
        phase1.operator, phase2.operator,
        phase1.capacity, phase2.capacity,
        phase1.Diffusion_coeff, phase2.Diffusion_coeff,
        ic, scheme
    )
    
    s.b = b_concentration_unsteady_diph(
        phase1.operator, phase2.operator,
        phase1.capacity, phase2.capacity,
        phase1.Diffusion_coeff, phase2.Diffusion_coeff,
        phase1.source, phase2.source, 
        ic, u0, Δt, 0.0, scheme
    )
    
    # Apply boundary conditions
    BC_border_concentration!(s.A, s.b, bc_b, n)
    
    return s
end

"""
    BC_border_concentration!(A, b, bc_b, n)

Apply concentration boundary conditions to the system.
"""
function BC_border_concentration!(A, b, bc_b, n)
    for (direction, bc) in bc_b.borders
        idx = direction == :top ? 1 : n
        val = bc isa Dirichlet ? bc.value : 0.0
        
        # Concentration in phase 1
        A[idx, :] .= 0.0
        A[idx, idx] = 1.0
        b[idx] = val
        
        # Concentration in phase 2
        A[2*n+idx, :] .= 0.0
        A[2*n+idx, 2*n+idx] = 1.0
        b[2*n+idx] = val
    end
end

"""
    solve_DiffusionUnsteadyConcentration!(s, phase1, phase2, xf, Δt, Tend, bc_b, ic, mesh; Newton_params=(1000, 1e-8, 1e-8, 0.8), method=Base.:/)

Solve the concentration diffusion problem with interface balance.
"""
function solve_DiffusionUnsteadyConcentration!(
    s::Solver, 
    phase1::Phase, phase2::Phase, xf,
    Δt::Float64, Tend::Float64, 
    bc_b::BorderConditions,
    ic::InterfaceConditions,
    mesh::AbstractMesh;
    Newton_params=(1000, 1e-8, 1e-8, 0.8),
    method=Base.:\,
    algorithm=nothing,
    kwargs...
)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end
    
    println("Solving concentration diffusion problem:")
    println("- Two-phase system")
    println("- Interface concentration balance")
    
    # Extract parameters
    flux_factor = ic.flux.value  # Interface flux coefficient
    
    max_iter = Newton_params[1]
    tol      = Newton_params[2]
    reltol   = Newton_params[3]
    α        = Newton_params[4]
    
    # Track residuals and interface position
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Float64[]
    
    # Determine dimensions
    dims = phase1.operator.size
    len_dims = length(dims)
    cap_index = len_dims
    
    if len_dims == 2
        # 1D problem
        nx, _ = dims
        n = nx
    else
        # 2D problem 
        nx, ny, _ = dims
        n = nx * ny
    end
    
    # Time loop initialization
    t = 0.0
    println("Time: $(t)")
    
    # Get initial state
    solve_system!(s; method=method, algorithm=algorithm, kwargs...)
    u = s.x
    
    push!(s.states, u)
    println("Max concentration: $(maximum(abs.(u)))")
    
    # Initial interface position
    current_xf = xf
    new_xf = current_xf
    push!(xf_log, current_xf)
    
    # Time stepping loop
    k = 1
    while t < Tend
        t += Δt
        k += 1
        println("Time: $(t)")
        
        # Newton iteration for interface position
        err = Inf
        iter = 0
        
        while (iter < max_iter) && (err > tol) && (err > reltol * abs(current_xf))
            iter += 1
            
            # 1. Reconstruct domains based on current interface position
            body1 = (x, tt=0, _=0) -> (x - new_xf)
            body2 = (x, tt=0, _=0) -> -(x - new_xf)
            
            STmesh = SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
            
            # Create new capacities and operators
            capacity1 = Capacity(body1, STmesh)
            capacity2 = Capacity(body2, STmesh)
            
            operator1 = DiffusionOps(capacity1)
            operator2 = DiffusionOps(capacity2)
            
            phase1 = Phase(capacity1, operator1, phase1.source, phase1.Diffusion_coeff)
            phase2 = Phase(capacity2, operator2, phase2.source, phase2.Diffusion_coeff)
            
            # 2. Rebuild system matrices with new geometry
            s.A = A_concentration_unsteady_diph(
                phase1.operator, phase2.operator,
                phase1.capacity, phase2.capacity,
                phase1.Diffusion_coeff, phase2.Diffusion_coeff,
                ic, "BE"
            )
            
            s.b = b_concentration_unsteady_diph(
                phase1.operator, phase2.operator,
                phase1.capacity, phase2.capacity,
                phase1.Diffusion_coeff, phase2.Diffusion_coeff,
                phase1.source, phase2.source,
                ic, u, Δt, t-Δt, "BE"
            )
            
            # Apply boundary conditions
            BC_border_concentration!(s.A, s.b, bc_b, n)
            
            # 3. Solve the system
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            u = s.x
            
            # Split into bulk and interface values
            Cₒ1, Cᵧ1 = u[1:n], u[n+1:2n]
            Cₒ2, Cᵧ2 = u[2n+1:3n], u[3n+1:4n]
            
            # 4. Extract capacity matrices for volume calculation
            Vn1_1 = phase1.capacity.A[cap_index][1:end÷2, 1:end÷2]
            Vn1   = phase1.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
            Hₙ   = sum(diag(Vn1))
            Hₙ₊₁ = sum(diag(Vn1_1))
            
            # 5. Compute interface fluxes for concentration
            # Phase 1
            W!1 = phase1.operator.Wꜝ[1:n, 1:n]
            G1 = phase1.operator.G[1:n, 1:n]
            H1 = phase1.operator.H[1:n, 1:n]
            Id1 = build_I_D(phase1.operator, phase1.Diffusion_coeff, phase1.capacity)
            Id1 = Id1[1:n, 1:n]
            
            flux_1 = Id1 * H1' * W!1 * G1 * Cₒ1 + Id1 * H1' * W!1 * H1 * Cᵧ1
            flux_1_sum = sum(flux_1)
            
            # Phase 2
            W!2 = phase2.operator.Wꜝ[1:n, 1:n]
            G2 = phase2.operator.G[1:n, 1:n]
            H2 = phase2.operator.H[1:n, 1:n]
            Id2 = build_I_D(phase2.operator, phase2.Diffusion_coeff, phase2.capacity)
            Id2 = Id2[1:n, 1:n]
            
            flux_2 = Id2 * H2' * W!2 * G2 * Cₒ2 + Id2 * H2' * W!2 * H2 * Cᵧ2
            flux_2_sum = sum(flux_2)
            
            # 6. Calculate interface movement based on concentration balance
            # (C1-C2)w = q1·n1 - q2·n2
            C_diff = (Cᵧ1 - Cᵧ2) 
            println("C_diff = $C_diff")
            
            Interface_term = (flux_1_sum - flux_2_sum) #./ (C_diff)

            # 7. Update interface position
            res = Hₙ₊₁ - Hₙ - Interface_term
            new_xf = current_xf + α * res
            err = abs(res)
            println("Iteration $iter | xf = $new_xf | error = $err | res = $res")
            
            # Store residuals
            if !haskey(residuals, k)
                residuals[k] = Float64[]
            end
            push!(residuals[k], err)
            
            # Check convergence
            if (err <= tol) || (err <= reltol * abs(current_xf))
                push!(xf_log, new_xf)
                break
            end
            
            # Update interface position for next iteration
            current_xf = new_xf
        end
        
        # Report convergence status
        if (err <= tol) || (err <= reltol * abs(current_xf))
            println("Converged after $iter iterations with xf = $new_xf, error = $err")
        else
            println("Reached max_iter = $max_iter with xf = $new_xf, error = $err")
        end
        
        # Store solution
        push!(s.states, u)
        println("Time: $(t)")
        println("Max concentration: $(maximum(abs.(u)))")
    end
    
    return s, residuals, xf_log
end