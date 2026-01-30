# Moving - Diffusion - Unsteady - Monophasic
"""
    MovingDiffusionUnsteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tₛ::Float, Tᵢ::Vector{Float64}, scheme::String)

Create a solver for the unsteady monophasic diffusion problem inside a moving body.

# Arguments
- `phase::Phase`: The phase object containing the capacity and operator.
- `bc_b::BorderConditions`: The border conditions.
- `bc_i::AbstractBoundary`: The interface conditions.
- `Δt::Float64`: The time step.
- `Tₛ::Float`: The start time.
- `Tᵢ::Vector{Float64}`: The initial temperature field.
- `scheme::String`: The time integration scheme. Either "CN" or "BE".
"""
function MovingDiffusionUnsteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tₛ::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Moving problem")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, [], [])

    s.x = copy(Tᵢ)
    s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, scheme)
    s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, Tₛ, scheme)

    BC_border_mono!(s.A, s.b, bc_b, mesh; t=Tₛ + Δt)
    return s
end

# Helper function to compute bulk value in a space time mesh
"""
function psip_cn(args::Vararg{T,2}) where {T<:Real}
    if all(iszero, args)
        0.0
    elseif all(!iszero, args)
        0.5
    else
        0.5
    end
end

function psim_cn(args::Vararg{T,2}) where {T<:Real}
    if all(iszero, args)
        0.0
    elseif all(!iszero, args)
        0.5
    else
        0.5
    end
end
"""

function psip_cn(args::Vararg{T,2}) where {T<:Real}
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

function psim_cn(args::Vararg{T,2}) where {T<:Real}
    if all(iszero, args)
        0.0
    elseif all(!iszero, args)
        0.5
    elseif iszero(args[1]) && !iszero(args[2]) # Fresh
        0.5
    elseif !iszero(args[1]) && iszero(args[2]) # Dead
        0.0
    else
        0.0
    end
end

function psip_be(args::Vararg{T,2}) where {T<:Real}
    if all(iszero, args)
        0.0
    elseif all(!iszero, args)
        1.0
    else
        1.0
    end
end

function psim_be(args::Vararg{T,2}) where {T<:Real}
    0.0
end

function A_mono_unstead_diff_moving(operator::DiffusionOps, capacity::Capacity, D, bc::AbstractBoundary, scheme::String)
    # Determine dimension (1D vs 2D) from operator.size
    dims = operator.size
    len_dims = length(dims)
    
    # Pick capacity.A index based on dimension
    #  => 2 for 1D: capacity.A[2]
    #  => 3 for 2D: capacity.A[3]
    #  => 4 for 3D: capacity.A[4]
    cap_index = len_dims
    
    # Vn_1 stands for V at time n+1 (to be solved for)
    # Vn   stands for V at time n (known)
    Vn_1 = capacity.A[cap_index][1:end÷2, 1:end÷2]
    Vn   = capacity.A[cap_index][end÷2+1:end, end÷2+1:end]

    # Select time integration weighting : For fresh/dead cells handling
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else
        psip, psim = psip_be, psim_be
    end
    Ψn1 = Diagonal(psip.(Vn, Vn_1))

    # Build boundary blocks
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Iᵧ     = capacity.Γ
    Id     = build_I_D(operator, D, capacity)

    # Adjust for dimension
    if len_dims == 2
        # 1D problem
        nx, nt = dims
        n = nx
    elseif len_dims == 3
        # 2D problem
        nx, ny, nt = dims
        n = nx*ny
    elseif len_dims == 4
        # 3D problem
        nx, ny, nz, nt = dims
        n = nx*ny*nz
    else
        error("A_mono_unstead_diff_moving_generic not supported for dimension $(len_dims).")
    end

    # VERY IMPORTANT : Extract operator sub-blocks
    W! = operator.Wꜝ[1:end÷2, 1:end÷2]
    G  = operator.G[1:end÷2, 1:end÷2]
    H  = operator.H[1:end÷2, 1:end÷2]
    Iᵦ = Iᵦ[1:end÷2, 1:end÷2]
    Iₐ = Iₐ[1:end÷2, 1:end÷2]
    Iᵧ = Iᵧ[1:end÷2, 1:end÷2]
    Id  = Id[1:end÷2, 1:end÷2]

    # Construct subblocks
    block1 = Vn_1 + Id * G' * W! * G * Ψn1
    block2 = -(Vn_1 - Vn) + Id * G' * W! * H * Ψn1
    block3 = Iᵦ * H' * W! * G
    block4 = Iᵦ * H' * W! * H + (Iₐ * Iᵧ)

    return [block1 block2; block3 block4]
end

function b_mono_unstead_diff_moving(operator::DiffusionOps, capacity::Capacity, D, f::Function, bc::AbstractBoundary, Tᵢ::Vector{Float64}, Δt::Float64, t::Float64, scheme::String)
    # Determine how many dimensions
    dims = operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Build common data
    fₒn  = build_source(operator, f, t,      capacity)
    fₒn1 = build_source(operator, f, t+Δt,  capacity)
    gᵧ   = build_g_g(operator, bc, capacity)
    Id   = build_I_D(operator, D, capacity)

    Vn_1 = capacity.A[cap_index][1:end÷2, 1:end÷2]
    Vn   = capacity.A[cap_index][end÷2+1:end, end÷2+1:end]

    # Time integration weighting
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else
        psip, psim = psip_be, psim_be
    end
    Ψn = Diagonal(psim.(Vn, Vn_1))

    # Create the 1D or 2D indices
    if len_dims == 2
        # 1D case
        nx, nt = dims
        n = nx
        
    elseif len_dims == 3
        # 2D case
        nx, ny, nt = dims
        n = nx*ny
    elseif len_dims == 4
        # 3D case
        nx, ny, nz, nt = dims
        n = nx*ny*nz
    else
        error("b_mono_unstead_diff_moving not supported for dimension $len_dims")
    end

    # Extract operator sub-blocks
    W! = operator.Wꜝ[1:end÷2, 1:end÷2]
    G  = operator.G[1:end÷2, 1:end÷2]
    H  = operator.H[1:end÷2, 1:end÷2]
    V  = operator.V[1:end÷2, 1:end÷2]
    Iᵧ_mat = capacity.Γ[1:end÷2, 1:end÷2]
    Tₒ = Tᵢ[1:end÷2]
    Tᵧ = Tᵢ[end÷2+1:end]
    fₒn, fₒn1 = fₒn[1:end÷2], fₒn1[1:end÷2]
    gᵧ = gᵧ[1:end÷2]
    Id = Id[1:end÷2, 1:end÷2]

    # Construct the right-hand side
    if scheme == "CN"
        b1 = (Vn - Id * G' * W! * G * Ψn)*Tₒ - 0.5 * Id * G' * W! * H * Tᵧ + 0.5 * V * (fₒn + fₒn1)
    else
        b1 = Vn * Tₒ + V * fₒn1
    end
    b2 = Iᵧ_mat * gᵧ

    return [b1; b2]
end

function solve_MovingDiffusionUnsteadyMono!(s::Solver, phase::Phase, body::Function, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, mesh::AbstractMesh, scheme::String; method = IterativeSolvers.gmres, algorithm=nothing, geometry_method="VOFI", kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the problem:")
    println("- Moving problem")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Store the initial condition at the start time
    t = Tₛ
    s.x === nothing && error("Initial condition is not set. Initialize solver with the initial state.")
    push!(s.states, copy(s.x))
    println("Time : $(t)")
    println("Solver Extremum : ", maximum(abs.(s.x)))
    Tᵢ = s.x
    
    # Guard against floating point drift when stepping to the final time
    tol = eps(Float64) * max(1.0, abs(Tₑ))

    # Time loop
    while t + tol < Tₑ
        step_dt = min(Δt, Tₑ - t)
        t_next = t + step_dt
        println("Time : $(t_next)")
        STmesh = Penguin.SpaceTimeMesh(mesh, [t, t_next], tag=mesh.tag)
        capacity = Capacity(body, STmesh; method=geometry_method, kwargs...)
        operator = DiffusionOps(capacity)

        s.A = A_mono_unstead_diff_moving(operator, capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(operator, capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, step_dt, t, scheme)

        BC_border_mono!(s.A, s.b, bc_b, mesh; t=t_next) 

        # Solve system
        solve_system!(s; method, algorithm=algorithm, kwargs...)

        push!(s.states, s.x)
        println("Solver Extremum : ", maximum(abs.(s.x)))
        Tᵢ = s.x
        t = t_next
    end

end

function solve_MovingDiffusionUnsteadyMono_cfl!(s::Solver, phase::Phase, body::Function, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, mesh::AbstractMesh, scheme::String, interface_velocity, cfl::Float64; method = IterativeSolvers.gmres, algorithm=nothing, geometry_method="VOFI", kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the problem:")
    println("- Moving problem")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Store the initial condition at the start time
    t = Tₛ
    s.x === nothing && error("Initial condition is not set. Initialize solver with the initial state.")
    push!(s.states, copy(s.x))
    println("Time : $(t)")
    println("Solver Extremum : ", maximum(abs.(s.x)))
    Tᵢ = s.x

    if length(mesh.nodes) == 1
        Δh_min = minimum(diff(mesh.nodes[1]))
    elseif length(mesh.nodes) == 2
        Δh_min = min(minimum(diff(mesh.nodes[1])), minimum(diff(mesh.nodes[2])))
    elseif length(mesh.nodes) == 3
        Δh_min = min(minimum(diff(mesh.nodes[1])), minimum(diff(mesh.nodes[2])), minimum(diff(mesh.nodes[3])))
    else
        error("Unsupported mesh dimension")
    end

    function interface_speed(velocity, time)
        v = velocity isa Function ? velocity(time) : velocity
        return v isa Number ? abs(v) : maximum(abs.(v))
    end

    Δt_prev = Δt
    # Guard against floating point drift when stepping to the final time
    tol = eps(Float64) * max(1.0, abs(Tₑ))
    # Time loop
    while t + tol < Tₑ
        v_abs = interface_speed(interface_velocity, t)
        if v_abs < 1e-12
            Δt_step = min(Δt_prev, Tₑ - t)
        else
            Δt_step = min(cfl * Δh_min / v_abs, Tₑ - t)
        end
        t_next = t + Δt_step

        println("Time : $(t_next)")
        STmesh = Penguin.SpaceTimeMesh(mesh, [t, t_next], tag=mesh.tag)
        capacity = Capacity(body, STmesh; method=geometry_method, kwargs...)
        operator = DiffusionOps(capacity)

        s.A = A_mono_unstead_diff_moving(operator, capacity, phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(operator, capacity, phase.Diffusion_coeff, phase.source, bc, Tᵢ, Δt_step, t, scheme)

        BC_border_mono!(s.A, s.b, bc_b, mesh; t=t_next)

        # Solve system
        solve_system!(s; method, algorithm=algorithm, kwargs...)

        push!(s.states, s.x)
        println("Solver Extremum : ", maximum(abs.(s.x)))
        Tᵢ = s.x
        t = t_next
        Δt_prev = Δt_step
    end

end


# Moving - Diffusion - Unsteady - Diphasic
function MovingDiffusionUnsteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Moving problem")
    println("- Diphasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Diphasic, Diffusion, nothing, nothing, nothing, [], [])

    s.x = copy(Tᵢ)
    
    if scheme == "CN"
        s.A = A_diph_unstead_diff_moving(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, "CN")
        s.b = b_diph_unstead_diff_moving(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, 0.0, "CN")
    else 
        s.A = A_diph_unstead_diff_moving(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, "BE")
        s.b = b_diph_unstead_diff_moving(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, Δt, 0.0, "BE")
    end
    BC_border_diph!(s.A, s.b, bc_b, mesh)
    return s
end

function A_diph_unstead_diff_moving(operator1::DiffusionOps, operator2::DiffusionOps, capacite1::Capacity, capacite2::Capacity, D1, D2, ic::InterfaceConditions, scheme::String)
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
        nx2, _ = operator2.size
        n = nx1  # used for sub-block sizing
    elseif len_dims1 == 3
        # 2D problem
        nx1, ny1, _ = dims1
        nx2, ny2, _ = operator2.size
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
    Id1  = Id1[1:end÷2, 1:end÷2]
    Id2  = Id2[1:end÷2, 1:end÷2]

    # Construct blocks
    block1 = Vn1_1 + Id1 * G1' * W!1 * G1 * Ψn1
    block2 = -(Vn1_1 - Vn1) + Id1 * G1' * W!1 * H1 * Ψn1
    block3 = Vn2_1 + Id2 * G2' * W!2 * G2 * Ψn2
    block4 = -(Vn2_1 - Vn2) + Id2 * G2' * W!2 * H2 * Ψn2

    block5 = Iᵦ1 * H1' * W!1 * G1 * Ψn1
    block6 = Iᵦ1 * H1' * W!1 * H1 * Ψn1 -(Vn1_1 - Vn1) 
    block7 = Iᵦ2 * H2' * W!2 * G2 * Ψn2
    block8 = Iᵦ2 * H2' * W!2 * H2 * Ψn2 -(Vn2_1 - Vn2)

    # Build the 4n×4n matrix
    A = spzeros(Float64, 4n, 4n)

    # Assign sub-blocks

    A1 = [block1 block2 spzeros(size(block2)) spzeros(size(block2))]


    A2 = [spzeros(size(block1)) Iₐ1 spzeros(size(block2)) -Iₐ2]


    A3 = [spzeros(size(block1)) spzeros(size(block2)) block3 block4]


    A4 = [block5 block6 block7 block8]

    # Combine all blocks into the final matrix
    A = [A1; A2; A3; A4]
    return A
end

function b_diph_unstead_diff_moving(operator1::DiffusionOps, operator2::DiffusionOps, capacity1::Capacity, capacity2::Capacity, D1, D2, f1::Function, f2::Function, ic::InterfaceConditions, Tᵢ::Vector{Float64}, Δt::Float64, t::Float64, scheme::String)
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
        b1 = (Vn1 - Id1 * G1' * W!1 * G1 * Ψn1) * Tₒ1 - Id1 * G1' * W!1 * H1 * Ψn1 * Tᵧ1 + 0.5 * V1 * (f1ₒn + f1ₒn1)
        b3 = (Vn2 - Id2 * G2' * W!2 * G2 * Ψn2) * Tₒ2 - Id2 * G2' * W!2 * H2 * Ψn2 * Tᵧ2 + 0.5 * V2 * (f2ₒn + f2ₒn1)
    else
        b1 = (Vn1 - Id1 * G1' * W!1 * G1 * Ψn1) * Tₒ1 - Id1 * G1' * W!1 * H1 * Ψn1 * Tᵧ1 + V1 * f1ₒn1
        b3 = (Vn2 - Id2 * G2' * W!2 * G2 * Ψn2) * Tₒ2 - Id2 * G2' * W!2 * H2 * Ψn2 * Tᵧ2 + V2 * f2ₒn1
    end

    # 10) Build boundary terms
    b2 = gᵧ
    b4 = Iᵧ2 * hᵧ

    # Final right-hand side
    return vcat(b1, b2, b3, b4)
end


function solve_MovingDiffusionUnsteadyDiph!(s::Solver, phase1::Phase, phase2::Phase, body::Function, body_c::Function, Δt::Float64, Tₑ::Float64, bc_b::BorderConditions, ic::InterfaceConditions, mesh::AbstractMesh, scheme::String; method = IterativeSolvers.gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the problem:")
    println("- Moving problem")
    println("- Diphasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Store the initial condition at the start time
    t = 0.0
    s.x === nothing && error("Initial condition is not set. Initialize solver with the initial state.")
    push!(s.states, copy(s.x))
    println("Time : $(t)")
    println("Solver Extremum : ", maximum(abs.(s.x)))
    Tᵢ = s.x

    # Guard against floating point drift when stepping to the final time
    tol = eps(Float64) * max(1.0, abs(Tₑ))

    # Time loop
    while t + tol < Tₑ
        step_dt = min(Δt, Tₑ - t)
        t_next = t + step_dt
        println("Time : $(t_next)")
        STmesh = Penguin.SpaceTimeMesh(mesh, [t, t_next], tag=mesh.tag)
        capacity1 = Capacity(body, STmesh)
        capacity2 = Capacity(body_c, STmesh)
        operator1 = DiffusionOps(capacity1)
        operator2 = DiffusionOps(capacity2)

        s.A = A_diph_unstead_diff_moving(operator1, operator2, capacity1, capacity2, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, scheme)
        s.b = b_diph_unstead_diff_moving(operator1, operator2, capacity1, capacity2, phase1.Diffusion_coeff, phase2.Diffusion_coeff, phase1.source, phase2.source, ic, Tᵢ, step_dt, t, scheme)

        BC_border_diph!(s.A, s.b, bc_b, mesh)

        # Solve system
        solve_system!(s; method, algorithm=algorithm, kwargs...)

        push!(s.states, s.x)
        println("Solver Extremum : ", maximum(abs.(s.x)))
        Tᵢ = s.x
        t = t_next

    end
end
