# Binary Diffusion - Two-phase Unsteady Solver
"""
    A_binary_unsteady_diph(operatorT1, operatorT2, operatorC1, operatorC2, capacityT1, capacityT2, capacityC1, capacityC2, DT1, DT2, DC1, DC2, icT, icC, m, k, scheme)

Construct the system matrix for binary diffusion two-phase problem with temperature and concentration coupling.

# Arguments
- operatorTx: Diffusion operators for temperature in phase x
- operatorCx: Diffusion operators for concentration in phase x
- capacityTx: Capacity for temperature in phase x
- capacityCx: Capacity for concentration in phase x
- DTx: Thermal diffusivity in phase x
- DCx: Mass diffusivity in phase x
- icT: Interface condition for temperature
- icC: Interface condition for concentration
- m: Liquidus slope
- k: Partition coefficient
- scheme: Time integration scheme ("BE" or "CN")
"""
function A_binary_unsteady_diph(
    operatorT1, operatorT2, operatorC1, operatorC2,
    capacityT1, capacityT2, capacityC1, capacityC2,
    DT1, DT2, DC1, DC2,
    icT, icC, m, k, scheme
)
    # Determine dimensions from operator
    dims = operatorT1.size
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
    jumpT, fluxT = icT.scalar, icT.flux
    jumpC, fluxC = icC.scalar, icC.flux
    
    # Interface condition weights
    IₐT1, IₐT2 = jumpT.α₁ * I(n), jumpT.α₂ * I(n)
    IᵦT1, IᵦT2 = fluxT.β₁ * I(n), fluxT.β₂ * I(n)
    
    IₐC1, IₐC2 = jumpC.α₁ * I(n), jumpC.α₂ * I(n)
    IᵦC1, IᵦC2 = fluxC.β₁ * I(n), fluxC.β₂ * I(n)
    
    # Build diffusion operators
    IdT1 = build_I_D(operatorT1, DT1, capacityT1)
    IdT2 = build_I_D(operatorT2, DT2, capacityT2)
    IdC1 = build_I_D(operatorC1, DC1, capacityC1)
    IdC2 = build_I_D(operatorC2, DC2, capacityC2)
    
    # Extract capacity matrices
    VnT1_1 = capacityT1.A[cap_index][1:end÷2, 1:end÷2]
    VnT1   = capacityT1.A[cap_index][end÷2+1:end, end÷2+1:end]
    VnT2_1 = capacityT2.A[cap_index][1:end÷2, 1:end÷2]
    VnT2   = capacityT2.A[cap_index][end÷2+1:end, end÷2+1:end]
    
    VnC1_1 = capacityC1.A[cap_index][1:end÷2, 1:end÷2]
    VnC1   = capacityC1.A[cap_index][end÷2+1:end, end÷2+1:end]
    VnC2_1 = capacityC2.A[cap_index][1:end÷2, 1:end÷2]
    VnC2   = capacityC2.A[cap_index][end÷2+1:end, end÷2+1:end]
    
    # Time integration weights
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else # BE
        psip, psim = psip_be, psim_be
    end
    
    ΨnT1 = Diagonal(psip.(VnT1, VnT1_1))
    ΨnT2 = Diagonal(psip.(VnT2, VnT2_1))
    ΨnC1 = Diagonal(psip.(VnC1, VnC1_1))
    ΨnC2 = Diagonal(psip.(VnC2, VnC2_1))
    
    # Operator sub-blocks
    # Temperature phase 1
    W!T1 = operatorT1.Wꜝ[1:n, 1:n]
    GT1 = operatorT1.G[1:n, 1:n]
    HT1 = operatorT1.H[1:n, 1:n]
    # Temperature phase 2
    W!T2 = operatorT2.Wꜝ[1:n, 1:n]
    GT2 = operatorT2.G[1:n, 1:n]
    HT2 = operatorT2.H[1:n, 1:n]
    # Concentration phase 1
    W!C1 = operatorC1.Wꜝ[1:n, 1:n]
    GC1 = operatorC1.G[1:n, 1:n]
    HC1 = operatorC1.H[1:n, 1:n]
    # Concentration phase 2
    W!C2 = operatorC2.Wꜝ[1:n, 1:n]
    GC2 = operatorC2.G[1:n, 1:n]
    HC2 = operatorC2.H[1:n, 1:n]
    
    # Pre-slice matrices to right dimensions
    IdT1 = IdT1[1:n, 1:n]
    IdT2 = IdT2[1:n, 1:n]
    IdC1 = IdC1[1:n, 1:n]
    IdC2 = IdC2[1:n, 1:n]
    
    IᵦT1 = IᵦT1[1:n, 1:n]
    IᵦT2 = IᵦT2[1:n, 1:n]
    IᵦC1 = IᵦC1[1:n, 1:n]
    IᵦC2 = IᵦC2[1:n, 1:n]
    
    IₐT1 = IₐT1[1:n, 1:n]
    IₐT2 = IₐT2[1:n, 1:n]
    IₐC1 = IₐC1[1:n, 1:n]
    IₐC2 = IₐC2[1:n, 1:n]
    
    # Construct blocks for temperature
    blockT1 = VnT1_1 + IdT1 * GT1' * W!T1 * GT1 * ΨnT1
    blockT2 = -(VnT1_1 - VnT1) + IdT1 * GT1' * W!T1 * HT1 * ΨnT1
    blockT3 = VnT2_1 + IdT2 * GT2' * W!T2 * GT2 * ΨnT2
    blockT4 = -(VnT2_1 - VnT2) + IdT2 * GT2' * W!T2 * HT2 * ΨnT2
    
    blockT5 = IᵦT1 * HT1' * W!T1 * GT1 * ΨnT1
    blockT6 = IᵦT1 * HT1' * W!T1 * HT1 * ΨnT1
    blockT7 = IᵦT2 * HT2' * W!T2 * GT2 * ΨnT2
    blockT8 = IᵦT2 * HT2' * W!T2 * HT2 * ΨnT2
    
    # Construct blocks for concentration
    blockC1 = VnC1_1 + IdC1 * GC1' * W!C1 * GC1 * ΨnC1
    blockC2 = -(VnC1_1 - VnC1) + IdC1 * GC1' * W!C1 * HC1 * ΨnC1
    blockC3 = VnC2_1 + IdC2 * GC2' * W!C2 * GC2 * ΨnC2
    blockC4 = -(VnC2_1 - VnC2) + IdC2 * GC2' * W!C2 * HC2 * ΨnC2
    
    blockC5 = IᵦC1 * HC1' * W!C1 * GC1 * ΨnC1
    blockC6 = IᵦC1 * HC1' * W!C1 * HC1 * ΨnC1
    blockC7 = IᵦC2 * HC2' * W!C2 * GC2 * ΨnC2
    blockC8 = IᵦC2 * HC2' * W!C2 * HC2 * ΨnC2
    
    # Build the 8n×8n matrix for T1, T1γ, T2, T2γ, C1, C1γ, C2, C2γ
    A = spzeros(Float64, 8n, 8n)
    
    # Temperature phase 1 (T1)
    A[1:n, 1:n] = blockT1
    A[1:n, n+1:2n] = blockT2
    
    # Interface temperature (T1γ) - Set to Tm directly
    A[n+1:2n, n+1:2n] = I(n)
    # No coupling to concentration, just fixed value Tm
    
    # Temperature phase 2 (T2)
    A[2n+1:3n, 2n+1:3n] = blockT3
    A[2n+1:3n, 3n+1:4n] = blockT4
    
    # Interface temperature (T2γ) - Set to Tm directly
    A[3n+1:4n, 3n+1:4n] = I(n)
    
    # Concentration phase 1 (C1)
    A[4n+1:5n, 4n+1:5n] = blockC1
    A[4n+1:5n, 5n+1:6n] = blockC2
    
    # Interface concentration (C1γ) - Set to cm directly
    A[5n+1:6n, 5n+1:6n] = I(n)
    
    # Concentration phase 2 (C2)
    A[6n+1:7n, 6n+1:7n] = blockC3
    A[6n+1:7n, 7n+1:8n] = blockC4
    
    # Interface concentration (C2γ) - Set to cm directly
    A[7n+1:8n, 7n+1:8n] = I(n)
    
    """
    # Energy balance (Stefan condition)
    A[n+1:2n, 1:n] = blockT5
    A[n+1:2n, n+1:2n] = blockT6
    A[n+1:2n, 2n+1:3n] = blockT7
    A[n+1:2n, 3n+1:4n] = blockT8
    
    # Mass conservation
    A[5n+1:6n, 4n+1:5n] = blockC5
    A[5n+1:6n, 5n+1:6n] = blockC6
    A[5n+1:6n, 6n+1:7n] = blockC7
    A[5n+1:6n, 7n+1:8n] = blockC8
    """

    return A
end

"""
    b_binary_unsteady_diph(operatorT1, operatorT2, operatorC1, operatorC2, capacityT1, capacityT2, capacityC1, capacityC2, DT1, DT2, DC1, DC2, fT1, fT2, fC1, fC2, icT, icC, m, k, Tᵢ, Cᵢ, Δt, t, scheme)

Construct the right-hand side vector for binary diffusion two-phase problem.
"""
function b_binary_unsteady_diph(
    operatorT1, operatorT2, operatorC1, operatorC2,
    capacityT1, capacityT2, capacityC1, capacityC2,
    DT1, DT2, DC1, DC2,
    fT1, fT2, fC1, fC2,
    icT, icC, m, k,
    Tᵢ, Cᵢ, Δt, t, scheme
)
    # Determine problem dimensions
    dims = operatorT1.size
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
    fT1ₒn  = build_source(operatorT1, fT1, t, capacityT1)
    fT1ₒn1 = build_source(operatorT1, fT1, t+Δt, capacityT1)
    fT2ₒn  = build_source(operatorT2, fT2, t, capacityT2)
    fT2ₒn1 = build_source(operatorT2, fT2, t+Δt, capacityT2)
    
    fC1ₒn  = build_source(operatorC1, fC1, t, capacityC1)
    fC1ₒn1 = build_source(operatorC1, fC1, t+Δt, capacityC1)
    fC2ₒn  = build_source(operatorC2, fC2, t, capacityC2)
    fC2ₒn1 = build_source(operatorC2, fC2, t+Δt, capacityC2)
    
    # Get interface data
    jumpT, fluxT = icT.scalar, icT.flux
    jumpC, fluxC = icC.scalar, icC.flux
    
    # Get the fixed temperature and concentration at the interface
    Tm = jumpT.value  # Fixed interface temperature
    Cm = jumpC.value  # Fixed interface concentration
    
    # Extract interface data
    IᵧT1, IᵧT2 = capacityT1.Γ, capacityT2.Γ
    IᵧC1, IᵧC2 = capacityC1.Γ, capacityC2.Γ
    
    gᵧT = build_g_g(operatorT1, jumpT, capacityT1)
    hᵧT = build_g_g(operatorT2, fluxT, capacityT2)
    
    gᵧC = build_g_g(operatorC1, jumpC, capacityC1)
    hᵧC = build_g_g(operatorC2, fluxC, capacityC2)
    
    # Extract capacity matrices
    VnT1_1 = capacityT1.A[cap_index][1:end÷2, 1:end÷2]
    VnT1   = capacityT1.A[cap_index][end÷2+1:end, end÷2+1:end]
    VnT2_1 = capacityT2.A[cap_index][1:end÷2, 1:end÷2]
    VnT2   = capacityT2.A[cap_index][end÷2+1:end, end÷2+1:end]
    
    VnC1_1 = capacityC1.A[cap_index][1:end÷2, 1:end÷2]
    VnC1   = capacityC1.A[cap_index][end÷2+1:end, end÷2+1:end]
    VnC2_1 = capacityC2.A[cap_index][1:end÷2, 1:end÷2]
    VnC2   = capacityC2.A[cap_index][end÷2+1:end, end÷2+1:end]
    
    # Time integration weights
    if scheme == "CN"
        psip, psim = psip_cn, psim_cn
    else # BE
        psip, psim = psip_be, psim_be
    end
    
    ΨnT1 = Diagonal(psim.(VnT1, VnT1_1))
    ΨnT2 = Diagonal(psim.(VnT2, VnT2_1))
    ΨnC1 = Diagonal(psim.(VnC1, VnC1_1))
    ΨnC2 = Diagonal(psim.(VnC2, VnC2_1))
    
    # Slice initial conditions
    Tₒ1 = Tᵢ[1:n]
    Tᵧ1 = Tᵢ[n+1:2n]
    Tₒ2 = Tᵢ[2n+1:3n]
    Tᵧ2 = Tᵢ[3n+1:4n]
    
    Cₒ1 = Cᵢ[1:n]
    Cᵧ1 = Cᵢ[n+1:2n]
    Cₒ2 = Cᵢ[2n+1:3n]
    Cᵧ2 = Cᵢ[3n+1:4n]
    
    # Slice and extract operators
    fT1ₒn  = fT1ₒn[1:n]
    fT1ₒn1 = fT1ₒn1[1:n]
    fT2ₒn  = fT2ₒn[1:n]
    fT2ₒn1 = fT2ₒn1[1:n]
    
    fC1ₒn  = fC1ₒn[1:n]
    fC1ₒn1 = fC1ₒn1[1:n]
    fC2ₒn  = fC2ₒn[1:n]
    fC2ₒn1 = fC2ₒn1[1:n]
    
    gᵧT = gᵧT[1:n]
    hᵧT = hᵧT[1:n]
    gᵧC = gᵧC[1:n]
    hᵧC = hᵧC[1:n]
    
    IᵧT1 = IᵧT1[1:n, 1:n]
    IᵧT2 = IᵧT2[1:n, 1:n]
    IᵧC1 = IᵧC1[1:n, 1:n]
    IᵧC2 = IᵧC2[1:n, 1:n]
    
    # Build IDT and IDC
    IdT1 = build_I_D(operatorT1, DT1, capacityT1)
    IdT2 = build_I_D(operatorT2, DT2, capacityT2)
    IdC1 = build_I_D(operatorC1, DC1, capacityC1)
    IdC2 = build_I_D(operatorC2, DC2, capacityC2)
    
    IdT1 = IdT1[1:n, 1:n]
    IdT2 = IdT2[1:n, 1:n]
    IdC1 = IdC1[1:n, 1:n]
    IdC2 = IdC2[1:n, 1:n]
    
    # Get remaining operators
    W!T1 = operatorT1.Wꜝ[1:n, 1:n]
    GT1 = operatorT1.G[1:n, 1:n]
    HT1 = operatorT1.H[1:n, 1:n]
    VT1 = operatorT1.V[1:n, 1:n]
    
    W!T2 = operatorT2.Wꜝ[1:n, 1:n]
    GT2 = operatorT2.G[1:n, 1:n]
    HT2 = operatorT2.H[1:n, 1:n]
    VT2 = operatorT2.V[1:n, 1:n]
    
    W!C1 = operatorC1.Wꜝ[1:n, 1:n]
    GC1 = operatorC1.G[1:n, 1:n]
    HC1 = operatorC1.H[1:n, 1:n]
    VC1 = operatorC1.V[1:n, 1:n]
    
    W!C2 = operatorC2.Wꜝ[1:n, 1:n]
    GC2 = operatorC2.G[1:n, 1:n]
    HC2 = operatorC2.H[1:n, 1:n]
    VC2 = operatorC2.V[1:n, 1:n]
    
    # Build RHS for temperature
    if scheme == "CN"
        bT1 = (VnT1 - IdT1 * GT1' * W!T1 * GT1 * ΨnT1) * Tₒ1 - 0.5 * IdT1 * GT1' * W!T1 * HT1 * Tᵧ1 + 0.5 * VT1 * (fT1ₒn + fT1ₒn1)
        bT3 = (VnT2 - IdT2 * GT2' * W!T2 * GT2 * ΨnT2) * Tₒ2 - 0.5 * IdT2 * GT2' * W!T2 * HT2 * Tᵧ2 + 0.5 * VT2 * (fT2ₒn + fT2ₒn1)
        
        bC1 = (VnC1 - IdC1 * GC1' * W!C1 * GC1 * ΨnC1) * Cₒ1 - 0.5 * IdC1 * GC1' * W!C1 * HC1 * Cᵧ1 + 0.5 * VC1 * (fC1ₒn + fC1ₒn1)
        bC3 = (VnC2 - IdC2 * GC2' * W!C2 * GC2 * ΨnC2) * Cₒ2 - 0.5 * IdC2 * GC2' * W!C2 * HC2 * Cᵧ2 + 0.5 * VC2 * (fC2ₒn + fC2ₒn1)
    else  # BE
        bT1 = VnT1 * Tₒ1 + VT1 * fT1ₒn1
        bT3 = VnT2 * Tₒ2 + VT2 * fT2ₒn1
        
        bC1 = VnC1 * Cₒ1 + VC1 * fC1ₒn1
        bC3 = VnC2 * Cₒ2 + VC2 * fC2ₒn1
    end
    
    # Modified interface conditions - fixed values
    # Temperature interface values
    bT2 = ones(n) * Tm  # T1γ = Tm
    bT4 = ones(n) * Tm  # T2γ = Tm
    
    # Concentration interface values
    bC2 = ones(n) * Cm  # C1γ = Cm
    bC4 = ones(n) * Cm  # C2γ = Cm
    
    # Combine all components
    return vcat(bT1, bT2, bT3, bT4, bC1, bC2, bC3, bC4)
end

"""
    DiffusionUnsteadyBinary(phaseT1, phaseT2, phaseC1, phaseC2, bc_bT, bc_bC, icT, icC, Δt, u0_T, u0_C, scheme, Newton_params)

Create a solver for binary diffusion with two phases, temperature and concentration.
"""
function DiffusionUnsteadyBinary(
    phaseT1::Phase, phaseT2::Phase, 
    phaseC1::Phase, phaseC2::Phase,
    bc_bT::BorderConditions, bc_bC::BorderConditions, 
    icT::InterfaceConditions, icC::InterfaceConditions, 
    Δt::Float64, u0_T::Vector{Float64}, u0_C::Vector{Float64}, 
    scheme::String,
    Newton_params::Tuple{Int64, Float64, Float64, Float64}
)
    println("Solver Creation:")
    println("- Two-phase Binary Diffusion")
    println("- Unsteady problem")
    println("- Coupling temperature and concentration")
    
    s = Solver(Unsteady, Diphasic, Diffusion, nothing, nothing, nothing, [], [])
    
    # Default physical parameters if not provided
    m = 1.0  # Liquidus slope
    k = 1.0  # Partition coefficient
    
    # Get dimensions and arrange initial conditions
    dims = phaseT1.operator.size
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
    
    s.A = A_binary_unsteady_diph(
        phaseT1.operator, phaseT2.operator, phaseC1.operator, phaseC2.operator,
        phaseT1.capacity, phaseT2.capacity, phaseC1.capacity, phaseC2.capacity,
        phaseT1.Diffusion_coeff, phaseT2.Diffusion_coeff, phaseC1.Diffusion_coeff, phaseC2.Diffusion_coeff,
        icT, icC, m, k, scheme
    )
    
    s.b = b_binary_unsteady_diph(
        phaseT1.operator, phaseT2.operator, phaseC1.operator, phaseC2.operator,
        phaseT1.capacity, phaseT2.capacity, phaseC1.capacity, phaseC2.capacity,
        phaseT1.Diffusion_coeff, phaseT2.Diffusion_coeff, phaseC1.Diffusion_coeff, phaseC2.Diffusion_coeff,
        phaseT1.source, phaseT2.source, phaseC1.source, phaseC2.source,
        icT, icC, m, k, 
        u0_T, u0_C, Δt, 0.0, scheme
    )
    
    # Apply boundary conditions
    BC_border_binary!(s.A, s.b, bc_bT, bc_bC, n)
    
    return s
end

"""
    BC_border_binary!(A, b, bc_bT, bc_bC, n)

Apply temperature and concentration boundary conditions to the binary system.
"""
function BC_border_binary!(A, b, bc_bT, bc_bC, n)
    # Apply temperature BCs
    for (direction, bc) in bc_bT.borders
        idx = direction == :top ? 1 : n
        val = bc isa Dirichlet ? bc.value : 0.0
        
        # Temperature in phase 1
        A[idx, :] .= 0.0
        A[idx, idx] = 1.0
        b[idx] = val
        
        # Temperature in phase 2
        A[2*n+idx, :] .= 0.0
        A[2*n+idx, 2*n+idx] = 1.0
        b[2*n+idx] = val
    end
    
    # Apply concentration BCs
    for (direction, bc) in bc_bC.borders
        idx = direction == :top ? 1 : n
        val = bc isa Dirichlet ? bc.value : 0.0
        
        # Concentration in phase 1
        A[4*n+idx, :] .= 0.0
        A[4*n+idx, 4*n+idx] = 1.0
        b[4*n+idx] = val
        
        # Concentration in phase 2
        A[6*n+idx, :] .= 0.0
        A[6*n+idx, 6*n+idx] = 1.0
        b[6*n+idx] = val
    end
end

"""
    solve_DiffusionUnsteadyBinary!(s, phaseT1, phaseT2, phaseC1, phaseC2, Δt, Tend, bc_bT, bc_bC, icT, icC; Newton_params=(1000, 1e-8, 1e-8, 0.8), method=Base.:/)

Solve the binary diffusion problem with temperature and concentration coupling.
"""
function solve_DiffusionUnsteadyBinary!(
    s::Solver, 
    phaseT1::Phase, phaseT2::Phase, 
    phaseC1::Phase, phaseC2::Phase, xf,
    Δt::Float64, Tend::Float64, 
    bc_bT::BorderConditions, bc_bC::BorderConditions,
    icT::InterfaceConditions, icC::InterfaceConditions,
    mesh::AbstractMesh;
    Newton_params=(1000, 1e-8, 1e-8, 0.8),
    method=Base.:\,
    algorithm=nothing,
    kwargs...
)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end
    
    println("Solving binary diffusion problem:")
    println("- Two-phase system")
    println("- Temperature and concentration coupling")
    
    # Extract parameters
    ρL = icT.flux.value     # Latent heat coefficient
    m = 1.0                # Liquidus slope
    k = 1.0                # Partition coefficient
    
    max_iter = Newton_params[1]
    tol      = Newton_params[2]
    reltol   = Newton_params[3]
    α        = Newton_params[4]
    
    # Track residuals and interface position
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Float64[]
    
    # Determine dimensions
    dims = phaseT1.operator.size
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
    
    # Get initial combined state
    solve_system!(s; method=method, algorithm=algorithm, kwargs...)
    u = s.x
    
    # Split into temperature and concentration components
    u_T = u[1:4*n]
    u_C = u[4*n+1:8*n]
    
    push!(s.states, u)
    println("Max temperature: $(maximum(abs.(u_T)))")
    println("Max concentration: $(maximum(abs.(u_C)))")
    
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
            
            STmesh1 = SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
            STmesh2 = SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
            
            # Create new capacities and operators
            capacityT1 = Capacity(body1, STmesh1)
            capacityT2 = Capacity(body2, STmesh2)
            capacityC1 = Capacity(body1, STmesh1)
            capacityC2 = Capacity(body2, STmesh2)
            
            operatorT1 = DiffusionOps(capacityT1)
            operatorT2 = DiffusionOps(capacityT2)
            operatorC1 = DiffusionOps(capacityC1)
            operatorC2 = DiffusionOps(capacityC2)
            
            phaseT1 = Phase(capacityT1, operatorT1, phaseT1.source, phaseT1.Diffusion_coeff)
            phaseT2 = Phase(capacityT2, operatorT2, phaseT2.source, phaseT2.Diffusion_coeff)
            phaseC1 = Phase(capacityC1, operatorC1, phaseC1.source, phaseC1.Diffusion_coeff)
            phaseC2 = Phase(capacityC2, operatorC2, phaseC2.source, phaseC2.Diffusion_coeff)
            
            # 2. Rebuild system matrices with new geometry
            s.A = A_binary_unsteady_diph(
                phaseT1.operator, phaseT2.operator, phaseC1.operator, phaseC2.operator,
                phaseT1.capacity, phaseT2.capacity, phaseC1.capacity, phaseC2.capacity,
                phaseT1.Diffusion_coeff, phaseT2.Diffusion_coeff, phaseC1.Diffusion_coeff, phaseC2.Diffusion_coeff,
                icT, icC, m, k, "BE"
            )
            
            s.b = b_binary_unsteady_diph(
                phaseT1.operator, phaseT2.operator, phaseC1.operator, phaseC2.operator,
                phaseT1.capacity, phaseT2.capacity, phaseC1.capacity, phaseC2.capacity,
                phaseT1.Diffusion_coeff, phaseT2.Diffusion_coeff, phaseC1.Diffusion_coeff, phaseC2.Diffusion_coeff,
                phaseT1.source, phaseT2.source, phaseC1.source, phaseC2.source,
                icT, icC, m, k, u_T, u_C, Δt, t-Δt, "BE"
            )
            
            # Apply boundary conditions
            BC_border_binary!(s.A, s.b, bc_bT, bc_bC, n)
            
            # 3. Solve the system
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            u = s.x
            
            # Split into temperature and concentration
            u_T = u[1:4*n]
            u_C = u[4*n+1:8*n]
            
            # 4. Extract capacity matrices for volume calculation
            VnT1_1 = phaseT1.capacity.A[cap_index][1:end÷2, 1:end÷2]
            VnT1   = phaseT1.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
            Hₙ   = sum(diag(VnT1))
            Hₙ₊₁ = sum(diag(VnT1_1))
            
            # 5. Compute interface fluxes for both fields
            # Temperature - phase 1
            W!T1 = phaseT1.operator.Wꜝ[1:n, 1:n]
            GT1 = phaseT1.operator.G[1:n, 1:n]
            HT1 = phaseT1.operator.H[1:n, 1:n]
            IdT1 = build_I_D(phaseT1.operator, phaseT1.Diffusion_coeff, phaseT1.capacity)
            IdT1 = IdT1[1:n, 1:n]
            
            Tₒ1, Tᵧ1 = u_T[1:n], u_T[n+1:2n]
            flux_T1 = IdT1 * HT1' * W!T1 * GT1 * Tₒ1 + IdT1 * HT1' * W!T1 * HT1 * Tᵧ1
            flux_T1_sum = sum(flux_T1)
            
            # Temperature - phase 2
            W!T2 = phaseT2.operator.Wꜝ[1:n, 1:n]
            GT2 = phaseT2.operator.G[1:n, 1:n]
            HT2 = phaseT2.operator.H[1:n, 1:n]
            IdT2 = build_I_D(phaseT2.operator, phaseT2.Diffusion_coeff, phaseT2.capacity)
            IdT2 = IdT2[1:n, 1:n]
            
            Tₒ2, Tᵧ2 = u_T[2n+1:3n], u_T[3n+1:4n]
            flux_T2 = IdT2 * HT2' * W!T2 * GT2 * Tₒ2 + IdT2 * HT2' * W!T2 * HT2 * Tᵧ2
            flux_T2_sum = sum(flux_T2)
            
            # Combined temperature flux for Stefan condition
            Interface_term_T = (flux_T1_sum + flux_T2_sum) / ρL
            
            # Concentration fluxes also impact interface motion through solutal effects
            # This is a simplified model; actual coupling would depend on specific physics
            W!C1 = phaseC1.operator.Wꜝ[1:n, 1:n]
            GC1 = phaseC1.operator.G[1:n, 1:n]
            HC1 = phaseC1.operator.H[1:n, 1:n]
            IdC1 = build_I_D(phaseC1.operator, phaseC1.Diffusion_coeff, phaseC1.capacity)
            IdC1 = IdC1[1:n, 1:n]
            
            # Additional solutal contribution would be calculated here
            
            # 6. Update interface position
            res = Hₙ₊₁ - Hₙ - Interface_term_T
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
        println("Max temperature: $(maximum(abs.(u_T)))")
        println("Max concentration: $(maximum(abs.(u_C)))")
    end
    
    return s, residuals, xf_log
end