# AdvectionDiffusion - Steady - Monophasic
"""
    AdvectionDiffusionSteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary)

Creates a solver for a steady-state monophasic advection-diffusion problem.

# Arguments
- `phase::Phase`: The phase object representing the physical properties of the system.
- `bc_b::BorderConditions`: The border conditions object representing the boundary conditions at the outer border.
- `bc_i::AbstractBoundary`: The boundary conditions object representing the boundary conditions at the inner border.
"""
function AdvectionDiffusionSteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary)
    println("Solver Creation:")
    println("- Monophasic problem")
    println("- Steady problem")
    println("- Advection-Diffusion problem")
    
    s = Solver(Steady, Monophasic, DiffusionAdvection, nothing, nothing, nothing, [], [])
    
    s.A = A_mono_stead_advdiff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i)
    s.b = b_mono_stead_advdiff(phase.operator, phase.source, phase.capacity, bc_i)

    BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh)

    return s
end

function A_mono_stead_advdiff(operator::ConvectionOps, capacite::Capacity, D, bc::AbstractBoundary)
    n = prod(operator.size)
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Iᵧ = capacite.Γ 
    Id = build_I_D(operator, D, capacite)

    C = operator.C # NTuple{N, SparseMatrixCSC{Float64, Int}}
    K = operator.K # NTuple{N, SparseMatrixCSC{Float64, Int}}

    A11 = (sum(C) + 0.5 * sum(K)) + Id * operator.G' * operator.Wꜝ * operator.G
    A12 = 0.5 * sum(K) + Id * operator.G' * operator.Wꜝ * operator.H
    A21 = Iᵦ * operator.H' * operator.Wꜝ * operator.G
    A22 = Iᵦ * operator.H' * operator.Wꜝ * operator.H + Iₐ * Iᵧ

    A = vcat(hcat(A11, A12), hcat(A21, A22))
    return A
end

function b_mono_stead_advdiff(operator::ConvectionOps, f, capacite::Capacity, bc::AbstractBoundary)
    N = prod(operator.size)
    b = zeros(2N)

    Iᵧ = capacite.Γ
    fₒ = build_source(operator, f, capacite)
    gᵧ = build_g_g(operator, bc, capacite)

    # Build the right-hand side
    b = vcat(operator.V*fₒ, Iᵧ * gᵧ)

    return b
end

function solve_AdvectionDiffusionSteadyMono!(s::Solver; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    solve_system!(s; method, algorithm=algorithm, kwargs...)
end


# AdvectionDiffusion - Steady - Diphasic
"""
    AdvectionDiffusionSteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions)

Creates a solver for a steady-state two-phase advection-diffusion problem.

# Arguments
- `phase1::Phase`: The first phase of the problem.
- `phase2::Phase`: The second phase of the problem.
- `bc_b::BorderConditions`: The boundary conditions of the problem.
- `ic::InterfaceConditions`: The conditions at the interface between the two phases.
"""
function AdvectionDiffusionSteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions)
    println("Création du solveur:")
    println("- Diphasic problem")
    println("- Steady problem")
    println("- Advection-Diffusion problem")
    
    s = Solver(Steady, Diphasic, DiffusionAdvection, nothing, nothing, nothing, [], [])
    
    s.A = A_diph_stead_advdiff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic)
    s.b = b_diph_stead_advdiff(phase1.operator, phase2.operator, phase1.source, phase2.source, phase1.capacity, phase2.capacity, ic)

    BC_border_diph!(s.A, s.b, bc_b, phase1.capacity, phase2.capacity)

    return s
end

function A_diph_stead_advdiff(operator1::ConvectionOps, operator2::ConvectionOps, capacite1::Capacity, capacite2::Capacity, D1, D2, ic::InterfaceConditions)
    n = prod(operator1.size)

    jump, flux = ic.scalar, ic.flux
    Iₐ1, Iₐ2 = jump.α₁*I(n), jump.α₂*I(n)
    Iᵦ1, Iᵦ2 = flux.β₁*I(n), flux.β₂*I(n)
    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    C1 = operator1.C # NTuple{N, SparseMatrixCSC{Float64, Int}}
    K1 = operator1.K # NTuple{N, SparseMatrixCSC{Float64, Int}}
    C2 = operator2.C # NTuple{N, SparseMatrixCSC{Float64, Int}}
    K2 = operator2.K # NTuple{N, SparseMatrixCSC{Float64, Int}

    block1 = Id1 * operator1.G' * operator1.Wꜝ * operator1.G + (sum(C1) + 0.5 * sum(K1))
    block2 = Id1 * operator1.G' * operator1.Wꜝ * operator1.H + 0.5 * sum(K1)
    block3 = Id2 * operator2.G' * operator2.Wꜝ * operator2.G + (sum(C2) + 0.5 * sum(K2))
    block4 = Id2 * operator2.G' * operator2.Wꜝ * operator2.H + 0.5 * sum(K2)
    block5 = operator1.H' * operator1.Wꜝ * operator1.G
    block6 = operator1.H' * operator1.Wꜝ * operator1.H 
    block7 = operator2.H' * operator2.Wꜝ * operator2.G
    block8 = operator2.H' * operator2.Wꜝ * operator2.H

    A = vcat(hcat(block1, block2, zeros(n, n), zeros(n, n)),
             hcat(zeros(n, n), Iₐ1, zeros(n, n), -Iₐ2),
             hcat(zeros(n, n), zeros(n, n), block3, block4),
             hcat(Iᵦ1*block5, Iᵦ1*block6, Iᵦ2*block7, Iᵦ2*block8))
    return A
end

function b_diph_stead_advdiff(operator1::ConvectionOps, operator2::ConvectionOps, f1, f2, capacite1::Capacity, capacite2::Capacity, ic::InterfaceConditions)
    N = prod(operator1.size)
    b = zeros(4N)

    jump, flux = ic.scalar, ic.flux
    Iᵧ1, Iᵧ2 = capacite1.Γ, capacite2.Γ
    gᵧ, hᵧ = build_g_g(operator1, jump, capacite1), build_g_g(operator2, flux, capacite2)

    fₒ1 = build_source(operator1, f1, capacite1)
    fₒ2 = build_source(operator2, f2, capacite2)

    # Build the right-hand side
    b = vcat(operator1.V*fₒ1, gᵧ, operator2.V*fₒ2, Iᵧ2*hᵧ)

    return b
end

function solve_AdvectionDiffusionSteadyDiph!(s::Solver; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    solve_system!(s; method, algorithm=algorithm, kwargs...)
end



# AdvectionDiffusion - Unsteady - Monophasic
"""
    AdvectionDiffusionUnsteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tₑ::Float64, Tᵢ::Vector{Float64})

Creates a solver for an unsteady monophasic advection-diffusion problem.

# Arguments
- `phase::Phase`: The phase object representing the physical properties of the system.
- `bc_b::BorderConditions`: The border conditions object representing the boundary conditions at the outer border.
- `bc_i::AbstractBoundary`: The boundary conditions object representing the boundary conditions at the inner border.
- `Δt::Float64`: The time step size.
- `Tᵢ::Vector{Float64}`: The initial temperature distribution.
"""
function AdvectionDiffusionUnsteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, scheme::String)
    println("Solver Creation:")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Advection-Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, DiffusionAdvection, nothing, nothing, nothing, [], [])
    
    s.A = A_mono_unstead_advdiff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, Δt, scheme)
    s.b = b_mono_unstead_advdiff(phase.operator, phase.source, phase.capacity, phase.Diffusion_coeff, bc_i, Tᵢ, Δt, 0.0, scheme)

    return s
end

function A_mono_unstead_advdiff(operator::ConvectionOps, capacite::Capacity, D, bc::AbstractBoundary, Δt::Float64, scheme::String)
    n = prod(operator.size)
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Iᵧ = capacite.Γ
    Id = build_I_D(operator, D, capacite)

    C = operator.C # NTuple{N, SparseMatrixCSC{Float64, Int}}
    K = operator.K # NTuple{N, SparseMatrixCSC{Float64, Int}}

    conv_bulk = sum(C)
    conv_iface = 0.5 * sum(K)
    diff_ωω = Id * operator.G' * operator.Wꜝ * operator.G
    diff_ωγ = Id * operator.G' * operator.Wꜝ * operator.H
    diff_γω = Iᵦ * operator.H' * operator.Wꜝ * operator.G
    diff_γγ = Iᵦ * operator.H' * operator.Wꜝ * operator.H
    tie_block = diff_γγ + Iₐ * Iᵧ

    if scheme == "CN"
        A11 = operator.V + Δt/2 * (conv_bulk + conv_iface + diff_ωω)
        A12 = Δt/2 * (conv_iface + diff_ωγ)
        A21 = Δt/2 * diff_γω
        A22 = Δt/2 * tie_block
    elseif scheme == "BE"
        A11 = operator.V + Δt * (conv_bulk + conv_iface + diff_ωω)
        A12 = Δt * (conv_iface + diff_ωγ)
        A21 = diff_γω
        A22 = tie_block
    else
        error("Unknown scheme.")
    end
    
    A = vcat(hcat(A11, A12), hcat(A21, A22))
    return A
end

function b_mono_unstead_advdiff(operator::ConvectionOps, f, capacite::Capacity, D, bc::AbstractBoundary, Tᵢ, Δt::Float64, t::Float64, scheme::String)
    N = prod(operator.size)
    b = zeros(2N)

    C = operator.C # NTuple{N, SparseMatrixCSC{Float64, Int}}
    K = operator.K # NTuple{N, SparseMatrixCSC{Float64, Int}}

    Iᵧ = capacite.Γ
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    fₒn, fₒn1 = build_source(operator, f, t, capacite), build_source(operator, f, t+Δt, capacite)
    gᵧn, gᵧn1 = build_g_g(operator, bc, capacite, t), build_g_g(operator, bc, capacite, t+Δt)

    Tₒ, Tᵧ = Tᵢ[1:N], Tᵢ[N+1:end]

    conv_bulk = sum(C)
    conv_iface = 0.5 * sum(K)
    diff_ωω = build_I_D(operator, D, capacite) * operator.G' * operator.Wꜝ * operator.G
    diff_ωγ = build_I_D(operator, D, capacite) * operator.G' * operator.Wꜝ * operator.H    
    diff_γω = Iᵦ * operator.H' * operator.Wꜝ * operator.G  
    tie_block = Iᵦ * operator.H' * operator.Wꜝ * operator.H + Iₐ * Iᵧ

    if scheme == "CN"
        b1 = (operator.V - Δt/2 * (conv_bulk + conv_iface + diff_ωω)) * Tₒ
        b1 .-= Δt/2 * (conv_iface + diff_ωγ) * Tᵧ
        b1 .+= Δt/2 * operator.V * (fₒn + fₒn1)

        b2 = Δt/2 * Iᵧ * (gᵧn + gᵧn1)
        b2 .-= Δt/2 * diff_γω * Tₒ
        b2 .-= Δt/2 * tie_block * Tᵧ
    elseif scheme == "BE"
        b1 = operator.V * Tₒ + Δt * operator.V * fₒn1
        b2 = Iᵧ * gᵧn1
    else
        error("Unknown scheme.")
    end

    b = vcat(b1, b2)
    return b
end

function solve_AdvectionDiffusionUnsteadyMono!(s::Solver, phase::Phase, Δt::Float64, Tₑ, bc_b::BorderConditions, bc::AbstractBoundary, scheme::String; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    # Solve for the initial time step
    t = 0.0
    println("Time: ", t)
    solve_system!(s; method, algorithm=algorithm, kwargs...)

    push!(s.states, s.x)
    println("Solver Extremum: ", maximum(abs.(s.x)))

    while t < Tₑ
        t += Δt
        println("Time: ", t)
        s.A = A_mono_unstead_advdiff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, Δt, scheme)
        s.b = b_mono_unstead_advdiff(phase.operator, phase.source, phase.capacity, bc, s.x, Δt, t, scheme)
        BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh; t=t)
        
        solve_system!(s; method, algorithm=algorithm, kwargs...)

        push!(s.states, s.x)
        println("Solver Extremum: ", maximum(abs.(s.x)))
        Tᵢ = s.x
    end
end



# AdvectionDiffusion - Unsteady - Diphasic
"""
    AdvectionDiffusionUnsteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions, Δt::Float64, Tₑ::Float64, Tᵢ::Vector{Float64})

Creates a solver for an unsteady two-phase advection-diffusion problem.

# Arguments
- `phase1::Phase`: The first phase of the problem.
- `phase2::Phase`: The second phase of the problem.
- `bc_b::BorderConditions`: The boundary conditions of the problem.
- `ic::InterfaceConditions`: The conditions at the interface between the two phases.
- `Δt::Float64`: The time interval.
- `Tᵢ::Vector{Float64}`: The vector of initial temperatures.
"""
function AdvectionDiffusionUnsteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions, Δt::Float64, Tᵢ::Vector{Float64}, scheme::String)
    println("Solver Creation:")
    println("- Diphasic problem")
    println("- Unsteady problem")
    println("- Advection-Diffusion problem")
    
    s = Solver(Unsteady, Diphasic, DiffusionAdvection, nothing, nothing, nothing, [], [])
    
    s.A = A_diph_unstead_advdiff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, Δt, scheme)
    s.b = b_diph_unstead_advdiff(phase1.operator, phase2.operator, phase1.source, phase2.source, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, Tᵢ, Δt, 0.0, scheme)

    return s
end

function A_diph_unstead_advdiff(operator1::ConvectionOps, operator2::ConvectionOps, capacite1::Capacity, capacite2::Capacity, D1, D2, ic::InterfaceConditions, Δt::Float64, scheme::String)
    n = prod(operator1.size)

    jump, flux = ic.scalar, ic.flux
    Iₐ1, Iₐ2 = jump.α₁*I(n), jump.α₂*I(n)
    Iᵦ1, Iᵦ2 = flux.β₁*I(n), flux.β₂*I(n)
    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    C1 = operator1.C # NTuple{N, SparseMatrixCSC{Float64, Int}}
    K1 = operator1.K # NTuple{N, SparseMatrixCSC{Float64, Int}}
    C2 = operator2.C # NTuple{N, SparseMatrixCSC{Float64, Int}}
    K2 = operator2.K # NTuple{N, SparseMatrixCSC{Float64, Int}

    if scheme == "CN"
        block1 = operator1.V + Δt/2 * (sum(C1) + 0.5 * sum(K1)) + Δt/2 * Id1 * operator1.G' * operator1.Wꜝ * operator1.G
        block2 = Δt/2 * 0.5 * sum(K1) + Δt/2 * Id1 * operator1.G' * operator1.Wꜝ * operator1.H
        block3 = operator2.V + Δt/2 * (sum(C2) + 0.5 * sum(K2)) + Δt/2 * Id2 * operator2.G' * operator2.Wꜝ * operator2.G
        block4 = Δt/2 * 0.5 * sum(K2) + Δt/2 * Id2 * operator2.G' * operator2.Wꜝ * operator2.H
        block5 = Iᵦ1 * operator1.H' * operator1.Wꜝ * operator1.G
        block6 = Iᵦ1 * operator1.H' * operator1.Wꜝ * operator1.H
        block7 = Iᵦ2 * operator2.H' * operator2.Wꜝ * operator2.G
        block8 = Iᵦ2 * operator2.H' * operator2.Wꜝ * operator2.H
    elseif scheme == "BE"
        block1 = operator1.V + Δt * (sum(C1) + 0.5 * sum(K1)) + Δt * Id1 * operator1.G' * operator1.Wꜝ * operator1.G
        block2 = Δt * 0.5 * sum(K1) + Δt * Id1 * operator1.G' * operator1.Wꜝ * operator1.H
        block3 = operator2.V + Δt * (sum(C2) + 0.5 * sum(K2)) + Δt * Id2 * operator2.G' * operator2.Wꜝ * operator2.G
        block4 = Δt * 0.5 * sum(K2) + Δt * Id2 * operator2.G' * operator2.Wꜝ * operator2.H
        block5 = Iᵦ1 * operator1.H' * operator1.Wꜝ * operator1.G
        block6 = Iᵦ1 * operator1.H' * operator1.Wꜝ * operator1.H
        block7 = Iᵦ2 * operator2.H' * operator2.Wꜝ * operator2.G
        block8 = Iᵦ2 * operator2.H' * operator2.Wꜝ * operator2.H
    else
        error("Unknown scheme.")
    end

    A = vcat(hcat(block1, block2, zeros(n, n), zeros(n, n)),
             hcat(zeros(n, n), Iₐ1, zeros(n, n), -Iₐ2),
             hcat(zeros(n, n), zeros(n, n), block3, block4),
             hcat(block5, block6, block7, block8))

    return A
end

function b_diph_unstead_advdiff(operator1::ConvectionOps, operator2::ConvectionOps, f1, f2, capacite1::Capacity, capacite2::Capacity, D1, D2, ic::InterfaceConditions, Tᵢ, Δt::Float64, t::Float64, scheme::String)
    N = prod(operator1.size)
    b = zeros(4N)

    jump, flux = ic.scalar, ic.flux
    Iᵧ1, Iᵧ2 = capacite1.Γ, capacite2.Γ 
    gᵧ, hᵧ = build_g_g(operator1, jump,capacite1), build_g_g(operator2, flux, capacite2)
    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    fₒn1, fₒn2 = build_source(operator1, f1, t, capacite1), build_source(operator2, f2, t, capacite2)
    fₒn1p1, fₒn2p1 = build_source(operator1, f1, t+Δt, capacite1), build_source(operator2, f2, t+Δt, capacite2)

    Tₒ1, Tᵧ1 = Tᵢ[1:N], Tᵢ[N+1:2N]
    Tₒ2, Tᵧ2 = Tᵢ[2N+1:3N], Tᵢ[3N+1:end]

    if scheme == "CN"
        b1 = (operator1.V - Δt/2 * sum(operator1.C) - Δt/2 * 0.5 * sum(operator1.K))*Tₒ1 - Δt/2 * 0.5 * sum(operator1.K) * Tᵧ1 + Δt/2 * operator1.V * (fₒn1 + fₒn1p1)
        b2 = gᵧ
        b3 = (operator2.V - Δt/2 * sum(operator2.C) - Δt/2 * 0.5 * sum(operator2.K))*Tₒ2 - Δt/2 * 0.5 * sum(operator2.K) * Tᵧ2 + Δt/2 * operator2.V * (fₒn2 + fₒn2p1)
        b4 = Iᵧ2*hᵧ
    elseif scheme == "BE"
        b1 = operator1.V * Tₒ1 + Δt * operator1.V * fₒn1p1
        b2 = gᵧ
        b3 = operator2.V * Tₒ2 + Δt * operator2.V * fₒn2p1
        b4 = Iᵧ2 * hᵧ
    else
        error("Unknown scheme.")
    end

    b = vcat(b1, b2, b3, b4)
    return b
end

function solve_AdvectionDiffusionUnsteadyDiph!(s::Solver, phase1::Phase, phase2::Phase, Δt::Float64, Tₑ, bc_b::BorderConditions, ic::InterfaceConditions, scheme::String; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    # Solve for the initial time step
    t = 0.0
    println("Time: ", t)
    solve_system!(s; method, algorithm=algorithm, kwargs...)

    push!(s.states, s.x)
    Tᵢ = s.x
    println("Solver Extremum: ", maximum(abs.(s.x)))

    # Loop over the time steps
    while t < Tₑ
        t += Δt
        println("Time: ", t)

        s.A = A_diph_unstead_advdiff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, Δt, scheme)
        s.b = b_diph_unstead_advdiff(phase1.operator, phase2.operator, phase1.source, phase2.source, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, Tᵢ, Δt, t, scheme)
        BC_border_diph!(s.A, s.b, bc_b, phase1.capacity, phase2.capacity)
        
        solve_system!(s; method, algorithm=algorithm, kwargs...)

        push!(s.states, s.x)
        Tᵢ = s.x
        println("Solver Extremum: ", maximum(abs.(s.x)))
    end
end
