"""
    DiffusionSteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary)

Create a solver for a steady-state monophasic diffusion problem.

Arguments:
- `phase` : Phase object representing the phase of the problem.
- `bc_b` : BorderConditions object representing the boundary conditions of the problem at the boundary.
- `bc_i` : AbstractBoundary object representing the internal boundary conditions of the problem.

Returns:
- `s` : Solver object representing the created solver.
"""
function DiffusionSteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary)
    println("Solver creation:")
    println("- Monophasic problem")
    println("- Steady problem")
    println("- Diffusion problem")
    
    s = Solver(Steady, Monophasic, Diffusion, nothing, nothing, nothing, [], [])
    
    s.A = A_mono_stead_diff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i)
    s.b = b_mono_stead_diff(phase.operator, phase.source, phase.capacity, bc_i)

    BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh)

    return s
end

"""
    DiffusionSteadyMonoVariable(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary; mean::Symbol = :harmonic)

Create a solver for a steady-state monophasic diffusion problem with variable diffusion coefficient.
"""
function DiffusionSteadyMonoVariable(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary; mean::Symbol = :harmonic)
    println("Solver creation:")
    println("- Monophasic problem")
    println("- Steady problem")
    println("- Diffusion problem")
    println("- Variable diffusion coefficient")

    s = Solver(Steady, Monophasic, Diffusion, nothing, nothing, nothing, [], [])

    s.A = A_mono_stead_diff_variable(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i; mean=mean)
    s.b = b_mono_stead_diff(phase.operator, phase.source, phase.capacity, bc_i)

    BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh)

    return s
end

function A_mono_stead_diff(operator::DiffusionOps, capacity::Capacity, D, bc::AbstractBoundary)
    n = prod(operator.size)
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Iᵧ =  capacity.Γ
    Id = build_I_D(operator, D, capacity)

    A1 = Id * operator.G' * operator.Wꜝ * operator.G
    A2 = Id * operator.G' * operator.Wꜝ * operator.H
    A3 = Iᵦ * operator.H' * operator.Wꜝ * operator.G
    A4 = Iᵦ * operator.H' * operator.Wꜝ * operator.H + Iₐ * Iᵧ

    A = vcat(hcat(A1, A2), hcat(A3, A4))
    return A
end

function A_mono_stead_diff_variable(operator::DiffusionOps, capacity::Capacity, D, bc::AbstractBoundary; mean::Symbol = :harmonic)
    n = prod(operator.size)
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Iᵧ = capacity.Γ
    Id = if mean == :harmonic
        build_I_D_harmonic(operator, D, capacity)
    elseif mean == :arithmetic
        build_I_D_arithmetic(operator, D, capacity)
    else
        error("Unknown mean $(mean). Use :harmonic or :arithmetic.")
    end

    A1 = operator.G' * Id * operator.Wꜝ * operator.G
    A2 = operator.G' * Id * operator.Wꜝ * operator.H
    A3 = Iᵦ * operator.H' * Id * operator.Wꜝ * operator.G
    A4 = Iᵦ * operator.H' * Id * operator.Wꜝ * operator.H + Iₐ * Iᵧ

    A = vcat(hcat(A1, A2), hcat(A3, A4))
    return A
end

function b_mono_stead_diff(operator::DiffusionOps, f::Function, capacite::Capacity, bc::AbstractBoundary)
    N = prod(operator.size)
    b = zeros(2N)

    Iᵧ = capacite.Γ 
    fₒ = build_source(operator, f, capacite)
    gᵧ = build_g_g(operator, bc, capacite)

    # Build the right-hand side
    b1 = operator.V * fₒ
    b2 = Iᵧ * gᵧ
    b = vcat(b1, b2)
    return b
end

function solve_DiffusionSteadyMono!(s::Solver; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the system:")
    println("- Monophasic problem")
    println("- Steady problem")
    println("- Diffusion problem")

    # Solve the system with the ability to use LinearSolve.jl
    solve_system!(s; method=method, algorithm=algorithm, kwargs...)
end



# Diffusion - Steady - Diphasic
"""
    DiffusionSteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions)

Creates a solver for a steady-state two-phase diffusion problem.

# Arguments
- `phase1::Phase`: The first phase of the problem.
- `phase2::Phase`: The second phase of the problem.
- `bc_b::BorderConditions`: The boundary conditions of the problem.
- `ic::InterfaceConditions`: The conditions at the interface between the two phases.
"""
function DiffusionSteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions)
    println("Solver creation:")
    println("- Diphasic problem")
    println("- Steady problem")
    println("- Diffusion problem")
    
    s = Solver(Steady, Diphasic, Diffusion, nothing, nothing, nothing, [], [])
    
    s.A = A_diph_stead_diff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, bc_b, ic)
    s.b = b_diph_stead_diff(phase1.operator, phase2.operator, phase1.source, phase2.source, phase1.capacity, phase2.capacity, bc_b, ic)

    BC_border_diph!(s.A, s.b, bc_b, phase1.capacity, phase2.capacity)

    return s
end

"""
    DiffusionSteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, bc_i::RobinJump, bc_f::FluxJump)

Creates a solver for a steady-state two-phase diffusion problem with a Robin jump
condition at the interface.
"""
function DiffusionSteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, bc_i::RobinJump, bc_f::FluxJump)
    println("Solver creation:")
    println("- Diphasic problem")
    println("- Steady problem")
    println("- Diffusion problem")
    println("- Robin jump at interface")

    s = Solver(Steady, Diphasic, Diffusion, nothing, nothing, nothing, [], [])

    s.A = A_diph_stead_diff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity,
        phase1.Diffusion_coeff, phase2.Diffusion_coeff, bc_b, bc_i, bc_f)
    s.b = b_diph_stead_diff(phase1.operator, phase2.operator, phase1.source, phase2.source,
        phase1.capacity, phase2.capacity, bc_b, bc_i, bc_f)

    BC_border_diph!(s.A, s.b, bc_b, phase1.capacity, phase2.capacity)

    return s
end

function A_diph_stead_diff(operator1::DiffusionOps, operator2::DiffusionOps, capacite1::Capacity, capacite2::Capacity, D1, D2, bc_b::BorderConditions, ic::InterfaceConditions)
    n = prod(operator1.size)

    jump, flux = ic.scalar, ic.flux
    Iₐ1, Iₐ2 = jump.α₁ * I(n), jump.α₂ * I(n)
    Iᵦ1, Iᵦ2 = flux.β₁ * I(n), flux.β₂ * I(n)
    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    block1 = Id1 * operator1.G' * operator1.Wꜝ * operator1.G
    block2 = Id1 * operator1.G' * operator1.Wꜝ * operator1.H
    block3 = Id2 * operator2.G' * operator2.Wꜝ * operator2.G
    block4 = Id2 * operator2.G' * operator2.Wꜝ * operator2.H
    block5 = operator1.H' * operator1.Wꜝ * operator1.G
    block6 = operator1.H' * operator1.Wꜝ * operator1.H 
    block7 = operator2.H' * operator2.Wꜝ * operator2.G
    block8 = operator2.H' * operator2.Wꜝ * operator2.H

    A = spzeros(Float64, 4n, 4n)

    @inbounds begin
        # Top-left blocks
        A[1:n, 1:n] += block1
        A[1:n, n+1:2n] += block2

        # Middle blocks
        A[n+1:2n, n+1:2n] += Iₐ1
        A[n+1:2n, 3n+1:4n] -= Iₐ2

        # Bottom-left blocks
        A[2n+1:3n, 2n+1:3n] += block3
        A[2n+1:3n, 3n+1:4n] += block4

        # Bottom blocks with Iᵦ
        A[3n+1:4n, 1:n] += Iᵦ1 * block5
        A[3n+1:4n, n+1:2n] += Iᵦ1 * block6
        A[3n+1:4n, 2n+1:3n] += Iᵦ2 * block7
        A[3n+1:4n, 3n+1:4n] += Iᵦ2 * block8
    end

    return A
end

function A_diph_stead_diff(operator1::DiffusionOps, operator2::DiffusionOps, capacite1::Capacity, capacite2::Capacity, D1, D2, bc_b::BorderConditions, bc_i::RobinJump, bc_f::FluxJump)
    n = prod(operator1.size)

    Iₐ = bc_i.α * I(n)
    Iᵦ = bc_i.β * I(n)
    Iᵦ1, Iᵦ2 = bc_f.β₁ * I(n), bc_f.β₂ * I(n)
    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    block1 = Id1 * operator1.G' * operator1.Wꜝ * operator1.G
    block2 = Id1 * operator1.G' * operator1.Wꜝ * operator1.H
    block3 = Id2 * operator2.G' * operator2.Wꜝ * operator2.G
    block4 = Id2 * operator2.G' * operator2.Wꜝ * operator2.H
    block5 = operator1.H' * operator1.Wꜝ * operator1.G
    block6 = operator1.H' * operator1.Wꜝ * operator1.H
    block7 = operator2.H' * operator2.Wꜝ * operator2.G
    block8 = operator2.H' * operator2.Wꜝ * operator2.H

    A = spzeros(Float64, 4n, 4n)

    @inbounds begin
        # Top-left blocks
        A[1:n, 1:n] += block1
        A[1:n, n+1:2n] += block2

        # Middle blocks (Robin jump)
        A[n+1:2n, 1:n] += Iᵦ * block5
        A[n+1:2n, n+1:2n] += Iₐ + Iᵦ * block6
        A[n+1:2n, 3n+1:4n] -= Iₐ

        # Bottom-left blocks
        A[2n+1:3n, 2n+1:3n] += block3
        A[2n+1:3n, 3n+1:4n] += block4

        # Bottom blocks with Iᵦ
        A[3n+1:4n, 1:n] += Iᵦ1 * block5
        A[3n+1:4n, n+1:2n] += Iᵦ1 * block6
        A[3n+1:4n, 2n+1:3n] += Iᵦ2 * block7
        A[3n+1:4n, 3n+1:4n] += Iᵦ2 * block8
    end

    return A
end

function b_diph_stead_diff(operator1::DiffusionOps, operator2::DiffusionOps, f1, f2, capacite1::Capacity, capacite2::Capacity, bc_b::BorderConditions, ic::InterfaceConditions)
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

function b_diph_stead_diff(operator1::DiffusionOps, operator2::DiffusionOps, f1, f2, capacite1::Capacity, capacite2::Capacity, bc_b::BorderConditions, bc_i::RobinJump, bc_f::FluxJump)
    N = prod(operator1.size)
    b = zeros(4N)

    Iᵧ1, Iᵧ2 = capacite1.Γ, capacite2.Γ
    gᵧ, hᵧ = build_g_g(operator1, bc_i, capacite1), build_g_g(operator2, bc_f, capacite2)

    fₒ1 = build_source(operator1, f1, capacite1)
    fₒ2 = build_source(operator2, f2, capacite2)

    # Build the right-hand side
    b = vcat(operator1.V*fₒ1, gᵧ, operator2.V*fₒ2, Iᵧ2*hᵧ)

    return b
end

function solve_DiffusionSteadyDiph!(s::Solver; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the system:")
    println("- Diphasic problem")
    println("- Steady problem")
    println("- Diffusion problem")

    # Solve the system
    solve_system!(s; method, algorithm=algorithm, kwargs...)
end



# Diffusion - Unsteady - Monophasic
"""
    DiffusionUnsteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tₑ::Float64, Tᵢ::Vector{Float64})

Constructs a solver for the unsteady monophasic diffusion problem.

# Arguments
- `phase::Phase`: The phase object representing the physical properties of the system.
- `bc_b::BorderConditions`: The border conditions object representing the boundary conditions at the outer border.
- `bc_i::AbstractBoundary`: The boundary conditions object representing the boundary conditions at the inner border.
- `Δt::Float64`: The time step size.
- `Tᵢ::Vector{Float64}`: The initial temperature distribution.
"""
function DiffusionUnsteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, scheme::String)
    println("Solver creation:")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, [], [])

    if scheme == "CN"
        s.A = A_mono_unstead_diff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, Δt, "CN")
        s.b = b_mono_unstead_diff(phase.operator, phase.source, phase.Diffusion_coeff, phase.capacity, bc_i, Tᵢ, Δt, 0.0, "CN")
    else
        s.A = A_mono_unstead_diff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, Δt, "BE")
        s.b = b_mono_unstead_diff(phase.operator, phase.source, phase.Diffusion_coeff, phase.capacity, bc_i, Tᵢ, Δt, 0.0, "BE")
    end
    BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh; t=0.0)

    return s
end

function A_mono_unstead_diff(operator::DiffusionOps, capacite::Capacity, D, bc::AbstractBoundary, Δt::Float64, scheme::String)
    n = prod(operator.size)
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Iᵧ = capacite.Γ # build_I_g(operator, bc)

    Id = build_I_D(operator, D, capacite)

    # Preallocate the sparse matrix A with 2n rows and 2n columns
    A = spzeros(Float64, 2n, 2n)

    # Compute blocks
    if scheme=="CN"
        block1 = operator.V + Δt / 2 * (Id * operator.G' * operator.Wꜝ * operator.G)
        block2 = Δt / 2 * (Id * operator.G' * operator.Wꜝ * operator.H)
        block3 = Δt/2 * Iᵦ * operator.H' * operator.Wꜝ * operator.G
        block4 = Δt/2 * Iᵦ * operator.H' * operator.Wꜝ * operator.H + Δt/2 * (Iₐ * Iᵧ)
    else
        block1 = operator.V + Δt * (Id * operator.G' * operator.Wꜝ * operator.G)
        block2 = Δt * (Id * operator.G' * operator.Wꜝ * operator.H)
        block3 = Iᵦ * operator.H' * operator.Wꜝ * operator.G
        block4 = Iᵦ * operator.H' * operator.Wꜝ * operator.H + (Iₐ * Iᵧ)
    end
    
    A[1:n, 1:n] = block1
    A[1:n, n+1:2n] = block2
    A[n+1:2n, 1:n] = block3
    A[n+1:2n, n+1:2n] = block4
    
    return A
end

function b_mono_unstead_diff(operator::DiffusionOps, f, D, capacite::Capacity, bc::AbstractBoundary, Tᵢ, Δt::Float64, t::Float64, scheme::String)
    N = prod(operator.size)
    b = zeros(2N)

    Iᵧ = capacite.Γ # build_I_g(operator, bc)
    fₒn, fₒn1 = build_source(operator, f, t, capacite), build_source(operator, f, t+Δt, capacite)
    gᵧn, gᵧn1 = build_g_g(operator, bc, capacite, t), build_g_g(operator, bc, capacite, t+Δt)
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Id = build_I_D(operator, D, capacite)

    Tₒ, Tᵧ = Tᵢ[1:N], Tᵢ[N+1:end]

    # Build the right-hand side
    if scheme=="CN"
        b1 = (operator.V - Δt/2 * Id * operator.G' * operator.Wꜝ * operator.G)*Tₒ - Δt/2 * Id * operator.G' * operator.Wꜝ * operator.H * Tᵧ + Δt/2 * operator.V * (fₒn + fₒn1)
        b2 = Δt/2 * Iᵧ * (gᵧn+gᵧn1) - Δt/2 * Iᵦ * operator.H' * operator.Wꜝ * operator.G * Tₒ - Δt/2 * Iᵦ * operator.H' * operator.Wꜝ * operator.H * Tᵧ - Δt/2 * Iₐ * Iᵧ * Tᵧ
    else
        b1 = (operator.V)*Tₒ + Δt * operator.V * (fₒn1)
        b2 = Iᵧ * gᵧn1
    end
    b = vcat(b1, b2)
   return b
end


function solve_DiffusionUnsteadyMono!(s::Solver, phase::Phase, Δt::Float64, Tₑ, bc_b::BorderConditions, bc::AbstractBoundary, scheme::String; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    # Guard against floating point drift when stepping to the final time
    tol = eps(Float64) * max(1.0, abs(Tₑ))
    current_dt = Δt

    # Solve the system for the initial time with the initial scheme
    t = 0.0
    solve_system!(s; method, algorithm=algorithm, kwargs...)

    push!(s.states, s.x)
    println("Time: ", t)
    println("Solver Extremum: ", maximum(abs.(s.x)))
    Tᵢ = s.x

    # Build once matrix for the new scheme
    s.A = A_mono_unstead_diff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, Δt, scheme)

    # Solve the system for the next times
    while t + tol < Tₑ
        step_dt = min(Δt, Tₑ - t)

        if step_dt != current_dt
            s.A = A_mono_unstead_diff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc, step_dt, scheme)
            current_dt = step_dt
        end

        t += step_dt
        println("Time: ", t)

        s.b = b_mono_unstead_diff(phase.operator, phase.source, phase.Diffusion_coeff, phase.capacity, bc, Tᵢ, step_dt, t, scheme)

        BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh; t=t)
        
        solve_system!(s; method, algorithm=algorithm, kwargs...)

        push!(s.states, s.x)
        println("Solver Extremum: ", maximum(abs.(s.x)))

        Tᵢ = s.x
    end
end



# Diffusion - Unsteady - Diphasic
"""
    DiffusionUnsteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions, Δt::Float64, Tₑ::Float64, Tᵢ::Vector{Float64})

Creates a solver for an unsteady two-phase diffusion problem.

## Arguments
- `phase1::Phase`: The first phase of the problem.
- `phase2::Phase`: The second phase of the problem.
- `bc_b::BorderConditions`: The boundary conditions of the problem.
- `ic::InterfaceConditions`: The conditions at the interface between the two phases.
- `Δt::Float64`: The time interval.
- `Tᵢ::Vector{Float64}`: The vector of initial temperatures.
"""
function DiffusionUnsteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions, Δt::Float64, Tᵢ::Vector{Float64}, scheme::String)
    println("Solver creation:")
    println("- Diphasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Diphasic, Diffusion, nothing, nothing, nothing, [], [])

    s.A = A_diph_unstead_diff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, Δt, scheme)
    s.b = b_diph_unstead_diff(phase1.operator, phase2.operator, phase1.source, phase2.source, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, Tᵢ, Δt, 0.0, scheme)

    BC_border_diph!(s.A, s.b, bc_b, phase1.capacity, phase2.capacity)
    return s
end

"""
    DiffusionUnsteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, bc_i::RobinJump, bc_f::FluxJump, Δt::Float64, Tᵢ::Vector{Float64}, scheme::String)

Creates a solver for an unsteady two-phase diffusion problem with a Robin jump
condition at the interface.
"""
function DiffusionUnsteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, bc_i::RobinJump, bc_f::FluxJump, Δt::Float64, Tᵢ::Vector{Float64}, scheme::String)
    println("Solver creation:")
    println("- Diphasic problem")
    println("- Unsteady problem")
    println("- Diffusion problem")
    println("- Robin jump at interface")

    s = Solver(Unsteady, Diphasic, Diffusion, nothing, nothing, nothing, [], [])

    s.A = A_diph_unstead_diff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity,
        phase1.Diffusion_coeff, phase2.Diffusion_coeff, bc_i, bc_f, Δt, scheme)
    s.b = b_diph_unstead_diff(phase1.operator, phase2.operator, phase1.source, phase2.source,
        phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff,
        bc_i, bc_f, Tᵢ, Δt, 0.0, scheme)

    BC_border_diph!(s.A, s.b, bc_b, phase1.capacity, phase2.capacity)
    return s
end

function A_diph_unstead_diff(operator1::DiffusionOps, operator2::DiffusionOps, capacite1::Capacity, capacite2::Capacity, D1, D2, ic::InterfaceConditions, Δt::Float64, scheme::String)
    n = prod(operator1.size)

    jump, flux = ic.scalar, ic.flux
    Iₐ1, Iₐ2 = jump.α₁ * I(n), jump.α₂ * I(n)
    Iᵦ1, Iᵦ2 = flux.β₁ * I(n), flux.β₂ * I(n)
    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    # Precompute repeated multiplications
    WG_G1 = operator1.Wꜝ * operator1.G
    WG_H1 = operator1.Wꜝ * operator1.H
    WG_G2 = operator2.Wꜝ * operator2.G
    WG_H2 = operator2.Wꜝ * operator2.H

    if scheme == "CN"
        block1 = operator1.V + Δt / 2 * Id1 * operator1.G' * WG_G1
        block2 = Δt / 2 * Id1 * operator1.G' * WG_H1
        block3 = operator2.V + Δt / 2 * Id2 * operator2.G' * WG_G2
        block4 = Δt / 2 * Id2 * operator2.G' * WG_H2
    else
        block1 = operator1.V + Δt * Id1 * operator1.G' * WG_G1
        block2 = Δt * Id1 * operator1.G' * WG_H1
        block3 = operator2.V + Δt * Id2 * operator2.G' * WG_G2
        block4 = Δt * Id2 * operator2.G' * WG_H2
    end
    block5 = Iᵦ1 * operator1.H' * WG_G1
    block6 = Iᵦ1 * operator1.H' * WG_H1
    block7 = Iᵦ2 * operator2.H' * WG_G2
    block8 = Iᵦ2 * operator2.H' * WG_H2

    # Preallocate the sparse matrix
    A = spzeros(Float64, 4n, 4n)

    # Assign blocks to the matrix
    A[1:n, 1:n] = block1
    A[1:n, n+1:2n] = block2
    A[1:n, 2n+1:3n] = spzeros(n, n)
    A[1:n, 3n+1:4n] = spzeros(n, n)

    A[n+1:2n, 1:n] = spzeros(n, n)
    A[n+1:2n, n+1:2n] = Iₐ1
    A[n+1:2n, 2n+1:3n] = spzeros(n, n)
    A[n+1:2n, 3n+1:4n] = -Iₐ2

    A[2n+1:3n, 1:n] = spzeros(n, n)
    A[2n+1:3n, n+1:2n] = spzeros(n, n)
    A[2n+1:3n, 2n+1:3n] = block3
    A[2n+1:3n, 3n+1:4n] = block4

    A[3n+1:4n, 1:n] = block5
    A[3n+1:4n, n+1:2n] = block6
    A[3n+1:4n, 2n+1:3n] = block7
    A[3n+1:4n, 3n+1:4n] = block8

    return A
end

function A_diph_unstead_diff(operator1::DiffusionOps, operator2::DiffusionOps, capacite1::Capacity, capacite2::Capacity, D1, D2, bc_i::RobinJump, bc_f::FluxJump, Δt::Float64, scheme::String)
    n = prod(operator1.size)

    Iₐ = bc_i.α * I(n)
    Iᵦ = bc_i.β * I(n)
    Iᵦ1, Iᵦ2 = bc_f.β₁ * I(n), bc_f.β₂ * I(n)
    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    # Precompute repeated multiplications
    WG_G1 = operator1.Wꜝ * operator1.G
    WG_H1 = operator1.Wꜝ * operator1.H
    WG_G2 = operator2.Wꜝ * operator2.G
    WG_H2 = operator2.Wꜝ * operator2.H

    if scheme == "CN"
        block1 = operator1.V + Δt / 2 * Id1 * operator1.G' * WG_G1
        block2 = Δt / 2 * Id1 * operator1.G' * WG_H1
        block3 = operator2.V + Δt / 2 * Id2 * operator2.G' * WG_G2
        block4 = Δt / 2 * Id2 * operator2.G' * WG_H2
    else
        block1 = operator1.V + Δt * Id1 * operator1.G' * WG_G1
        block2 = Δt * Id1 * operator1.G' * WG_H1
        block3 = operator2.V + Δt * Id2 * operator2.G' * WG_G2
        block4 = Δt * Id2 * operator2.G' * WG_H2
    end
    block5 = operator1.H' * WG_G1
    block6 = operator1.H' * WG_H1
    block7 = Iᵦ2 * operator2.H' * WG_G2
    block8 = Iᵦ2 * operator2.H' * WG_H2

    # Preallocate the sparse matrix
    A = spzeros(Float64, 4n, 4n)

    # Assign blocks to the matrix
    A[1:n, 1:n] = block1
    A[1:n, n+1:2n] = block2
    A[1:n, 2n+1:3n] = spzeros(n, n)
    A[1:n, 3n+1:4n] = spzeros(n, n)

    A[n+1:2n, 1:n] = Iᵦ * block5
    A[n+1:2n, n+1:2n] = Iₐ + Iᵦ * block6
    A[n+1:2n, 2n+1:3n] = spzeros(n, n)
    A[n+1:2n, 3n+1:4n] = -Iₐ

    A[2n+1:3n, 1:n] = spzeros(n, n)
    A[2n+1:3n, n+1:2n] = spzeros(n, n)
    A[2n+1:3n, 2n+1:3n] = block3
    A[2n+1:3n, 3n+1:4n] = block4

    A[3n+1:4n, 1:n] = Iᵦ1 * block5
    A[3n+1:4n, n+1:2n] = Iᵦ1 * block6
    A[3n+1:4n, 2n+1:3n] = block7
    A[3n+1:4n, 3n+1:4n] = block8

    return A
end

function b_diph_unstead_diff(operator1::DiffusionOps, operator2::DiffusionOps, f1, f2, capacite1::Capacity, capacite2::Capacity, D1, D2, ic::InterfaceConditions, Tᵢ, Δt::Float64, t::Float64, scheme::String)
    N = prod(operator1.size)
    b = zeros(4N)

    jump, flux = ic.scalar, ic.flux
    Iᵧ1, Iᵧ2 = capacite1.Γ, capacite2.Γ
    gᵧ, hᵧ = build_g_g(operator1, jump,capacite1), build_g_g(operator2, flux, capacite2)

    fₒn1, fₒn2 = build_source(operator1, f1, t, capacite1), build_source(operator2, f2, t, capacite2)
    fₒn1p1, fₒn2p1 = build_source(operator1, f1, t+Δt, capacite1), build_source(operator2, f2, t+Δt, capacite2)

    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    Tₒ1, Tᵧ1 = Tᵢ[1:N], Tᵢ[N+1:2N]
    Tₒ2, Tᵧ2 = Tᵢ[2N+1:3N], Tᵢ[3N+1:end]

    # Build the right-hand side
    if scheme == "CN"
        b1 = (operator1.V - Δt/2 * Id1 * operator1.G' * operator1.Wꜝ * operator1.G)*Tₒ1 - Δt/2 * Id1 * operator1.G' * operator1.Wꜝ * operator1.H * Tᵧ1 + Δt/2 * operator1.V * (fₒn1 + fₒn1p1)
        b3 = (operator2.V - Δt/2 * Id2 * operator2.G' * operator2.Wꜝ * operator2.G)*Tₒ2 - Δt/2 * Id2 * operator2.G' * operator2.Wꜝ * operator2.H * Tᵧ2 + Δt/2 * operator2.V * (fₒn2 + fₒn2p1)
    else
        b1 = (operator1.V)*Tₒ1 + Δt * operator1.V * (fₒn1p1)
        b3 = (operator2.V)*Tₒ2 + Δt * operator2.V * (fₒn2p1)
    end
    b2 = gᵧ
    b4 = Iᵧ2*hᵧ
    b = vcat(b1, b2, b3, b4)

    return b
end

function b_diph_unstead_diff(operator1::DiffusionOps, operator2::DiffusionOps, f1, f2, capacite1::Capacity, capacite2::Capacity, D1, D2, bc_i::RobinJump, bc_f::FluxJump, Tᵢ, Δt::Float64, t::Float64, scheme::String)
    N = prod(operator1.size)
    b = zeros(4N)

    Iᵧ1, Iᵧ2 = capacite1.Γ, capacite2.Γ
    gᵧ, hᵧ = build_g_g(operator1, bc_i, capacite1), build_g_g(operator2, bc_f, capacite2)

    fₒn1, fₒn2 = build_source(operator1, f1, t, capacite1), build_source(operator2, f2, t, capacite2)
    fₒn1p1, fₒn2p1 = build_source(operator1, f1, t+Δt, capacite1), build_source(operator2, f2, t+Δt, capacite2)

    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    Tₒ1, Tᵧ1 = Tᵢ[1:N], Tᵢ[N+1:2N]
    Tₒ2, Tᵧ2 = Tᵢ[2N+1:3N], Tᵢ[3N+1:end]

    # Build the right-hand side
    if scheme == "CN"
        b1 = (operator1.V - Δt/2 * Id1 * operator1.G' * operator1.Wꜝ * operator1.G)*Tₒ1 - Δt/2 * Id1 * operator1.G' * operator1.Wꜝ * operator1.H * Tᵧ1 + Δt/2 * operator1.V * (fₒn1 + fₒn1p1)
        b3 = (operator2.V - Δt/2 * Id2 * operator2.G' * operator2.Wꜝ * operator2.G)*Tₒ2 - Δt/2 * Id2 * operator2.G' * operator2.Wꜝ * operator2.H * Tᵧ2 + Δt/2 * operator2.V * (fₒn2 + fₒn2p1)
    else
        b1 = (operator1.V)*Tₒ1 + Δt * operator1.V * (fₒn1p1)
        b3 = (operator2.V)*Tₒ2 + Δt * operator2.V * (fₒn2p1)
    end
    b2 = gᵧ
    b4 = Iᵧ2*hᵧ
    b = vcat(b1, b2, b3, b4)

    return b
end

function solve_DiffusionUnsteadyDiph!(s::Solver, phase1::Phase, phase2::Phase, Δt::Float64, Tₑ::Float64, bc_b::BorderConditions, ic::InterfaceConditions, scheme::String; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    # Guard against floating point drift when stepping to the final time
    tol = eps(Float64) * max(1.0, abs(Tₑ))
    current_dt = Δt

    t = 0.0
    println("Time: ", t)
    # Solve for the initial condition with the initial scheme
    solve_system!(s; method, algorithm=algorithm, kwargs...)

    push!(s.states, s.x)
    println("Solver Extremum: ", maximum(abs.(s.x)))
    Tᵢ = s.x

    # Build once matrix for the new scheme
    s.A = A_diph_unstead_diff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, Δt, scheme)

    # Solve for the next times
    while t + tol < Tₑ
        step_dt = min(Δt, Tₑ - t)

        if step_dt != current_dt
            s.A = A_diph_unstead_diff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, step_dt, scheme)
            current_dt = step_dt
        end

        t += step_dt
        println("Time: ", t)

        s.b = b_diph_unstead_diff(phase1.operator, phase2.operator, phase1.source, phase2.source, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic, Tᵢ, step_dt, t, scheme)

        BC_border_diph!(s.A, s.b, bc_b, phase1.capacity, phase2.capacity)
        
        solve_system!(s; method, algorithm=algorithm, kwargs...)

        push!(s.states, s.x)
        println("Solver Extremum: ", maximum(abs.(s.x)))
        Tᵢ = s.x
    end
end

function solve_DiffusionUnsteadyDiph!(s::Solver, phase1::Phase, phase2::Phase, Δt::Float64, Tₑ::Float64, bc_b::BorderConditions, bc_i::RobinJump, bc_f::FluxJump, scheme::String; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    # Guard against floating point drift when stepping to the final time
    tol = eps(Float64) * max(1.0, abs(Tₑ))
    current_dt = Δt

    t = 0.0
    println("Time: ", t)
    # Solve for the initial condition with the initial scheme
    solve_system!(s; method, algorithm=algorithm, kwargs...)

    push!(s.states, s.x)
    println("Solver Extremum: ", maximum(abs.(s.x)))
    Tᵢ = s.x

    # Build once matrix for the new scheme
    s.A = A_diph_unstead_diff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity,
        phase1.Diffusion_coeff, phase2.Diffusion_coeff, bc_i, bc_f, Δt, scheme)

    # Solve for the next times
    while t + tol < Tₑ
        step_dt = min(Δt, Tₑ - t)

        if step_dt != current_dt
            s.A = A_diph_unstead_diff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity,
                phase1.Diffusion_coeff, phase2.Diffusion_coeff, bc_i, bc_f, step_dt, scheme)
            current_dt = step_dt
        end

        t += step_dt
        println("Time: ", t)

        s.b = b_diph_unstead_diff(phase1.operator, phase2.operator, phase1.source, phase2.source,
            phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff,
            bc_i, bc_f, Tᵢ, step_dt, t, scheme)

        BC_border_diph!(s.A, s.b, bc_b, phase1.capacity, phase2.capacity)
        
        solve_system!(s; method, algorithm=algorithm, kwargs...)

        push!(s.states, s.x)
        println("Solver Extremum: ", maximum(abs.(s.x)))
        Tᵢ = s.x
    end
end
