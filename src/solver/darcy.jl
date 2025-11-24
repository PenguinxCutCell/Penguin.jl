function DarcyFlow(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary)
    println("Solver Creation:")
    println("- Darcy Flow")
    println("- Steady problem")
    println("- Monophasic")

    s = Solver(Steady, Monophasic, Diffusion, nothing, nothing, nothing, [], [])

    s.A = A_mono_stead_diff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i)
    s.b = b_mono_stead_diff(phase.operator, phase.source, phase.capacity, bc_i)

    BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh)

    return s
end

function solve_DarcyFlow!(s::Solver; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    solve_system!(s; method, algorithm=algorithm, kwargs...)
    push!(s.states, s.x)
end

function solve_darcy_velocity(solver, Fluide; state_i=1)
    cell_types = Fluide.capacity.cell_types
    pₒ = solver.states[state_i][1:div(end,2)]
    pᵧ = solver.states[state_i][div(end,2)+1:end]

    pₒ[cell_types .== 0] .= NaN
    pᵧ[cell_types .== 0] .= NaN
    pᵧ[cell_types .== 1] .= NaN

    p = vcat(pₒ, pᵧ)

    # Compute the velocity field
    u = - ∇(Fluide.operator, p)
    return u
end



# Unsteady Darcy Flow
function DarcyFlowUnsteady(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, scheme::String)
    println("Création du solveur:")
    println("- Darcy Flow")
    println("- Unsteady problem")
    println("- Monophasic")

    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, [], [])

    s.A = A_mono_unstead_diff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, Δt, scheme)
    s.b = b_mono_unstead_diff(phase.operator, phase.source, phase.Diffusion_coeff, phase.capacity, bc_i, Tᵢ, Δt, 0.0, scheme)

    BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh; t=0.0)
    return s
end

function solve_DarcyFlowUnsteady!(s::Solver, phase::Phase, Δt::Float64, Tₑ::Float64, bc_b::BorderConditions, bc_i::AbstractBoundary, scheme::String; method::Function = gmres, algorithm=nothing, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    # Solve for the initial condition
    t=0.0
    println("Time: ", t)
    solve_system!(s; method, algorithm=algorithm, kwargs...)

    push!(s.states, s.x)
    println("Solver Extremum: ", maximum(abs.(s.x)))
    Tᵢ = s.x

    # Solve for the rest of the time
    while t<Tₑ
        t+=Δt
        println("Time: ", t)

        s.A = A_mono_unstead_diff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, Δt, scheme)
        s.b = b_mono_unstead_diff(phase.operator, phase.source, phase.Diffusion_coeff, phase.capacity, bc_i, Tᵢ, Δt, t, scheme)

        BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh; t=t)

        solve_system!(s; method, algorithm=algorithm, kwargs...)

        push!(s.states, s.x)
        println("Solver Extremum: ", maximum(abs.(s.x)))
        Tᵢ = s.x
    end
end