"""
    mutable struct StreamVorticity{N}

Solver dedicated to two-dimensional streamfunction–vorticity simulations on
cut-cell meshes. The discretisation follows the scalar layout already used in
`diffusion.jl`/`advectiondiffusion.jl`: the unknown vectors are split between
cell-centred (`⋅^ω`) and interface (`⋅^γ`) degrees of freedom. All operators are
assembled from the existing `DiffusionOps`/`ConvectionOps` definitions so that
embedded boundaries are handled exactly the same way as for scalar transport
equations.

# Fields
- `capacity::Capacity{N}`: Cut-cell capacity description.
- `operator::DiffusionOps{N}`: Differential operators built from the capacity.
- `ν::Union{Float64, Function}`: Kinematic viscosity appearing in the
  vorticity diffusion term.
- `Δt::Float64`: Time step used for the vorticity transport equation.
- `bc_stream::AbstractBoundary`: Interface boundary condition for the
  streamfunction.
- `bc_vorticity::AbstractBoundary`: Interface boundary condition for the
  vorticity.
- `bc_stream_border::BorderConditions`: Outer domain boundary conditions for
  the streamfunction.
- `bc_vorticity_border::BorderConditions`: Outer domain boundary conditions for
  the vorticity.
- `ψ::Vector{Float64}`: Current streamfunction vector `[ψ^ω; ψ^γ]`.
- `ω::Vector{Float64}`: Current vorticity vector `[ω^ω; ω^γ]`.
- `velocity::NTuple{N, Vector{Float64}}`: Velocity components reconstructed
  from the streamfunction on the staggered layout.
- `source::Function`: Optional body-force term in the vorticity equation.
- `time::Float64`: Current simulation time.
- `states::Vector{NamedTuple}`: History of `(time, ψ, ω)` tuples.
- `Aψ::SparseMatrixCSC{Float64, Int}`: Poisson matrix for the streamfunction.
- `last_convection::Union{Nothing, ConvectionOps{N}}`: Cached convection
  operators built from the most recent velocity field.
"""
mutable struct StreamVorticity{N}
    capacity::Capacity{N}
    operator::DiffusionOps{N}
    ν::Union{Float64, Function}
    Δt::Float64
    bc_stream::AbstractBoundary
    bc_vorticity::AbstractBoundary
    bc_stream_border::BorderConditions
    bc_vorticity_border::BorderConditions
    ψ::Vector{Float64}
    ω::Vector{Float64}
    velocity::NTuple{N, Vector{Float64}}
    source::Function
    time::Float64
    states::Vector{NamedTuple}
    Aψ::SparseMatrixCSC{Float64, Int}
    last_convection::Union{Nothing, ConvectionOps{N}}
end

"""
    StreamVorticity(capacity::Capacity{2}, ν, Δt; kwargs...)

Convenience constructor for two-dimensional streamfunction–vorticity problems.
Optional keyword arguments:

- `bc_stream::AbstractBoundary=Dirichlet(0.0)`
- `bc_vorticity::AbstractBoundary=Dirichlet(0.0)`
- `bc_stream_border::BorderConditions=BorderConditions(Dict()))`
- `bc_vorticity_border::BorderConditions=BorderConditions(Dict()))`
- `ψ0::Union{Nothing, Vector{Float64}}=nothing`
- `ω0::Union{Nothing, Vector{Float64}}=nothing`
- `source::Function=(args...)->0.0`

The constructor pre-assembles the Poisson operator for the streamfunction using
the supplied interface boundary condition.
"""
function StreamVorticity(capacity::Capacity{2}, ν, Δt;
                         bc_stream::AbstractBoundary = Dirichlet(0.0),
                         bc_vorticity::AbstractBoundary = Dirichlet(0.0),
                         bc_stream_border::BorderConditions = BorderConditions(Dict{Symbol,AbstractBoundary}()),
                         bc_vorticity_border::BorderConditions = BorderConditions(Dict{Symbol,AbstractBoundary}()),
                         ψ0::Union{Nothing,Vector{Float64}} = nothing,
                         ω0::Union{Nothing,Vector{Float64}} = nothing,
                         source::Function = (args...)->0.0)

    operator = DiffusionOps(capacity)
    n = prod(operator.size)
    ψ_init = isnothing(ψ0) ? zeros(2n) : copy(ψ0)
    ω_init = isnothing(ω0) ? zeros(2n) : copy(ω0)

    velocity = (zeros(n), zeros(n))

    Aψ = assemble_laplacian(operator, capacity, bc_stream, 1.0)

    states = NamedTuple[(time = 0.0, ψ = copy(ψ_init), ω = copy(ω_init))]

    return StreamVorticity{2}(capacity, operator, ν, Δt,
                              bc_stream, bc_vorticity,
                              bc_stream_border, bc_vorticity_border,
                              ψ_init, ω_init, velocity,
                              source, 0.0, states, Aψ, nothing)
end

"""
    assemble_laplacian(operator, capacity, bc, coeff)

Build the cut-cell Laplacian matrix used in the diffusion and Poisson solves.
"""
function assemble_laplacian(operator::DiffusionOps, capacity::Capacity,
                            bc::AbstractBoundary, coeff)
    Id = build_I_D(operator, coeff, capacity)
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Iᵧ = capacity.Γ

    A11 = Id * operator.G' * operator.Wꜝ * operator.G
    A12 = Id * operator.G' * operator.Wꜝ * operator.H
    A21 = Iᵦ * operator.H' * operator.Wꜝ * operator.G
    A22 = Iᵦ * operator.H' * operator.Wꜝ * operator.H + Iₐ * Iᵧ

    return vcat(hcat(A11, A12), hcat(A21, A22))
end

"""
    poisson_rhs(solver, t)

Assemble the right-hand side of the Poisson equation
    ∇²ψ = -ω
using the current vorticity field and interface boundary conditions.
"""
function poisson_rhs(s::StreamVorticity, t::Float64)
    operator = s.operator
    capacity = s.capacity
    n = prod(operator.size)

    ωω = view(s.ω, 1:n)
    rhs_bulk = -operator.V * ωω

    gᵧ = build_g_g(operator, s.bc_stream, capacity, t)
    rhs_γ = capacity.Γ * gᵧ

    return vcat(rhs_bulk, rhs_γ)
end

"""
    update_velocity!(solver)

Reconstruct the velocity field from the streamfunction using the cut-cell
gradient operator. Only the two-dimensional case is supported at the moment.
"""
function update_velocity!(s::StreamVorticity{2})
    gradψ = ∇(s.operator, s.ψ)
    n = prod(s.operator.size)

    ∂ψ∂x = view(gradψ, 1:n)
    ∂ψ∂y = view(gradψ, n+1:2n)

    u = copy(∂ψ∂y)
    v = -copy(∂ψ∂x)

    s.velocity = (u, v)
    s.last_convection = nothing
    return s.velocity
end

"""
    build_convection(solver)

Create or reuse the convection operators associated with the current velocity
field.
"""
function build_convection(s::StreamVorticity{2})
    if s.last_convection !== nothing
        return s.last_convection
    end

    u = s.velocity[1]
    v = s.velocity[2]
    n = length(u)

    u_bulk = (u, v)
    uᵧ = vcat(u, v)

    conv = ConvectionOps(s.capacity, u_bulk, uᵧ)
    s.last_convection = conv
    return conv
end

"""
    _solve_streamfunction!(solver; kwargs...)

Internal helper solving the Poisson problem for the streamfunction using the
current vorticity as right-hand side. Keyword arguments are forwarded to
`solve_system!` for linear algebra control.
"""
function _solve_streamfunction!(s::StreamVorticity{2};
                                method::Function = Base.:\,
                                algorithm = nothing,
                                kwargs...)
    Aψ = copy(s.Aψ)
    bψ = poisson_rhs(s, s.time)

    BC_border_mono!(Aψ, bψ, s.bc_stream_border, s.capacity.mesh)

    solver = Solver(Steady, Monophasic, Diffusion, Aψ, bψ, nothing, Any[], [])
    solve_system!(solver; method=method, algorithm=algorithm, kwargs...)

    s.ψ = solver.x
    update_velocity!(s)
    return s.ψ
end

"""
    _step!(solver; kwargs...)

Advance the coupled streamfunction–vorticity system by one time step. The
vorticity transport is advanced with the backward Euler option of the existing
advection–diffusion assembly. Keyword arguments are forwarded to the linear
solvers.
"""
function _step!(s::StreamVorticity{2};
                scheme::String = "BE",
                method::Function = Base.:\,
                algorithm = nothing,
                kwargs...)

    _solve_streamfunction!(s; method=method, algorithm=algorithm, kwargs...)

    convection = build_convection(s)

    Aω = A_mono_unstead_advdiff(convection, s.capacity, s.ν,
                                 s.bc_vorticity, s.Δt, scheme)
    bω = b_mono_unstead_advdiff(convection, s.source, s.capacity,
                                 s.bc_vorticity, s.ω, s.Δt, s.time, scheme)

    BC_border_mono!(Aω, bω, s.bc_vorticity_border, s.capacity.mesh)

    solver = Solver(Unsteady, Monophasic, DiffusionAdvection,
                    Aω, bω, nothing, Any[], [])
    solve_system!(solver; method=method, algorithm=algorithm, kwargs...)

    s.ω = solver.x
    s.time += s.Δt
    push!(s.states, (time = s.time, ψ = copy(s.ψ), ω = copy(s.ω)))

    return s.ω
end

"""
    _run!(solver, steps; kwargs...)

Integrate the solver for a given number of time steps.
"""
function _run!(s::StreamVorticity, steps::Integer; kwargs...)
    for _ in 1:steps
        _step!(s; kwargs...)
    end
    return s
end

"""
    _run_until!(solver, t_end; kwargs...)

Advance the solution until the simulation time reaches `t_end`.
"""
function _run_until!(s::StreamVorticity, t_end::Float64; kwargs...)
    while s.time < t_end - 1e-12
        _step!(s; kwargs...)
    end
    return s
end

"""
    solve_StreamVorticity!(solver; kwargs...)

Solve the Poisson problem for the streamfunction with the current vorticity.
"""
function solve_StreamVorticity!(s::StreamVorticity; kwargs...)
    return _solve_streamfunction!(s; kwargs...)
end

"""
    step_StreamVorticity!(solver; kwargs...)

Advance the coupled streamfunction–vorticity system by one time step.
"""
function step_StreamVorticity!(s::StreamVorticity; kwargs...)
    return _step!(s; kwargs...)
end

"""
    run_StreamVorticity!(solver, steps; kwargs...)

Integrate the solver for a prescribed number of time steps.
"""
function run_StreamVorticity!(s::StreamVorticity, steps::Integer; kwargs...)
    return _run!(s, steps; kwargs...)
end

"""
    run_until_StreamVorticity!(solver, t_end; kwargs...)

Advance the solution until the simulation time reaches `t_end`.
"""
function run_until_StreamVorticity!(s::StreamVorticity, t_end::Float64; kwargs...)
    return _run_until!(s, t_end; kwargs...)
end
