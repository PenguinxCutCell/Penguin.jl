using Penguin
using Test
using LinearAlgebra
using IterativeSolvers

const P = Penguin

@testset "Streamfunction–Vorticity Solver (uniform)" begin
    nx, ny = 12, 12
    Lx, Ly = 1.0, 1.0
    mesh = Mesh((nx, ny), (Lx, Ly), (0.0, 0.0))

    body = (x, y, _=0.0) -> -1.0
    capacity = Capacity(body, mesh)

    zero_dirichlet = Dirichlet(0.0)
    border_bc = BorderConditions(Dict(
        :left => zero_dirichlet,
        :right => zero_dirichlet,
        :bottom => zero_dirichlet,
        :top => zero_dirichlet,
    ))

    ν = 0.01
    Δt = 5.0e-3
    solver = StreamVorticity(capacity, ν, Δt;
        bc_stream = zero_dirichlet,
        bc_vorticity = zero_dirichlet,
        bc_stream_border = border_bc,
        bc_vorticity_border = border_bc,
    )

    n = prod(solver.operator.size)
    ω_bulk = [sinpi(c[1]) * sinpi(c[2]) for c in solver.capacity.C_ω]
    solver.ω .= vcat(ω_bulk, zeros(n))

    solve_StreamVorticity!(solver; method = gmres)

    Aψ = copy(solver.Aψ)
    bψ = P.poisson_rhs(solver, solver.time)
    P.BC_border_mono!(Aψ, bψ, solver.bc_stream_border, solver.capacity.mesh)
    residual = norm(Aψ * solver.ψ - bψ) / max(norm(bψ), 1.0)
    @test residual ≤ 1e-8

    u, v = solver.velocity
    @test length(u) == n
    @test length(v) == n

    conv_first = P.build_convection(solver)
    conv_second = P.build_convection(solver)
    @test conv_first === conv_second
end

@testset "Streamfunction–Vorticity Step" begin
    nx, ny = 10, 10
    mesh = Mesh((nx, ny), (1.0, 1.0), (0.0, 0.0))
    body = (x, y, _=0.0) -> -1.0
    capacity = Capacity(body, mesh)

    zero_dirichlet = Dirichlet(0.0)
    border_bc = BorderConditions(Dict(
        :left => zero_dirichlet,
        :right => zero_dirichlet,
        :bottom => zero_dirichlet,
        :top => zero_dirichlet,
    ))

    ν = 0.02
    Δt = 1.0e-2
    n = prod(DiffusionOps(capacity).size)
    ω0 = zeros(2n)

    solver = StreamVorticity(capacity, ν, Δt;
        bc_stream = zero_dirichlet,
        bc_vorticity = zero_dirichlet,
        bc_stream_border = border_bc,
        bc_vorticity_border = border_bc,
        ω0 = ω0,
    )

    step_StreamVorticity!(solver; method = gmres)
    @test solver.time ≈ Δt atol = 1e-12
    @test length(solver.states) == 2
    @test norm(solver.ω) ≤ 1e-12

    run_StreamVorticity!(solver, 2; method = gmres)
    @test solver.time ≈ 3Δt atol = 1e-12
    @test length(solver.states) == 4

    times = [state.time for state in solver.states]
    @test issorted(times)
end

@testset "Cut-cell streamfunction–vorticity evolution" begin
    nx, ny = 24, 24
    mesh = Mesh((nx, ny), (1.0, 1.0), (0.0, 0.0))

    centre = (0.5, 0.5)
    radius = 0.2
    circle = (x, y, _=0.0) -> sqrt((x - centre[1])^2 + (y - centre[2])^2) - radius
    capacity = Capacity(circle, mesh)

    zero_dirichlet = Dirichlet(0.0)
    border_bc = BorderConditions(Dict(
        :left => zero_dirichlet,
        :right => zero_dirichlet,
        :bottom => zero_dirichlet,
        :top => zero_dirichlet,
    ))

    ν = 5.0e-3
    Δt = 5.0e-3
    n = prod(DiffusionOps(capacity).size)

    ω_ring = [exp(-((c[1] - centre[1])^2 + (c[2] - centre[2])^2) / (radius^2)) for c in capacity.C_ω]
    ω0 = vcat(ω_ring, zeros(n))

    solver = StreamVorticity(capacity, ν, Δt;
        bc_stream = zero_dirichlet,
        bc_vorticity = zero_dirichlet,
        bc_stream_border = border_bc,
        bc_vorticity_border = border_bc,
        ω0 = ω0,
    )

    solve_StreamVorticity!(solver; method = gmres)
    u, v = solver.velocity
    @test maximum(abs.(u)) > 0
    @test maximum(abs.(v)) > 0

    step_StreamVorticity!(solver; method = gmres)
    @test solver.last_convection isa ConvectionOps
    @test all(isfinite, solver.ω)
end
