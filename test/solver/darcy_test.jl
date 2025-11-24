using Penguin
using Test

@testset "Darcy Test" begin
    nx, ny = 20, 20
    lx, ly = 2.0, 2.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = (sqrt((x-0.5)^2 + (y-0.5)^2) - 0.5)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Neumann(0.0)
    bc_10 = Dirichlet(10.0)
    bc_20 = Dirichlet(20.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc_10, :right => bc_20))
    f(x,y,_=0) = 0.0
    D(x,y,_=0) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DarcyFlow(Fluide, bc_b, bc)
    solve_DarcyFlow!(solver; method=Base.:\)
    uo = solver.x[1:end÷2]
    ug = solver.x[end÷2+1:end]
    @test maximum(uo) ≈ 20.0 atol=1e-2
end

@testset "Darcy Unsteady Test" begin
    nx, ny = 20, 20
    lx, ly = 2.0, 2.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    LS(x,y,_=0) = (sqrt((x-0.5)^2 + (y-0.5)^2) - 0.5)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Neumann(0.0)
    bc_10 = Dirichlet(10.0)
    bc_20 = Dirichlet(20.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc_10, :right => bc_20))
    f(x,y,z,t) = 0.0
    D(x,y,_=0) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    u0ₒ = fill(10.0, (nx+1)*(ny+1))
    u0ᵧ = fill(10.0, (nx+1)*(ny+1))
    u0 = vcat(u0ₒ, u0ᵧ)
    Δt = 0.1 * (lx/nx)^2
    Tend = 0.2
    solver = DarcyFlowUnsteady(Fluide, bc_b, bc, Δt, u0, "BE")
    solve_DarcyFlowUnsteady!(solver, Fluide, Δt, Tend, bc_b, bc, "BE"; method=IterativeSolvers.gmres)
    uo = solver.states[end][1:end÷2]
    ug = solver.states[end][end÷2+1:end]
    @test maximum(uo) ≈ 20.0 atol=1e-2
end

@testset "Darcy Velocity Test" begin
    nx, ny = 20, 20
    lx, ly = 2.0, 2.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    LS(x,y,_=0) = (sqrt((x-0.5)^2 + (y-0.5)^2) - 0.5)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Neumann(0.0)
    bc_10 = Dirichlet(10.0)
    bc_20 = Dirichlet(20.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc_10, :right => bc_20))
    f(x,y,z) = 0.0
    D(x,y,_=0) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DarcyFlow(Fluide, bc_b, bc)
    solve_DarcyFlow!(solver; method=Base.:\)
    u = solve_darcy_velocity(solver, Fluide)
    u_no_nan = filter(!isnan, abs.(u))
    @test maximum(u_no_nan) < 1e2
end
