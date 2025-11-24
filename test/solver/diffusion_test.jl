using Penguin
using Test
using SparseArrays

@testset "Diffusion Monophasic" begin
    nx, ny = 20, 20
    lx, ly = 2.0, 2.0
    x0, y0 = 0.0, 0.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = (sqrt((x-0.5)^2 + (y-0.5)^2) - 0.5)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Dirichlet(1.0)
    bc1 = Dirichlet(1.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))
    f(x,y,_=0) = 0.0
    D(x,y,_=0) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    solve_DiffusionSteadyMono!(solver)
    uo = solver.x[1:end÷2]
    ug = solver.x[end÷2+1:end]
    @test maximum(uo) ≈ 1.0 atol=1e-2
    @test maximum(ug) ≈ 1.0 atol=1e-2
end

@testset "Diffusion Diphasic" begin
    nx, ny = 80, 80
    lx, ly = 4.0, 4.0
    x0, y0 = 0.0, 0.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    LS(x,y,_=0) = (sqrt((x-2.0)^2 + (y-2.0)^2) - 1.0)
    LS_c(x,y,_=0) = -(sqrt((x-2.0)^2 + (y-2.0)^2) - 1.0)
    capacity = Capacity(LS, mesh, method="VOFI")
    capacity_c = Capacity(LS_c, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    operator_c = DiffusionOps(capacity_c)
    bc1 = Dirichlet(0.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))
    ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))
    f(x,y,_=0) = 1.0
    D(x,y,_=0) = 1.0
    Fluide1 = Phase(capacity, operator, f, D)
    Fluide2 = Phase(capacity_c, operator_c, f, D)
    solver = DiffusionSteadyDiph(Fluide1, Fluide2, bc_b, ic)
    solve_DiffusionSteadyDiph!(solver, method=Base.:\)
    u1 = solver.x[1:end÷2]
    u2 = solver.x[end÷2+1:end]
    u1o = u1[1:end÷2]
    u1g = u1[end÷2+1:end]
    u2o = u2[1:end÷2]
    u2g = u2[end÷2+1:end]
    @test maximum(u1o) ≈ 1.15 atol=1e-2
end

@testset "Heat Monophasic" begin
    nx, ny = 20, 20
    lx, ly = 4.0, 4.0
    x0, y0 = 0.0, 0.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    circle = (x,y,_=0)->(sqrt((x-2.0)^2 + (y-2.0)^2) - 1.0)
    capacity = Capacity(circle, mesh)
    operator = DiffusionOps(capacity)
    bc1 = Dirichlet(1.0)
    bc0 = Dirichlet(0.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc0, :right => bc0, :top => bc0, :bottom => bc0))
    f = (x,y,z,t)->0.0
    D = (x,y,z)->1.0
    Fluide = Phase(capacity, operator, f, D)
    u0ₒ = zeros((nx+1)*(ny+1))
    u0ᵧ = ones((nx+1)*(ny+1))
    u0 = vcat(u0ₒ, u0ᵧ)
    Δt = 0.25 * (lx/nx)^2
    Tend = 0.01
    solver = DiffusionUnsteadyMono(Fluide, bc_b, bc1, Δt, u0, "BE")
    solve_DiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, bc1, "BE"; method=Base.:\)
    uo = solver.x[1:end÷2]
    ug = solver.x[end÷2+1:end]
    @test maximum(ug) ≈ 1.0 atol=1e-2
end