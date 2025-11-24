using Penguin
using Test

@testset "1D Phase" begin
    nx = 10
    lx = 2.0
    x0 = 0.0
    mesh = Mesh((nx,), (lx,), (x0,))
    Φ(X) = sqrt(X[1]^2) - 0.5
    LS(x,_=0) = sqrt(x^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    operators = DiffusionOps(capacity)
    f(x,_=0) = 0.0
    D(x,_=0) = 1.0
    Fluide = Phase(capacity, operators, f, D)
    @test Fluide.capacity == capacity
    @test Fluide.operator == operators
    @test Fluide.source == f
    @test Fluide.Diffusion_coeff == D
end

@testset "2D Phase" begin
    nx, ny = 10, 10
    lx, ly = 2.0, 2.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = sqrt(x^2 + y^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    operators = DiffusionOps(capacity)
    f(x,y,_=0) = 0.0
    D(x,y,_=0) = 1.0
    Fluide = Phase(capacity, operators, f, D)
    @test Fluide.capacity == capacity
    @test Fluide.operator == operators
    @test Fluide.source == f
    @test Fluide.Diffusion_coeff == D
end