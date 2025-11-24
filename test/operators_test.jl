using Penguin
using Test

@testset "Gradient Divergence" begin
    nx, ny = 20, 20
    lx, ly = 1.0, 1.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = sqrt(x^2 + y^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    operators = DiffusionOps(capacity)
    grad = ∇(operators, ones(2*length(mesh.nodes[1])*length(mesh.nodes[2])))
    @test grad[2] == 0.0
    div = ∇₋(operators, ones(2*length(mesh.nodes[1])*length(mesh.nodes[2])), ones(2*length(mesh.nodes[1])*length(mesh.nodes[2])))
    @test div[2] == 0.0
end

@testset "Diffusion Operators 1D" begin
    nx = 20
    lx = 1.0
    x0 = 0.0
    mesh = Mesh((nx,), (lx,), (x0,))
    Φ(X) = sqrt(X[1]^2) - 0.5
    LS(x,_=0) = sqrt(x^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    operators = DiffusionOps(capacity)
    @test size(operators.G'*operators.Wꜝ*operators.G) == (length(mesh.nodes[1]), length(mesh.nodes[1]))
    @test operators.size == (length(mesh.nodes[1]),)
end

@testset "Diffusion Operators 2D " begin
    nx, ny = 20, 20
    lx, ly = 1.0, 1.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = sqrt(x^2 + y^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    operators = DiffusionOps(capacity)
    @test size(operators.G'*operators.Wꜝ*operators.G) == (length(mesh.nodes[1])*length(mesh.nodes[2]), length(mesh.nodes[1])*length(mesh.nodes[2]))
    @test operators.size == (length(mesh.nodes[1]), length(mesh.nodes[2]))
end

@testset "Diffusion Operators 3D" begin
    nx, ny, nz = 20, 20, 20
    lx, ly, lz = 1.0, 1.0, 1.0
    x0, y0, z0 = 0.0, 0.0, 0.0
    mesh = Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2 + X[3]^2) - 0.5
    LS(x,y,z) = sqrt(x^2 + y^2 + z^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    operators = DiffusionOps(capacity)
    @test size(operators.G'*operators.Wꜝ*operators.G) == (length(mesh.nodes[1])*length(mesh.nodes[2])*length(mesh.nodes[3]), length(mesh.nodes[1])*length(mesh.nodes[2])*length(mesh.nodes[3]))
    @test operators.size == (length(mesh.nodes[1]), length(mesh.nodes[2]), length(mesh.nodes[3]))
end

@testset "Convection Operator 1D" begin
    nx = 20
    lx = 1.0
    x0 = 0.0
    mesh = Mesh((nx,), (lx,), (x0,))
    Φ(X) = sqrt(X[1]^2) - 0.5
    LS(x,_=0) = sqrt(x^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    nx = length(mesh.nodes[1])
    operators = ConvectionOps(capacity, (ones(nx),), zeros(nx))
    @test size(operators.G'*operators.Wꜝ*operators.G) == (length(mesh.nodes[1]), length(mesh.nodes[1]))
    @test operators.size == (length(mesh.nodes[1]),)
end

@testset "Convection Operator 2D" begin
    nx, ny = 20, 20
    lx, ly = 1.0, 1.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = sqrt(x^2 + y^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    nx, ny = length(mesh.nodes[1]), length(mesh.nodes[2])
    operators = ConvectionOps(capacity, (ones(nx*ny), ones(nx*ny)), zeros(2*nx*ny))
    @test size(operators.G'*operators.Wꜝ*operators.G) == (length(mesh.nodes[1])*length(mesh.nodes[2]), length(mesh.nodes[1])*length(mesh.nodes[2]))
    @test operators.size == (length(mesh.nodes[1]), length(mesh.nodes[2]))
end