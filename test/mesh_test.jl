using Penguin
using Test

@testset "1D mesh" begin
    nx = 5
    lx = 1.0
    x0 = 0.0
    mesh1D = Mesh((nx,), (lx,), (x0,))
    borders1D = mesh1D.tag.border_cells
    @test mesh1D.centers[1] ≈ [0.1, 0.3, 0.5, 0.7, 0.9]
    @test nC(mesh1D) == 5
    @test size(mesh1D, 1) == 5
    @test length(borders1D) == 2
    @test borders1D[1][1] == CartesianIndex((1,))
    @test borders1D[1][2] ≈ (0.1,)
    @test borders1D[2][1] == CartesianIndex((5,))
    @test borders1D[2][2] ≈ (0.9,)
end

@testset "2D mesh" begin
    nx, ny = 5, 5
    lx, ly = 1.0, 1.0
    x0, y0 = 0.0, 0.0
    mesh2D = Mesh((nx, ny), (lx, ly), (x0, y0))
    borders2D = mesh2D.tag.border_cells
    @test mesh2D.centers[1] ≈ [0.1, 0.3, 0.5, 0.7, 0.9]
    @test mesh2D.centers[2] ≈ [0.1, 0.3, 0.5, 0.7, 0.9]
    @test nC(mesh2D) == 25
    @test length(borders2D) == 16
    @test borders2D[1][1] == CartesianIndex((1, 1))
    @test all(isapprox.(borders2D[1][2], (0.1, 0.1)))
    @test borders2D[2][1] == CartesianIndex(1, 2)
    @test all(isapprox.(borders2D[2][2], (0.1, 0.3)))
end

@testset "3D mesh" begin
    nx, ny, nz = 5, 5, 5
    lx, ly, lz = 1.0, 1.0, 1.0
    x0, y0, z0 = 0.0, 0.0, 0.0
    mesh3D = Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))
    borders3D = mesh3D.tag.border_cells
    @test mesh3D.centers[1] ≈ [0.1, 0.3, 0.5, 0.7, 0.9]
    @test mesh3D.centers[2] ≈ [0.1, 0.3, 0.5, 0.7, 0.9]
    @test mesh3D.centers[3] ≈ [0.1, 0.3, 0.5, 0.7, 0.9]
    @test nC(mesh3D) == 125
    @test length(borders3D) == 98
    @test borders3D[1][1] == CartesianIndex((1, 1, 1))
    @test all(isapprox.(borders3D[1][2], (0.1, 0.1, 0.1)))
    @test borders3D[2][1] == CartesianIndex((1, 2, 1))
    @test all(isapprox.(borders3D[2][2], (0.1, 0.3, 0.1)))
end

@testset "4D mesh" begin
    nx, ny, nz, nw = 5, 5, 5, 5
    lx, ly, lz, lw = 1.0, 1.0, 1.0, 1.0
    x0, y0, z0, w0 = 0.0, 0.0, 0.0, 0.0
    mesh4D = Mesh((nx, ny, nz, nw), (lx, ly, lz, lw), (x0, y0, z0, w0))
    borders4D = mesh4D.tag.border_cells
    expected_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    @test mesh4D.centers[1] ≈ expected_centers
    @test mesh4D.centers[2] ≈ expected_centers
    @test mesh4D.centers[3] ≈ expected_centers
    @test mesh4D.centers[4] ≈ expected_centers
    @test nC(mesh4D) == 625
    @test length(borders4D) == 544
    @test borders4D[1][1] == CartesianIndex((1, 1, 1, 1))
    @test all(isapprox.(borders4D[1][2], (0.1, 0.1, 0.1, 0.1)))
    @test borders4D[2][1] == CartesianIndex((1, 2, 1, 1))
    @test all(isapprox.(borders4D[2][2], (0.1, 0.3, 0.1, 0.1)))
end

@testset "1D+1 SpaceTime mesh" begin
    nx = 5
    lx = 1.0
    x0 = 0.0
    mesh1D = Mesh((nx,), (lx,), (x0,))
    STmesh1D = SpaceTimeMesh(mesh1D, [0.0, 0.1], tag=mesh1D.tag)
    @test STmesh1D.centers[1] ≈ [0.1, 0.3, 0.5, 0.7, 0.9]
    @test STmesh1D.centers[2] ≈ [0.05]
    @test STmesh1D.nodes[1] ≈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    @test STmesh1D.nodes[2] ≈ [0.0, 0.1]
    @test STmesh1D.tag == mesh1D.tag
    @test nC(STmesh1D) == 5
end

@testset "1D+1 SpaceTime mesh with Spatial Mesh" begin
    nx, nt = 5, 1
    lx, lt = 1.0, 0.05
    x0, t0 = 0.0, 0.0
    mesh1D1 = Mesh((nx,nt,), (lx, lt), (x0, t0))
    mesh1D = Mesh((nx,), (lx,), (x0,))
    STmesh1D = SpaceTimeMesh(mesh1D, [0.0, 0.1], tag=mesh1D.tag)
    @test STmesh1D.centers[1] ≈ [0.1, 0.3, 0.5, 0.7, 0.9]
    @test STmesh1D.centers[2] ≈ [0.05]
    @test STmesh1D.nodes[1] ≈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    @test STmesh1D.nodes[2] ≈ [0.0, 0.1]
    @test STmesh1D.tag == mesh1D.tag
    @test nC(STmesh1D) == 5
    @test size(STmesh1D, 1) == 5
    @test size(STmesh1D, 2) == 1
end