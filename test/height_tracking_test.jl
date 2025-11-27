using Penguin
using Test

@testset "Height tracking utilities" begin
    # spatial_shape_from_dims
    @testset "spatial_shape_from_dims" begin
        dims = (10, 20, 2) # last entry time stencil
        shape = Penguin.spatial_shape_from_dims(dims)
        @test shape == (10, 20)

        dims1 = (8, 2)
        shape1 = Penguin.spatial_shape_from_dims(dims1)
        @test shape1 == (8,)

        # 3D case
        dims3 = (5, 6, 7, 2)  # nx, ny, nz, time stencil
        shape3 = Penguin.spatial_shape_from_dims(dims3)
        @test shape3 == (5, 6, 7)
    end

    # column_height_profile with 1D, 2D, and 3D inputs
    @testset "column_height_profile" begin
        v1 = [1.0, 2.0, 3.0]
    @test Penguin.column_height_profile(v1) == v1

        V2 = [1.0 2.0 3.0; 4.0 5.0 6.0]
        # sum along streamwise (rows) -> column sums
    expected = vec(sum(V2, dims=1))
    @test Penguin.column_height_profile(V2) == expected

        # 3D case: sum along first dimension, return 2D matrix
        V3 = zeros(2, 3, 4)  # nx=2, ny=3, nz=4
        for i in 1:2, j in 1:3, k in 1:4
            V3[i, j, k] = i + 10*j + 100*k
        end
        expected3 = dropdims(sum(V3, dims=1), dims=1)  # (3, 4) matrix
        result3 = Penguin.column_height_profile(V3)
        @test size(result3) == (3, 4)
        @test result3 == expected3
    end

    # interface_positions_from_heights for 1D mesh
    @testset "interface_positions_from_heights 1D" begin
        nx = 5
        lx = 1.0
        x0 = 0.0
        mesh = Penguin.Mesh((nx,), (lx,), (x0,))
        # heights per cell (columns) -> scalar position
        heights = fill(0.1, nx)
    pos = Penguin.interface_positions_from_heights(heights, mesh)
        # Δx = lx/nx => expected x0 + sum(heights)/Δx
    Δx = mesh.nodes[1][2] - mesh.nodes[1][1]
    x0_mesh = mesh.nodes[1][1]
    @test pos ≈ x0_mesh + sum(heights) / Δx atol=1e-12
    end

    # interface_positions_from_heights for 2D mesh
    @testset "interface_positions_from_heights 2D" begin
        nx, ny = 4, 3
        lx, ly = 2.0, 1.0
        x0, y0 = 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
        # heights is a vector of column heights (ny values)
        heights = collect(0.1:0.1:0.1*ny)
    positions = Penguin.interface_positions_from_heights(heights, mesh)
        # positions should be a vector of length ny with x0 + heights/Δy
    Δy = mesh.nodes[2][2] - mesh.nodes[2][1]
    x0_mesh = mesh.nodes[1][1]
    expected = x0_mesh .+ heights ./ Δy
        @test length(positions) == length(expected)
        @test positions ≈ expected atol=1e-12
    end

    # interface_positions_from_heights for 3D mesh
    @testset "interface_positions_from_heights 3D" begin
        nx, ny, nz = 4, 3, 2
        lx, ly, lz = 2.0, 1.0, 1.5
        x0, y0, z0 = 0.0, 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))
        # heights is a 2D matrix of (ny x nz) interface positions
        heights = [0.1 0.2; 0.15 0.25; 0.2 0.3]  # (3, 2) matrix
        positions = Penguin.interface_positions_from_heights(heights, mesh)
        # positions should be a matrix with x0 + heights/(Δy*Δz)
        Δy = mesh.nodes[2][2] - mesh.nodes[2][1]
        Δz = mesh.nodes[3][2] - mesh.nodes[3][1]
        x0_mesh = mesh.nodes[1][1]
        expected = x0_mesh .+ heights ./ (Δy * Δz)
        @test size(positions) == size(expected)
        @test positions ≈ expected atol=1e-12
    end

    # ensure_periodic!
    @testset "ensure_periodic!" begin
        v = [0.0, 1.0, 2.0]
    Penguin.ensure_periodic!(v)
        @test v[end] == v[1]

        empty_v = Float64[]
    Penguin.ensure_periodic!(empty_v)
        @test isempty(empty_v)
    end

    # ensure_periodic_3d!
    @testset "ensure_periodic_3d!" begin
        M = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        Penguin.ensure_periodic_3d!(M)
        # Last row should equal first row
        @test M[end, :] == M[1, :]
        # Last column should equal first column
        @test M[:, end] == M[:, 1]
    end
end

