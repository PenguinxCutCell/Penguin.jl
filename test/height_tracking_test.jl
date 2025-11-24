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
    end

    # column_height_profile with 1D and 2D inputs
    @testset "column_height_profile" begin
        v1 = [1.0, 2.0, 3.0]
    @test Penguin.column_height_profile(v1) == v1

        V2 = [1.0 2.0 3.0; 4.0 5.0 6.0]
        # sum along streamwise (rows) -> column sums
    expected = vec(sum(V2, dims=1))
    @test Penguin.column_height_profile(V2) == expected
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

    # ensure_periodic!
    @testset "ensure_periodic!" begin
        v = [0.0, 1.0, 2.0]
    Penguin.ensure_periodic!(v)
        @test v[end] == v[1]

        empty_v = Float64[]
    Penguin.ensure_periodic!(empty_v)
        @test isempty(empty_v)
    end
end
