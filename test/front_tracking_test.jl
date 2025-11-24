using Test
using LinearAlgebra
using Statistics
using LibGEOS
using Penguin

@testset "Julia FrontTracking Tests" begin
    
    @testset "Basic Construction" begin
        # Create an empty interface
        front = FrontTracker()
        @test isempty(front.markers)
        @test front.is_closed == true
        @test front.interface === nothing
        @test front.interface_poly === nothing
        
        # Create interface with markers
        markers = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        front = FrontTracker(markers)
        # The implementation adds a closing point
        @test length(front.markers) == 5  
        @test front.is_closed == true
        @test front.interface !== nothing
        @test front.interface_poly !== nothing
    end
    
    @testset "Shape Creation" begin
        # Test circle creation
        front = FrontTracker()
        create_circle!(front, 0.5, 0.5, 0.3, 20)
        markers = get_markers(front)
        @test length(markers) == 21  # With closing point
        
        # Check that markers are approximately on a circle
        for (x, y) in markers
            distance = sqrt((x - 0.5)^2 + (y - 0.5)^2)
            @test isapprox(distance, 0.3, atol=1e-10)
        end
        
        # Test rectangle creation
        front = FrontTracker()
        create_rectangle!(front, 0.1, 0.2, 0.8, 0.9)
        markers = get_markers(front)
        @test length(markers) == 97

        
        # Test ellipse creation
        front = FrontTracker()
        create_ellipse!(front, 0.5, 0.5, 0.3, 0.2, 20)
        markers = get_markers(front)
        @test length(markers) == 21  # With closing point
        
        # Check that markers are approximately on an ellipse
        for (x, y) in markers
            normalized_distance = ((x - 0.5)/0.3)^2 + ((y - 0.5)/0.2)^2
            @test isapprox(normalized_distance, 1.0, atol=1e-10)
        end
    end
    
    @testset "Point Inside Tests" begin
        # Create a square
        front = FrontTracker()
        create_rectangle!(front, 0.0, 0.0, 1.0, 1.0)
        
        # Test points inside
        @test is_point_inside(front, 0.5, 0.5) == true
        @test is_point_inside(front, 0.1, 0.1) == true
        @test is_point_inside(front, 0.9, 0.9) == true
        
        # Test points outside
        @test is_point_inside(front, -0.5, 0.5) == false
        @test is_point_inside(front, 1.5, 0.5) == false
        @test is_point_inside(front, 0.5, -0.5) == false
        @test is_point_inside(front, 0.5, 1.5) == false
    end
    
    @testset "SDF Calculation" begin
        # Create a square centered at origin
        front = FrontTracker()
        create_rectangle!(front, -1.0, -1.0, 1.0, 1.0)
        
        # Test points inside (should be negative)
        @test sdf(front, 0.0, 0.0) < 0
        @test isapprox(sdf(front, 0.0, 0.0), -1.0, atol=1e-2)
        
        # Test points outside (should be positive)
        @test sdf(front, 2.0, 0.0) > 0
        @test isapprox(sdf(front, 2.0, 0.0), 1.0, atol=1e-2)
        
        # Test points on the boundary (should be approximately zero)
        @test isapprox(sdf(front, 1.0, 0.0), 0.0, atol=1e-2)
        @test isapprox(sdf(front, 0.0, 1.0), 0.0, atol=1e-2)
        
        # Create a circle for additional tests
        front = FrontTracker()
        create_circle!(front, 0.0, 0.0, 1.0)
        
        # Test points at various distances from circle
        @test isapprox(sdf(front, 0.0, 0.0), -1.0, atol=1e-2)  # Center should be -radius
        @test isapprox(sdf(front, 2.0, 0.0), 1.0, atol=1e-2)   # Outside, distance = 1
        @test isapprox(sdf(front, 0.5, 0.0), -0.5, atol=1e-2)  # Inside, distance = 0.5
    end
    

    @testset "Normals and Curvature" begin
        @testset "Circle Normals" begin
            # Create a circle centered at origin
            radius = 0.5
            center_x, center_y = 0.0, 0.0
            front = FrontTracker()
            create_circle!(front, center_x, center_y, radius, 32)
            
            # Calculate normals
            markers = get_markers(front)
            normals = compute_marker_normals(front, markers)
            
            # For a circle, normals should point away from center with unit length
            for i in 1:length(markers)-1 # Skip last point (duplicate of first for closed shape)
                x, y = markers[i]
                nx, ny = normals[i]
                
                # Expected normal: unit vector from center to point
                dist = sqrt((x - center_x)^2 + (y - center_y)^2)
                expected_nx = (x - center_x) / dist
                expected_ny = (y - center_y) / dist
                
                # Check that normal is a unit vector
                @test isapprox(nx^2 + ny^2, 1.0, atol=1e-6)
                
                # Check normal direction
                @test isapprox(nx, expected_nx, atol=1e-6)
                @test isapprox(ny, expected_ny, atol=1e-6)
            end
        end
    end
    
    @testset "Volume Jacobian" begin
        # Create a circular interface
        front = FrontTracker()
        create_circle!(front, 0.5, 0.5, 0.3)
        
        # Create a simple mesh grid
        x_faces = 0.0:0.1:1.0
        y_faces = 0.0:0.1:1.0
        
        # Compute the volume Jacobian
        jacobian = compute_volume_jacobian(front, x_faces, y_faces)
        
        # Basic checks
        @test isa(jacobian, Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}})
        
        # At least some cells should have non-zero Jacobian entries
        @test any(length(values) > 0 for (_, values) in jacobian)
        
        # Cells far from the interface should have zero Jacobian entries
        # Cells at corner (1,1) and (10,10) should be outside the circle
        @test length(get(jacobian, (1, 1), [])) == 0
        @test length(get(jacobian, (10, 10), [])) == 0
        
        # Cells near the interface should have non-zero Jacobian entries
        # Find where the interface crosses cells
        has_nonzero_entries = false
        for i in 1:length(x_faces)-1
            for j in 1:length(y_faces)-1
                if haskey(jacobian, (i, j)) && !isempty(jacobian[(i, j)])
                    has_nonzero_entries = true
                    break
                end
            end
            if has_nonzero_entries
                break
            end
        end
        @test has_nonzero_entries

    end
end

@testset "FrontTracker1D Basic Functionality" begin
    # Test constructors
    ft_empty = FrontTracker1D()
    @test isempty(get_markers(ft_empty))

    ft = FrontTracker1D([0.3, 0.7])
    @test get_markers(ft) == [0.3, 0.7]

    # Test add_marker!
    add_marker!(ft, 0.5)
    @test get_markers(ft) == sort([0.3, 0.5, 0.7])

    # Test set_markers!
    set_markers!(ft, [0.1, 0.9])
    @test get_markers(ft) == [0.1, 0.9]

    # Test is_point_inside
    @test !is_point_inside(ft, 0.0)
    @test is_point_inside(ft, 0.5)
    @test !is_point_inside(ft, 1.0)

    # Test sdf
    @test sdf(ft, 0.0) > 0
    @test sdf(ft, 0.5) < 0
    @test sdf(ft, 1.0) > 0
end

@testset "FrontTracker1D Capacity Calculation" begin
    # Simple mesh and front
    using Penguin
    nx = 10
    lx = 1.0
    mesh = Penguin.Mesh((nx,), (lx,), (0.0,))
    ft = FrontTracker1D([0.3, 0.7])

    caps = compute_capacities_1d(mesh, ft)
    @test length(caps[:fractions]) == nx+1
    @test length(caps[:volumes]) == nx+1
    @test length(caps[:centroids_x]) == nx+1
    @test length(caps[:cell_types]) == nx+1
    @test length(caps[:Ax]) == nx+1
    @test length(caps[:Wx]) == nx+1
    @test length(caps[:Bx]) == nx+1

    # Check that fluid fractions are between 0 and 1
    @test all(0 .<= caps[:fractions] .<= 1)
end

@testset "FrontTracker1D Space-Time Capacity Calculation" begin
    using Penguin
    nx = 10
    lx = 1.0
    mesh = Penguin.Mesh((nx,), (lx,), (0.0,))
    ft_n = FrontTracker1D([0.3, 0.7])
    ft_np1 = FrontTracker1D([0.35, 0.75])
    dt = 0.1

    st_caps = compute_spacetime_capacities_1d(mesh, ft_n, ft_np1, dt)
    @test length(st_caps[:Ax_spacetime]) == nx+1
    @test length(st_caps[:V_spacetime]) == nx+1
    @test length(st_caps[:Bx_spacetime]) == nx+1
    @test length(st_caps[:Wx_spacetime]) == nx+1
    @test length(st_caps[:ST_centroids]) == nx
    @test length(st_caps[:ms_cases]) == nx
    @test length(st_caps[:edge_types]) == nx+1
    @test length(st_caps[:t_crosses]) == nx+1
    @test length(st_caps[:center_types]) == nx
    @test length(st_caps[:t_crosses_center]) == nx
end
