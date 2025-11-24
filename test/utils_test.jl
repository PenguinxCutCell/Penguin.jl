using Penguin
using Test
using LinearAlgebra
using SparseArrays

@testset "Utils Test" begin
    #=========================
    Temperature Initialization
    =========================#
    @testset "Temperature Initialization" begin
        nx = 10
        ny = 10
        x_coords = collect(range(0.0, 1.0, length=nx+1))
        y_coords = collect(range(0.0, 1.0, length=ny+1))
        T0ₒ = zeros((nx+1)*(ny+1))
        T0ᵧ = zeros((nx+1)*(ny+1))

        # Test uniform initialization
        @testset "Uniform Initialization" begin
            initialize_temperature_uniform!(T0ₒ, T0ᵧ, 1.0)
            @test all(T0ₒ .== 1.0)
            @test all(T0ᵧ .== 1.0)
            
            # Test with negative values
            initialize_temperature_uniform!(T0ₒ, T0ᵧ, -2.5)
            @test all(T0ₒ .== -2.5)
            @test all(T0ᵧ .== -2.5)
            
            # Test with zero
            initialize_temperature_uniform!(T0ₒ, T0ᵧ, 0.0)
            @test all(T0ₒ .== 0.0)
            @test all(T0ᵧ .== 0.0)
        end

        # Test square initialization
        @testset "Square Initialization" begin
            # Center square
            T0ₒ = zeros((nx+1)*(ny+1))
            T0ᵧ = zeros((nx+1)*(ny+1))
            initialize_temperature_square!(T0ₒ, T0ᵧ, x_coords, y_coords, (0.5, 0.5), 2, 1.0, nx, ny)
            @test T0ₒ[1] == 0.0  # Corner should be outside square
            @test T0ᵧ[1] == 0.0
            @test T0ₒ[(nx+1)*(ny+1)] == 0.0  # Opposite corner should be outside
            @test T0ᵧ[(nx+1)*(ny+1)] == 0.0
            
            # Corner square
            T0ₒ = zeros((nx+1)*(ny+1))
            T0ᵧ = zeros((nx+1)*(ny+1))
            initialize_temperature_square!(T0ₒ, T0ᵧ, x_coords, y_coords, (0.0, 0.0), 2, 2.5, nx, ny)
            @test T0ₒ[1] == 2.5  # Bottom-left corner should be inside
            @test T0ᵧ[1] == 2.5
            @test T0ₒ[nx+1] == 0.0  # Bottom-right corner should be outside
            
            # Test with half_width 0 (single point)
            T0ₒ = zeros((nx+1)*(ny+1))
            T0ᵧ = zeros((nx+1)*(ny+1))
            initialize_temperature_square!(T0ₒ, T0ᵧ, x_coords, y_coords, (0.5, 0.5), 0, 3.0, nx, ny)
            center_i = findfirst(x -> x >= 0.5, x_coords)
            center_j = findfirst(y -> y >= 0.5, y_coords)
            center_idx = center_i + (center_j - 1) * (nx + 1)
            @test T0ₒ[center_idx] == 3.0  # Only center point should be set
            @test sum(T0ₒ) == 3.0  # No other points should be set
        end

        # Test circle initialization
        @testset "Circle Initialization" begin
            # Centered circle
            T0ₒ = zeros((nx+1)*(ny+1))
            T0ᵧ = zeros((nx+1)*(ny+1))
            initialize_temperature_circle!(T0ₒ, T0ᵧ, x_coords, y_coords, (0.5, 0.5), 0.3, 1.0, nx, ny)
            @test T0ₒ[1] == 0.0  # Corner should be outside circle
            @test sum(T0ₒ) > 0  # Some points should be inside
            @test sum(T0ₒ) < (nx+1)*(ny+1)  # Not all points should be inside
            
            # Corner circle (partial)
            T0ₒ = zeros((nx+1)*(ny+1))
            T0ᵧ = zeros((nx+1)*(ny+1))
            initialize_temperature_circle!(T0ₒ, T0ᵧ, x_coords, y_coords, (0.0, 0.0), 0.3, 2.0, nx, ny)
            @test T0ₒ[1] == 2.0  # Origin should be inside
            @test T0ₒ[nx+1] == 0.0  # Right edge should be outside
            
            # Large circle covering whole domain
            T0ₒ = zeros((nx+1)*(ny+1))
            T0ᵧ = zeros((nx+1)*(ny+1))
            initialize_temperature_circle!(T0ₒ, T0ᵧ, x_coords, y_coords, (0.5, 0.5), 1.0, 3.0, nx, ny)
            @test all(T0ₒ[1:nx-1] .== 3.0)  # All points should be inside (or exactly on edge)
        end

        # Test function initialization
        @testset "Function Initialization" begin
            # Linear function
            T0ₒ = zeros((nx+1)*(ny+1))
            T0ᵧ = zeros((nx+1)*(ny+1))
            initialize_temperature_function!(T0ₒ, T0ᵧ, x_coords, y_coords, (x, y) -> x + y, nx, ny)
            @test T0ₒ[1] ≈ 0.0 atol=1e-10  # f(0,0) = 0

            
            # Quadratic function
            T0ₒ = zeros((nx+1)*(ny+1))
            T0ᵧ = zeros((nx+1)*(ny+1))
            initialize_temperature_function!(T0ₒ, T0ᵧ, x_coords, y_coords, (x, y) -> x^2 + y^2, nx, ny)
            @test T0ₒ[1] ≈ 0.0 atol=1e-10  # f(0,0) = 0
            
            # Constant function
            T0ₒ = zeros((nx+1)*(ny+1))
            T0ᵧ = zeros((nx+1)*(ny+1))
            initialize_temperature_function!(T0ₒ, T0ᵧ, x_coords, y_coords, (x, y) -> 5.0, nx, ny)
        end
    end

    #========================
    Velocity Initialization
    ========================#
    @testset "Velocity Field Initialization" begin
        # Setup for 2D tests
        nx, ny = 20, 20
        lx, ly = 1.0, 1.0
        x0, y0 = 0.0, 0.0
        
        @testset "Rotating Velocity Field" begin
            magnitude = 1.0
            uₒx, uₒy = initialize_rotating_velocity_field(nx, ny, lx, ly, x0, y0, magnitude)
            
            # Check dimensions
            @test length(uₒx) == (nx+1) * (ny+1)
            @test length(uₒy) == (nx+1) * (ny+1)
            
            # Check center point (should be close to zero)
            center_i, center_j = nx÷2, ny÷2
            center_idx = center_i + center_j * (nx + 1) + 1
            @test abs(uₒx[center_idx]) < 0.1
            @test abs(uₒy[center_idx]) < 0.1
            
            # Check a point away from center
            i, j = 0, 0  # Bottom-left corner
            idx = i + j * (nx + 1) + 1
            @test uₒx[idx] ≈ (y0 - ly/2) * -magnitude atol=1e-10
            @test uₒy[idx] ≈ (x0 - lx/2) * magnitude atol=1e-10
            
            # Verify rotation direction (counterclockwise)
            for j in 0:ny
                for i in 0:nx
                    idx = i + j * (nx + 1) + 1
                    x = x0 + i * (lx / nx)
                    y = y0 + j * (ly / ny)
                    
                    # Skip points very near the center
                    if abs(x - lx/2) > 1e-10 || abs(y - ly/2) > 1e-10
                        # For counterclockwise rotation, u × (r - r₀) should be positive
                        # where r₀ is the center point
                        cross_product = uₒx[idx] * (y - ly/2) - uₒy[idx] * (x - lx/2)
                        @test cross_product < 0  # Should be negative for clockwise rotation
                    end
                end
            end
        end
        
        @testset "Poiseuille Velocity Field" begin
            uₒx, uₒy = initialize_poiseuille_velocity_field(nx, ny, lx, ly, x0, y0)
            
            # Check dimensions
            @test length(uₒx) == (nx+1) * (ny+1)
            @test length(uₒy) == (nx+1) * (ny+1)
            
            # Check y-velocity is zero everywhere
            @test all(uₒy .== 0.0)
            
            # Check parabolic profile in x-direction
            for j in 0:ny
                for i in 0:nx
                    idx = i + j * (nx + 1) + 1
                    x = x0 + i * (lx / nx)
                    
                    # Should follow x(1-x) pattern
                    @test uₒx[idx] ≈ x * (1 - x) atol=1e-10
                    
                    # Check maximum at center, zero at boundaries
                    if i == 0 || i == nx
                        @test abs(uₒx[idx]) < 1e-10
                    elseif i == nx÷2
                        @test uₒx[idx] > uₒx[idx-1]  # Increasing up to center
                    elseif i > nx÷2
                        @test uₒx[idx] < uₒx[idx-1]  # Decreasing after center
                    end
                end
            end
        end
        
        @testset "Radial Velocity Field" begin
            center = (0.5, 0.5)
            magnitude = 2.0
            uₒx, uₒy = initialize_radial_velocity_field(nx, ny, lx, ly, x0, y0, center, magnitude)
            
            # Check dimensions
            @test length(uₒx) == (nx+1) * (ny+1)
            @test length(uₒy) == (nx+1) * (ny+1)
            
            # Find center point index
            center_i = Int(center[1] * nx)
            center_j = Int(center[2] * ny)
            center_idx = center_i + center_j * (nx + 1) + 1
            
            # Check velocities point outward from center
            for j in 0:ny
                for i in 0:nx
                    idx = i + j * (nx + 1) + 1
                    x = x0 + i * (lx / nx)
                    y = y0 + j * (ly / ny)
                    
                    # Skip center point (undefined direction)
                    if idx != center_idx
                        # Calculate vector from center
                        dx = x - center[1]
                        dy = y - center[2]
                        r = sqrt(dx^2 + dy^2)
                        
                        # Velocity should be in same direction as position vector
                        if r > 1e-10
                            cosine = (uₒx[idx] * dx + uₒy[idx] * dy) / (magnitude * r)
                            @test cosine ≈ 1.0 atol=1e-10
                        end
                        
                        # Check magnitude
                        v_mag = sqrt(uₒx[idx]^2 + uₒy[idx]^2)
                        @test v_mag ≈ magnitude atol=1e-10
                    end
                end
            end
        end
    end
    
    #========================
    Volume Redefinition
    ========================#
    @testset "Volume Redefinition" begin
        # Setup 1D case
        nx = 20
        lx = 1.0
        
        # Create a simple 1D level-set function
        center = 0.5
        radius = 0.21
        body = (x, _=0) -> abs(x - center) - radius
        
        # Create mesh and capacity
        mesh = Penguin.Mesh((nx,), (lx,), (0.0,))
        capacity = Capacity(body, mesh)
        operator = DiffusionOps(capacity)
        
        # Store original values
        W_orig = copy(capacity.W[1])
        V_orig = copy(capacity.V)
        
        # Apply volume redefinition
        volume_redefinition!(capacity, operator)
        
        # Verify changes
        @test W_orig != capacity.W[1]  # Weights should change
        @test V_orig != capacity.V     # Volume matrix should change
        
        # Check diagonal property is maintained
        @test isa(capacity.W[1], SparseMatrixCSC)
        @test isa(capacity.V, SparseMatrixCSC)
    end
end