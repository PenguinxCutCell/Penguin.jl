using Penguin
using Test
using LinearAlgebra
using SparseArrays

@testset "GeometricMoments vs VOFI Single Circle Comparison" begin
    nx, ny = 20, 20
    mesh = Mesh((nx, ny), (1.0, 1.0), (0.0, 0.0))

    # Define a simple circle shape
    center_x, center_y = 0.5, 0.5
    radius = 0.3
    
    # Scalar version for both methods
    function circle_scalar(x, y, _=0)
        return sqrt((x - center_x)^2 + (y - center_y)^2) - radius
    end

    # Calculate capacities using both methods
    geo_capacity = Capacity(circle_scalar, mesh, method="ImplicitIntegration")
    vofi_capacity = Capacity(circle_scalar, mesh, method="VOFI")
    
    # Volumes should be similar
    geo_vol = sum(diag(geo_capacity.V))
    vofi_vol = sum(diag(vofi_capacity.V))
    @test isapprox(geo_vol, vofi_vol, rtol=0.05)
    
    # Interface lengths should be similar
    geo_interface = sum(diag(geo_capacity.Γ))
    vofi_interface = sum(diag(vofi_capacity.Γ))
    @test isapprox(geo_interface, vofi_interface, rtol=0.1)
    
    # Cell type classification should be consistent
    geo_cut = count(x -> x == -1, geo_capacity.cell_types)
    vofi_cut = count(x -> x == -1, vofi_capacity.cell_types)
    @test isapprox(geo_cut, vofi_cut, rtol=0.2)

    # Test Types
    @test typeof(geo_capacity.A) == NTuple{2, SparseMatrixCSC{Float64, Int64}}
    @test typeof(geo_capacity.B) == NTuple{2, SparseMatrixCSC{Float64, Int64}}
    @test typeof(geo_capacity.V) == SparseMatrixCSC{Float64, Int64}

    # Test size consistency
    @test size(geo_capacity.A[1]) == size(vofi_capacity.A[1])
    @test size(geo_capacity.A[2]) == size(vofi_capacity.A[2])
    @test size(geo_capacity.B[1]) == size(vofi_capacity.B[1])
    @test size(geo_capacity.B[2]) == size(vofi_capacity.B[2])
    @test size(geo_capacity.V) == size(vofi_capacity.V)

    # Test element-wise consistency
    for d in 1:2
        @test isapprox(geo_capacity.A[d], vofi_capacity.A[d], rtol=0.1)
        @test isapprox(geo_capacity.B[d], vofi_capacity.B[d], rtol=0.1)
        @test isapprox(geo_capacity.W[d], vofi_capacity.W[d], rtol=0.1)
    end

    # Test bulk centroid consistency with VOFI
    @test isapprox(geo_capacity.C_ω, vofi_capacity.C_ω, rtol=0.1)

    
    # Face capacities (A) should be similar
    for d in 1:2
        geo_a = sum(diag(geo_capacity.A[d]))
        vofi_a = sum(diag(vofi_capacity.A[d]))
        @test isapprox(geo_a, vofi_a, rtol=0.1)
    end
    
    # Compare with analytical values
    expected_area = π * radius^2
    @test isapprox(geo_vol, expected_area, rtol=0.05)
    
    expected_perimeter = 2 * π * radius
    @test isapprox(geo_interface, expected_perimeter, rtol=0.1)
    
    # Check centroid accuracy
    cut_cells = findall(x -> x == -1, geo_capacity.cell_types)
    for idx in cut_cells
        if !all(iszero, geo_capacity.C_γ[idx])  # Skip any zero centroids
            centroid = geo_capacity.C_γ[idx]
            distance = sqrt((centroid[1] - center_x)^2 + (centroid[2] - center_y)^2)
            @test isapprox(distance, radius, atol=0.05)
        end
    end
end

@testset "GeometricMoments 3D Sphere Test" begin
    nx, ny, nz = 10, 10, 10  # Lower resolution for 3D
    mesh = Mesh((nx, ny, nz), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    
    # Sphere centered at (0.5, 0.5, 0.5) with radius 0.3
    center_x, center_y, center_z = 0.5, 0.5, 0.5
    radius = 0.3
    
    function sphere(x, y, z)
        return sqrt((x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2) - radius
    end
    
    # Calculate capacities using both methods
    geo_capacity = Capacity(sphere, mesh, method="ImplicitIntegration")
    vofi_capacity = Capacity(sphere, mesh, method="VOFI")
    
    # Compare with analytical values
    expected_volume = (4/3) * π * radius^3
    geo_vol = sum(diag(geo_capacity.V))
    vofi_vol = sum(diag(vofi_capacity.V))
    
    @test isapprox(geo_vol, expected_volume, rtol=0.1)
    @test isapprox(geo_vol, vofi_vol, rtol=0.05)
    
    # Surface area should be similar
    expected_area = 4 * π * radius^2
    geo_interface = sum(diag(geo_capacity.Γ))
    vofi_interface = sum(diag(vofi_capacity.Γ))
    
    @test isapprox(geo_interface, expected_area, rtol=0.1)
    @test isapprox(geo_interface, vofi_interface, rtol=0.1)
    
    # Center coordinates should be close to specified center
    cut_cells = findall(x -> x == -1, geo_capacity.cell_types)
    centroids_sum = zeros(3)
    count_valid = 0
    
    for idx in cut_cells
        if !all(isnan, geo_capacity.C_γ[idx]) && !all(iszero, geo_capacity.C_γ[idx])
            centroid = geo_capacity.C_γ[idx]
            distance = sqrt((centroid[1] - center_x)^2 + 
                           (centroid[2] - center_y)^2 + 
                           (centroid[3] - center_z)^2)
            
            # Interface centroids should be on the sphere surface
            @test isapprox(distance, radius, atol=0.1)
            
            centroids_sum .+= [centroid[1], centroid[2], centroid[3]]
            count_valid += 1
        end
    end
    
    # Average of interface centroids should be close to sphere center
    if count_valid > 0
        avg_centroid = centroids_sum ./ count_valid
        @test isapprox(avg_centroid[1], center_x, atol=0.1)
        @test isapprox(avg_centroid[2], center_y, atol=0.1)
        @test isapprox(avg_centroid[3], center_z, atol=0.1)
    end
end

@testset "1D Capacity" begin
    nx = 10
    lx = 2.0
    x0 = 0.0
    mesh = Mesh((nx,), (lx,), (x0,))
    Φ(X) = sqrt(X[1]^2) - 0.5
    LS(x,_=0) = sqrt(x^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    @test capacity.mesh == mesh
    @test capacity.body == LS
    @test length(capacity.A) == 1
    @test length(capacity.B) == 1
    @test length(capacity.W) == 1
end

@testset "2D Capacity" begin
    nx, ny = 10, 10
    lx, ly = 2.0, 2.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = sqrt(x^2 + y^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    @test capacity.mesh == mesh
    @test capacity.body == LS
    @test length(capacity.A) == 2
    @test length(capacity.B) == 2
    @test length(capacity.W) == 2
end

@testset "3D Capacity" begin
    nx, ny, nz = 10, 10, 10
    lx, ly, lz = 2.0, 2.0, 2.0
    x0, y0, z0 = 0.0, 0.0, 0.0
    mesh = Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2 + X[3]^2) - 0.5
    LS(x,y,z) = sqrt(x^2 + y^2 + z^2) - 0.5
    capacity = Capacity(LS, mesh, method="VOFI")
    @test capacity.mesh == mesh
    @test capacity.body == LS
    @test length(capacity.A) == 3
    @test length(capacity.B) == 3
    @test length(capacity.W) == 3
end

@testset "Interface Centroids 1D" begin
    nx = 20
    lx = 1.0
    x0 = 0.0
    mesh = Mesh((nx,), (lx,), (x0,))
    
    # Circle centered at 0.5 with radius 0.3
    center = 0.5
    radius = 0.3
    LS(x,_=0) = abs(x - center) - radius
    
    # Test with centroids computed
    capacity = Capacity(LS, mesh, method="VOFI", compute_centroids=true)
    @test !isempty(capacity.C_γ)
    
    # Expected interface positions
    expected_left = center - radius  # 0.2
    expected_right = center + radius  # 0.8
    
    # Find cells containing interfaces and check positions
    has_interface = findall(x -> x > 0, diag(capacity.Γ))
    @test length(has_interface) == 2  # Should have 2 interfaces
    
    # Get interface centroids for cells with interfaces
    centroids = [capacity.C_γ[i][1] for i in has_interface]
    
    # Check that the interface positions are approximately correct
    # (within one cell width, which is 0.05 for 20 cells on [0,1])
    @test any(c -> abs(c - expected_left) < 0.05, centroids)
    @test any(c -> abs(c - expected_right) < 0.05, centroids)
    
    # Test with centroids disabled
    capacity_no_centroids = Capacity(LS, mesh, method="VOFI", compute_centroids=false)
    @test isempty(capacity_no_centroids.C_γ)
end

@testset "Interface Centroids 2D" begin
    nx, ny = 30, 30
    lx, ly = 1.0, 1.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    
    # Circle centered at (0.5, 0.5) with radius 0.3
    center_x, center_y = 0.51, 0.51
    radius = 0.3
    LS(x, y, _=0) = sqrt((x - center_x)^2 + (y - center_y)^2) - radius
    
    # Compute capacity
    capacity = Capacity(LS, mesh, method="VOFI", compute_centroids=true)
    @test !isempty(capacity.C_γ)
    
    # Find cells containing interfaces using cell_types
    cut_cells = findall(x -> x == -1, capacity.cell_types)
    @test length(cut_cells) > 0
    
    # Verify that interface centroids lie approximately on the circle
    for idx in cut_cells
        centroid = capacity.C_γ[idx]
        distance = sqrt((centroid[1] - center_x)^2 + (centroid[2] - center_y)^2)
        # Check if centroid is close to circle radius (with tolerance)
        @test abs(distance - radius) < 0.05
    end
    
    # Test consistency between cell_types and interface norm matrix
    norm_interfaces = findall(x -> x > 0, diag(capacity.Γ))
    @test sort(cut_cells) == sort(norm_interfaces)
end

@testset "Interface Centroids 3D" begin
    nx, ny, nz = 10, 10, 10  # Lower resolution for 3D
    lx, ly, lz = 1.0, 1.0, 1.0
    x0, y0, z0 = 0.0, 0.0, 0.0
    mesh = Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))
    
    # Sphere centered at (0.5, 0.5, 0.5) with radius 0.3
    center_x, center_y, center_z = 0.5, 0.5, 0.5
    radius = 0.3
    LS(x, y, z) = sqrt((x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2) - radius
    
    # Compute capacity
    capacity = Capacity(LS, mesh, method="VOFI", compute_centroids=true)
    @test !isempty(capacity.C_γ)
    
    # Find cells containing interfaces using cell_types
    cut_cells = findall(x -> x == -1, capacity.cell_types)
    @test length(cut_cells) > 0
    
    # Verify interface centroid behavior - centroids should be close to sphere surface
    for idx in cut_cells
        centroid = capacity.C_γ[idx]
        distance = sqrt(sum((centroid .- [center_x, center_y, center_z]).^2))
        # Check if centroid is close to sphere radius (with larger tolerance for coarser grid)
        @test abs(distance - radius) < 0.1
    end
end

@testset "1D ImplicitIntegration" begin
    nx = 10
    lx = 2.0
    x0 = 0.0
    mesh = Mesh((nx,), (lx,), (x0,))
    
    # Circle centered at 0.5 with radius 0.3
    center = 0.5
    radius = 0.3
    LS(x,_=0) = abs(x - center) - radius
    
    # Scalar version for ImplicitIntegration
    Φ(x,_=0) = abs(x - center) - radius
    
    capacity = Capacity(Φ, mesh, method="ImplicitIntegration")
    
    # Basic structure tests
    @test capacity.mesh == mesh
    @test length(capacity.A) == 1
    @test length(capacity.B) == 1
    @test length(capacity.W) == 1
    
    # Find cut cells
    cut_cells = findall(x -> x == -1, capacity.cell_types)
    @test length(cut_cells) == 2  # Should have 2 interfaces
    
    # Verify volumes
    fluid_cells = findall(x -> x == 1, capacity.cell_types)
    solid_cells = findall(x -> x == 0, capacity.cell_types)
    
    # Check that fluid cells have volume near 1
    for idx in fluid_cells
        @test isapprox(capacity.V[idx, idx], lx/nx, atol=1e-10)
    end
    
    # Check that solid cells have volume near 0
    for idx in solid_cells
        @test isapprox(capacity.V[idx, idx], 0.0, atol=1e-10)
    end
    
    # Test centroids computation
    @test !isempty(capacity.C_γ)
 
end

@testset "2D ImplicitIntegration" begin
    nx, ny = 20, 20
    lx, ly = 1.0, 1.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    
    # Circle centered at (0.5, 0.5) with radius 0.3
    center_x, center_y = 0.5, 0.5
    radius = 0.3
    
    # Scalar version for ImplicitIntegration
    Φ(x,y,_=0) = sqrt((x - center_x)^2 + (y - center_y)^2) - radius
    
    capacity = Capacity(Φ, mesh, method="ImplicitIntegration")
    
    # Basic structure tests
    @test length(capacity.A) == 2
    @test length(capacity.B) == 2
    @test length(capacity.W) == 2
    
    # Find cut cells
    cut_cells = findall(x -> x == -1, capacity.cell_types)
    @test length(cut_cells) > 0
    
    # Verify centroids are on interface
    for idx in cut_cells
        if !all(isnan, capacity.C_γ[idx])  # Skip any NaN centroids
            centroid = capacity.C_γ[idx]
            distance = sqrt((centroid[1] - center_x)^2 + (centroid[2] - center_y)^2)
            @test isapprox(distance, radius, atol=0.05)
        end
    end
    
    # Check volume conservation
    total_volume = sum(diag(capacity.V))
    expected_volume = π * radius^2  # Circle area
    @test isapprox(total_volume, expected_volume, rtol=0.05)
    
    # Check consistency between interface norm and cell types
    norm_interfaces = findall(x -> x > 0, diag(capacity.Γ))
    @test sort(cut_cells) == sort(norm_interfaces)
end

@testset "3D ImplicitIntegration" begin
    nx, ny, nz = 10, 10, 10  # Lower resolution for 3D
    lx, ly, lz = 1.0, 1.0, 1.0
    x0, y0, z0 = 0.0, 0.0, 0.0
    mesh = Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))
    
    # Sphere centered at (0.5, 0.5, 0.5) with radius 0.3
    center_x, center_y, center_z = 0.5, 0.5, 0.5
    radius = 0.3
    
    # Scalar version for ImplicitIntegration
    Φ(x,y,z) = sqrt((x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2) - radius
    
    capacity = Capacity(Φ, mesh, method="ImplicitIntegration")
    
    # Basic structure tests
    @test length(capacity.A) == 3
    @test length(capacity.B) == 3
    @test length(capacity.W) == 3
    
    # Find cut cells
    cut_cells = findall(x -> x == -1, capacity.cell_types)
    @test length(cut_cells) > 0
    
    # Verify some interface properties
    interface_area = sum(diag(capacity.Γ))
    expected_area = 4 * π * radius^2  # Sphere surface area
    @test isapprox(interface_area, expected_area, rtol=0.1)
    
    # Check volume conservation
    total_volume = sum(diag(capacity.V))
    expected_volume = (4/3) * π * radius^3  # Sphere volume
    @test isapprox(total_volume, expected_volume, rtol=0.1)
    
    # Test comparison between methods
    LS(x, y, z) = sqrt((x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2) - radius
    vofi_capacity = Capacity(LS, mesh, method="VOFI")
    
    # Volumes from both methods should be similar
    vofi_volume = sum(diag(vofi_capacity.V))
    implicit_volume = sum(diag(capacity.V))
    @test isapprox(vofi_volume, implicit_volume, rtol=0.05)
end

@testset "ImplicitIntegration vs VOFI Single Circle Comparison" begin
    nx, ny = 20, 20
    mesh = Mesh((nx, ny), (1.0, 1.0), (0.0, 0.0))
    
    # Define a simple circle shape
    center_x, center_y = 0.5, 0.5
    radius = 0.3

    
    # Scalar version for VOFI
    function single_circle_scalar(x, y, _=0)
        return sqrt((x - center_x)^2 + (y - center_y)^2) - radius
    end
    
    # Calculate capacities using both methods
    implicit_capacity = Capacity(single_circle_scalar, mesh, method="ImplicitIntegration")
    vofi_capacity = Capacity(single_circle_scalar, mesh, method="VOFI")
    
    # Volumes should be similar
    implicit_vol = sum(diag(implicit_capacity.V))
    vofi_vol = sum(diag(vofi_capacity.V))
    @test isapprox(implicit_vol, vofi_vol, rtol=0.05)
    
    # Interface lengths should be similar
    implicit_interface = sum(diag(implicit_capacity.Γ))
    vofi_interface = sum(diag(vofi_capacity.Γ))
    @test isapprox(implicit_interface, vofi_interface, rtol=0.1)
    
    # Cell type classification should be consistent
    implicit_cut = count(x -> x == -1, implicit_capacity.cell_types)
    vofi_cut = count(x -> x == -1, vofi_capacity.cell_types)
    @test isapprox(implicit_cut, vofi_cut, rtol=0.2)
    
    # Face capacities (A) should be similar
    for d in 1:2
        implicit_a = sum(diag(implicit_capacity.A[d]))
        vofi_a = sum(diag(vofi_capacity.A[d]))
        @test isapprox(implicit_a, vofi_a, rtol=0.1)
    end
    
    # Compare with analytical values
    expected_area = π * radius^2
    @test isapprox(implicit_vol, expected_area, rtol=0.05)
    @test isapprox(vofi_vol, expected_area, rtol=0.05)
    
    expected_perimeter = 2 * π * radius
    @test isapprox(implicit_interface, expected_perimeter, rtol=0.1)
    @test isapprox(vofi_interface, expected_perimeter, rtol=0.1)
end
