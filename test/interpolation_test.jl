using Penguin
using Test

@testset "Interpolation Tests" begin
    # Setup a simple test case
    nx = 5
    lx = 1.0
    dx = lx/nx
    x_mesh = collect(0:dx:lx)
    
    @testset "Basic Functions" begin
        # Constant function
        H_const = fill(2.0, nx)
        h_lin = lin_interpol(x_mesh, H_const)
        h_quad = quad_interpol(x_mesh, H_const)
        h_cubic = cubic_interpol(x_mesh, H_const)
        
        @test h_lin(0.0) ≈ 2.0 atol=1e-10
        @test h_lin(0.5) ≈ 2.0 atol=1e-10
        @test h_quad(0.0) ≈ 2.0 atol=1e-10
        @test h_cubic(0.5) ≈ 2.0 atol=1e-10
        
        # Linear function
        H_lin = [(i-0.5)*dx for i in 1:nx]  # Linear function x
        h_lin = lin_interpol(x_mesh, H_lin)
        h_quad = quad_interpol(x_mesh, H_lin)
        h_cubic = cubic_interpol(x_mesh, H_lin)
        
        for x in [0.1, 0.3, 0.7, 0.9]
            @test h_lin(x) ≈ x atol=0.1  # Linear interp has larger error
            @test h_quad(x) ≈ x atol=0.1
            @test h_cubic(x) ≈ x atol=0.1
        end
    end
    
    @testset "Extrapolation" begin
        # Simple dataset
        H_values = [1.0, 2.0, 1.0, 3.0, 2.0]
        h_lin = lin_interpol(x_mesh, H_values)
        h_quad = quad_interpol(x_mesh, H_values)
        h_cubic = cubic_interpol(x_mesh, H_values)
        
        # Test extrapolation outside domain
        @test h_lin(-0.1) ≈ h_lin(0.0) - 0.1*((h_lin(dx) - h_lin(0.0))/dx) atol=1e-10
        @test h_lin(lx+0.1) ≈ h_lin(lx) + 0.1*((h_lin(lx) - h_lin(lx-dx))/dx) atol=1e-10
        
        # Test with extrapolation disabled
        h_lin_no_ext = lin_interpol(x_mesh, H_values, extrapolate=false)
        h_quad_no_ext = quad_interpol(x_mesh, H_values, extrapolate=false)
        h_cubic_no_ext = cubic_interpol(x_mesh, H_values, extrapolate=false)
  
    end
    
    @testset "Continuity" begin
        # Arbitrary function with discontinuous appearance
        H_values = [0.0, 5.0, 2.0, 1.0, 4.0]
        h_lin = lin_interpol(x_mesh, H_values)
        h_quad = quad_interpol(x_mesh, H_values)
        h_cubic = cubic_interpol(x_mesh, H_values)
        
        # Check continuity at interior points
        for i in 1:nx-1
            x = x_mesh[i+1]
            h = 1e-10
            
            # Value continuity
            @test h_lin(x-h) ≈ h_lin(x+h) atol=1e-2
            @test h_quad(x-h) ≈ h_quad(x+h) atol=1e-2
            @test h_cubic(x-h) ≈ h_cubic(x+h) atol=1e-2
            
            # Derivative continuity (quad and cubic)
            quad_left = (h_quad(x) - h_quad(x-h))/h
            quad_right = (h_quad(x+h) - h_quad(x))/h
            @test quad_left ≈ quad_right atol=1e-2
            
            cubic_left = (h_cubic(x) - h_cubic(x-h))/h
            cubic_right = (h_cubic(x+h) - h_cubic(x))/h
            @test cubic_left ≈ cubic_right atol=1e-2
        end
    end
    
    @testset "Random Values" begin
        # Test with the specific values from the original test
        H_values = [0.25230897722466805, 0.3361875379944873, 0.7283705636359846, 
                   0.6249949877617716, 0.03991558949506924]
        x_mesh = collect(0:dx:lx)
        
        h_lin = lin_interpol(x_mesh, H_values)
        h_quad = quad_interpol(x_mesh, H_values)
        h_cubic = cubic_interpol(x_mesh, H_values)
        
        # Test at specific points
        x_test = 0.5 * (dx/2 + dx + dx/2)
        @test h_lin(x_test) ≈ 0.3361875379944873 atol=0.2
        
        # Test extrapolation
        @test h_lin(-dx) != 0.0  # Should extrapolate
        @test h_quad(lx + dx) != 0.0  # Should extrapolate
    end

    @testset "Bilinear Interpolation 3D" begin
        # Test bilinear interpolation for 3D interface tracking
        y_coords = [0.0, 0.5, 1.0]
        z_coords = [0.0, 0.5, 1.0]
        
        # Constant function
        values_const = fill(2.0, 3, 3)
        interp_const = bilinear_interpolation_3d(y_coords, z_coords, values_const)
        @test interp_const(0.0, 0.0) ≈ 2.0 atol=1e-10
        @test interp_const(0.5, 0.5) ≈ 2.0 atol=1e-10
        @test interp_const(0.25, 0.75) ≈ 2.0 atol=1e-10
        
        # Linear function in y: f(y,z) = y
        values_y = [0.0 0.0 0.0; 0.5 0.5 0.5; 1.0 1.0 1.0]
        interp_y = bilinear_interpolation_3d(y_coords, z_coords, values_y)
        @test interp_y(0.0, 0.5) ≈ 0.0 atol=1e-10
        @test interp_y(0.5, 0.5) ≈ 0.5 atol=1e-10
        @test interp_y(1.0, 0.5) ≈ 1.0 atol=1e-10
        @test interp_y(0.25, 0.5) ≈ 0.25 atol=1e-10
        
        # Linear function in z: f(y,z) = z
        values_z = [0.0 0.5 1.0; 0.0 0.5 1.0; 0.0 0.5 1.0]
        interp_z = bilinear_interpolation_3d(y_coords, z_coords, values_z)
        @test interp_z(0.5, 0.0) ≈ 0.0 atol=1e-10
        @test interp_z(0.5, 0.5) ≈ 0.5 atol=1e-10
        @test interp_z(0.5, 1.0) ≈ 1.0 atol=1e-10
        @test interp_z(0.5, 0.25) ≈ 0.25 atol=1e-10
        
        # Bilinear function: f(y,z) = y + z
        values_yz = [0.0 0.5 1.0; 0.5 1.0 1.5; 1.0 1.5 2.0]
        interp_yz = bilinear_interpolation_3d(y_coords, z_coords, values_yz)
        @test interp_yz(0.0, 0.0) ≈ 0.0 atol=1e-10
        @test interp_yz(0.5, 0.5) ≈ 1.0 atol=1e-10
        @test interp_yz(1.0, 1.0) ≈ 2.0 atol=1e-10
        @test interp_yz(0.25, 0.25) ≈ 0.5 atol=1e-10
    end
end