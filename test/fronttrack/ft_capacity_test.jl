using Penguin
using LibGEOS
using Statistics
using CairoMakie
using SparseArrays

# Update the difference calculations to use relative error

# Helper function for safe relative error calculation
function safe_relative_error(ft_val, vofi_val; epsilon=1e-10)
    if abs(vofi_val) < epsilon
        # For near-zero vofi values
        return abs(ft_val) < epsilon ? 0.0 : 1.0  # 0% or 100% error
    else
        return abs(ft_val - vofi_val) / abs(vofi_val)
    end
end

# Apply this function element-wise
function relative_error_matrix(ft_mat, vofi_mat; epsilon=1e-10)
    result = zeros(size(ft_mat))
    for i in eachindex(ft_mat)
        result[i] = safe_relative_error(ft_mat[i], vofi_mat[i], epsilon=epsilon)
    end
    return result
end

"""
Compare volume and surface capacity calculations between Front Tracking (LibGEOS) and VOFI
"""
function compare_volume_capacities()
    # Parameters for the test case
    nx, ny = 10, 10  # Reduced resolution for better visualization
    lx, ly = 1.0, 1.0
    
    # Create mesh
    mesh = Penguin.Mesh((nx, ny), (lx, ly))
    
    # Create figure - expanded to accommodate more plots
    fig = Figure(size=(1800, 2400))  # Increased height for more plots
    
    # 1. CIRCLE TEST CASE
    center_x, center_y = 0.55, 0.55
    radius = 0.31
    
    # Create front tracker for circle
    front_circle = FrontTracker()
    create_circle!(front_circle, center_x, center_y, radius, 5000)
    
    # Calculate capacities using Front Tracking
    ft_capacities_circle = compute_capacities(mesh, front_circle)
    
    # Define equivalent level-set function for VOFI
    circle_ls(x, y, _=0.0) = sqrt((x - center_x)^2 + (y - center_y)^2) - radius
    
    # Calculate capacities using VOFI
    vofi_capacity_circle = Capacity(circle_ls, mesh, method="VOFI")
    
    # ----- VOLUME CAPACITIES -----
    # Convert sparse matrix to dense matrix
    V_vofi = Array(SparseArrays.diag(vofi_capacity_circle.V))
    vofi_V_dense_circle = Matrix(reshape(V_vofi, (nx+1, ny+1)))
    
    # Make sure ft_volumes is a dense matrix
    ft_volumes = Matrix(ft_capacities_circle[:volumes])
    
    # Plot volume fractions (row 1)
    ax1 = Axis(fig[1, 1], title="Front Tracking - Volume Fraction", 
            aspect=DataAspect(), xlabel="x", ylabel="y")
    hm1 = heatmap!(ax1, 0:lx/nx:lx, 0:ly/ny:ly, 
                ft_volumes, colormap=:viridis)
    Colorbar(fig[1, 2], hm1)
    
    ax2 = Axis(fig[1, 3], title="VOFI - Volume Fraction", 
            aspect=DataAspect(), xlabel="x", ylabel="y")
    hm2 = heatmap!(ax2, 0:lx/nx:lx, 0:ly/ny:ly, 
                vofi_V_dense_circle, colormap=:viridis)
    Colorbar(fig[1, 4], hm2)
    
        # Calculate and plot relative error
        rel_error_vol = relative_error_matrix(ft_volumes, vofi_V_dense_circle)
        ax3 = Axis(fig[1, 5], title="Relative Error (|FT - VOFI|/|VOFI|)", 
                aspect=DataAspect(), xlabel="x", ylabel="y")
        hm3 = heatmap!(ax3, 0:lx/nx:lx, 0:ly/ny:ly, 
                rel_error_vol, colormap=:viridis)
        Colorbar(fig[1, 6], hm3)

        # Print volume statistics
        println("Circle volume statistics:")
        println("  Max relative error: $(maximum(rel_error_vol))")
        println("  Mean relative error: $(mean(rel_error_vol))")


    # ----- Ax CAPACITIES (row 2) -----
    # Convert VOFI Ax to dense matrix
    vofi_Ax = Array(SparseArrays.diag(vofi_capacity_circle.A[1]))
    vofi_Ax = Matrix(reshape(vofi_Ax, (nx+1, ny+1)))
    
    # Get Front Tracking Ax and ensure it's a dense matrix
    ft_Ax = Matrix(ft_capacities_circle[:Ax])
    
    # Plot Ax capacities
    ax4 = Axis(fig[2, 1], title="Front Tracking - Ax (Vertical Faces)", 
            aspect=DataAspect(), xlabel="x", ylabel="y")
    hm4 = heatmap!(ax4, 0:lx/nx:lx, 0:ly/ny:ly, 
                ft_Ax, colormap=:viridis)
    Colorbar(fig[2, 2], hm4)
    
    ax5 = Axis(fig[2, 3], title="VOFI - Ax (Vertical Faces)", 
            aspect=DataAspect(), xlabel="x", ylabel="y")
    hm5 = heatmap!(ax5, 0:lx/nx:lx, 0:ly/ny:ly, 
                vofi_Ax, colormap=:viridis)
    Colorbar(fig[2, 4], hm5)
        
        # Calculate and plot Ax relative error
        rel_error_Ax = relative_error_matrix(ft_Ax, vofi_Ax)
        ax6 = Axis(fig[2, 5], title="Ax Relative Error (|FT - VOFI|/|VOFI|)", 
                aspect=DataAspect(), xlabel="x", ylabel="y")
        hm6 = heatmap!(ax6, 0:lx/nx:lx, 0:ly/ny:ly, 
                rel_error_Ax, colormap=:viridis)
        Colorbar(fig[2, 6], hm6)

        # Print Ax capacity statistics
        println("\nCircle Ax capacity statistics:")
        println("  Max relative error: $(maximum(rel_error_Ax))")
        println("  Mean relative error: $(mean(rel_error_Ax))")

    # ----- Ay CAPACITIES (row 3) -----
    # Convert VOFI Ay to dense matrix
    vofi_Ay = Array(SparseArrays.diag(vofi_capacity_circle.A[2]))
    vofi_Ay = Matrix(reshape(vofi_Ay, (nx+1, ny+1)))
    
    # Get Front Tracking Ay and ensure it's a dense matrix
    ft_Ay = Matrix(ft_capacities_circle[:Ay])
    
    # Plot Ay capacities
    ax7 = Axis(fig[3, 1], title="Front Tracking - Ay (Horizontal Faces)", 
            aspect=DataAspect(), xlabel="x", ylabel="y")
    hm7 = heatmap!(ax7, 0:lx/nx:lx, 0:ly/ny:ly, 
                ft_Ay, colormap=:viridis)
    Colorbar(fig[3, 2], hm7)
    
    ax8 = Axis(fig[3, 3], title="VOFI - Ay (Horizontal Faces)", 
            aspect=DataAspect(), xlabel="x", ylabel="y")
    hm8 = heatmap!(ax8, 0:lx/nx:lx, 0:ly/ny:ly, 
                vofi_Ay, colormap=:viridis)
    Colorbar(fig[3, 4], hm8)
        
        # Calculate and plot Ay relative error
        rel_error_Ay = relative_error_matrix(ft_Ay, vofi_Ay)
        ax9 = Axis(fig[3, 5], title="Ay Relative Error (|FT - VOFI|/|VOFI|)", 
                aspect=DataAspect(), xlabel="x", ylabel="y")
        hm9 = heatmap!(ax9, 0:lx/nx:lx, 0:ly/ny:ly, 
                rel_error_Ay, colormap=:viridis)
        Colorbar(fig[3, 6], hm9)

        # Print Ay capacity statistics
        println("\nCircle Ay capacity statistics:")
        println("  Max relative error: $(maximum(rel_error_Ay))")
        println("  Mean relative error: $(mean(rel_error_Ay))")

    # ----- Wx CAPACITIES (row 4) -----
    # Convert VOFI Wx to dense matrix
    vofi_Wx = Array(SparseArrays.diag(vofi_capacity_circle.W[1]))
    vofi_Wx = Matrix(reshape(vofi_Wx, (nx+1, ny+1)))
    
    # Get Front Tracking Wx and ensure it's a dense matrix
    ft_Wx = Matrix(ft_capacities_circle[:Wx])
    
    
    # Plot Wx capacities
    ax10 = Axis(fig[4, 1], title="Front Tracking - Wx", 
             aspect=DataAspect(), xlabel="x", ylabel="y")
    hm10 = heatmap!(ax10, 0:lx/nx:lx, 0:ly/ny:ly, 
                 ft_Wx, colormap=:viridis)
    Colorbar(fig[4, 2], hm10)
    
    ax11 = Axis(fig[4, 3], title="VOFI - Wx", 
              aspect=DataAspect(), xlabel="x", ylabel="y")
    hm11 = heatmap!(ax11, 0:lx/nx:lx, 0:ly/ny:ly, 
                  vofi_Wx, colormap=:viridis)
    Colorbar(fig[4, 4], hm11)
    
        # Calculate and plot Wx relative error
        rel_error_Wx = relative_error_matrix(ft_Wx, vofi_Wx)
        ax12 = Axis(fig[4, 5], title="Wx Relative Error (|FT - VOFI|/|VOFI|)", 
                aspect=DataAspect(), xlabel="x", ylabel="y")
        hm12 = heatmap!(ax12, 0:lx/nx:lx, 0:ly/ny:ly, 
                rel_error_Wx, colormap=:viridis)
        Colorbar(fig[4, 6], hm12)

        # Print Wx capacity statistics
        println("\nCircle Wx capacity statistics:")
        nonzero_wx = rel_error_Wx[1:nx, :]
        println("  Max relative error: $(maximum(nonzero_wx))")
        println("  Mean relative error: $(mean(nonzero_wx))")

    # ----- Wy CAPACITIES (row 5) -----
    # Convert VOFI Wy to dense matrix
    vofi_Wy = Array(SparseArrays.diag(vofi_capacity_circle.W[2]))
    vofi_Wy = Matrix(reshape(vofi_Wy, (nx+1, ny+1)))
    
    # Get Front Tracking Wy and ensure it's a dense matrix
    ft_Wy = Matrix(ft_capacities_circle[:Wy])

    
    # Plot Wy capacities
    ax13 = Axis(fig[5, 1], title="Front Tracking - Wy", 
             aspect=DataAspect(), xlabel="x", ylabel="y")
    hm13 = heatmap!(ax13, 0:lx/nx:lx, 0:ly/ny:ly, 
                 ft_Wy, colormap=:viridis)
    Colorbar(fig[5, 2], hm13)
    
    ax14 = Axis(fig[5, 3], title="VOFI - Wy", 
              aspect=DataAspect(), xlabel="x", ylabel="y")
    hm14 = heatmap!(ax14, 0:lx/nx:lx, 0:ly/ny:ly, 
                  vofi_Wy, colormap=:viridis)
    Colorbar(fig[5, 4], hm14)
        
        # Calculate and plot Wy relative error
        rel_error_Wy = relative_error_matrix(ft_Wy, vofi_Wy)
        ax15 = Axis(fig[5, 5], title="Wy Relative Error (|FT - VOFI|/|VOFI|)", 
                aspect=DataAspect(), xlabel="x", ylabel="y")
        hm15 = heatmap!(ax15, 0:lx/nx:lx, 0:ly/ny:ly, 
                rel_error_Wy, colormap=:viridis)
        Colorbar(fig[5, 6], hm15)

        # Print Wy capacity statistics
        println("\nCircle Wy capacity statistics:")
        nonzero_wy = rel_error_Wy[:, 1:ny]
        println("  Max relative error: $(maximum(nonzero_wy))")
        println("  Mean relative error: $(mean(nonzero_wy))")

    # ----- Bx CAPACITIES (row 6) -----
    # Convert VOFI Bx to dense matrix
    vofi_Bx = Array(SparseArrays.diag(vofi_capacity_circle.B[1]))
    vofi_Bx = Matrix(reshape(vofi_Bx, (nx+1, ny+1)))
    
    # Get Front Tracking Bx and ensure it's a dense matrix
    ft_Bx = Matrix(ft_capacities_circle[:Bx])
    
    # Plot Bx capacities
    ax16 = Axis(fig[6, 1], title="Front Tracking - Bx", 
             aspect=DataAspect(), xlabel="x", ylabel="y")
    hm16 = heatmap!(ax16, 0:lx/nx:lx, 0:ly/ny:ly, 
                 ft_Bx, colormap=:viridis)
    Colorbar(fig[6, 2], hm16)
    
    ax17 = Axis(fig[6, 3], title="VOFI - Bx", 
              aspect=DataAspect(), xlabel="x", ylabel="y")
    hm17 = heatmap!(ax17, 0:lx/nx:lx, 0:ly/ny:ly, 
                  vofi_Bx, colormap=:viridis)
    Colorbar(fig[6, 4], hm17)

        # Calculate and plot Bx relative error
        rel_error_Bx = relative_error_matrix(ft_Bx, vofi_Bx)
        ax18 = Axis(fig[6, 5], title="Bx Relative Error (|FT - VOFI|/|VOFI|)", 
                aspect=DataAspect(), xlabel="x", ylabel="y")
        hm18 = heatmap!(ax18, 0:lx/nx:lx, 0:ly/ny:ly, 
                rel_error_Bx, colormap=:viridis)
        Colorbar(fig[6, 6], hm18)

        # Print Bx capacity statistics
        println("\nCircle Bx capacity statistics:")
        println("  Max relative error: $(maximum(rel_error_Bx))")
        println("  Mean relative error: $(mean(rel_error_Bx))")

    # ----- By CAPACITIES (row 7) -----
    # Convert VOFI By to dense matrix
    vofi_By = Array(SparseArrays.diag(vofi_capacity_circle.B[2]))
    vofi_By = Matrix(reshape(vofi_By, (nx+1, ny+1)))
    
    # Get Front Tracking By and ensure it's a dense matrix
    ft_By = Matrix(ft_capacities_circle[:By])
    
    # Plot By capacities
    ax19 = Axis(fig[7, 1], title="Front Tracking - By", 
             aspect=DataAspect(), xlabel="x", ylabel="y")
    hm19 = heatmap!(ax19, 0:lx/nx:lx, 0:ly/ny:ly, 
                 ft_By, colormap=:viridis)
    Colorbar(fig[7, 2], hm19)
    
    ax20 = Axis(fig[7, 3], title="VOFI - By", 
              aspect=DataAspect(), xlabel="x", ylabel="y")
    hm20 = heatmap!(ax20, 0:lx/nx:lx, 0:ly/ny:ly, 
                  vofi_By, colormap=:viridis)
    Colorbar(fig[7, 4], hm20)
    
        # Calculate and plot By relative error
        rel_error_By = relative_error_matrix(ft_By, vofi_By)
        ax21 = Axis(fig[7, 5], title="By Relative Error (|FT - VOFI|/|VOFI|)", 
                aspect=DataAspect(), xlabel="x", ylabel="y")
        hm21 = heatmap!(ax21, 0:lx/nx:lx, 0:ly/ny:ly, 
                rel_error_By, colormap=:viridis)
        Colorbar(fig[7, 6], hm21)

        # Print By capacity statistics
        println("\nCircle By capacity statistics:")
        println("  Max relative error: $(maximum(rel_error_By))")
        println("  Mean relative error: $(mean(rel_error_By))")

    return fig
end

# Run the comparison
fig = compare_volume_capacities()
display(fig)