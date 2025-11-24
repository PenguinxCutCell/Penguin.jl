using Test
using LinearAlgebra
using LibGEOS
using Penguin
using CairoMakie
using Statistics
using SparseArrays

"""
Compare space-time volumes computed using Front Tracking and VOFI with 3D level-set (x,y,t)
"""
function compare_spacetime_volumes_vofi3d()
    # Parameters
    nx, ny = 20, 20
    lx, ly = 2.0, 2.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))
    dt = 1.0  # Time step
    
    # Create figure
    fig = Figure(size=(1500, 1000))
    
    # Choose test case: expanding circle
    center_x, center_y = 1.0, 1.0
    radius_n = 0.4
    radius_np1 = 0.5
    
    # Create front trackers for both time steps
    front_n = FrontTracker()
    front_np1 = FrontTracker()
    
    create_circle!(front_n, center_x, center_y, radius_n, 32)
    create_circle!(front_np1, center_x, center_y, radius_np1, 32)
    
    # Define equivalent level-set functions for both time steps
    ls_n(x, y, _=0.0) = sqrt((x - center_x)^2 + (y - center_y)^2) - radius_n
    ls_np1(x, y, _=0.0) = sqrt((x - center_x)^2 + (y - center_y)^2) - radius_np1
    
    # Define 3D space-time level-set function
    # This interpolates between ls_n and ls_np1 based on t
    function spacetime_ls(x, y, t)
        # Normalize t to [0,1] range relative to dt
        t_ratio = t / dt
        # Linear interpolation between level sets
        return (1.0 - t_ratio) * ls_n(x, y) + t_ratio * ls_np1(x, y)
    end
    
    # Create a 3D mesh with t as the third dimension
    # For VOFI, we'll use a 3D mesh with one cell in the t direction
    nx3d, ny3d, nt = nx, ny, 1
    lx3d, ly3d, lt = lx, ly, dt
    mesh3d = Penguin.Mesh((nx3d, ny3d, nt), (lx3d, ly3d, lt), (0.0, 0.0, 0.0))
    
    # Compute front tracking space-time volumes
    ft_st_volumes = compute_spacetime_volumes(mesh, front_n, front_np1, dt)
    
    # Compute VOFI space-time volumes using 3D level-set
    vofi_capacity_3d = Capacity(spacetime_ls, mesh3d, method="VOFI")
    
    # Extract volume capacities from the 3D VOFI computation
    V_vofi_3d = Array(SparseArrays.diag(vofi_capacity_3d.V))
    vofi_V_dense_3d = reshape(V_vofi_3d, (nx+1, ny+1, nt+1))
    
    # Sum over the time dimension to get space-time volumes
    vofi_st_volumes = dropdims(sum(vofi_V_dense_3d, dims=3), dims=3) * dt
    
    # ----- VISUALIZATION -----
    # Extract cell properties at both time steps for visualization
    _, volumes_n, _, _, cell_types_n = fluid_cell_properties(mesh, front_n)
    _, volumes_np1, _, _, cell_types_np1 = fluid_cell_properties(mesh, front_np1)
    
    # Row 1: Initial and final states
    ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Initial State (t=0)",
               xlabel="x", ylabel="y")
    ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="Final State (t=dt)",
               xlabel="x", ylabel="y")
    
    # Create colormap for cell types
    cell_type_cmap = Dict(
        0 => :white,     # Empty
        1 => :skyblue,   # Full
        -1 => :orange    # Cut
    )
    
    # Extract mesh properties
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    
    # Initial state
    for i in 1:nx
        for j in 1:ny
            cell_type = cell_types_n[i, j]
            color = cell_type_cmap[cell_type]
            opacity = cell_type == 0 ? 0.0 : 0.5
            
            poly!(ax1, [
                Point2f(x_nodes[i], y_nodes[j]),
                Point2f(x_nodes[i+1], y_nodes[j]),
                Point2f(x_nodes[i+1], y_nodes[j+1]),
                Point2f(x_nodes[i], y_nodes[j+1])
            ], color=(color, opacity))
        end
    end
    
    # Draw grid lines
    for x in x_nodes
        lines!(ax1, [x, x], [minimum(y_nodes), maximum(y_nodes)], color=:gray, linewidth=0.5)
    end
    for y in y_nodes
        lines!(ax1, [minimum(x_nodes), maximum(x_nodes)], [y, y], color=:gray, linewidth=0.5)
    end
    
    # Draw interface at t=0
    markers_n = get_markers(front_n)
    lines!(ax1, first.(markers_n), last.(markers_n), color=:blue, linewidth=2)
    
    # Final state
    for i in 1:nx
        for j in 1:ny
            cell_type = cell_types_np1[i, j]
            color = cell_type_cmap[cell_type]
            opacity = cell_type == 0 ? 0.0 : 0.5
            
            poly!(ax2, [
                Point2f(x_nodes[i], y_nodes[j]),
                Point2f(x_nodes[i+1], y_nodes[j]),
                Point2f(x_nodes[i+1], y_nodes[j+1]),
                Point2f(x_nodes[i], y_nodes[j+1])
            ], color=(color, opacity))
        end
    end
    
    # Draw grid lines
    for x in x_nodes
        lines!(ax2, [x, x], [minimum(y_nodes), maximum(y_nodes)], color=:gray, linewidth=0.5)
    end
    for y in y_nodes
        lines!(ax2, [minimum(x_nodes), maximum(x_nodes)], [y, y], color=:gray, linewidth=0.5)
    end
    
    # Draw interface at t=dt
    markers_np1 = get_markers(front_np1)
    lines!(ax2, first.(markers_np1), last.(markers_np1), color=:red, linewidth=2)
    
    # Row 2: Space-time volumes
    ax3 = Axis(fig[2, 1], aspect=DataAspect(), title="Front Tracking Space-Time Volumes",
               xlabel="x", ylabel="y")
    ax4 = Axis(fig[2, 2], aspect=DataAspect(), title="VOFI 3D Level-Set Space-Time Volumes",
               xlabel="x", ylabel="y")
    ax5 = Axis(fig[2, 3], aspect=DataAspect(), title="Relative Difference",
               xlabel="x", ylabel="y")
    
    # Prepare cell centers for heatmap
    cell_centers_x = [(x_nodes[i] + x_nodes[i+1])/2 for i in 1:nx]
    cell_centers_y = [(y_nodes[j] + y_nodes[j+1])/2 for j in 1:ny]
    
    # Set fixed color range for better comparison
    max_vol = max(maximum(ft_st_volumes[1:nx, 1:ny]), maximum(vofi_st_volumes[1:nx, 1:ny]))
    
    # Plot heatmaps
    hm1 = heatmap!(ax3, cell_centers_x, cell_centers_y, ft_st_volumes[1:nx, 1:ny]', 
                  colormap=:viridis, colorrange=(0, max_vol))
    cb1 = Colorbar(fig[2, 4], hm1, label="Volume")
    
    hm2 = heatmap!(ax4, cell_centers_x, cell_centers_y, vofi_st_volumes[1:nx, 1:ny]', 
                  colormap=:viridis, colorrange=(0, max_vol))
    cb2 = Colorbar(fig[2, 5], hm2, label="Volume")
    
    # Draw both interfaces on the space-time plots
    lines!(ax3, first.(markers_n), last.(markers_n), color=:blue, linewidth=1, linestyle=:dash)
    lines!(ax3, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1, linestyle=:dash)
    
    lines!(ax4, first.(markers_n), last.(markers_n), color=:blue, linewidth=1, linestyle=:dash)
    lines!(ax4, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1, linestyle=:dash)
    
    # Calculate relative difference
    rel_diff = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            if abs(vofi_st_volumes[i, j]) > 1e-10
                rel_diff[i, j] = abs(ft_st_volumes[i, j] - vofi_st_volumes[i, j]) / abs(vofi_st_volumes[i, j])
            else
                rel_diff[i, j] = abs(ft_st_volumes[i, j]) > 1e-10 ? 1.0 : 0.0
            end
        end
    end
    
    # Plot relative difference with log scale to better visualize small differences
    log_rel_diff = log10.(rel_diff .+ 1e-10)  # Add small value to avoid log(0)
    hm3 = heatmap!(ax5, cell_centers_x, cell_centers_y, log_rel_diff', 
                  colormap=:plasma)
    cb3 = Colorbar(fig[2, 6], hm3, label="log10(Relative Difference)")
    
    # Row 3: 1D profiles and cross-sections
    # Create slices through the domain to compare volume profiles
    mid_x = nx ÷ 2
    mid_y = ny ÷ 2
    
    # X-Profile (horizontal slice)
    ax6 = Axis(fig[3, 1], title="Volume Profile at y=$(cell_centers_y[mid_y])",
               xlabel="x", ylabel="Space-Time Volume")
    
    lines!(ax6, cell_centers_x, ft_st_volumes[1:nx, mid_y], color=:blue, label="Front Tracking")
    lines!(ax6, cell_centers_x, vofi_st_volumes[1:nx, mid_y], color=:red, label="VOFI 3D")
    axislegend(ax6)
    
    # Y-Profile (vertical slice)
    ax7 = Axis(fig[3, 2], title="Volume Profile at x=$(cell_centers_x[mid_x])",
               xlabel="y", ylabel="Space-Time Volume")
    
    lines!(ax7, cell_centers_y, ft_st_volumes[mid_x, 1:ny], color=:blue, label="Front Tracking")
    lines!(ax7, cell_centers_y, vofi_st_volumes[mid_x, 1:ny], color=:red, label="VOFI 3D")
    axislegend(ax7)
    
    # Scatter plot of Front-Tracking vs VOFI volumes
    ax8 = Axis(fig[3, 3], title="Front Tracking vs VOFI 3D Volumes",
               xlabel="VOFI Volume", ylabel="Front Tracking Volume")
    
    # Create scatter data
    scatter_x = vec(vofi_st_volumes[1:nx, 1:ny])
    scatter_y = vec(ft_st_volumes[1:nx, 1:ny])
    
    # Plot scatter and ideal line
    max_val = max(maximum(scatter_x), maximum(scatter_y))
    scatter!(ax8, scatter_x, scatter_y, color=:black, markersize=4)
    lines!(ax8, [0, max_val], [0, max_val], color=:red, linestyle=:dash, label="y=x")
    
    # Compute correlation
    correlation = cor(scatter_x, scatter_y)
    text!(ax8, 0.05, 0.95, text="Correlation: $(round(correlation, digits=5))", 
          align=(:left, :top), space=:relative)
    
    # Row 4: Statistics
    ax9 = Axis(fig[4, 1:3], title="Relative Error Statistics",
              xlabel="Statistic", ylabel="Value")
    
    # Calculate error statistics
    error_stats = [
        maximum(rel_diff),
        mean(rel_diff),
        median(rel_diff)
    ]
    
    stat_labels = ["Maximum", "Mean", "Median"]
    
    barplot!(ax9, 1:length(error_stats), error_stats, width=0.6)
    ax9.xticks = (1:length(error_stats), stat_labels)
    
    # Add total volume comparison
    total_ft = sum(ft_st_volumes[1:nx, 1:ny])
    total_vofi = sum(vofi_st_volumes[1:nx, 1:ny])
    total_rel_diff = abs(total_ft - total_vofi) / abs(total_vofi)
    
    Label(fig[4, 4:6], "Total Space-Time Volumes:
           Front Tracking: $(round(total_ft, digits=6))
           VOFI 3D: $(round(total_vofi, digits=6))
           Relative Difference: $(round(100 * total_rel_diff, digits=3))%", 
           tellwidth=false)
    
    # Set overall figure title
    Label(fig[0, :], "Space-Time Volume Comparison: Front Tracking vs VOFI 3D Level-Set", fontsize=20)
    
    # Save figure
    save("spacetime_vofi3d_comparison.png", fig)
    
    return fig, ft_st_volumes, vofi_st_volumes
end

"""
Compare space-time volumes with different test cases
"""
function run_comparison_tests()
    test_cases = [
        # (name, center_n, center_np1, radius_n, radius_np1)
        ("Expanding Circle", (1.0, 1.0), (1.0, 1.0), 0.4, 0.6),
    ]
    
    results = []
    
    for (name, center_n, center_np1, radius_n, radius_np1) in test_cases
        println("Running test case: $name")
        
        # Parameters
        nx, ny = 20, 20
        lx, ly = 2.0, 2.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))
        dt = 0.1
        
        # Create front trackers
        front_n = FrontTracker()
        front_np1 = FrontTracker()
        
        create_circle!(front_n, center_n[1], center_n[2], radius_n, 32)
        create_circle!(front_np1, center_np1[1], center_np1[2], radius_np1, 32)
        
        # Define level-set functions
        ls_n(x, y, _=0.0) = sqrt((x - center_n[1])^2 + (y - center_n[2])^2) - radius_n
        ls_np1(x, y, _=0.0) = sqrt((x - center_np1[1])^2 + (y - center_np1[2])^2) - radius_np1
        
        # Define 3D space-time level-set function
        function spacetime_ls(x, y, t)
            t_ratio = t / dt
            return (1.0 - t_ratio) * ls_n(x, y) + t_ratio * ls_np1(x, y)
        end
        
        # Create 3D mesh
        nx3d, ny3d, nt = nx, ny, 1
        lx3d, ly3d, lt = lx, ly, dt
        mesh3d = Penguin.Mesh((nx3d, ny3d, nt), (lx3d, ly3d, lt), (0.0, 0.0, 0.0))
        
        # Compute volumes
        ft_st_volumes = compute_spacetime_volumes(mesh, front_n, front_np1, dt)
        
        # Compute VOFI space-time volumes
        vofi_capacity_3d = Capacity(spacetime_ls, mesh3d, method="VOFI")
        V_vofi_3d = Array(SparseArrays.diag(vofi_capacity_3d.V))
        vofi_V_dense_3d = reshape(V_vofi_3d, (nx+1, ny+1, nt+1))
        vofi_st_volumes = dropdims(sum(vofi_V_dense_3d, dims=3), dims=3) * dt
        
        # Calculate statistics
        rel_diff = similar(ft_st_volumes[1:nx, 1:ny])
        for i in 1:nx
            for j in 1:ny
                if abs(vofi_st_volumes[i, j]) > 1e-10
                    rel_diff[i, j] = abs(ft_st_volumes[i, j] - vofi_st_volumes[i, j]) / abs(vofi_st_volumes[i, j])
                else
                    rel_diff[i, j] = abs(ft_st_volumes[i, j]) > 1e-10 ? 1.0 : 0.0
                end
            end
        end
        
        total_ft = sum(ft_st_volumes[1:nx, 1:ny])
        total_vofi = sum(vofi_st_volumes[1:nx, 1:ny])
        total_rel_diff = abs(total_ft - total_vofi) / abs(total_vofi)
        
        stats = Dict(
            "name" => name,
            "max_error" => maximum(rel_diff),
            "mean_error" => mean(rel_diff),
            "median_error" => median(rel_diff),
            "total_ft" => total_ft,
            "total_vofi" => total_vofi,
            "total_relative_diff" => total_rel_diff
        )
        
        push!(results, stats)
        
        println("  Max error: $(stats["max_error"])")
        println("  Mean error: $(stats["mean_error"])")
        println("  Total vol (FT): $(stats["total_ft"])")
        println("  Total vol (VOFI): $(stats["total_vofi"])")
        println("  Total relative diff: $(stats["total_relative_diff"] * 100)%")
    end
    
    return results
end

# Run detailed comparison for visualization
fig, ft_vols, vofi_vols = compare_spacetime_volumes_vofi3d()
display(fig)

# Run multiple test cases and compare statistics
results = run_comparison_tests()