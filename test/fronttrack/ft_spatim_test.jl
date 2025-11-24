using Test
using LinearAlgebra
using LibGEOS
using Penguin
using CairoMakie
using Statistics
using SparseArrays

"""
Test and visualize space-time surface capacities (Ax_st and Ay_st)
"""
function test_spacetime_surfaces()
    # Parameters
    nx, ny = 20, 20
    lx, ly = 2.0, 2.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))
    dt = 1.0
    
    # Create figure
    fig = Figure(size=(1500, 1200))
    
    # Choose test case: expanding circle
    center_x, center_y = 1.0, 1.0
    radius_n = 0.4
    radius_np1 = 0.6
    
    # Create front trackers for both time steps
    front_n = FrontTracker()
    front_np1 = FrontTracker()
    
    create_circle!(front_n, center_x, center_y, radius_n, 32)
    create_circle!(front_np1, center_x, center_y, radius_np1, 32)
    
    # Compute capacities at initial and final times
    Ax_n, Ay_n = compute_surface_capacities(mesh, front_n)
    Ax_np1, Ay_np1 = compute_surface_capacities(mesh, front_np1)
    
    # Compute space-time capacities
    Ax_st, Ay_st = compute_spacetime_surfaces(mesh, front_n, front_np1, dt)
    
    # Also compute volumes for comparison
    V_st = compute_spacetime_volumes(mesh, front_n, front_np1, dt)
    centroids_st = compute_spacetime_centroid(mesh, front_n, front_np1, dt)
    
    # Extract mesh properties
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    
    # Get cell types for visualization
    _, _, _, _, cell_types_n = fluid_cell_properties(mesh, front_n)
    _, _, _, _, cell_types_np1 = fluid_cell_properties(mesh, front_np1)
    
    # Create colormap for cell types
    cell_type_cmap = Dict(
        0 => :white,     # Empty
        1 => :skyblue,   # Full
        -1 => :orange    # Cut
    )

    # Calculate centerline capacities (Bx, By) at initial and final times
    _, _, centroids_x_n, centroids_y_n, _ = fluid_cell_properties(mesh, front_n)
    _, _, Bx_n, By_n = compute_second_type_capacities(mesh, front_n, centroids_x_n, centroids_y_n)
    
    _, _, centroids_x_np1, centroids_y_np1, _ = fluid_cell_properties(mesh, front_np1)
    _, _, Bx_np1, By_np1 = compute_second_type_capacities(mesh, front_np1, centroids_x_np1, centroids_y_np1)
    
    # Compute space-time centerline capacities
    Bx_st, By_st = compute_spacetime_centerline_lengths(mesh, front_n, front_np1, dt)

    # -----------------------------------
    # Row 1: Initial and final states
    # -----------------------------------
    ax1 = Axis(fig[1, 1], aspect=DataAspect(), title="Initial State (t=0)",
               xlabel="x", ylabel="y")
    ax2 = Axis(fig[1, 2], aspect=DataAspect(), title="Final State (t=dt)",
               xlabel="x", ylabel="y")
    
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
    
    # -----------------------------------
    # Row 2: Surface capacities at initial and final times
    # -----------------------------------
    ax3 = Axis(fig[2, 1], aspect=DataAspect(), title="Ax at t=0",
               xlabel="x", ylabel="y")
    ax4 = Axis(fig[2, 2], aspect=DataAspect(), title="Ax at t=dt",
               xlabel="x", ylabel="y")
    ax5 = Axis(fig[2, 3], aspect=DataAspect(), title="Ax Space-Time",
               xlabel="x", ylabel="y")
    
    # Remove unused variables
    # ax_face_x = x_nodes
    # ax_face_y = [(y_nodes[j] + y_nodes[j+1])/2 for j in 1:ny]
    # max_ax = max(maximum(Ax_n), maximum(Ax_np1))
    
    max_ax_st = maximum(Ax_st)
    
    # Draw Ax at t=0
    for i in 1:nx+1
        for j in 1:ny
            if Ax_n[i, j] > 0
                lines!(ax3, [x_nodes[i], x_nodes[i]], 
                      [y_nodes[j], y_nodes[j] + Ax_n[i, j]], 
                      color=:blue, linewidth=2.5)
            end
        end
    end
    
    # Draw Ax at t=dt
    for i in 1:nx+1
        for j in 1:ny
            if Ax_np1[i, j] > 0
                lines!(ax4, [x_nodes[i], x_nodes[i]], 
                      [y_nodes[j], y_nodes[j] + Ax_np1[i, j]], 
                      color=:red, linewidth=2.5)
            end
        end
    end
    
    # Draw Ax Space-Time (using width for visualization)
    for i in 1:nx+1
        for j in 1:ny
            if Ax_st[i, j] > 0
                # Use line width proportional to the space-time capacity
                width = 1.0 + 4.0 * Ax_st[i, j] / max_ax_st
                lines!(ax5, [x_nodes[i], x_nodes[i]], 
                      [y_nodes[j], y_nodes[j+1]], 
                      color=:purple, linewidth=width)
            end
        end
    end
    
    # Draw interfaces
    lines!(ax3, first.(markers_n), last.(markers_n), color=:black, linewidth=1, linestyle=:dash)
    lines!(ax4, first.(markers_np1), last.(markers_np1), color=:black, linewidth=1, linestyle=:dash)
    lines!(ax5, first.(markers_n), last.(markers_n), color=:blue, linewidth=1, linestyle=:dash)
    lines!(ax5, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1, linestyle=:dash)
    
    # -----------------------------------
    # Row 3: Ay capacities
    # -----------------------------------
    ax6 = Axis(fig[3, 1], aspect=DataAspect(), title="Ay at t=0",
               xlabel="x", ylabel="y")
    ax7 = Axis(fig[3, 2], aspect=DataAspect(), title="Ay at t=dt",
               xlabel="x", ylabel="y")
    ax8 = Axis(fig[3, 3], aspect=DataAspect(), title="Ay Space-Time",
               xlabel="x", ylabel="y")
    
    # Remove unused variables
    # ay_face_x = [(x_nodes[i] + x_nodes[i+1])/2 for i in 1:nx]
    # ay_face_y = y_nodes
    # max_ay = max(maximum(Ay_n), maximum(Ay_np1))
    
    max_ay_st = maximum(Ay_st)
    
    # Draw Ay at t=0
    for i in 1:nx
        for j in 1:ny+1
            if Ay_n[i, j] > 0
                lines!(ax6, [x_nodes[i], x_nodes[i] + Ay_n[i, j]], 
                      [y_nodes[j], y_nodes[j]], 
                      color=:blue, linewidth=2.5)
            end
        end
    end
    
    # Draw Ay at t=dt
    for i in 1:nx
        for j in 1:ny+1
            if Ay_np1[i, j] > 0
                lines!(ax7, [x_nodes[i], x_nodes[i] + Ay_np1[i, j]], 
                      [y_nodes[j], y_nodes[j]], 
                      color=:red, linewidth=2.5)
            end
        end
    end
    
    # Draw Ay Space-Time (using width for visualization)
    for i in 1:nx
        for j in 1:ny+1
            if Ay_st[i, j] > 0
                # Use line width proportional to the space-time capacity
                width = 1.0 + 4.0 * Ay_st[i, j] / max_ay_st
                lines!(ax8, [x_nodes[i], x_nodes[i+1]], 
                      [y_nodes[j], y_nodes[j]], 
                      color=:purple, linewidth=width)
            end
        end
    end
    
    # Draw interfaces
    lines!(ax6, first.(markers_n), last.(markers_n), color=:black, linewidth=1, linestyle=:dash)
    lines!(ax7, first.(markers_np1), last.(markers_np1), color=:black, linewidth=1, linestyle=:dash)
    lines!(ax8, first.(markers_n), last.(markers_n), color=:blue, linewidth=1, linestyle=:dash)
    lines!(ax8, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1, linestyle=:dash)
    
    # -----------------------------------
    # Row 4: Space-Time Volumes and Centroids
    # -----------------------------------
    ax9 = Axis(fig[4, 1:3], aspect=DataAspect(), title="Space-Time Volumes and Centroids",
               xlabel="x", ylabel="y")
    
    # Prepare cell centers for heatmap
    cell_centers_x = [(x_nodes[i] + x_nodes[i+1])/2 for i in 1:nx]
    cell_centers_y = [(y_nodes[j] + y_nodes[j+1])/2 for j in 1:ny]
    
    # Draw volume heatmap
    hm = heatmap!(ax9, cell_centers_x, cell_centers_y, V_st[1:nx, 1:ny]', 
                  colormap=:viridis)
    
    # Fix: Store colorbar in _ to avoid unused warning
    _ = Colorbar(fig[4, 4], hm, label="Volume")
    
    # Draw interfaces
    lines!(ax9, first.(markers_n), last.(markers_n), color=:blue, linewidth=1, linestyle=:dash)
    lines!(ax9, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1, linestyle=:dash)
    
    # Draw centroids
    centroid_x = Float64[]
    centroid_y = Float64[]
    centroid_t = Float64[]  # For color
    
    for i in 1:nx
        for j in 1:ny
            if haskey(centroids_st, (i, j)) && V_st[i, j] > 1e-10
                cx, cy, ct = centroids_st[(i, j)]
                push!(centroid_x, cx)
                push!(centroid_y, cy)
                push!(centroid_t, ct / dt)  # Normalize to [0,1]
            end
        end
    end
    
    # Fix: Use a proper colormap function to map time values to colors
    # This fixes the "Can't convert 0.931543 to a colorant" error
    cmap = cgrad(:coolwarm)
    centroid_colors = [cmap[t] for t in centroid_t]
    
    scatter!(ax9, centroid_x, centroid_y, 
             color=centroid_colors,  # Use the pre-computed colors
             markersize=10)
    
    # Add a colorbar for time (store in _ to avoid unused warning)
    _ = Colorbar(fig[4, 5], colorrange=(0, 1), colormap=:coolwarm, 
                label="Normalized Time")
    
    # -----------------------------------
    # Row 5: Profile analysis
    # -----------------------------------
    # Create slices through the domain to compare profiles
    mid_x = nx ÷ 2
    mid_y = ny ÷ 2
    
    ax10 = Axis(fig[5, 1], title="Ax Profile at y=$(cell_centers_y[mid_y])",
                xlabel="x", ylabel="Surface Capacity")
    
    # Plot Ax values at different times
    lines!(ax10, x_nodes, Ax_n[:, mid_y], color=:blue, label="t=0")
    lines!(ax10, x_nodes, Ax_np1[:, mid_y], color=:red, label="t=dt")
    lines!(ax10, x_nodes, Ax_st[:, mid_y] ./ dt, color=:purple, label="Space-Time (mean)")
    axislegend(ax10)
    
    ax11 = Axis(fig[5, 2], title="Ay Profile at x=$(cell_centers_x[mid_x])",
                xlabel="y", ylabel="Surface Capacity")
    
    # Plot Ay values at different times
    lines!(ax11, y_nodes, Ay_n[mid_x, :], color=:blue, label="t=0")
    lines!(ax11, y_nodes, Ay_np1[mid_x, :], color=:red, label="t=dt")
    lines!(ax11, y_nodes, Ay_st[mid_x, :] ./ dt, color=:purple, label="Space-Time (mean)")
    axislegend(ax11)

    # -----------------------------------
    # Row 6: Centerline capacities (Bx, By)
    # -----------------------------------
    ax12 = Axis(fig[6, 1], aspect=DataAspect(), title="Bx at t=0",
               xlabel="x", ylabel="y")
    ax13 = Axis(fig[6, 2], aspect=DataAspect(), title="Bx at t=dt",
               xlabel="x", ylabel="y")
    ax14 = Axis(fig[6, 3], aspect=DataAspect(), title="Bx Space-Time",
               xlabel="x", ylabel="y")
    
    max_bx_st = maximum(Bx_st)
    
    # Draw Bx at t=0
    for i in 1:nx
        for j in 1:ny
            if Bx_n[i, j] > 0
                cx = (x_nodes[i] + x_nodes[i+1])/2
                lines!(ax12, [cx, cx], 
                      [y_nodes[j], y_nodes[j] + Bx_n[i, j]], 
                      color=:blue, linewidth=2.5)
            end
        end
    end
    
    # Draw Bx at t=dt
    for i in 1:nx
        for j in 1:ny
            if Bx_np1[i, j] > 0
                cx = (x_nodes[i] + x_nodes[i+1])/2
                lines!(ax13, [cx, cx], 
                      [y_nodes[j], y_nodes[j] + Bx_np1[i, j]], 
                      color=:red, linewidth=2.5)
            end
        end
    end
    
    # Draw Bx Space-Time (using width for visualization)
    for i in 1:nx
        for j in 1:ny
            if Bx_st[i, j] > 0
                cx = (x_nodes[i] + x_nodes[i+1])/2
                # Use line width proportional to the space-time capacity
                width = 1.0 + 4.0 * Bx_st[i, j] / max_bx_st
                lines!(ax14, [cx, cx], 
                      [y_nodes[j], y_nodes[j+1]], 
                      color=:purple, linewidth=width)
            end
        end
    end
    
    # Draw interfaces on all plots
    lines!(ax12, first.(markers_n), last.(markers_n), color=:black, linewidth=1, linestyle=:dash)
    lines!(ax13, first.(markers_np1), last.(markers_np1), color=:black, linewidth=1, linestyle=:dash)
    lines!(ax14, first.(markers_n), last.(markers_n), color=:blue, linewidth=1, linestyle=:dash)
    lines!(ax14, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1, linestyle=:dash)
    
    # Row 7: By capacities
    ax15 = Axis(fig[7, 1], aspect=DataAspect(), title="By at t=0",
               xlabel="x", ylabel="y")
    ax16 = Axis(fig[7, 2], aspect=DataAspect(), title="By at t=dt",
               xlabel="x", ylabel="y")
    ax17 = Axis(fig[7, 3], aspect=DataAspect(), title="By Space-Time",
               xlabel="x", ylabel="y")
    
    max_by_st = maximum(By_st)
    
    # Draw By at t=0
    for i in 1:nx
        for j in 1:ny
            if By_n[i, j] > 0
                cy = (y_nodes[j] + y_nodes[j+1])/2
                lines!(ax15, [x_nodes[i], x_nodes[i] + By_n[i, j]], 
                      [cy, cy], 
                      color=:blue, linewidth=2.5)
            end
        end
    end
    
    # Draw By at t=dt
    for i in 1:nx
        for j in 1:ny
            if By_np1[i, j] > 0
                cy = (y_nodes[j] + y_nodes[j+1])/2
                lines!(ax16, [x_nodes[i], x_nodes[i] + By_np1[i, j]], 
                      [cy, cy], 
                      color=:red, linewidth=2.5)
            end
        end
    end
    
    # Draw By Space-Time (using width for visualization)
    for i in 1:nx
        for j in 1:ny
            if By_st[i, j] > 0
                cy = (y_nodes[j] + y_nodes[j+1])/2
                # Use line width proportional to the space-time capacity
                width = 1.0 + 4.0 * By_st[i, j] / max_by_st
                lines!(ax17, [x_nodes[i], x_nodes[i+1]], 
                      [cy, cy], 
                      color=:purple, linewidth=width)
            end
        end
    end
    
    # Draw interfaces on all plots
    lines!(ax15, first.(markers_n), last.(markers_n), color=:black, linewidth=1, linestyle=:dash)
    lines!(ax16, first.(markers_np1), last.(markers_np1), color=:black, linewidth=1, linestyle=:dash)
    lines!(ax17, first.(markers_n), last.(markers_n), color=:blue, linewidth=1, linestyle=:dash)
    lines!(ax17, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1, linestyle=:dash)
    
    # Add general statistics
    total_ax_n = sum(Ax_n)
    total_ax_np1 = sum(Ax_np1)
    total_ax_st = sum(Ax_st)
    
    total_ay_n = sum(Ay_n)
    total_ay_np1 = sum(Ay_np1)
    total_ay_st = sum(Ay_st)
    
    total_bx_n = sum(Bx_n)
    total_bx_np1 = sum(Bx_np1)
    total_bx_st = sum(Bx_st)
    
    total_by_n = sum(By_n)
    total_by_np1 = sum(By_np1)
    total_by_st = sum(By_st)
    
    stats_text = "Surface Capacity Totals:
                  Ax at t=0: $(round(total_ax_n, digits=4))
                  Ax at t=dt: $(round(total_ax_np1, digits=4))
                  Ax Space-Time: $(round(total_ax_st, digits=4))
                  
                  Ay at t=0: $(round(total_ay_n, digits=4))
                  Ay at t=dt: $(round(total_ay_np1, digits=4))
                  Ay Space-Time: $(round(total_ay_st, digits=4))
                  
                  Bx at t=0: $(round(total_bx_n, digits=4))
                  Bx at t=dt: $(round(total_bx_np1, digits=4))
                  Bx Space-Time: $(round(total_bx_st, digits=4))
                  
                  By at t=0: $(round(total_by_n, digits=4))
                  By at t=dt: $(round(total_by_np1, digits=4))
                  By Space-Time: $(round(total_by_st, digits=4))
                  
                  Ratio Ax_st/(Ax_n+Ax_np1)*dt/2: $(round(total_ax_st / (total_ax_n + total_ax_np1) * 2 / dt, digits=4))
                  Ratio Ay_st/(Ay_n+Ay_np1)*dt/2: $(round(total_ay_st / (total_ay_n + total_ay_np1) * 2 / dt, digits=4))
                  Ratio Bx_st/(Bx_n+Bx_np1)*dt/2: $(round(total_bx_st / (total_bx_n + total_bx_np1) * 2 / dt, digits=4))
                  Ratio By_st/(By_n+By_np1)*dt/2: $(round(total_by_st / (total_by_n + total_by_np1) * 2 / dt, digits=4))"
                  
                  
    Label(fig[5, 3], stats_text, tellwidth=false)
    
    # Set overall figure title
    Label(fig[0, :], "Space-Time Surface Capacities Analysis", fontsize=20)
    
    # Save figure
    save("spacetime_surfaces_analysis.png", fig)
    
    return fig, Ax_st, Ay_st, Bx_st, By_st, V_st, centroids_st
end
"""
Compare space-time surface capacities with different test cases
"""
function run_surface_tests()
    test_cases = [
        # (name, center_n, center_np1, radius_n, radius_np1)
        ("Expanding Circle", (1.0, 1.0), (1.0, 1.0), 0.4, 0.6),
        ("Moving Circle", (0.8, 1.0), (1.2, 1.0), 0.3, 0.3),
        ("Complex Case", (0.9, 0.9), (1.1, 1.1), 0.5, 0.4)
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
        
        # Compute surface capacities
        Ax_n, Ay_n = compute_surface_capacities(mesh, front_n)
        Ax_np1, Ay_np1 = compute_surface_capacities(mesh, front_np1)
        Ax_st, Ay_st = compute_spacetime_surfaces(mesh, front_n, front_np1, dt)
        
        # Calculate statistics
        total_ax_n = sum(Ax_n)
        total_ax_np1 = sum(Ax_np1)
        total_ax_st = sum(Ax_st)
        ratio_ax = total_ax_st / ((total_ax_n + total_ax_np1) * dt / 2)
        
        total_ay_n = sum(Ay_n)
        total_ay_np1 = sum(Ay_np1)
        total_ay_st = sum(Ay_st)
        ratio_ay = total_ay_st / ((total_ay_n + total_ay_np1) * dt / 2)
        
        stats = Dict(
            "name" => name,
            "total_ax_n" => total_ax_n,
            "total_ax_np1" => total_ax_np1,
            "total_ax_st" => total_ax_st,
            "ratio_ax" => ratio_ax,
            "total_ay_n" => total_ay_n,
            "total_ay_np1" => total_ay_np1,
            "total_ay_st" => total_ay_st,
            "ratio_ay" => ratio_ay
        )
        
        push!(results, stats)
        
        println("  Total Ax (t=0): $(stats["total_ax_n"])")
        println("  Total Ax (t=dt): $(stats["total_ax_np1"])")
        println("  Total Ax (space-time): $(stats["total_ax_st"])")
        println("  Ratio Ax: $(stats["ratio_ax"])")
        println("  Total Ay (t=0): $(stats["total_ay_n"])")
        println("  Total Ay (t=dt): $(stats["total_ay_np1"])")
        println("  Total Ay (space-time): $(stats["total_ay_st"])")
        println("  Ratio Ay: $(stats["ratio_ay"])")
    end
    
    return results
end

# Run the test and visualization
fig, Ax_st, Ay_st, Bx_st, By_st, V_st, centroids_st = test_spacetime_surfaces()
display(fig)

# Run multiple test cases and compare statistics
results = run_surface_tests()