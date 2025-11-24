using Penguin
using CairoMakie
using SparseArrays
using Statistics

"""
Comparison of 1D space-time capacities: Front Tracking vs VOFI approach
"""

# 1. Define the 1D spatial mesh
nx = 100
lx = 2.0
x0 = 0.0
mesh_1d = Penguin.Mesh((nx,), (lx,), (x0,))
x_nodes = mesh_1d.nodes[1]
dx = lx/nx

# 2. Define time step
dt = 0.1

# 3. Create front trackers at two different time steps (moving interface)
# Linear motion of the interface from x=0.5 to x=1.0
interface_pos_n = 0.51
interface_pos_np1 = 1.01

front_n = FrontTracker1D([interface_pos_n])
front_np1 = FrontTracker1D([interface_pos_np1])

# 4. Compute space-time capacities using Front Tracking
st_capacities = compute_spacetime_capacities_1d(mesh_1d, front_n, front_np1, dt)

# 5. Define a space-time level set function for comparison with VOFI
# Time becomes the second dimension (y)
function spacetime_ls(x, y, z=0)
    # Convert y to normalized time [0,1]
    t_normalized = y / dt
    
    # Linear interpolation of interface position
    interface_t = interface_pos_n + t_normalized * (interface_pos_np1 - interface_pos_n)
    
    # Return signed distance to interface
    return -(x - interface_t)
end

# 6. Create a 2D space-time mesh for VOFI
STmesh = Penguin.SpaceTimeMesh(mesh_1d, [0.0, dt], tag=mesh_1d.tag)

# 7. Compute VOFI capacities on the space-time mesh
vofi_capacity = Capacity(spacetime_ls, STmesh, method="VOFI")

# Extract the Ax component from VOFI (integrated over time)
Ax_vofi = Array(SparseArrays.diag(vofi_capacity.A[1]))
Ax_vofi = reshape(Ax_vofi, (nx+1, 2))[:, 1]  # Extract Ax component at t=0

# 8. Visualize and compare results
fig = Figure(size=(1000, 800), fontsize=12)

# 8.1 Plot showing the interface movement
ax1 = Axis(fig[1, 1:2], 
          title="Interface Movement",
          xlabel="Position (x)",
          ylabel="Time (t)")

# Plot the interface positions
scatter!(ax1, [interface_pos_n], [0.0], color=:blue, markersize=10, label="t = 0")
scatter!(ax1, [interface_pos_np1], [dt], color=:red, markersize=10, label="t = dt")

# Connect with line to show movement
lines!(ax1, [interface_pos_n, interface_pos_np1], [0.0, dt], color=:black, linestyle=:dash)

# Add horizontal lines at both time steps
lines!(ax1, [x0, lx], [0.0, 0.0], color=:blue, linestyle=:dash, alpha=0.5)
lines!(ax1, [x0, lx], [dt, dt], color=:red, linestyle=:dash, alpha=0.5)

# Add legend
axislegend(ax1)

# 8.2 Plot the space-time Ax capacities from both methods
ax2 = Axis(fig[2, 1:2], 
          title="Space-Time Capacities: Ax",
          xlabel="Position (x)",
          ylabel="Capacity Value")

lines!(ax2, x_nodes, st_capacities[:Ax_spacetime], color=:blue, linewidth=2, label="Front Tracking")
lines!(ax2, x_nodes, Ax_vofi, color=:red, linewidth=2, linestyle=:dash, label="VOFI")

# Add vertical markers at interface positions
vlines!(ax2, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5, label="Interface t=0")
vlines!(ax2, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5, label="Interface t=dt")

# Add legend
axislegend(ax2, position=:lt)

# 8.3 Plot the difference between methods
ax3 = Axis(fig[3, 1:2], 
          title="Absolute Difference: |Front Tracking - VOFI|",
          xlabel="Position (x)",
          ylabel="Difference")

abs_diff = abs.(st_capacities[:Ax_spacetime] - Ax_vofi)
lines!(ax3, x_nodes, abs_diff, color=:purple, linewidth=2)

# Add vertical markers at interface positions
vlines!(ax3, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5)
vlines!(ax3, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5)


# 9. Print statistics
mean_diff = mean(abs_diff)
max_diff = maximum(abs_diff)
total_ft = sum(st_capacities[:Ax_spacetime])
total_vofi = sum(Ax_vofi)
rel_total_diff = abs(total_ft - total_vofi) / total_vofi * 100

stats_text = """
Comparison Statistics:
Mean Absolute Difference: $(round(mean_diff, digits=6))
Maximum Absolute Difference: $(round(max_diff, digits=6))
Total Front Tracking: $(round(total_ft, digits=6))
Total VOFI: $(round(total_vofi, digits=6))
Relative Difference in Total: $(round(rel_total_diff, digits=2))%
"""

# Add stats text to the figure
Label(fig[5, 1:2], stats_text, tellwidth=false)

# Save the figure
save("spacetime_1d_comparison.png", fig)

# Display the figure
display(fig)

# Print detailed statistics
println("Space-Time Capacities Comparison (Front Tracking vs VOFI)")
println("=" ^ 50)
println("Mean Absolute Difference: $(mean_diff)")
println("Max Absolute Difference: $(max_diff)")
println("Total Front Tracking: $(total_ft)")
println("Total VOFI: $(total_vofi)")
println("Relative Difference in Total: $(rel_total_diff)%")
println()

# Print table of values at key positions
println("Values at key positions:")
println("=" ^ 30)
println("x\tFront Tracking\tVOFI\tDiff")
println("-" ^ 30)

for i in 1:length(x_nodes)
    # Print values around the interfaces
    if abs(x_nodes[i] - interface_pos_n) < 0.1 || abs(x_nodes[i] - interface_pos_np1) < 0.1
        println("$(round(x_nodes[i], digits=3))\t$(round(st_capacities[:Ax_spacetime][i], digits=6))\t$(round(Ax_vofi[i], digits=6))\t$(round(abs_diff[i], digits=6))")
    end
end

# 10. Compute space-time volumes using Front Tracking
V_spacetime_ft = st_capacities[:V_spacetime]
edge_types = st_capacities[:edge_types]

# 11. Extract volumes from VOFI
# In the SpaceTimeMesh, the volumes are stored in the D component
V_spacetime_vofi = Array(SparseArrays.diag(vofi_capacity.V))
V_spacetime_vofi = reshape(V_spacetime_vofi, (nx+1, 2))[:, 1]  # Extract Ax component at t=0

# 12. Compare volumes
abs_vol_diff = abs.(V_spacetime_ft - V_spacetime_vofi)
cell_centers = mesh_1d.nodes[1]

# Create a new figure for volume comparison
fig2 = Figure(size=(1000, 800), fontsize=12)

# 12.1 Plot the space-time volumes from both methods
ax1 = Axis(fig2[1, 1:2], 
          title="Space-Time Volumes",
          xlabel="Position (x)",
          ylabel="Volume")

scatter!(ax1, cell_centers, V_spacetime_ft, color=:blue, markersize=8, label="Front Tracking")
scatter!(ax1, cell_centers, V_spacetime_vofi, color=:red, markersize=4, label="VOFI")
stairs!(ax1, [x_nodes[1]; cell_centers; x_nodes[end]], [0; V_spacetime_ft; 0], color=:blue, alpha=0.5)
stairs!(ax1, [x_nodes[1]; cell_centers; x_nodes[end]], [0; V_spacetime_vofi; 0], color=:red, alpha=0.3, linestyle=:dash)

# Add vertical markers at interface positions
vlines!(ax1, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5, label="Interface t=0")
vlines!(ax1, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5, label="Interface t=dt")

# Add legend
axislegend(ax1)

# 12.2 Plot the absolute difference in volumes
ax2 = Axis(fig2[2, 1:2], 
          title="Absolute Difference in Space-Time Volumes",
          xlabel="Position (x)",
          ylabel="Difference")

scatter!(ax2, cell_centers, abs_vol_diff, color=:purple, markersize=8)
stairs!(ax2, [x_nodes[1]; cell_centers; x_nodes[end]], [0; abs_vol_diff; 0], color=:purple, alpha=0.5)

# Add vertical markers at interface positions
vlines!(ax2, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5)
vlines!(ax2, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5)

# 12.3 Display edge types
ax3 = Axis(fig2[3, 1:2],
          title="Edge Types for Front Tracking",
          xlabel="Position (x)",
          ylabel="Type")

# Map edge types to numerical values for visualization
edge_type_values = zeros(length(edge_types))
for i in 1:length(edge_types)
    if edge_types[i] == :empty
        edge_type_values[i] = 0
    elseif edge_types[i] == :dead
        edge_type_values[i] = 1
    elseif edge_types[i] == :fresh
        edge_type_values[i] = 2
    else # :full
        edge_type_values[i] = 3
    end
end

scatter!(ax3, x_nodes, edge_type_values, color=edge_type_values, colormap=:plasma, markersize=10)
#text!(ax3, x_nodes, edge_type_values .+ 0.2, text=string.(Symbol.(edge_types)), textsize=8, align=(:center, :bottom))

# 12.4 Print volume statistics
mean_vol_diff = mean(abs_vol_diff)
max_vol_diff = maximum(abs_vol_diff)
total_ft_vol = sum(V_spacetime_ft)
total_vofi_vol = sum(V_spacetime_vofi)
rel_total_vol_diff = abs(total_ft_vol - total_vofi_vol) / total_vofi_vol * 100

vol_stats_text = """
Volume Comparison Statistics:
Mean Absolute Difference: $(round(mean_vol_diff, digits=6))
Maximum Absolute Difference: $(round(max_vol_diff, digits=6))
Total Front Tracking Volume: $(round(total_ft_vol, digits=6))
Total VOFI Volume: $(round(total_vofi_vol, digits=6))
Relative Difference in Total: $(round(rel_total_vol_diff, digits=2))%
"""

Label(fig2[4, 1:2], vol_stats_text, tellwidth=false)

# Save the volume comparison figure
save("spacetime_volume_comparison.png", fig2)

# Display the figure
display(fig2)

# 13. Print detailed volume statistics
println("\nSpace-Time Volume Comparison (Front Tracking vs VOFI)")
println("=" ^ 50)
println("Mean Absolute Difference: $(mean_vol_diff)")
println("Max Absolute Difference: $(max_vol_diff)")
println("Total Front Tracking Volume: $(total_ft_vol)")
println("Total VOFI Volume: $(total_vofi_vol)")
println("Relative Difference in Total: $(rel_total_vol_diff)%")

# Print table of volumes at key positions
println("\nVolume Values at key positions:")
println("=" ^ 40)
println("x (cell center)\tFront Tracking\tVOFI\tDiff")
println("-" ^ 40)

for i in 1:nx
    # Print values around the interfaces
    cell_center = (x_nodes[i] + x_nodes[i+1])/2
    if abs(cell_center - interface_pos_n) < 0.1 || abs(cell_center - interface_pos_np1) < 0.1
        println("$(round(cell_center, digits=3))\t$(round(V_spacetime_ft[i], digits=6))\t$(round(V_spacetime_vofi[i], digits=6))\t$(round(abs_vol_diff[i], digits=6))")
    end
end

# 14. Optional: Visualize the space-time fluid regions
fig3 = Figure(size=(800, 400), fontsize=12)
ax = Axis(fig3[1, 1], 
          title="Space-Time Fluid Regions",
          xlabel="Position (x)",
          ylabel="Time (t)")

# Draw cell boundaries
for i in 1:nx+1
    lines!(ax, [x_nodes[i], x_nodes[i]], [0, dt], color=:gray, alpha=0.5)
end
lines!(ax, [x_nodes[1], x_nodes[end]], [0, 0], color=:gray, alpha=0.5)
lines!(ax, [x_nodes[1], x_nodes[end]], [dt, dt], color=:gray, alpha=0.5)

# Draw fluid areas with color intensity proportional to the normalized volume
for i in 1:nx
    x_min, x_max = x_nodes[i], x_nodes[i+1]
    cell_width = x_max - x_min
    rect = Rect(x_min, 0, cell_width, dt)
    
    # Normalize the volume by the maximum possible volume (cell_width * dt)
    normalized_volume = V_spacetime_ft[i] / (cell_width * dt)
    
    # Draw the rectangle with opacity based on fluid volume
    poly!(ax, rect, color=(:blue, normalized_volume))
end

# Show interface positions
scatter!(ax, [interface_pos_n], [0.0], color=:red, markersize=10)
scatter!(ax, [interface_pos_np1], [dt], color=:red, markersize=10)
lines!(ax, [interface_pos_n, interface_pos_np1], [0.0, dt], color=:red, linestyle=:dash)

# Save and display the third figure
save("spacetime_fluid_regions.png", fig3)
display(fig3)

# 15. Visualize the space-time centroids
fig4 = Figure(size=(800, 400), fontsize=12)
ax = Axis(fig4[1, 1], 
          title="Space-Time Centroids",
          xlabel="Position (x)",
          ylabel="Time (t)")

# Draw cell boundaries
for i in 1:nx+1
    lines!(ax, [x_nodes[i], x_nodes[i]], [0, dt], color=:gray, alpha=0.3)
end
lines!(ax, [x_nodes[1], x_nodes[end]], [0, 0], color=:gray, alpha=0.3)
lines!(ax, [x_nodes[1], x_nodes[end]], [dt, dt], color=:gray, alpha=0.3)

# Extract centroid coordinates
centroids = st_capacities[:ST_centroids]

# Draw fluid areas with color intensity proportional to the normalized volume
for i in 1:nx
    x_min, x_max = x_nodes[i], x_nodes[i+1]
    cell_width = x_max - x_min
    rect = Rect(x_min, 0, cell_width, dt)
    
    # Normalize the volume by the maximum possible volume (cell_width * dt)
    normalized_volume = V_spacetime_ft[i] / (cell_width * dt)
    
    # Draw the rectangle with opacity based on fluid volume
    poly!(ax, rect, color=(:blue, normalized_volume))
    
    # Draw the centroid if the cell has fluid
    if normalized_volume > 0.01
        scatter!(ax, [centroids[i][1]], [centroids[i][2]], color=:red, markersize=6)
        
        # Draw a line connecting the cell center to the centroid
        cell_center_x = (x_min + x_max) / 2
        cell_center_t = dt / 2
        lines!(ax, [cell_center_x, centroids[i][1]], [cell_center_t, centroids[i][2]], 
               color=:black, linewidth=1, alpha=0.5)
    end
end

# Show interface positions
lines!(ax, [interface_pos_n, interface_pos_np1], [0.0, dt], color=:red, linestyle=:dash)
scatter!(ax, [interface_pos_n], [0.0], color=:red, markersize=10)
scatter!(ax, [interface_pos_np1], [dt], color=:red, markersize=10)

# Add legend elements
scatter!(ax, [NaN], [NaN], color=:red, markersize=6, label="Centroid")
lines!(ax, [NaN, NaN], [NaN, NaN], color=:black,  linewidth=1, label="Shift from center")
axislegend(ax, position=:lt)

# Save and display
save("spacetime_centroids.png", fig4)
display(fig4)

# After the existing Ax comparison (around line 121), add:

# Extract the Bx component from VOFI (integrated over time)
# VOFI stores B differently depending on dimension
Bx_vofi = Array(SparseArrays.diag(vofi_capacity.B[1]))
# Reshape to match the mesh structure - extract first time step values
Bx_vofi = reshape(Bx_vofi, (nx+1, 2))[:, 1]

# Create a new figure for Bx comparison
fig_bx = Figure(size=(1000, 800), fontsize=12)

# Plot the space-time Bx capacities from both methods
ax_bx1 = Axis(fig_bx[1, 1:2], 
          title="Space-Time Centerline Capacities: Bx",
          xlabel="Position (x)",
          ylabel="Capacity Value")

# Calculate cell centers for plotting
cell_centers = mesh_1d.nodes[1]

# Plot both methods
scatter!(ax_bx1, cell_centers, st_capacities[:Bx_spacetime], color=:blue, 
        markersize=8, label="Front Tracking")
scatter!(ax_bx1, cell_centers, Bx_vofi, color=:red, 
        markersize=4, label="VOFI")

# Connect points with lines
lines!(ax_bx1, cell_centers, st_capacities[:Bx_spacetime], 
      color=:blue, alpha=0.5)
lines!(ax_bx1, cell_centers, Bx_vofi, 
      color=:red, alpha=0.3, linestyle=:dash)

# Add vertical markers at interface positions
vlines!(ax_bx1, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5, 
       label="Interface t=0")
vlines!(ax_bx1, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5, 
       label="Interface t=dt")

# Add legend
axislegend(ax_bx1, position=:lt)

# Plot the difference between methods
ax_bx2 = Axis(fig_bx[2, 1:2], 
          title="Absolute Difference: |Front Tracking - VOFI| for Bx",
          xlabel="Position (x)",
          ylabel="Difference")

# Calculate absolute difference
abs_diff_bx = abs.(st_capacities[:Bx_spacetime] - Bx_vofi)
scatter!(ax_bx2, cell_centers, abs_diff_bx, color=:purple, markersize=8)
lines!(ax_bx2, cell_centers, abs_diff_bx, color=:purple, alpha=0.5)

# Add vertical markers at interface positions
vlines!(ax_bx2, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5)
vlines!(ax_bx2, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5)

# Print statistics
mean_diff_bx = mean(abs_diff_bx)
max_diff_bx = maximum(abs_diff_bx)
total_ft_bx = sum(st_capacities[:Bx_spacetime])
total_vofi_bx = sum(Bx_vofi)
rel_total_diff_bx = abs(total_ft_bx - total_vofi_bx) / (total_vofi_bx + 1e-10) * 100

bx_stats_text = """
Bx Comparison Statistics:
Mean Absolute Difference: $(round(mean_diff_bx, digits=6))
Maximum Absolute Difference: $(round(max_diff_bx, digits=6))
Total Front Tracking Bx: $(round(total_ft_bx, digits=6))
Total VOFI Bx: $(round(total_vofi_bx, digits=6))
Relative Difference in Total: $(round(rel_total_diff_bx, digits=2))%
"""

# Add stats text to the figure
Label(fig_bx[3, 1:2], bx_stats_text, tellwidth=false)

# Save and display the Bx comparison figure
save("spacetime_bx_comparison.png", fig_bx)
display(fig_bx)

# Print detailed Bx statistics
println("\nSpace-Time Bx Capacities Comparison (Front Tracking vs VOFI)")
println("=" ^ 50)
println("Mean Absolute Difference: $(mean_diff_bx)")
println("Max Absolute Difference: $(max_diff_bx)")
println("Total Front Tracking Bx: $(total_ft_bx)")
println("Total VOFI Bx: $(total_vofi_bx)")
println("Relative Difference in Total: $(rel_total_diff_bx)%")

# Print table of Bx values at key positions
println("\nBx Values at key positions:")
println("=" ^ 40)
println("x (cell center)\tFront Tracking\tVOFI\tDiff")
println("-" ^ 40)

for i in 1:nx
    cell_center = (x_nodes[i] + x_nodes[i+1])/2
    if abs(cell_center - interface_pos_n) < 0.1 || abs(cell_center - interface_pos_np1) < 0.1
        println("$(round(cell_center, digits=3))\t$(round(st_capacities[:Bx_spacetime][i], digits=6))\t$(round(Bx_vofi[i], digits=6))\t$(round(abs_diff_bx[i], digits=6))")
    end
end

# Extract the Wx component from VOFI (integrated over time)
Wx_vofi = Array(SparseArrays.diag(vofi_capacity.W[1]))
# Reshape to match the mesh structure - extract first time step values
Wx_vofi = reshape(Wx_vofi, (nx+1, 2))[:, 1]

# Create a new figure for Wx comparison
fig_wx = Figure(size=(1000, 800), fontsize=12)

# Plot the space-time Wx capacities from both methods
ax_wx1 = Axis(fig_wx[1, 1:2], 
          title="Space-Time Connection Capacities: Wx",
          xlabel="Position (x)",
          ylabel="Capacity Value")

# Plot both methods
scatter!(ax_wx1, x_nodes, st_capacities[:Wx_spacetime], color=:blue, 
        markersize=8, label="Front Tracking")
scatter!(ax_wx1, x_nodes, Wx_vofi, color=:red, 
        markersize=4, label="VOFI")

# Connect points with lines
lines!(ax_wx1, x_nodes, st_capacities[:Wx_spacetime], 
      color=:blue, alpha=0.5)
lines!(ax_wx1, x_nodes, Wx_vofi, 
      color=:red, alpha=0.3, linestyle=:dash)

# Add vertical markers at interface positions
vlines!(ax_wx1, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5, 
       label="Interface t=0")
vlines!(ax_wx1, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5, 
       label="Interface t=dt")

# Add legend
axislegend(ax_wx1, position=:lt)

# Plot the difference between methods
ax_wx2 = Axis(fig_wx[2, 1:2], 
          title="Absolute Difference: |Front Tracking - VOFI| for Wx",
          xlabel="Position (x)",
          ylabel="Difference")

# Calculate absolute difference
abs_diff_wx = abs.(st_capacities[:Wx_spacetime] - Wx_vofi)
scatter!(ax_wx2, x_nodes, abs_diff_wx, color=:purple, markersize=8)
lines!(ax_wx2, x_nodes, abs_diff_wx, color=:purple, alpha=0.5)

# Add vertical markers at interface positions
vlines!(ax_wx2, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5)
vlines!(ax_wx2, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5)

# Print statistics
mean_diff_wx = mean(abs_diff_wx)
max_diff_wx = maximum(abs_diff_wx)
total_ft_wx = sum(st_capacities[:Wx_spacetime])
total_vofi_wx = sum(Wx_vofi)
rel_total_diff_wx = abs(total_ft_wx - total_vofi_wx) / (total_vofi_wx + 1e-10) * 100

wx_stats_text = """
Wx Comparison Statistics:
Mean Absolute Difference: $(round(mean_diff_wx, digits=6))
Maximum Absolute Difference: $(round(max_diff_wx, digits=6))
Total Front Tracking Wx: $(round(total_ft_wx, digits=6))
Total VOFI Wx: $(round(total_vofi_wx, digits=6))
Relative Difference in Total: $(round(rel_total_diff_wx, digits=2))%
"""

# Add stats text to the figure
Label(fig_wx[3, 1:2], wx_stats_text, tellwidth=false)

# Save and display the Wx comparison figure
save("spacetime_wx_comparison.png", fig_wx)
display(fig_wx)

# Print detailed Wx statistics
println("\nSpace-Time Wx Capacities Comparison (Front Tracking vs VOFI)")
println("=" ^ 50)
println("Mean Absolute Difference: $(mean_diff_wx)")
println("Max Absolute Difference: $(max_diff_wx)")
println("Total Front Tracking Wx: $(total_ft_wx)")
println("Total VOFI Wx: $(total_vofi_wx)")
println("Relative Difference in Total: $(rel_total_diff_wx)%")

# Print table of Wx values at key positions
println("\nWx Values at key positions:")
println("=" ^ 40)
println("x (face position)\tFront Tracking\tVOFI\tDiff")
println("-" ^ 40)

# Create a visualization of the Wx connection strengths
fig_wx_viz = Figure(size=(800, 400), fontsize=12)
ax_viz = Axis(fig_wx_viz[1, 1], 
          title="Space-Time Wx Connections",
          xlabel="Position (x)",
          ylabel="Time (t)")

# Draw cell boundaries
for i in 1:nx+1
    lines!(ax_viz, [x_nodes[i], x_nodes[i]], [0, dt], color=:gray, alpha=0.3)
end
lines!(ax_viz, [x_nodes[1], x_nodes[end]], [0, 0], color=:gray, alpha=0.3)
lines!(ax_viz, [x_nodes[1], x_nodes[end]], [dt, dt], color=:gray, alpha=0.3)

# Get the centroids
centroids = st_capacities[:ST_centroids]

# Draw connections between adjacent cells
for i in 2:nx
    # Skip connections with zero capacity
    if st_capacities[:Wx_spacetime][i] <= 0.0
        continue
    end
    
    # Extract centroid coordinates
    x_left = centroids[i-1][1]
    x_right = centroids[i][1]
    t_left = centroids[i-1][2]
    t_right = centroids[i][2]
    
    # Normalize connection strength for visualization
    max_possible_wx = dx * dt
    connection_strength = st_capacities[:Wx_spacetime][i] / max_possible_wx
    
    # Draw line with thickness proportional to connection strength
    lines!(ax_viz, [x_left, x_right], [t_left, t_right], 
           color=:blue, linewidth=3*connection_strength + 1, alpha=0.7)
    
    # Add points at the centroids
    scatter!(ax_viz, [x_left], [t_left], color=:red, markersize=4)
    scatter!(ax_viz, [x_right], [t_right], color=:red, markersize=4)
end

# Show interface positions
lines!(ax_viz, [interface_pos_n, interface_pos_np1], [0.0, dt], color=:red, linestyle=:dash)
scatter!(ax_viz, [interface_pos_n], [0.0], color=:red, markersize=10)
scatter!(ax_viz, [interface_pos_np1], [dt], color=:red, markersize=10)

# Save and display
save("spacetime_wx_connections.png", fig_wx_viz)
display(fig_wx_viz)

# Find positions with largest differences
for i in 1:nx+1
    # Print values near the interfaces or where differences are significant
    if abs(x_nodes[i] - interface_pos_n) < 0.1 || 
       abs(x_nodes[i] - interface_pos_np1) < 0.1 ||
       abs_diff_wx[i] > 0.5 * max_diff_wx
        println("$(round(x_nodes[i], digits=3))\t$(round(st_capacities[:Wx_spacetime][i], digits=6))\t$(round(Wx_vofi[i], digits=6))\t$(round(abs_diff_wx[i], digits=6))")
    end
end