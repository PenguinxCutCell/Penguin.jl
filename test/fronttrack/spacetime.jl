using Penguin
using CairoMakie
using LibGEOS
using SparseArrays
using Statistics
"""
Simplified comparison of space-time capacities using Space-Time Mesh with VOFI
"""

# 1. Define the mesh parameters
nx, ny = 20, 20
lx, ly = 10.0, 10.0
x0, y0 = -5.0, -5.0
dt = 0.1

# Create the spatial mesh
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# 2. Create front trackers at two different time steps (expanding circle)
# Time n: Circle at center
front_n = FrontTracker()
nmarkers = 500  # Number of markers for the circle
radius_n = 2.5
center_x, center_y = 0.0, 0.0
create_circle!(front_n, center_x, center_y, radius_n, nmarkers)

# Time n+1: Circle slightly larger
front_np1 = FrontTracker()
radius_np1 = 2.7
create_circle!(front_np1, center_x, center_y, radius_np1, nmarkers)

# 3. Compute space-time capacities using Front Tracking
ft_spacetime_capacities = compute_spacetime_capacities(mesh, front_n, front_np1, dt)
Ax_st_ft = ft_spacetime_capacities[:Ax_spacetime]
Ay_st_ft = ft_spacetime_capacities[:Ay_spacetime]

# 4. Create a space-time level set function
function spacetime_ls(x, y, t)
    # Linear interpolation between circles
    α = t / dt
    radius_t = (1-α) * radius_n + α * radius_np1
    return sqrt((x - center_x)^2 + (y - center_y)^2) - radius_t
end

# 5. Create a SpaceTimeMesh and compute VOFI capacities directly
# Create time interval [0, dt]
times = [0.0, dt]
st_mesh = Penguin.SpaceTimeMesh(mesh, times)

# Get VOFI capacity on the space-time mesh
vofi_st_capacity = Capacity(spacetime_ls, st_mesh, method="VOFI")

# Extract Ax and Ay components (these should be space-time integrated)
Ax_st_vofi = Array(SparseArrays.diag(vofi_st_capacity.A[1]))
Ax_st_vofi = reshape(Ax_st_vofi, (nx+1, ny+1,2))
Ay_st_vofi = Array(SparseArrays.diag(vofi_st_capacity.A[2]))
Ay_st_vofi = reshape(Ay_st_vofi, (nx+1, ny+1,2))

Ax_st_vofi = Ax_st_vofi[:, :, 1]  # Extract Ax component
Ay_st_vofi = Ay_st_vofi[:, :, 1]  # Extract Ay component

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

# Calculate relative differences between methods
Ax_rel_diff = relative_error_matrix(Ax_st_ft, Ax_st_vofi)
Ay_rel_diff = relative_error_matrix(Ay_st_ft, Ay_st_vofi)

# 6. Create visualization for comparison
fig = Figure(size=(1500, 800))  # Wider figure to accommodate difference plots

# 6.1 Row 1: Show interfaces at both time steps
ax1 = Axis(fig[1, 1:3], title="Interface Movement", 
          xlabel="x", ylabel="y", aspect=DataAspect())

# Plot the interfaces
markers_n = get_markers(front_n)
markers_np1 = get_markers(front_np1)

lines!(ax1, first.(markers_n), last.(markers_n), color=:blue, linewidth=2,
      label="Interface at t=0")
lines!(ax1, first.(markers_np1), last.(markers_np1), color=:red, linewidth=2,
      label="Interface at t=dt")

axislegend(ax1, position=:rt)

# 6.2 Row 2: Compare Ax capacities
ax2 = Axis(fig[2, 1], title="Front Tracking Ax_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax3 = Axis(fig[2, 2], title="VOFI Space-Time Ax_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_diff_x = Axis(fig[2, 3], title="Relative Difference (Ax)",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Get mesh node positions for plotting
x_nodes = mesh.nodes[1]
y_nodes = mesh.nodes[2]

# Set consistent color range for Ax
max_Ax = maximum([maximum(Ax_st_ft), maximum(Ax_st_vofi)])

# Plot Ax capacities
hm_Ax_ft = heatmap!(ax2, x_nodes, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                  Ax_st_ft', colormap=:viridis, colorrange=(0, max_Ax))
Colorbar(fig[2, 4], hm_Ax_ft, label="Ax_st (Front Tracking)")

hm_Ax_vofi = heatmap!(ax3, x_nodes, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                     Ax_st_vofi', colormap=:viridis, colorrange=(0, max_Ax))
Colorbar(fig[2, 5], hm_Ax_vofi, label="Ax_st (VOFI)")

# Plot relative difference for Ax (capped at 0.2 or 20%)
hm_Ax_diff = heatmap!(ax_diff_x, x_nodes, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                     Ax_rel_diff', colormap=:plasma)
Colorbar(fig[2, 6], hm_Ax_diff, label="Relative Difference")

# Add interfaces to the Ax plots
lines!(ax2, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax2, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

lines!(ax3, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax3, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

lines!(ax_diff_x, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_diff_x, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

# 6.3 Row 3: Compare Ay capacities
ax4 = Axis(fig[3, 1], title="Front Tracking Ay_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax5 = Axis(fig[3, 2], title="VOFI Space-Time Ay_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_diff_y = Axis(fig[3, 3], title="Relative Difference (Ay)",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Set consistent color range for Ay
max_Ay = maximum([maximum(Ay_st_ft), maximum(Ay_st_vofi)])

# Plot Ay capacities
hm_Ay_ft = heatmap!(ax4, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes, 
                  Ay_st_ft, colormap=:viridis, colorrange=(0, max_Ay))
Colorbar(fig[3, 4], hm_Ay_ft, label="Ay_st (Front Tracking)")

hm_Ay_vofi = heatmap!(ax5, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes, 
                     Ay_st_vofi, colormap=:viridis, colorrange=(0, max_Ay))
Colorbar(fig[3, 5], hm_Ay_vofi, label="Ay_st (VOFI)")

# Plot relative difference for Ay (capped at 0.2 or 20%)
hm_Ay_diff = heatmap!(ax_diff_y, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes, 
                     Ay_rel_diff, colormap=:plasma)
Colorbar(fig[3, 6], hm_Ay_diff, label="Relative Difference")

# Add interfaces to the Ay plots
lines!(ax4, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax4, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

lines!(ax5, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax5, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

lines!(ax_diff_y, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_diff_y, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

# Add row for statistics of differences
ax_stats = Axis(fig[4, 1:3], title="Relative Error Statistics", 
               xlabel="Relative Error", ylabel="Frequency")

# Create histograms of difference values
hist!(ax_stats, filter(x -> x > 0 && x < 0.5, vec(Ax_rel_diff)), bins=20, color=:blue, label="Ax Relative Error")
hist!(ax_stats, filter(x -> x > 0 && x < 0.5, vec(Ay_rel_diff)), bins=20, color=:red, label="Ay Relative Error")
axislegend(ax_stats, position=:rt)

# Add statistical summary text
# Using valid_values to filter out potential NaN or Inf values
valid_ax = filter(isfinite, vec(Ax_rel_diff))
valid_ay = filter(isfinite, vec(Ay_rel_diff))

stats_text = """
Statistical Summary:
Mean Relative Error Ax: $(round(mean(valid_ax)*100, digits=2))%
Mean Relative Error Ay: $(round(mean(valid_ay)*100, digits=2))%
Median Relative Error Ax: $(round(median(valid_ax)*100, digits=2))%
Median Relative Error Ay: $(round(median(valid_ay)*100, digits=2))%
"""

Label(fig[4, 4:6], stats_text, tellwidth=false)

# Save and display the figure
save("spacetime_capacities_with_relative_differences.png", fig)
display(fig)

# Print total capacities for comparison
println("Total Ax (Front Tracking): $(sum(Ax_st_ft))")
println("Total Ax (VOFI): $(sum(Ax_st_vofi))")
println("Relative Error: $(round((sum(Ax_st_ft) - sum(Ax_st_vofi))/sum(Ax_st_vofi)*100, digits=2))%")
println()
println("Total Ay (Front Tracking): $(sum(Ay_st_ft))")
println("Total Ay (VOFI): $(sum(Ay_st_vofi))")
println("Relative Error: $(round((sum(Ay_st_ft) - sum(Ay_st_vofi))/sum(Ay_st_vofi)*100, digits=2))%")

# Extract volume data from front tracking method
V_st_ft = ft_spacetime_capacities[:V_spacetime]

# Get VOFI volumes from space-time mesh
V_st_vofi = Array(SparseArrays.diag(vofi_st_capacity.V))
V_st_vofi = reshape(V_st_vofi, (nx+1, ny+1, 2))
V_st_vofi = V_st_vofi[:, :, 1]  # Extract V component

# Calculate relative differences for volumes
V_rel_diff = relative_error_matrix(V_st_ft, V_st_vofi)
#V_rel_diff = abs.(V_st_ft - V_st_vofi)

# Create a focused figure just for interface and volume visualization
fig_volumes = Figure(size=(1200, 800))

# Row 1: Show interfaces at both time steps
ax_interfaces = Axis(fig_volumes[1, 1:3], 
                    title="Interface Movement", 
                    xlabel="x", ylabel="y", aspect=DataAspect())

# Plot the interfaces
markers_n = get_markers(front_n)
markers_np1 = get_markers(front_np1)

lines!(ax_interfaces, first.(markers_n), last.(markers_n), color=:blue, linewidth=2,
      label="Interface at t=0")
lines!(ax_interfaces, first.(markers_np1), last.(markers_np1), color=:red, linewidth=2,
      label="Interface at t=dt")

axislegend(ax_interfaces, position=:rt)

# Row 2: Volume visualization
ax_v_ft = Axis(fig_volumes[2, 1], title="Front Tracking V_st",
              xlabel="x", ylabel="y", aspect=DataAspect())
ax_v_vofi = Axis(fig_volumes[2, 2], title="VOFI Space-Time V_st",
                xlabel="x", ylabel="y", aspect=DataAspect())
ax_diff_v = Axis(fig_volumes[2, 3], title="Relative Difference (V)",
                xlabel="x", ylabel="y", aspect=DataAspect())

# Set consistent color range for volumes
max_V = maximum([maximum(V_st_ft), maximum(V_st_vofi)])

# Plot volume capacities
hm_V_ft = heatmap!(ax_v_ft, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                  V_st_ft', colormap=:viridis, colorrange=(0, max_V))
Colorbar(fig_volumes[2, 4], hm_V_ft, label="V_st (Front Tracking)")

hm_V_vofi = heatmap!(ax_v_vofi, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                    V_st_vofi, colormap=:viridis, colorrange=(0, max_V))
Colorbar(fig_volumes[2, 5], hm_V_vofi, label="V_st (VOFI)")

# Plot relative difference for volumes
hm_V_diff = heatmap!(ax_diff_v, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                    V_rel_diff', colormap=:plasma)
Colorbar(fig_volumes[2, 6], hm_V_diff, label="Relative Difference")

# Add interfaces to the volume plots
lines!(ax_v_ft, first.(markers_n), last.(markers_n), color=:blue, linewidth=1.5, label="t=0")
lines!(ax_v_ft, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1.5, label="t=dt")

lines!(ax_v_vofi, first.(markers_n), last.(markers_n), color=:blue, linewidth=1.5)
lines!(ax_v_vofi, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1.5)

lines!(ax_diff_v, first.(markers_n), last.(markers_n), color=:blue, linewidth=1.5)
lines!(ax_diff_v, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1.5)

# Save and display the figure
save("spacetime_capacities_with_volumes.png", fig_volumes)
display(fig_volumes)

# Print summary statistics for volume
println("Volume Statistical Summary:")
println("Total V (Front Tracking): $(sum(V_st_ft))")
println("Total V (VOFI): $(sum(V_st_vofi))")
println("Relative Error: $(round((sum(V_st_ft) - sum(V_st_vofi))/sum(V_st_vofi)*100, digits=2))%")

# Print total volume statistics at the end
println()
println("Total V (Front Tracking): $(sum(V_st_ft))")
println("Total V (VOFI): $(sum(V_st_vofi))")
println("Volume Relative Error: $(round((sum(V_st_ft) - sum(V_st_vofi))/sum(V_st_vofi)*100, digits=2))%")

# Extract Bx, By, Wx, Wy data from front tracking method
Bx_st_ft = ft_spacetime_capacities[:Bx_spacetime]
By_st_ft = ft_spacetime_capacities[:By_spacetime]
Wx_st_ft = ft_spacetime_capacities[:Wx_spacetime]
Wy_st_ft = ft_spacetime_capacities[:Wy_spacetime]

# Get corresponding VOFI values from space-time mesh
# For B capacities (assuming they're accessible in the B field)
Bx_st_vofi = Array(SparseArrays.diag(vofi_st_capacity.B[1]))
Bx_st_vofi = reshape(Bx_st_vofi, (nx+1, ny+1, 2))
Bx_st_vofi = Bx_st_vofi[:, :, 1]

By_st_vofi = Array(SparseArrays.diag(vofi_st_capacity.B[2]))
By_st_vofi = reshape(By_st_vofi, (nx+1, ny+1, 2))
By_st_vofi = By_st_vofi[:, :, 1]

# For W capacities (assuming they're accessible in the W field)
Wx_st_vofi = Array(SparseArrays.diag(vofi_st_capacity.W[1]))
Wx_st_vofi = reshape(Wx_st_vofi, (nx+1, ny+1, 2))
Wx_st_vofi = Wx_st_vofi[:, :, 1]

Wy_st_vofi = Array(SparseArrays.diag(vofi_st_capacity.W[2]))
Wy_st_vofi = reshape(Wy_st_vofi, (nx+1, ny+1, 2))
Wy_st_vofi = Wy_st_vofi[:, :, 1]

# Calculate relative differences
Bx_rel_diff = relative_error_matrix(Bx_st_ft, Bx_st_vofi)
By_rel_diff = relative_error_matrix(By_st_ft, By_st_vofi)
Wx_rel_diff = relative_error_matrix(Wx_st_ft, Wx_st_vofi)
Wy_rel_diff = relative_error_matrix(Wy_st_ft, Wy_st_vofi)

# Create a new figure with more rows for the additional capacities
fig = Figure(size=(1500, 1800))  # Increased height to accommodate more rows

# Add rows for Bx visualization (Row 7)
ax_bx_ft = Axis(fig[7, 1], title="Front Tracking Bx_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_bx_vofi = Axis(fig[7, 2], title="VOFI Space-Time Bx_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_diff_bx = Axis(fig[7, 3], title="Relative Difference (Bx)",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Set consistent color range for Bx
max_Bx = maximum([maximum(Bx_st_ft), maximum(Bx_st_vofi)])

# Plot Bx capacities
hm_Bx_ft = heatmap!(ax_bx_ft, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                 Bx_st_ft', colormap=:viridis, colorrange=(0, max_Bx))
Colorbar(fig[7, 4], hm_Bx_ft, label="Bx_st (Front Tracking)")

hm_Bx_vofi = heatmap!(ax_bx_vofi, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                    Bx_st_vofi', colormap=:viridis, colorrange=(0, max_Bx))
Colorbar(fig[7, 5], hm_Bx_vofi, label="Bx_st (VOFI)")

# Plot relative difference
hm_Bx_diff = heatmap!(ax_diff_bx, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                    Bx_rel_diff', colormap=:plasma)
Colorbar(fig[7, 6], hm_Bx_diff, label="Relative Difference")

# Add interfaces to the Bx plots
lines!(ax_bx_ft, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_bx_ft, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)
lines!(ax_bx_vofi, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_bx_vofi, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)
lines!(ax_diff_bx, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_diff_bx, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

# Add rows for By visualization (Row 8)
ax_by_ft = Axis(fig[8, 1], title="Front Tracking By_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_by_vofi = Axis(fig[8, 2], title="VOFI Space-Time By_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_diff_by = Axis(fig[8, 3], title="Relative Difference (By)",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Set consistent color range for By
max_By = maximum([maximum(By_st_ft), maximum(By_st_vofi)])

# Plot By capacities
hm_By_ft = heatmap!(ax_by_ft, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                 By_st_ft', colormap=:viridis, colorrange=(0, max_By))
Colorbar(fig[8, 4], hm_By_ft, label="By_st (Front Tracking)")

hm_By_vofi = heatmap!(ax_by_vofi, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                    By_st_vofi', colormap=:viridis, colorrange=(0, max_By))
Colorbar(fig[8, 5], hm_By_vofi, label="By_st (VOFI)")

# Plot relative difference
hm_By_diff = heatmap!(ax_diff_by, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                    By_rel_diff', colormap=:plasma)
Colorbar(fig[8, 6], hm_By_diff, label="Relative Difference")

# Add interfaces to the By plots
lines!(ax_by_ft, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_by_ft, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)
lines!(ax_by_vofi, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_by_vofi, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)
lines!(ax_diff_by, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_diff_by, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

# Add rows for Wx visualization (Row 9)
ax_wx_ft = Axis(fig[9, 1], title="Front Tracking Wx_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_wx_vofi = Axis(fig[9, 2], title="VOFI Space-Time Wx_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_diff_wx = Axis(fig[9, 3], title="Relative Difference (Wx)",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Set consistent color range for Wx
max_Wx = maximum([maximum(Wx_st_ft), maximum(Wx_st_vofi)])

# Plot Wx capacities
hm_Wx_ft = heatmap!(ax_wx_ft, x_nodes, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                 Wx_st_ft', colormap=:viridis, colorrange=(0, max_Wx))
Colorbar(fig[9, 4], hm_Wx_ft, label="Wx_st (Front Tracking)")

hm_Wx_vofi = heatmap!(ax_wx_vofi, x_nodes, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                    Wx_st_vofi', colormap=:viridis, colorrange=(0, max_Wx))
Colorbar(fig[9, 5], hm_Wx_vofi, label="Wx_st (VOFI)")

# Plot relative difference
hm_Wx_diff = heatmap!(ax_diff_wx, x_nodes, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                    Wx_rel_diff', colormap=:plasma)
Colorbar(fig[9, 6], hm_Wx_diff, label="Relative Difference")

# Add interfaces to the Wx plots
lines!(ax_wx_ft, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_wx_ft, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)
lines!(ax_wx_vofi, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_wx_vofi, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)
lines!(ax_diff_wx, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_diff_wx, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

# Add rows for Wy visualization (Row 10)
ax_wy_ft = Axis(fig[10, 1], title="Front Tracking Wy_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_wy_vofi = Axis(fig[10, 2], title="VOFI Space-Time Wy_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_diff_wy = Axis(fig[10, 3], title="Relative Difference (Wy)",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Set consistent color range for Wy
max_Wy = maximum([maximum(Wy_st_ft), maximum(Wy_st_vofi)])

# Plot Wy capacities
hm_Wy_ft = heatmap!(ax_wy_ft, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes, 
                 Wy_st_ft, colormap=:viridis, colorrange=(0, max_Wy))
Colorbar(fig[10, 4], hm_Wy_ft, label="Wy_st (Front Tracking)")

hm_Wy_vofi = heatmap!(ax_wy_vofi, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes, 
                    Wy_st_vofi, colormap=:viridis, colorrange=(0, max_Wy))
Colorbar(fig[10, 5], hm_Wy_vofi, label="Wy_st (VOFI)")

# Plot relative difference
hm_Wy_diff = heatmap!(ax_diff_wy, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes, 
                    Wy_rel_diff, colormap=:plasma)
Colorbar(fig[10, 6], hm_Wy_diff, label="Relative Difference")

# Add interfaces to the Wy plots
lines!(ax_wy_ft, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_wy_ft, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)
lines!(ax_wy_vofi, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_wy_vofi, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)
lines!(ax_diff_wy, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_diff_wy, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

# Add histogram of all secondary capacity errors
ax_stats_secondary = Axis(fig[11, 1:3], title="Secondary Capacities Relative Error Statistics", 
               xlabel="Relative Error", ylabel="Frequency")

# Create histograms of difference values
hist!(ax_stats_secondary, filter(x -> x > 0 && x < 0.5, vec(Bx_rel_diff)), bins=20, color=:orange, label="Bx Relative Error")
hist!(ax_stats_secondary, filter(x -> x > 0 && x < 0.5, vec(By_rel_diff)), bins=20, color=:purple, label="By Relative Error")
hist!(ax_stats_secondary, filter(x -> x > 0 && x < 0.5, vec(Wx_rel_diff)), bins=20, color=:teal, label="Wx Relative Error")
hist!(ax_stats_secondary, filter(x -> x > 0 && x < 0.5, vec(Wy_rel_diff)), bins=20, color=:pink, label="Wy Relative Error")
axislegend(ax_stats_secondary, position=:rt)

# Calculate statistics for secondary capacities
valid_bx = filter(isfinite, vec(Bx_rel_diff))
valid_by = filter(isfinite, vec(By_rel_diff))
valid_wx = filter(isfinite, vec(Wx_rel_diff))
valid_wy = filter(isfinite, vec(Wy_rel_diff))

stats_text_secondary = """
Secondary Capacities Statistical Summary:
Mean Relative Error Bx: $(round(mean(valid_bx)*100, digits=2))%
Mean Relative Error By: $(round(mean(valid_by)*100, digits=2))%
Mean Relative Error Wx: $(round(mean(valid_wx)*100, digits=2))%
Mean Relative Error Wy: $(round(mean(valid_wy)*100, digits=2))%
Median Relative Error: Bx: $(round(median(valid_bx)*100, digits=2))% By: $(round(median(valid_by)*100, digits=2))% 
                        Wx: $(round(median(valid_wx)*100, digits=2))% Wy: $(round(median(valid_wy)*100, digits=2))%
"""

Label(fig[11, 4:6], stats_text_secondary, tellwidth=false)

# Save the updated figure
save("spacetime_all_capacities_comparison.png", fig)

# Print total capacity statistics for secondary capacities
println()
println("Total Bx (Front Tracking): $(sum(Bx_st_ft))")
println("Total Bx (VOFI): $(sum(Bx_st_vofi))")
println("Bx Relative Error: $(round((sum(Bx_st_ft) - sum(Bx_st_vofi))/sum(Bx_st_vofi)*100, digits=2))%")
println()
println("Total By (Front Tracking): $(sum(By_st_ft))")
println("Total By (VOFI): $(sum(By_st_vofi))")
println("By Relative Error: $(round((sum(By_st_ft) - sum(By_st_vofi))/sum(By_st_vofi)*100, digits=2))%")
println()
println("Total Wx (Front Tracking): $(sum(Wx_st_ft))")
println("Total Wx (VOFI): $(sum(Wx_st_vofi))")
println("Wx Relative Error: $(round((sum(Wx_st_ft) - sum(Wx_st_vofi))/sum(Wx_st_vofi)*100, digits=2))%")
println()
println("Total Wy (Front Tracking): $(sum(Wy_st_ft))")
println("Total Wy (VOFI): $(sum(Wy_st_vofi))")
println("Wy Relative Error: $(round((sum(Wy_st_ft) - sum(Wy_st_vofi))/sum(Wy_st_vofi)*100, digits=2))%")