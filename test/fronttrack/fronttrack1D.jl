using Penguin
using CairoMakie
using SparseArrays
using StaticArrays
using Statistics
using LinearAlgebra
"""
Compare 1D Front Tracking with VOFI method
"""

# 1. Define the mesh parameters
nx = 100
lx = 10.0
x0 = -5.0

# Create the mesh
mesh = Penguin.Mesh((nx,), (lx,), (x0,))
x_nodes = mesh.nodes[1]
x_centers = mesh.nodes[1]

# 2. Create front tracker with two interface points
interface_points = [1.53]
front = FrontTracker1D(interface_points)

# 3. Compute capacities using Front Tracking
ft_capacities = compute_capacities_1d(mesh, front)
ft_volumes = ft_capacities[:volumes][1:nx+1]
ft_ax = ft_capacities[:Ax][1:nx+1]
ft_bx = ft_capacities[:Bx][1:nx+1]
ft_wx = ft_capacities[:Wx][1:nx+1]

# 4. Create an equivalent level set function for VOFI
# This function returns positive outside fluid, negative inside
function level_set_1d(x, y=0.0, z=0.0)
    return -(x - interface_points[1])  # Distance from the interface point
end
# 5. Compute capacities using VOFI
vofi_capacity = Capacity(level_set_1d, mesh, method="VOFI"; compute_centroids=false)

# Extract the diagonal elements from sparse matrices
vofi_volumes = Array(diag(vofi_capacity.V))
vofi_ax = Array(diag(vofi_capacity.A[1]))
vofi_bx = Array(diag(vofi_capacity.B[1]))
vofi_wx = Array(diag(vofi_capacity.W[1]))

# Extract centroids
ft_centroids = ft_capacities[:centroids_x][1:nx]
vofi_centroids = [vofi_capacity.C_ω[i][1] for i in 1:nx]  # Extract x-coordinate from SVector

# Print Wx Vofi vs Front Tracking
println(length(ft_volumes), length(vofi_volumes))
println("Front Tracking Volumes: ", ft_volumes)
println("VOFI Volumes: ", vofi_volumes)
println("Wx VOFI: ", vofi_wx)
println("Wx Front Tracking: ", ft_wx)

# 6. Create visualization for comparison
fig = Figure(size=(1000, 900))

# 6.1 Show interface positions
ax1 = Axis(fig[1, 1], 
           title="Interface Positions",
           ylabel="Fluid (1) / Solid (0)", 
           xlabel="x coordinate")

# Create a dense set of points for visualization
x_dense = range(x_nodes[1], x_nodes[end], length=500)
fluid_indicator(x) = is_point_inside(front, x) ? 0.8 : 0.2
y_indicators = fluid_indicator.(x_dense)

# Plot background shading for fluid regions
band!(ax1, x_dense, zeros(length(x_dense)), y_indicators, color=(:blue, 0.3))

# Add interface positions
scatter!(ax1, interface_points, fill(0.5, length(interface_points)), 
         color=:red, markersize=15)

# Set axis limits
ylims!(ax1, 0, 1)

# 6.2 Compare cell volumes
ax2 = Axis(fig[2, 1], 
           title="Cell Volumes (V)",
           ylabel="Volume", 
           xlabel="x coordinate")

# Plot front tracking volumes
scatter!(ax2, x_centers, ft_volumes, color=:blue, label="Front Tracking")

# Plot VOFI volumes
scatter!(ax2, x_centers, vofi_volumes, color=:red, marker=:diamond, 
         markersize=8, strokewidth=1, label="VOFI")

# Add a legend
axislegend(ax2, position=:rb)

# 6.3 Compare face capacities (Ax)
ax3 = Axis(fig[3, 1], 
           title="Face Capacities (Ax)",
           ylabel="Capacity", 
           xlabel="x coordinate")

# Plot front tracking Ax
scatter!(ax3, x_nodes, ft_ax, color=:blue, label="Front Tracking")

# Plot VOFI Ax
scatter!(ax3, x_nodes, vofi_ax, color=:red, marker=:diamond, 
         markersize=8, strokewidth=1, label="VOFI")

# Add a legend
axislegend(ax3, position=:rb)

# 6.4 Compare center-line capacities (Bx)
ax4 = Axis(fig[4, 1], 
           title="Center-line Capacities (Bx)",
           ylabel="Capacity", 
           xlabel="x coordinate")

# Plot front tracking Bx
scatter!(ax4, x_centers, ft_bx, color=:blue, label="Front Tracking")

# Plot VOFI Bx
scatter!(ax4, x_centers, vofi_bx, color=:red, marker=:diamond, 
         markersize=8, strokewidth=1, label="VOFI")

# Add a legend
axislegend(ax4, position=:rb)

# 6.5 Compare staggered volumes (Wx)
ax5 = Axis(fig[5, 1], 
           title="Staggered Volumes (Wx)",
           ylabel="Volume", 
           xlabel="x coordinate")

# Plot front tracking Wx
scatter!(ax5, x_nodes, ft_wx, color=:blue, label="Front Tracking")

# Plot VOFI Wx
scatter!(ax5, x_nodes, vofi_wx, color=:red, marker=:diamond, 
         markersize=8, strokewidth=1, label="VOFI")

# Add a legend
axislegend(ax5, position=:rb)

# Add a new plot for centroid comparison (after the Wx comparison)
ax6 = Axis(fig[6, 1], 
           title="Cell Centroids",
           ylabel="Position", 
           xlabel="x coordinate")

# Plot front tracking centroids
scatter!(ax6, mesh.centers[1], ft_centroids, color=:blue, label="Front Tracking")

# Plot VOFI centroids
scatter!(ax6, mesh.centers[1], vofi_centroids, color=:red, marker=:diamond, 
         markersize=8, strokewidth=1, label="VOFI")

# Add a legend
axislegend(ax6, position=:rb)


# 7. Add statistics panel
stats_panel = "Statistical Comparison:\n" *
             "Total Volume (FT): $(round(sum(ft_volumes), digits=4))\n" *
             "Total Volume (VOFI): $(round(sum(vofi_volumes), digits=4))\n" *
             "Volume Diff: $(round(100*(sum(ft_volumes)-sum(vofi_volumes))/sum(vofi_volumes), digits=2))%\n\n" *
             "Max Volume Diff: $(round(100*maximum(abs.(ft_volumes.-vofi_volumes))/maximum(vofi_volumes), digits=2))%\n" *
             "Mean Volume Diff: $(round(100*mean(abs.(ft_volumes.-vofi_volumes)./max.(vofi_volumes, 1e-10)), digits=2))%\n\n" *
             "Max Ax Diff: $(round(100*maximum(abs.(ft_ax.-vofi_ax))/maximum(vofi_ax), digits=2))%\n" *
             "Max Bx Diff: $(round(100*maximum(abs.(ft_bx.-vofi_bx))/maximum(vofi_bx), digits=2))%\n" *
             "Max Wx Diff: $(round(100*maximum(abs.(ft_wx.-vofi_wx))/maximum(vofi_wx), digits=2))%" *
            "Centroid Comparison:\n" *
             "Max Centroid Diff: $(round(maximum(abs.(ft_centroids.-vofi_centroids)), digits=6))\n" *
             "Mean Centroid Diff: $(round(mean(abs.(ft_centroids.-vofi_centroids)), digits=6))"


Label(fig[1:2, 2], stats_panel, tellwidth=false)

# Save and display the figure
save("front_tracking_vs_vofi_1d.png", fig)
display(fig)

# Print summary of differences
println("Front Tracking vs VOFI Comparison Summary:")
println("==========================================")
println("Total Volume (Front Tracking): $(sum(ft_volumes))")
println("Total Volume (VOFI): $(sum(vofi_volumes))")
println("Relative Difference: $(round((sum(ft_volumes) - sum(vofi_volumes))/sum(vofi_volumes)*100, digits=2))%")

println("\nCell-by-cell comparison:")
println("  Max Abs Volume Difference: $(round(maximum(abs.(ft_volumes .- vofi_volumes)), digits=6))")
println("  Max Rel Volume Difference: $(round(maximum(abs.(ft_volumes .- vofi_volumes) ./ max.(vofi_volumes, 1e-10)) * 100, digits=2))%")

println("\nCapacities comparison:")
println("  Max Ax Difference: $(round(maximum(abs.(ft_ax .- vofi_ax)) * 100, digits=2))%")
println("  Max Bx Difference: $(round(maximum(abs.(ft_bx .- vofi_bx)) * 100, digits=2))%")
println("  Max Wx Difference: $(round(maximum(abs.(ft_wx .- vofi_wx)) * 100, digits=2))%")


# Add to the printed summary
println("\nCentroid comparison:")
println("  Max Abs Centroid Difference: $(round(maximum(abs.(ft_centroids .- vofi_centroids)), digits=6))")
println("  Mean Abs Centroid Difference: $(round(mean(abs.(ft_centroids .- vofi_centroids)), digits=6))")