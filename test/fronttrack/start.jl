using Penguin
using CairoMakie
using LibGEOS

"""
Demonstration of Front Tracking with mesh visualization
Shows how to create a mesh with uniform spacing, 
initialize front tracking markers, and visualize them together
"""

# 1. Define the mesh parameters
nx, ny = 20, 20        # Number of cells
lx, ly = 10.0, 10.0    # Domain size
x0, y0 = -5.0, -5.0    # Domain origin

# Create the mesh using ranges
x_range = range(x0, x0 + lx, length=nx+1)
y_range = range(y0, y0 + ly, length=ny+1)
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# 2. Create a front tracking object with a circle
front = FrontTracker()
radius = 3.0
center_x, center_y = 0.0, 0.0
n_markers = 20  # Number of markers along the interface

# Initialize circle markers
create_circle!(front, center_x, center_y, radius, n_markers)

# 3. Create the visualization
fig = Figure(size=(1000, 800))

# Plot the mesh grid
ax1 = Axis(fig[1, 1], title="Mesh Grid and Front Tracking Interface",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Plot mesh grid lines
for x in x_range
    lines!(ax1, [x, x], [y_range[1], y_range[end]], 
          color=:lightgray, linestyle=:dash, linewidth=1)
end

for y in y_range
    lines!(ax1, [x_range[1], x_range[end]], [y, y], 
          color=:lightgray, linestyle=:dash, linewidth=1)
end

# Plot the front tracking interface
markers = get_markers(front)
marker_x = [m[1] for m in markers]
marker_y = [m[2] for m in markers]

# Draw the interface line
lines!(ax1, marker_x, marker_y, color=:blue, linewidth=2,
      label="Interface")

# Plot the markers
scatter!(ax1, marker_x, marker_y, color=:red, markersize=6,
        label="Markers")

# Label the first marker to show orientation
scatter!(ax1, [marker_x[1]], [marker_y[1]], color=:green, markersize=8,
        label="First Marker")

# Add a legend
axislegend(ax1, position=:rt)

# 4. Create a SDF visualization
ax2 = Axis(fig[1, 2], title="Signed Distance Function",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Create a grid for SDF evaluation
resolution = 100
x_sdf = range(x0, x0 + lx, length=resolution)
y_sdf = range(y0, y0 + ly, length=resolution)
sdf_values = zeros(resolution, resolution)

# Calculate SDF values
for i in 1:resolution
    for j in 1:resolution
        sdf_values[i, j] = sdf(front, x_sdf[i], y_sdf[j])
    end
end

# Plot SDF as heatmap
hm = heatmap!(ax2, x_sdf, y_sdf, sdf_values, colormap=:viridis)
Colorbar(fig[1, 3], hm, label="Signed Distance")

# Plot contours at specific levels
contour!(ax2, x_sdf, y_sdf, sdf_values, 
        levels=[-3, -2, -1, 0, 1, 2, 3], 
        linewidth=1)

# Plot the interface contour (zero level) more prominently
contour!(ax2, x_sdf, y_sdf, sdf_values, 
        levels=[0], 
        color=:white, linewidth=2)


# Display the figure
display(fig)

# Save the figure
save("front_tracking_visualization.png", fig)
println("Visualization saved as 'front_tracking_visualization.png'")