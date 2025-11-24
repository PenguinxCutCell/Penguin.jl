using Penguin
using CairoMakie
using SparseArrays
using LinearAlgebra
using Statistics
using LinearSolve
using Printf

"""
2D Stokes Poiseuille flow in inclined cut-cell channels: error vs inclination angle.

This script evaluates how cut-cell accuracy depends on the channel inclination angle.
For each angle, we solve the Stokes equations in an inclined channel and compute
the volume-weighted L2 error following the Taylor-Green methodology.
Also plots velocity profiles like in poiseuille_2d_cut.jl.
"""

###########
# Parameters
###########
nx, ny = 64, 64
Lx, Ly = 2.0, 2.0
x0, y0 = 0.0, 0.0
channel_height = 0.6
Umax = 1.0
μ = 1.0
ρ = 1.0

# Angles to test (in degrees)
angles_deg = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
angles_rad = [deg * π / 180 for deg in angles_deg]

# Storage for results
results = []
profile_data = []  # Store velocity profiles for plotting

###########
# Helper functions
###########

# Interior indices (exclude boundary)
interior_indices(n) = 2:(n-1)

# Create inclined channel body function
function create_inclined_channel(θ, h, center_x=1.0, center_y=1.0)
    # Channel walls are parallel lines at distance h apart, rotated by angle θ
    # Normal vector to channel walls: n = (-sin(θ), cos(θ))
    nx, ny = -sin(θ), cos(θ)
    
    return (x, y, _=0) -> begin
        # Distance from point to channel centerline
        dx, dy = x - center_x, y - center_y
        dist_to_center = abs(nx * dx + ny * dy)
        
        # Negative inside channel, positive outside
        return dist_to_center - h/2
    end
end

# Analytical solution for inclined Poiseuille flow
function analytical_velocity_inclined(θ, h, Umax)
    return (x, y, center_x=1.0, center_y=1.0) -> begin
        # Transform to channel coordinates
        dx, dy = x - center_x, y - center_y
        n_coord = -dx * sin(θ) + dy * cos(θ) + h/2  # Distance from bottom wall
        
        # Check if inside channel
        if n_coord < 0 || n_coord > h
            return (0.0, 0.0)
        end
        
        # Parabolic profile magnitude
        u_mag = 4 * Umax * n_coord * (h - n_coord) / h^2
        
        # Convert to x,y components
        ux = u_mag * cos(θ)
        uy = u_mag * sin(θ)
        
        return (ux, uy)
    end
end

# Volume-weighted L2 error computation (Taylor-Green style)
function compute_weighted_L2_by_celltype(num_2D, exact_2D, cell_types_2D, Vdiag, 
                                        ix_range, iy_range, target_types)
    ni, nj = size(num_2D, 1), size(num_2D, 2)
    total_volume = 0.0
    weighted_error_sq = 0.0
    n_cells = 0
    
    for j in 1:nj, i in 1:ni
        if (i in ix_range) && (j in iy_range) && (cell_types_2D[i,j] in target_types)
            lin_idx = (j-1)*ni + i
            vol = Vdiag[lin_idx]
            err = num_2D[i,j] - exact_2D[i,j]
            
            weighted_error_sq += vol * err^2
            total_volume += vol
            n_cells += 1
        end
    end
    
    l2_error = sqrt(weighted_error_sq)
    l2_normalized = total_volume > 0 ? sqrt(weighted_error_sq / total_volume) : 0.0
    
    return l2_error, l2_normalized, n_cells, total_volume
end

# Extract velocity profile across channel (perpendicular to flow direction)
function extract_channel_profile(Ux_2D, Uy_2D, xs, ys, θ, center_x, center_y, h)
    # Find points that cross the channel perpendicular to flow direction
    # Use the channel center line and extract profile perpendicular to it
    
    # Profile line passes through center, perpendicular to flow
    # Direction perpendicular to flow: (-sin(θ), cos(θ))
    perp_x, perp_y = -sin(θ), cos(θ)
    
    profile_coords = Float64[]
    profile_ux = Float64[]
    profile_uy = Float64[]
    profile_u_mag = Float64[]
    
    # Sample along perpendicular direction through center
    n_samples = 100
    s_range = range(-h, h, length=n_samples)
    
    for s in s_range
        # Point along perpendicular line
        px = center_x + s * perp_x
        py = center_y + s * perp_y
        
        # Find nearest grid point
        i = argmin(abs.(xs .- px))
        j = argmin(abs.(ys .- py))
        
        if 1 <= i <= length(xs) && 1 <= j <= length(ys)
            ux_val = Ux_2D[i, j]
            uy_val = Uy_2D[i, j]
            u_mag = sqrt(ux_val^2 + uy_val^2)
            
            push!(profile_coords, s + h/2)  # Distance from bottom wall
            push!(profile_ux, ux_val)
            push!(profile_uy, uy_val) 
            push!(profile_u_mag, u_mag)
        end
    end
    
    return profile_coords, profile_ux, profile_uy, profile_u_mag
end

###########
# Main convergence study
###########
println("=== Inclined Channel Cut-Cell Error Analysis ===")
println("Angles tested: $(angles_deg) degrees")
println("Resolution: $(nx) × $(ny)")
println("Channel height: $(channel_height)")
println()

for (i, θ) in enumerate(angles_rad)
    θ_deg = angles_deg[i]
    println("Solving for θ = $(θ_deg)°...")
    
    # Create geometry
    body = create_inclined_channel(θ, channel_height)
    analytical_vel = analytical_velocity_inclined(θ, channel_height, Umax)
    
    # Meshes
    mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
    dx, dy = Lx/nx, Ly/ny
    mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
    mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))
    
    # Capacities and operators
    capacity_ux = Capacity(body, mesh_ux; compute_centroids=false)
    capacity_uy = Capacity(body, mesh_uy; compute_centroids=false)
    capacity_p = Capacity(body, mesh_p; compute_centroids=false)
    operator_ux = DiffusionOps(capacity_ux)
    operator_uy = DiffusionOps(capacity_uy)
    operator_p = DiffusionOps(capacity_p)
    
    # Boundary conditions (enforce analytical solution at boundaries)
    ux_bc_func = (x, y) -> analytical_vel(x, y)[1]
    uy_bc_func = (x, y) -> analytical_vel(x, y)[2]
    
    bc_ux = BorderConditions(Dict(
        :left => Dirichlet(ux_bc_func),
        :right => Dirichlet(ux_bc_func), 
        :bottom => Dirichlet(ux_bc_func),
        :top => Dirichlet(ux_bc_func)
    ))
    
    bc_uy = BorderConditions(Dict(
        :left => Dirichlet(uy_bc_func),
        :right => Dirichlet(uy_bc_func),
        :bottom => Dirichlet(uy_bc_func), 
        :top => Dirichlet(uy_bc_func)
    ))
    
    pressure_gauge = PinPressureGauge()  # Gauge fixing
    u_bc = Dirichlet(0.0)  # No-slip on cut boundaries
    
    # Sources and material
    fᵤ = (x, y, z=0.0) -> 0.0
    fₚ = (x, y, z=0.0) -> 0.0
    
    # Create fluid and solver
    fluid = Fluid((mesh_ux, mesh_uy),
                  (capacity_ux, capacity_uy),
                  (operator_ux, operator_uy),
                  mesh_p, capacity_p, operator_p,
                  μ, ρ, fᵤ, fₚ)
    
    nu = prod(operator_ux.size)
    np = prod(operator_p.size)
    x0_init = zeros(4*nu + np)
    
    solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, u_bc; x0=x0_init)
    solve_StokesMono!(solver; algorithm=UMFPACKFactorization())
    
    # Extract solution
    uωx = solver.x[1:nu]
    uωy = solver.x[2*nu+1:3*nu]
    
    # Error analysis
    xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
    cell_types_x = capacity_ux.cell_types
    cell_types_y = capacity_uy.cell_types
    Vux = diag(operator_ux.V)
    Vuy = diag(operator_uy.V)
    
    # Reshape for 2D operations
    cell_types_x_2D = reshape(cell_types_x, (length(xs), length(ys)))
    cell_types_y_2D = reshape(cell_types_y, (length(xs), length(ys)))
    Ux_num = reshape(uωx, (length(xs), length(ys)))
    Uy_num = reshape(uωy, (length(xs), length(ys)))
    
    # Compute exact solution on grids
    Ux_exact = [analytical_vel(xs[i], ys[j])[1] for i in 1:length(xs), j in 1:length(ys)]
    Uy_exact = [analytical_vel(xs[i], ys[j])[2] for i in 1:length(xs), j in 1:length(ys)]
    
    # Interior ranges
    ix_range = interior_indices(length(xs))
    iy_range = interior_indices(length(ys))
    
    # Compute errors by cell type
    all_fluid_types = [1, -1]
    cut_cell_types = [-1]
    full_cell_types = [1]
    
    # Combined X and Y velocity errors
    l2_ux_all, _, _, _ = compute_weighted_L2_by_celltype(
        Ux_num, Ux_exact, cell_types_x_2D, Vux, ix_range, iy_range, all_fluid_types)
    l2_uy_all, _, _, _ = compute_weighted_L2_by_celltype(
        Uy_num, Uy_exact, cell_types_y_2D, Vuy, ix_range, iy_range, all_fluid_types)
    
    l2_ux_cut, _, n_ux_cut, _ = compute_weighted_L2_by_celltype(
        Ux_num, Ux_exact, cell_types_x_2D, Vux, ix_range, iy_range, cut_cell_types)
    l2_uy_cut, _, n_uy_cut, _ = compute_weighted_L2_by_celltype(
        Uy_num, Uy_exact, cell_types_y_2D, Vuy, ix_range, iy_range, cut_cell_types)
    
    l2_ux_full, _, n_ux_full, _ = compute_weighted_L2_by_celltype(
        Ux_num, Ux_exact, cell_types_x_2D, Vux, ix_range, iy_range, full_cell_types)
    l2_uy_full, _, n_uy_full, _ = compute_weighted_L2_by_celltype(
        Uy_num, Uy_exact, cell_types_y_2D, Vuy, ix_range, iy_range, full_cell_types)
    
    # Combined velocity magnitude error
    l2_total_all = sqrt(l2_ux_all^2 + l2_uy_all^2)
    l2_total_cut = sqrt(l2_ux_cut^2 + l2_uy_cut^2)
    l2_total_full = sqrt(l2_ux_full^2 + l2_uy_full^2)
    
    # compute full-field velocity magnitude for plotting
    Umag_num = sqrt.(Ux_num .^ 2 .+ Uy_num .^ 2)
    
    # Extract velocity profiles across channel
    profile_coords, profile_ux_num, profile_uy_num, profile_u_mag_num = extract_channel_profile(
        Ux_num, Uy_num, xs, ys, θ, 1.0, 1.0, channel_height)
    
    # Analytical profile
    profile_u_mag_exact = [4 * Umax * coord * (channel_height - coord) / channel_height^2 
                          for coord in profile_coords]
    
    # Store results
    result = Dict(
        :angle_deg => θ_deg,
        :angle_rad => θ,
        :l2_total_all => l2_total_all,
        :l2_total_cut => l2_total_cut,
        :l2_total_full => l2_total_full,
        :n_cut_cells => n_ux_cut + n_uy_cut,
        :n_full_cells => n_ux_full + n_uy_full
    )
    
    profile_result = Dict(
        :angle_deg => θ_deg,
        :coords => profile_coords,
        :u_mag_numerical => profile_u_mag_num,
        :u_mag_exact => profile_u_mag_exact,
        :ux_numerical => profile_ux_num,
        :uy_numerical => profile_uy_num,
        :Umag_grid => Umag_num    # <-- added: full-field magnitude for heatmap
    )
    
    push!(results, result)
    push!(profile_data, profile_result)
    
    @printf("  θ = %3d°: L2_total = %.2e (cut: %.2e, full: %.2e), cut cells: %d\n", 
            θ_deg, l2_total_all, l2_total_cut, l2_total_full, n_ux_cut + n_uy_cut)
end

###########
# Results analysis and visualization
###########
println("\n=== Summary Results ===")
println("Angle[°]  L2_total   L2_cut     L2_full    Cut/Full   #Cut")
println("------------------------------------------------------")

for r in results
    ratio = r[:n_full_cells] > 0 ? r[:l2_total_cut] / r[:l2_total_full] : NaN
    @printf("%6.0f %9.2e %9.2e %9.2e %8.2f %6d\n", 
            r[:angle_deg], r[:l2_total_all], r[:l2_total_cut], 
            r[:l2_total_full], ratio, r[:n_cut_cells])
end


# Plot velocity profiles for all angles (like poiseuille_2d_cut.jl)
fig_profiles = Figure(resolution=(1400, 1000))

# Main profile comparison plot
ax1 = Axis(fig_profiles[1,1:2], xlabel="Distance from bottom wall", ylabel="Velocity magnitude |u|", 
           title="Velocity Profiles Across Inclined Channels")

colors = [:blue, :red, :green, :purple, :orange, :brown, :pink, :cyan, :magenta, :olive]
for (i, pdata) in enumerate(profile_data)
    if !isempty(pdata[:coords])
        lines!(ax1, pdata[:coords], pdata[:u_mag_numerical], 
               label="θ=$(pdata[:angle_deg])° (num)", color=colors[i], linewidth=2)
        lines!(ax1, pdata[:coords], pdata[:u_mag_exact], 
               color=colors[i], linestyle=:dash, linewidth=1, alpha=0.7)
    end
end
axislegend(ax1, position=:rt, nbanks=2)


# Error convergence plot
fig_error = Figure(size=(1200, 800))

angles = [r[:angle_deg] for r in results]
l2_total_all = [r[:l2_total_all] for r in results]
l2_total_cut = [r[:l2_total_cut] for r in results]
l2_total_full = [r[:l2_total_full] for r in results]

# Main error plot
ax1 = Axis(fig_error[1,1], xlabel="Inclination angle [°]", ylabel="L2 error", 
           title="Cut-Cell Error vs Channel Inclination", yscale=log10)

lines!(ax1, angles, l2_total_all, label="All cells", linewidth=2)
scatter!(ax1, angles, l2_total_all, marker=:circle, color=:black)
lines!(ax1, angles, l2_total_cut, label="Cut cells", linewidth=2, color=:red)
scatter!(ax1, angles, l2_total_cut, marker=:square, color=:red)
lines!(ax1, angles, l2_total_full, label="Full cells", linewidth=2, color=:blue)
scatter!(ax1, angles, l2_total_full, marker=:diamond, color=:blue)
axislegend(ax1, position=:rt)

# Error ratio plot  
ax2 = Axis(fig_error[1,2], xlabel="Inclination angle [°]", ylabel="Error ratio (Cut/Full)", 
           title="Relative Cut-Cell Error")

valid_ratios = []
valid_angles = []
for r in results
    if r[:n_full_cells] > 0 && r[:l2_total_full] > 0
        push!(valid_ratios, r[:l2_total_cut] / r[:l2_total_full])
        push!(valid_angles, r[:angle_deg])
    end
end

if !isempty(valid_ratios)
    lines!(ax2, valid_angles, valid_ratios,  linewidth=2, color=:purple)
    scatter!(ax2, valid_angles, valid_ratios, marker=:star, color=:purple)
    hlines!(ax2, [1.0], color=:gray, linestyle=:dash, label="Equal error")
    axislegend(ax2)
end

# Number of cut cells
ax3 = Axis(fig_error[2,1], xlabel="Inclination angle [°]", ylabel="Number of cut cells",
           title="Cut Cells vs Angle")

n_cut = [r[:n_cut_cells] for r in results]
lines!(ax3, angles, n_cut, linewidth=2, color=:orange)
scatter!(ax3, angles, n_cut, marker=:utriangle, color=:orange)

display(fig_error)
save("inclined_poiseuille_error_analysis.png", fig_error)

println("\n=== Analysis Complete ===")
println("Results saved to:")
println("  - inclined_poiseuille_velocity_profiles.png")
println("  - inclined_poiseuille_error_analysis.png")

# Find worst and best angles
worst_idx = argmax(l2_total_all)
best_idx = argmin(l2_total_all)

println("Worst error:  $(angles[worst_idx])° (L2 = $(l2_total_all[worst_idx]))")
println("Best error:   $(angles[best_idx])° (L2 = $(l2_total_all[best_idx]))")
println("Error variation: $(maximum(l2_total_all) / minimum(l2_total_all))×")

# Plot velocity profiles for all angles (like poiseuille_2d_cut.jl)
# replaced: produce heatmap of velocity magnitude for each tested angle
fig_heat = Figure(size=(1200, 900))
ncols = 3
nrows = ceil(Int, length(profile_data) / ncols)
dx = Lx / nx
dy = Ly / ny
# mesh_ux was created with origin x0 - 0.5*dx, y0
xs = [x0 - 0.5*dx + (i-1)*dx for i in 1:nx]
ys = [y0 + (j-1)*dy for j in 1:ny]

for (i, pdata) in enumerate(profile_data)
    row = div(i-1, ncols) + 1
    col = mod(i-1, ncols) + 1
    ax = Axis(fig_heat[row, col], title = "θ=$(pdata[:angle_deg])°", xlabel="x", ylabel="y")
    Umag = pdata[:Umag_grid]
    # ensure dense matrix and proper orientation
    Umag_plot = Matrix(Umag)
    hm = heatmap!(ax, xs, ys, Umag_plot'; colormap=:plasma)
    # overlay channel centerline or walls not trivial for rotated channel; skip
    Colorbar(fig_heat[row, ncols+1], hm)  # one colorbar per row block (approx)
end

display(fig_heat)
save("inclined_poiseuille_velocity_heatmaps.png", fig_heat)
