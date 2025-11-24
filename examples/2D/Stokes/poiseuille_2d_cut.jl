using Penguin
using CairoMakie
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Statistics
using LinearSolve

"""
2D Stokes Poiseuille flow (steady) with cut-cell channel: u = (Ux(y), 0), driven by pressure gradient.

This example demonstrates the impact of cut cells by defining a channel geometry via level-set.
The channel walls are at y = y_wall_bot and y = y_wall_top, creating cut cells at the boundaries.

Domain: [0, Lx] × [0, Ly] (full computational domain)
Channel: y ∈ [y_wall_bot, y_wall_top] (fluid region defined by level-set)
BCs: u_x = 0 on channel walls (via cut-cell interface), u_y = 0 everywhere.
Left/Right: enforce parabolic profile to avoid incompatibility.
"""

###########
# Grids
###########
nx, ny = 64, 64
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, 0.0

# Channel geometry (walls at y = 0.2 and y = 0.8)
y_wall_bot = 0.2
y_wall_top = 0.8
channel_height = y_wall_top - y_wall_bot

# Body function: distance to channel walls (negative = fluid, positive = solid)
body = (x, y, _=0) -> begin
    if y < y_wall_bot
        return y_wall_bot - y   # positive outside below
    elseif y > y_wall_top
        return y - y_wall_top   # positive outside above
    else
        # Inside the channel: return negative signed distance (fluid region)
        return -min(y - y_wall_bot, y_wall_top - y)
    end
end

# Pressure grid (cell-centered)
mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))

# Component-wise staggered velocity grids
dx, dy = Lx/nx, Ly/ny
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

###########
# Capacities and operators (per component) - with compute_centroids=false
###########
capacity_ux = Capacity(body, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(body, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(body, mesh_p; compute_centroids=false)
operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# BCs
###########
Umax = 1.0
# Parabolic profile for the channel (referenced to channel walls)
parabola = (x, y) -> begin
    if y < y_wall_bot || y > y_wall_top
        return 0.0  # Zero velocity outside channel
    else
        y_local = y - y_wall_bot  # Local coordinate in channel
        return 4Umax * y_local * (channel_height - y_local) / (channel_height^2)
    end
end

ux_left  = Dirichlet(parabola)
ux_right = Dirichlet(parabola)
ux_bot   = Dirichlet((x, y)-> 0.0)
ux_top   = Dirichlet((x, y)-> 0.0)
bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right, :bottom=>ux_bot, :top=>ux_top
))

uy_zero = Dirichlet((x, y)-> 0.0)
bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero
))

# Pressure gauge
pressure_gauge = PinPressureGauge()

# Cut-cell / interface BC for uγ (no-slip on channel walls)
u_bc = Dirichlet(0.0)

###########
# Sources and material
###########
fᵤ = (x, y, z=0.0) -> 0.0
fₚ = (x, y, z=0.0) -> 0.0
μ = 1.0
ρ = 1.0

# Fluid with per-component (ux, uy) meshes/capacities/operators
fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

###########
# Initial guess
###########
nu = prod(operator_ux.size)
np = prod(operator_p.size)
x0 = zeros(4*nu + np)

###########
# Solver and solve
###########
solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, u_bc; x0=x0)
solve_StokesMono!(solver; algorithm=UMFPACKFactorization())

println("2D Cut-Cell Poiseuille solved. Unknowns = ", length(solver.x))

# Extract components
uωx = solver.x[1:nu]; uγx = solver.x[nu+1:2nu]
uωy = solver.x[2nu+1:3nu]; uγy = solver.x[3nu+1:4nu]
pω  = solver.x[4nu+1:end]

###########
# Post-processing and plots
###########
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
xp = mesh_p.nodes[1];  yp = mesh_p.nodes[2]
LIux = LinearIndices((length(xs), length(ys)))
LIp  = LinearIndices((length(xp), length(yp)))

# Mid-column u_x(y) profile and analytical parabola
icol = Int(cld(length(xs), 2))
ux_profile = [uωx[LIux[icol, j]] for j in 1:length(ys)]
ux_analytical = [parabola(0.0, y) for y in ys]

# Plot capacity volumes to visualize cut cells
capacity_volumes = diag(capacity_ux.V)
capacity_field = reshape(capacity_volumes, (length(xs), length(ys)))

fig = Figure(resolution=(1400, 800))

# Profile comparison
ax1 = Axis(fig[1,1], xlabel="u_x", ylabel="y", title="Cut-Cell Poiseuille: mid x profile vs analytical")
lines!(ax1, ux_profile, ys, label="numerical")
lines!(ax1, ux_analytical, ys, color=:red, linestyle=:dash, label="analytical")
# Mark channel walls
hlines!(ax1, [y_wall_bot, y_wall_top], color=:black, linestyle=:dot, linewidth=2, label="channel walls")
axislegend(ax1, position=:rb)

# Capacity visualization
ax2 = Axis(fig[1,2], xlabel="x", ylabel="y", title="Cut-Cell Capacities")
hm_cap = heatmap!(ax2, xs, ys, Matrix(capacity_field'); colormap=:viridis)
# Mark channel walls
hlines!(ax2, [y_wall_bot, y_wall_top], color=:red, linewidth=2)
Colorbar(fig[1,3], hm_cap, label="Volume Fraction")

# Velocity field
Ux = reshape(uωx, (length(xs), length(ys)))
Ux_dense = Matrix(Ux)  # Convert to dense matrix if sparse
ax3 = Axis(fig[1,4], xlabel="x", ylabel="y", title="u_x field")
hm_ux = heatmap!(ax3, xs, ys, Ux_dense'; colormap=:plasma)
# Mark channel walls
hlines!(ax3, [y_wall_bot, y_wall_top], color=:white, linewidth=2)
Colorbar(fig[1,5], hm_ux, label="u_x")

display(fig)
save("stokes2d_poiseuille_cut_comparison.png", fig)

# Error metrics on the profile (only in fluid region)
fluid_indices = findall(y -> y_wall_bot <= y <= y_wall_top, ys)
profile_err = ux_profile[fluid_indices] .- ux_analytical[fluid_indices]
ℓ2_profile = sqrt(sum(abs2, profile_err) / length(profile_err))
ℓinf_profile = maximum(abs, profile_err)
println("Profile L2 error (fluid region) = ", ℓ2_profile, ", Linf = ", ℓinf_profile)


# Pressure field
P = reshape(pω, (length(xp), length(yp)))
P_dense = Matrix(P)  # Convert to dense matrix if sparse
fig2 = Figure(resolution=(800, 400))
ax4 = Axis(fig2[1,1], xlabel="x", ylabel="y", title="Pressure field (cut-cell)")
hm_p = heatmap!(ax4, xp, yp, P_dense; colormap=:balance)
hlines!(ax4, [y_wall_bot, y_wall_top], color=:white, linewidth=2)
Colorbar(fig2[1,2], hm_p, label="Pressure")
display(fig2)
save("stokes2d_poiseuille_cut_pressure.png", fig2)

# Global sanity checks
Uy = reshape(uωy, (length(xs), length(ys)))
Uy_max = maximum(abs, Uy)
P_std = std(P)
println("Sanity: max |u_y| = ", Uy_max, ", std(p) = ", P_std)

# Compare with non-cut-cell solution at channel centerline
y_center = (y_wall_bot + y_wall_top) / 2
j_center = argmin(abs.(ys .- y_center))
ux_centerline = ux_profile[j_center]
ux_analytical_centerline = parabola(0.0, y_center)
println("Centerline velocity: numerical = ", ux_centerline, ", analytical = ", ux_analytical_centerline)
println("Relative error at centerline: ", abs(ux_centerline - ux_analytical_centerline) / ux_analytical_centerline)


# Volume-integrated L2 error analysis by cell type (following Taylor-Green pattern)
println("\n=== Volume-Integrated L2 Error Analysis ===")

# Helper function: exclude boundary indices (same as Taylor-Green)
interior_indices(n) = 2:(n-1)

# Get cell types and operator volumes (not just capacity diagonal)
cell_types = capacity_ux.cell_types
Vux = diag(operator_ux.V)  # Use operator volumes like Taylor-Green

# Reshape for 2D grid operations
cell_types_2D = reshape(cell_types, (length(xs), length(ys)))
Ux_numerical = reshape(uωx, (length(xs), length(ys)))

# Compute analytical solution on 2D grid
Ux_exact = [parabola(xs[i], ys[j]) for i in 1:length(xs), j in 1:length(ys)]

# Interior ranges (exclude boundary nodes)
ix_range = interior_indices(length(xs))
iy_range = interior_indices(length(ys))

# Taylor-Green style volume-weighted L2 error with cell type filtering
function compute_weighted_L2_by_celltype(num_2D, exact_2D, cell_types_2D, Vdiag, 
                                        ix_range, iy_range, target_types)
    ni, nj = size(num_2D, 1), size(num_2D, 2)
    total_volume = 0.0
    weighted_error_sq = 0.0
    n_cells = 0
    
    for j in 1:nj, i in 1:ni
        # Only consider interior cells of target types
        if (i in ix_range) && (j in iy_range) && (cell_types_2D[i,j] in target_types)
            lin_idx = (j-1)*ni + i  # Same indexing as Taylor-Green
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

# Compute errors by cell type (interior cells only)
all_fluid_types = [1, -1]  # Full and cut cells
full_cell_types = [1]      # Full cells only  
cut_cell_types = [-1]      # Cut cells only

l2_all, l2_all_norm, n_all, vol_all = compute_weighted_L2_by_celltype(
    Ux_numerical, Ux_exact, cell_types_2D, Vux, ix_range, iy_range, all_fluid_types)

l2_full, l2_full_norm, n_full, vol_full = compute_weighted_L2_by_celltype(
    Ux_numerical, Ux_exact, cell_types_2D, Vux, ix_range, iy_range, full_cell_types)

l2_cut, l2_cut_norm, n_cut, vol_cut = compute_weighted_L2_by_celltype(
    Ux_numerical, Ux_exact, cell_types_2D, Vux, ix_range, iy_range, cut_cell_types)

println("Interior cell classification:")
println("  All fluid cells (interior): $(n_all)")
println("  Full cells (interior): $(n_full)")  
println("  Cut cells (interior): $(n_cut)")

println("\nVolume-weighted L2 errors (u_x, interior only):")
println("  All fluid cells:  ||e||_L2 = $(l2_all), normalized = $(l2_all_norm)")
println("  Full cells only:  ||e||_L2 = $(l2_full), normalized = $(l2_full_norm)")
println("  Cut cells only:   ||e||_L2 = $(l2_cut), normalized = $(l2_cut_norm)")

# Error ratio analysis
if n_full > 0 && n_cut > 0
    ratio = l2_cut_norm / l2_full_norm
    println("  Cut/Full error ratio: $(ratio)")
    if ratio > 2.0
        println("  → Cut cells have significantly higher error")
    elseif ratio < 0.5
        println("  → Cut cells have better accuracy (unexpected)")
    else
        println("  → Cut and full cells have comparable accuracy")
    end
else
    println("  → Cannot compare: insufficient cells of one type")
end

# Volume statistics (interior only)
println("\nInterior volume statistics:")
println("  Total fluid volume: $(vol_all)")
if vol_all > 0
    println("  Full cell volume:   $(vol_full) ($(100*vol_full/vol_all)%)")
    println("  Cut cell volume:    $(vol_cut) ($(100*vol_cut/vol_cut)%)")
end

# Cut-cell capacity analysis (all cells, not just interior)
all_cut_indices = findall(cell_types .== -1)
if !isempty(all_cut_indices)
    cut_volumes = Vux[all_cut_indices]
    min_cut_vol = minimum(cut_volumes)
    max_cut_vol = maximum(cut_volumes) 
    mean_cut_vol = sum(cut_volumes) / length(cut_volumes)
    
    println("\nCut-cell volume statistics (all cut cells):")
    println("  Number of cut cells: $(length(all_cut_indices))")
    println("  Minimum volume: $(min_cut_vol)")
    println("  Maximum volume: $(max_cut_vol)")
    println("  Mean volume: $(mean_cut_vol)")
    
    if min_cut_vol < 0.1 * (dx * dy)  # Compare to full cell volume
        println("  ⚠ Warning: Very small cut cells detected")
    end
end
