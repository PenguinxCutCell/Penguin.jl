using Penguin
using SparseArrays
using LinearAlgebra
using Statistics
using IterativeSolvers
using Printf

"""
2D Navier–Stokes Poiseuille flow in inclined cut-cell channels.

This benchmark mirrors `examples/2D/Stokes/poiseuille_2d_cut_inclined.jl`
but uses the steady Navier–Stokes mono-domain solver. For a set of
inclination angles it solves the flow, computes volume-weighted errors,
and prints diagnostic checks instead of plotting.
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

###########
# Helper functions
###########

# Interior indices (exclude boundary)
interior_indices(n) = 2:(n-1)

# Create inclined channel body function
function create_inclined_channel(θ, h, center_x=1.0, center_y=1.0)
    # Channel walls are parallel lines at distance h apart, rotated by angle θ
    nx, ny = -sin(θ), cos(θ)

    return (x, y, _=0) -> begin
        dx, dy = x - center_x, y - center_y
        dist_to_center = abs(nx * dx + ny * dy)
        return dist_to_center - h/2
    end
end

# Analytical solution for inclined Poiseuille flow
function analytical_velocity_inclined(θ, h, Umax)
    return (x, y, center_x=1.0, center_y=1.0) -> begin
        dx, dy = x - center_x, y - center_y
        n_coord = -dx * sin(θ) + dy * cos(θ) + h/2  # distance from bottom wall

        if n_coord < 0 || n_coord > h
            return (0.0, 0.0)
        end

        u_mag = 4 * Umax * n_coord * (h - n_coord) / h^2
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

println("=== Inclined Channel Navier–Stokes Benchmark ===")
println("Angles tested: $(angles_deg) degrees")
println("Resolution: $(nx) × $(ny)")
println("Channel height: $(channel_height)")
println()

results = Vector{Dict{Symbol,Any}}()

for (i, θ) in enumerate(angles_rad)
    θ_deg = angles_deg[i]
    println("Solving for θ = $(θ_deg)° ...")

    # Geometry and analytical solution
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
    capacity_p  = Capacity(body, mesh_p;  compute_centroids=false)
    operator_ux = DiffusionOps(capacity_ux)
    operator_uy = DiffusionOps(capacity_uy)
    operator_p  = DiffusionOps(capacity_p)

    # Boundary conditions (enforce analytical solution at boundaries)
    ux_bc_func = (x, y) -> analytical_vel(x, y)[1]
    uy_bc_func = (x, y) -> analytical_vel(x, y)[2]

    bc_ux = BorderConditions(Dict(
        :left   => Dirichlet(ux_bc_func),
        :right  => Dirichlet(ux_bc_func),
        :bottom => Dirichlet(ux_bc_func),
        :top    => Dirichlet(ux_bc_func),
    ))

    bc_uy = BorderConditions(Dict(
        :left   => Dirichlet(uy_bc_func),
        :right  => Dirichlet(uy_bc_func),
        :bottom => Dirichlet(uy_bc_func),
        :top    => Dirichlet(uy_bc_func),
    ))

    pressure_gauge = PinPressureGauge()
    u_bc = Dirichlet(0.0)  # no-slip on cut boundaries

    # Sources and material
    fᵤ = (x, y, z=0.0) -> 0.0
    fₚ = (x, y, z=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (capacity_ux, capacity_uy),
                  (operator_ux, operator_uy),
                  mesh_p, capacity_p, operator_p,
                  μ, ρ, fᵤ, fₚ)

    nu = prod(operator_ux.size)
    np = prod(operator_p.size)
    x0_init = zeros(4*nu + np)

    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, u_bc; x0=x0_init)
    _, iters, res = solve_NavierStokesMono_steady!(solver; tol=1e-6, maxiter=100, relaxation=1.0)
    println("  steady solve iterations=$(iters), residual=$(res)")

    # Extract solution
    uωx = solver.x[1:nu]
    uωy = solver.x[2nu+1:3nu]

    # Error analysis
    xs = mesh_ux.nodes[1]
    ys = mesh_ux.nodes[2]
    cell_types_x = capacity_ux.cell_types
    cell_types_y = capacity_uy.cell_types
    Vux = diag(operator_ux.V)
    Vuy = diag(operator_uy.V)

    cell_types_x_2D = reshape(cell_types_x, (length(xs), length(ys)))
    cell_types_y_2D = reshape(cell_types_y, (length(xs), length(ys)))
    Ux_num = reshape(uωx, (length(xs), length(ys)))
    Uy_num = reshape(uωy, (length(xs), length(ys)))

    Ux_exact = [analytical_vel(xs[i], ys[j])[1] for i in 1:length(xs), j in 1:length(ys)]
    Uy_exact = [analytical_vel(xs[i], ys[j])[2] for i in 1:length(xs), j in 1:length(ys)]

    ix_range = interior_indices(length(xs))
    iy_range = interior_indices(length(ys))
    all_fluid_types = [1, -1]
    cut_cell_types = [-1]
    full_cell_types = [1]

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

    l2_total_all = sqrt(l2_ux_all^2 + l2_uy_all^2)
    l2_total_cut = sqrt(l2_ux_cut^2 + l2_uy_cut^2)
    l2_total_full = sqrt(l2_ux_full^2 + l2_uy_full^2)

    ratio = (n_ux_full + n_uy_full) > 0 ? l2_total_cut / max(l2_total_full, eps()) : NaN

    println(@sprintf("  L2_total=%.3e  (cut=%.3e, full=%.3e)", l2_total_all, l2_total_cut, l2_total_full))
    println("  checks:")
    println("    - residual < 1e-6 ", res < 1e-6)
    println("    - cut/full error ratio < 10? ", ratio < 10)
    println("    - any cut cells? ", (n_ux_cut + n_uy_cut) > 0)

    push!(results, Dict(
        :angle_deg => θ_deg,
        :l2_total_all => l2_total_all,
        :l2_total_cut => l2_total_cut,
        :l2_total_full => l2_total_full,
        :n_cut_cells => n_ux_cut + n_uy_cut,
        :n_full_cells => n_ux_full + n_uy_full,
        :residual => res,
        :iterations => iters,
        :ratio => ratio,
    ))
end

println()
println("=== Summary Results ===")
println("Angle[°]  iterations  residual     L2_total     cut/full ratio  #cut")
println("---------------------------------------------------------------------")
for r in results
    @printf("%6.0f %11d %10.3e %11.3e %15.3e %6d\n",
            r[:angle_deg], r[:iterations], r[:residual], r[:l2_total_all], r[:ratio], r[:n_cut_cells])
end
println()
println("Benchmark completed.")
