using Penguin
using LinearAlgebra
using Statistics
using DelimitedFiles
using Printf
using CairoMakie

"""
Steady lid-driven cavity benchmark at Re = 1000 using the Navier–Stokes mono solver.

The setup matches the classical Ghia et al. (1982) configuration:
unit square cavity, no-slip walls, and unit tangential lid velocity.
The script solves the flow, reports solver diagnostics, and compares
centerline velocity profiles against the reference data in `ghia/*.ghia`.
"""

###########
# Geometry and discretisation
###########
nx, ny = 128, 128
Lx, Ly = 1.0, 1.0
x0, y0 = 0.0, 0.0

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

###########
# Capacities and operators
###########
full_body = (x, y, _=0.0) -> -1.0
capacity_ux = Capacity(full_body, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(full_body, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(full_body, mesh_p;  compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
lid_speed = 1.0

ux_top    = Dirichlet((x, y, t=0.0) -> lid_speed)
ux_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
ux_left   = Dirichlet((x, y, t=0.0) -> 0.0)
ux_right  = Dirichlet((x, y, t=0.0) -> 0.0)

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left,
    :right=>ux_right,
    :bottom=>ux_bottom,
    :top=>ux_top,
))

bc_uy = BorderConditions(Dict(
    :left=>uy_zero,
    :right=>uy_zero,
    :bottom=>uy_zero,
    :top=>uy_zero,
))

pressure_gauge = PinPressureGauge()
cut_bc = Dirichlet(0.0)

###########
# Fluid properties and forcing (Re = lid_speed * L / ν = 1000)
###########
ν = 1e-3
ρ = 1.0
μ = ν * ρ
fᵤ = (x, y, z=0.0) -> 0.0
fₚ = (x, y, z=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

###########
# Solver setup
###########
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
x0_vec = zeros(2 * (nu_x + nu_y) + np)  # NavierStokesMono state layout

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, cut_bc; x0=x0_vec)

println("=== Lid-driven cavity (Navier–Stokes mono) ===")
println("Grid: $(nx) × $(ny), ν = $(ν), lid speed = $(lid_speed)")

# Picard iterations to reach a good initial state before Newton refinement.
_, picard_iters, picard_res = solve_NavierStokesMono_steady!(
    solver;
    tol=1e-6,
    maxiter=200,
    relaxation=0.7,
    nlsolve_method=:picard,
)
println("Picard iterations: iters=$(picard_iters), residual=$(picard_res)")

# Newton polish from Picard state.
_, newton_iters, newton_res = solve_NavierStokesMono_steady!(
    solver;
    tol=1e-9,
    maxiter=40,
    nlsolve_method=:newton,
)
println("Newton iterations: iters=$(newton_iters), residual=$(newton_res)")

###########
# Diagnostics
###########
blocks = Penguin.navierstokes2D_blocks(solver)
nu_x = blocks.nu_x
nu_y = blocks.nu_y

uωx = solver.x[1:nu_x]
uγx = solver.x[nu_x+1:2nu_x]
uωy = solver.x[2nu_x+1:2nu_x+nu_y]
uγy = solver.x[2nu_x+nu_y+1:2*(nu_x+nu_y)]
pω  = solver.x[2*(nu_x + nu_y)+1:end]

mass_residual = blocks.div_x_ω * Vector{Float64}(uωx) +
                blocks.div_x_γ * Vector{Float64}(uγx) +
                blocks.div_y_ω * Vector{Float64}(uωy) +
                blocks.div_y_γ * Vector{Float64}(uγy)
println(@sprintf("‖div(u)‖∞ = %.3e", maximum(abs, mass_residual)))

ke = 0.5 * (sum(abs2, uωx) + sum(abs2, uωy))
println(@sprintf("Kinetic energy = %.6e", ke))

println(@sprintf("Velocity max(|u_x|)=%.3e, max(|u_y|)=%.3e",
                 maximum(abs, uωx), maximum(abs, uωy)))
println(@sprintf("Pressure range: min=%.3e, max=%.3e",
                 minimum(pω), maximum(pω)))

###########
# Centerline profiles
###########
xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]
nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))

Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))

# Remove duplicate last entries on staggered grids when exporting profiles.
xs_vis = xs[1:end-1]
ys_vis = ys[1:end-1]

icol = nearest_index(xs, x0 + Lx / 2)
irow = nearest_index(ys, y0 + Ly / 2)
ux_centerline = vec(Ux[icol, 1:length(ys_vis)])
uy_centerline = vec(Uy[1:length(xs_vis), irow])

###########
# Ghia reference comparison
###########
function load_ghia_profile(path)
    data = readdlm(path)
    coords = data[:, 1] .+ 0.5  # stored as (coord - 0.5)
    values = data[:, 2]
    return coords, values
end

ghia_y, ghia_ux = load_ghia_profile(joinpath(@__DIR__, "ghia", "xprof.ghia"))
ghia_x, ghia_uy = load_ghia_profile(joinpath(@__DIR__, "ghia", "yprof.ghia"))

function sample_profile(grid_coords, field_vals, ref_coords)
    sampled = similar(ref_coords)
    for (k, coord) in pairs(ref_coords)
        idx = nearest_index(grid_coords, coord)
        sampled[k] = field_vals[idx]
    end
    return sampled
end

ux_sampled = sample_profile(ys_vis, ux_centerline, ghia_y)
uy_sampled = sample_profile(xs_vis, uy_centerline, ghia_x)

ux_diff = ux_sampled .- ghia_ux
uy_diff = uy_sampled .- ghia_uy

println("Centerline comparisons vs Ghia et al. (1982):")
println(@sprintf("  vertical line (u_x): Linf=%.3e, L2=%.3e",
                 maximum(abs, ux_diff), sqrt(mean(abs2, ux_diff))))
println(@sprintf("  horizontal line (u_y): Linf=%.3e, L2=%.3e",
                 maximum(abs, uy_diff), sqrt(mean(abs2, uy_diff))))

println("Sampled points (y, u_x_num, u_x_ghia):")
for k in eachindex(ghia_y)
    println(@sprintf("  %.4f  %.6f  %.6f", ghia_y[k], ux_sampled[k], ghia_ux[k]))
end

println("Sampled points (x, u_y_num, u_y_ghia):")
for k in eachindex(ghia_x)
    println(@sprintf("  %.4f  %.6f  %.6f", ghia_x[k], uy_sampled[k], ghia_uy[k]))
end

###########
# Profile plot
###########
#fig = Figure(resolution=(900, 400))

#ax_vert = Axis(fig[1,1], xlabel="u_x", ylabel="y", title="Vertical centerline (x = 0.5)")
#lines!(ax_vert, ux_centerline, ys_vis; label="numerical")
#scatter!(ax_vert, ghia_ux, ghia_y; color=:red, markersize=6, label="Ghia et al. (1982)")
#axislegend(ax_vert, position=:lb)

#ax_horiz = Axis(fig[1,2], xlabel="x", ylabel="u_y", title="Horizontal centerline (y = 0.5)")
#lines!(ax_horiz, xs_vis, uy_centerline; label="numerical")
#scatter!(ax_horiz, ghia_x, ghia_uy; color=:red, markersize=6, label="Ghia et al. (1982)")
#axislegend(ax_horiz, position=:rt)

#fig_path = joinpath(@__DIR__, "navierstokes2d_lidcavity_profiles_Re1000.png")
#save(fig_path, fig)
#println("Saved profile comparison plot to $(fig_path)")

println("Benchmark completed.")
