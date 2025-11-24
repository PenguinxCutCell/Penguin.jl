using Penguin
using CairoMakie
using LinearAlgebra

"""
Steady Navier–Stokes flow past a circular obstacle solved with both Picard
and Newton nonlinear solvers. This script runs the steady solver twice (first
with Picard, then Newton), records residual histories and the mass residual
(div u), and creates comparison plots.
"""

###########
# Geometry
###########
nx, ny = 128, 64
channel_length = 4.0
channel_height = 1.0
x0, y0 = -0.5, -0.5

circle_center = (0.5, 0.0)
circle_radius = 0.2
circle_body = (x, y, _=0.0) -> circle_radius - sqrt((x - circle_center[1])^2 + (y - circle_center[2])^2)

###########
# Meshes
###########
mesh_p  = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0 - 0.5 * dy))

###########
# Capacities & operators
###########
capacity_ux = Capacity(circle_body, mesh_ux)
capacity_uy = Capacity(circle_body, mesh_uy)
capacity_p  = Capacity(circle_body, mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
Umax = 1.0
parabolic = (x, y, t=0.0) -> begin
    ξ = (y - (y0 + channel_height / 2)) / (channel_height / 2)
    Umax * (1 - ξ^2)
end

ux_left   = Dirichlet((x, y, t=0.0) -> parabolic(x, y, t))
ux_right  = Dirichlet((x, y, t=0.0) -> parabolic(x, y, t))
ux_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
ux_top    = Dirichlet((x, y, t=0.0) -> 0.0)

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right, :bottom=>ux_bottom, :top=>ux_top
))
bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero
))
pressure_gauge = MeanPressureGauge()

interface_bc = Dirichlet(0.0)

###########
# Physics
###########
μ = 0.01 
ρ = 1.0
println("Re=", ρ * Umax * (2 * circle_radius) / μ)
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

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

x0_vec = zeros(2 * (nu_x + nu_y) + np)

# helper to run steady solve with specified nonlinear method
function run_steady(method::Symbol; tol=1e-8, maxiter=50, relaxation=1.0)
    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)
    println("Running steady solver with method=", method)
    iters, final_res = solve_NavierStokesMono_steady!(solver; nlsolve_method=method, tol=tol, maxiter=maxiter, relaxation=relaxation)

    data = Penguin.navierstokes2D_blocks(solver)

    # extract velocity and mass residual
    uωx = solver.x[1:nu_x]
    uωy = solver.x[2nu_x+1:2nu_x+nu_y]
    pω  = solver.x[2*(nu_x + nu_y)+1:end]

    # compute mass residual (div u) using block matrices if available
    mass_residual = try
        data.div_x_ω * Vector{Float64}(uωx) + data.div_x_γ * zeros(length(uωx)) +
        data.div_y_ω * Vector{Float64}(uωy) + data.div_y_γ * zeros(length(uωy))
    catch
        zeros(np)
    end

    return (solver, iters, final_res, solver.residual_history, mass_residual)
end

# Run Picard and Newton independently (do not reuse Picard solution as Newton's initial guess)
picard_solver, picard_iters, picard_res, picard_hist, picard_mass = run_steady(:picard; tol=eps(), maxiter=30, relaxation=1.0)

solver_newton, newton_iters, newton_res, newton_hist, newton_mass = run_steady(:newton; tol=eps(), maxiter=30, relaxation=1.1)

# Diagnostics
println("Picard: iters=", picard_iters, ", final residual=", picard_res)
println("Newton: iters=", newton_iters, ", final residual=", newton_res)

# Mass residual norms
picard_mass_norm = maximum(abs, picard_mass)
newton_mass_norm = maximum(abs, newton_mass)

println("max|div u| (Picard) = ", picard_mass_norm)
println("max|div u| (Newton)  = ", newton_mass_norm)

###########
# Plots
###########
# Convergence histories
fig_conv = Figure(resolution=(800,300))
ax1 = Axis(fig_conv[1,1], xlabel="iteration", ylabel="residual (max|Δu|)", yscale=log10, title="Nonlinear solver convergence")
lines!(ax1, 1:length(picard_hist), picard_hist, label="Picard", linewidth=2)
lines!(ax1, 1:length(newton_hist), newton_hist, label="Newton", linewidth=2)
axislegend(ax1)

# Mass residual comparison (reshape to pressure grid if possible)
Xs = mesh_p.nodes[1]
Ys = mesh_p.nodes[2]

mass_pic_mat = try
    reshape(picard_mass, (length(Xs), length(Ys)))
catch
    reshape(picard_mass, (length(Ys), length(Xs)))'
end

# reshape Newton mass residual (we computed it already as newton_mass)
mass_new_mat = try
    reshape(newton_mass, (length(Xs), length(Ys)))
catch
    reshape(newton_mass, (length(Ys), length(Xs)))'
end

# Remove ghost cells for visualization (assume last two entries are ghost rows/cols)
mass_pic_mat = abs.(mass_pic_mat[2:end-2, 2:end-2])
mass_new_mat = abs.(mass_new_mat[2:end-2, 2:end-2])

# Adjust coordinate vectors accordingly
Xs_vis = Xs[2:end-2]
Ys_vis = Ys[2:end-2]

fig_mass = Figure(resolution=(1000,400))
axp = Axis(fig_mass[1,2], title="Mass residual (Picard)")
hm1 = heatmap!(axp, Xs, Ys, mass_pic_mat; colormap=:balance)
axp = Axis(fig_mass[1,1], title="Mass residual (Picard)")
hm1 = heatmap!(axp, Xs_vis, Ys_vis, mass_pic_mat; colormap=:balance)
Colorbar(fig_mass[1,2], hm1)

axn = Axis(fig_mass[1,3], title="Mass residual (Newton)")
hm2 = heatmap!(axn, Xs_vis, Ys_vis, mass_new_mat; colormap=:balance)
Colorbar(fig_mass[1,4], hm2)

save("navierstokes2d_steady_convergence.png", fig_conv)
save("navierstokes2d_steady_mass_residuals.png", fig_mass)

println("Plots saved: navierstokes2d_steady_convergence.png, navierstokes2d_steady_mass_residuals.png")

###########
# Streamline plot for the Newton steady solution
###########
# Extract velocity from Newton solver
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
uωx_new = solver_newton.x[1:nu_x]
uωy_new = solver_newton.x[2nu_x+1:2nu_x+nu_y]

xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]

Ux_new = reshape(uωx_new, (length(xs), length(ys)))
Uy_new = reshape(uωy_new, (length(xs), length(ys)))

nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))
velocity_field_new(x, y) = Point2f(Ux_new[nearest_index(xs, x), nearest_index(ys, y)],
                                    Uy_new[nearest_index(xs, x), nearest_index(ys, y)])
fig_stream = Figure(resolution=(800,600))
ax_s = Axis(fig_stream[1,1], xlabel="x", ylabel="y", title="Steady streamlines (Newton)")

# Use a callable color function (Makie expects a function), map every point to 0.0 and use a gray colormap
streamplot!(ax_s, velocity_field_new, xs[1]..xs[end], ys[1]..ys[end]; density=2.0, color=(p)->0.0)
contour!(ax_s, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:red, linewidth=2)
display(fig_stream)
save("navierstokes2d_steady_streamlines.png", fig_stream)

println("Saved steady streamline plot: navierstokes2d_steady_streamlines.png")
