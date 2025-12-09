using Penguin
using CairoMakie
using LinearAlgebra
using LinearSolve

"""
Laplace-pressure jump test for the diphasic Stokes solver.

Two fluids separated by a circular interface of radius `R`. No external forcing
and no-slip on the outer boundaries; the only driving term is the surface-tension
traction jump `sigma * kappa * n` (kappa = 1/R for a circle). The expected
pressure jump is `p_in - p_out = sigma * kappa`.
"""

###########
# Geometry
###########
nx, ny = 32, 32
Lx, Ly = 2.0, 2.0
x0, y0 = -1.0, -1.0

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

radius = 0.5
center = (0.0, 0.0)
circle = (x, y, _=0.0) -> sqrt((x - center[1])^2 + (y - center[2])^2) - radius    # negative inside
circle_out = (x, y, _=0.0) -> radius - sqrt((x - center[1])^2 + (y - center[2])^2) # negative outside

###########
# Capacities and operators
###########
cap_ux_in  = Capacity(circle, mesh_ux; compute_centroids=true)
cap_uy_in  = Capacity(circle, mesh_uy; compute_centroids=true)
cap_p_in   = Capacity(circle, mesh_p; compute_centroids=true)
cap_ux_out = Capacity(circle_out, mesh_ux; compute_centroids=true)
cap_uy_out = Capacity(circle_out, mesh_uy; compute_centroids=true)
cap_p_out  = Capacity(circle_out, mesh_p; compute_centroids=true)

op_ux_in  = DiffusionOps(cap_ux_in);  op_uy_in  = DiffusionOps(cap_uy_in);  op_p_in  = DiffusionOps(cap_p_in)
op_ux_out = DiffusionOps(cap_ux_out); op_uy_out = DiffusionOps(cap_uy_out); op_p_out = DiffusionOps(cap_p_out)

###########
# Boundary conditions
###########
ux_zero = Dirichlet(0.0)
uy_zero = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(
    :left => ux_zero, :right => ux_zero,
    :bottom => ux_zero, :top => ux_zero,
))
bc_uy = BorderConditions(Dict(
    :left => uy_zero, :right => uy_zero,
    :bottom => uy_zero, :top => uy_zero,
))

bc_ux_in, bc_ux_out = bc_ux, bc_ux
bc_uy_in, bc_uy_out = bc_uy, bc_uy

pressure_gauges = (PinPressureGauge(), PinPressureGauge())

###########
# Physics
###########
sigma = 2.0
kappa = 1.0 / radius
f_u = (x, y, z=0.0) -> 0.0
f_p = (x, y, z=0.0) -> 0.0
mu_in, mu_out = 1.0, 1.0
rho_in, rho_out = 1.0, 1.0

fluid_in = Fluid((mesh_ux, mesh_uy),
                 (cap_ux_in, cap_uy_in),
                 (op_ux_in, op_uy_in),
                 mesh_p,
                 cap_p_in,
                 op_p_in,
                 mu_in, rho_in, f_u, f_p)

fluid_out = Fluid((mesh_ux, mesh_uy),
                  (cap_ux_out, cap_uy_out),
                  (op_ux_out, op_uy_out),
                  mesh_p,
                  cap_p_out,
                  op_p_out,
                  mu_out, rho_out, f_u, f_p)

###########
# Interface conditions (surface tension)
###########
normal_x = (x, y) -> (x - center[1]) / sqrt((x - center[1])^2 + (y - center[2])^2 + eps())
normal_y = (x, y) -> (y - center[2]) / sqrt((x - center[1])^2 + (y - center[2])^2 + eps())

# Traction jump: beta2 * (T_out dot n) - beta1 * (T_in dot n) = sigma * kappa * n.
# Use beta1 = beta2 = 1 so the pressure part enforces (p_out - p_in) = sigma * kappa.
flux_jump_x = FluxJump(1.0, 1.0, (x, y, _) -> sigma * kappa * normal_x(x, y))
flux_jump_y = FluxJump(1.0, 1.0, (x, y, _) -> sigma * kappa * normal_y(x, y))

ic_x = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), flux_jump_x)
ic_y = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), flux_jump_y)

###########
# Solve
###########
solver = StokesDiph(fluid_in, fluid_out,
                    (bc_ux_in, bc_uy_in),
                    (bc_ux_out, bc_uy_out),
                    (ic_x, ic_y),
                    pressure_gauges)
solve_StokesDiph!(solver; algorithm=UMFPACKFactorization())

println("Surface-tension Laplace test solved. Unknowns = ", length(solver.x))

###########
# Extract fields
###########
nu_x = prod(op_ux_in.size); nu_y = prod(op_uy_in.size)
np_in = prod(op_p_in.size); np_out = prod(op_p_out.size)
sum_nu = nu_x + nu_y
off_p_in = 2 * sum_nu
off_phase2 = off_p_in + np_in

u_in_omega_x = solver.x[1:nu_x];         u_in_gamma_x = solver.x[nu_x+1:2nu_x]
u_in_omega_y = solver.x[2nu_x+1:2nu_x+nu_y]; u_in_gamma_y = solver.x[2nu_x+nu_y+1:2nu_x+2nu_y]
p_in    = solver.x[off_p_in+1:off_p_in+np_in]

u_out_omega_x = solver.x[off_phase2+1:off_phase2+nu_x];           u_out_gamma_x = solver.x[off_phase2+nu_x+1:off_phase2+2nu_x]
u_out_omega_y = solver.x[off_phase2+2nu_x+1:off_phase2+2nu_x+nu_y]; u_out_gamma_y = solver.x[off_phase2+2nu_x+nu_y+1:off_phase2+2nu_x+2nu_y]
p_out   = solver.x[off_phase2+2 * (nu_x + nu_y) + 1:off_phase2+2 * (nu_x + nu_y) + np_out]

xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
xp = mesh_p.nodes[1];  yp = mesh_p.nodes[2]

Ux_in = reshape(u_in_omega_x, (length(xs), length(ys)))
Uy_in = reshape(u_in_omega_y, (length(xs), length(ys)))
Ux_out = reshape(u_out_omega_x, (length(xs), length(ys)))
Uy_out = reshape(u_out_omega_y, (length(xs), length(ys)))
P_in  = reshape(p_in,   (length(xp), length(yp)))
P_out = reshape(p_out,  (length(xp), length(yp)))

###########
# Diagnostics
###########
weights_in = diag(cap_p_in.V);  weights_out = diag(cap_p_out.V)
mean_p_in = dot(p_in, weights_in) / sum(weights_in)
mean_p_out = dot(p_out, weights_out) / sum(weights_out)
jump_expected = sigma * kappa
jump_measured = mean_p_in - mean_p_out

max_speed = maximum(abs, vcat(u_in_omega_x, u_in_omega_y, u_out_omega_x, u_out_omega_y))

println("Target pressure jump (p_in - p_out): ", jump_expected)
println("Measured pressure jump           : ", jump_measured)
println("Max |u| over both phases         : ", max_speed)

###########
# Plots
###########
fig_p = Figure(resolution=(1000, 420))
ax_p1 = Axis(fig_p[1, 1], xlabel="x", ylabel="y", title="Pressure inside phase")
hm_p1 = heatmap!(ax_p1, xp, yp, P_in'; colormap=:viridis)
Colorbar(fig_p[1, 2], hm_p1)

ax_p2 = Axis(fig_p[1, 3], xlabel="x", ylabel="y", title="Pressure outside phase")
hm_p2 = heatmap!(ax_p2, xp, yp, P_out'; colormap=:viridis)
Colorbar(fig_p[1, 4], hm_p2)

theta = range(0, 2pi; length=200)
circ_x = center[1] .+ radius .* cos.(theta)
circ_y = center[2] .+ radius .* sin.(theta)
lines!(ax_p1, circ_x, circ_y, color=:white, linewidth=2)
lines!(ax_p2, circ_x, circ_y, color=:white, linewidth=2)

fig_u = Figure(resolution=(1000, 420))
ax_u = Axis(fig_u[1, 1], xlabel="x", ylabel="y", title="|u| magnitude (inside)")
speed_in = sqrt.(Ux_in.^2 .+ Uy_in.^2)
hm_u = heatmap!(ax_u, xs, ys, speed_in'; colormap=:plasma)
Colorbar(fig_u[1, 2], hm_u)

ax_u2 = Axis(fig_u[1, 3], xlabel="x", ylabel="y", title="|u| magnitude (outside)")
speed_out = sqrt.(Ux_out.^2 .+ Uy_out.^2)
hm_u2 = heatmap!(ax_u2, xs, ys, speed_out'; colormap=:plasma)
Colorbar(fig_u[1, 4], hm_u2)
lines!(ax_u, circ_x, circ_y, color=:white, linewidth=2)
lines!(ax_u2, circ_x, circ_y, color=:white, linewidth=2)

display(fig_p)
display(fig_u)
save("stokes_diph_laplace_pressure.png", fig_p)
save("stokes_diph_laplace_velocity.png", fig_u)
