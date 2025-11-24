using Penguin
using CairoMakie
using Statistics
using Printf

"""
Couette–Poiseuille flow in a cut-cell channel solved with Navier–Stokes (steady).

Channel: y ∈ [y_bot, y_top] embedded in the box [0, Lx] × [0, Ly], level-set defined by
    φ(x, y) = y - y_top
Boundary conditions:
  - Analytical Couette–Poiseuille profile applied at left/right and on the walls.
  - No-slip (Dirichlet) on the cut interface.
Velocity field analytical form (with moving top wall U_top and pressure gradient G):
    u_exact(y) = U_top * (y - y_bot) / h - (G / (2μ)) * (y - y_bot) * (h - (y - y_bot))
    v_exact(y) = 0
Parameters chosen: h = 0.6, y_bot = 0.2, y_top = 0.8, U_top = 1.0, G = -2.0, μ = 1.0.
The script solves the steady Navier–Stokes equations and compares the numerical
mid-line profile against the analytical solution.
"""

###########
# Geometry
###########
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, 0.0
nx, ny = 96, 96

y_bot = 0.0
y_top = 0.8
h = y_top - y_bot

# Couette–Poiseuille parameters
U_top = 1.0
G = -2.0  # imposed pressure gradient
μ = 1.0
ρ = 1.0

u_exact(y) = begin
    ξ = clamp(y, y_bot, y_top) - y_bot
    U_top * ξ / h - (G / (2μ)) * ξ * (h - ξ)
end
v_exact(y) = 0.0

body = (x, y, _=0.0) -> begin
    return y-y_top
end

mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

capacity_ux = Capacity(body, mesh_ux; compute_centroids=true)
capacity_uy = Capacity(body, mesh_uy; compute_centroids=true)
capacity_p  = Capacity(body, mesh_p; compute_centroids=true)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
parabola = (x, y, t=0.0) -> begin
    if y <= y_bot
        return 0.0
    else
        return u_exact(y)
    end
end

ux_left  = Dirichlet(parabola)
ux_right = Dirichlet(parabola)
ux_bottom = Dirichlet((x, y, t=0.0) -> u_exact(y_bot))
ux_top    = Dirichlet((x, y, t=0.0) -> u_exact(y_top))

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(:left=>ux_left, :right=>ux_right, :bottom=>ux_bottom, :top=>ux_top))
bc_uy = BorderConditions(Dict(:left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero))

pressure_gauge = PinPressureGauge()
cut_bc = Dirichlet((x, y, t=0.0) -> u_exact(y))

   

###########
# Fluid and solver
###########
Gx = -G
fᵤ = (x, y, z=0.0, t=0.0) -> (0.0 <= x <= Lx && y <= y_top) ? -G : 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
x0_vec = zeros(2 * (nu_x + nu_y) + np)

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, cut_bc; x0=x0_vec)

println("Solving steady Couette–Poiseuille with cut cells...")
_, picard_iters, picard_res = solve_NavierStokesMono_steady!(solver;
    nlsolve_method=:picard,
    tol=1e-10,
    maxiter=100,
    relaxation=0.8)
println(@sprintf("Picard iterations = %d, residual = %.3e", picard_iters, picard_res))

###########
# Post-processing
###########
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
Ux = reshape(solver.x[1:nu_x], (length(xs), length(ys)))

icol = Int(cld(length(xs), 2))
ux_profile = Ux[icol, :]
ux_analytical = [parabola(0.0, y) for y in ys]

fluid_mask = (ys .>= y_bot) .& (ys .<= y_top)
profile_err = ux_profile[fluid_mask] .- ux_analytical[fluid_mask]
L2_err = sqrt(mean(abs2, profile_err))
Linf_err = maximum(abs.(profile_err))
println(@sprintf("Profile L2 error = %.4e, Linf error = %.4e", L2_err, Linf_err))

fig = Figure(resolution=(900, 400))
ax = Axis(fig[1, 1], xlabel="u", ylabel="y", title="Mid-line profile vs analytical")
lines!(ax, ux_profile, ys; label="numerical")
lines!(ax, ux_analytical, ys; color=:red, linestyle=:dash, label="analytical")
axislegend(ax, position=:rb)

ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", title="Speed magnitude")
Uy = reshape(solver.x[2nu_x+1:2nu_x+nu_y], (length(xs), length(ys)))
speed = sqrt.(Ux.^2 .+ Uy.^2)
heatmap!(ax2, xs, ys, speed; colormap=:plasma)
contour!(ax2, xs, ys, [body(coord[1], coord[2]) for coord in Iterators.product(mesh_ux.nodes...)]; levels=[0.0], color=:white)

save("couette_poiseuille_cut_profile.png", fig)
println("Saved plot to couette_poiseuille_cut_profile.png")
