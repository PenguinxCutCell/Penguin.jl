using Penguin
using CairoMakie
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Statistics
using LinearSolve

"""
2D Stokes Poiseuille flow (steady): u = (Ux(y), 0), driven by pressure gradient.

Domain: [0, Lx] × [0, Ly]
BCs: u_x(x, 0) = 0, u_x(x, Ly) = 0; u_y = 0 on all walls.
Left/Right: enforce the fully developed parabolic profile to avoid incompatibility.
"""

###########
# Grids
###########
nx, ny = 128, 128
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, 0.0

# Pressure grid (cell-centered)
mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))

# Component-wise staggered velocity grids
dx, dy = Lx/nx, Ly/ny
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

###########
# Capacities and operators (per component)
###########
body = (x, y, _=0) -> -1.0
capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# BCs
###########
Umax = 1.0
parabola = (x, y) -> 4Umax * (y - y0) * (Ly - (y - y0)) / (Ly^2)

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

# Pressure gauge (zero reference)
pressure_gauge = PinPressureGauge()

# Cut-cell / interface BC for uγ (not used here)
u_bc = Dirichlet(0.0)

###########
# Sources and material
###########
# Body force set to zero; the solver determines a pressure gradient consistent with the profile
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

println("2D Poiseuille solved. Unknowns = ", length(solver.x))

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

fig = Figure(resolution=(1000, 420))
ax1 = Axis(fig[1,1], xlabel="u_x", ylabel="y", title="Poiseuille: mid x profile vs analytical")
lines!(ax1, ux_profile, ys, label="numerical")
lines!(ax1, ux_analytical, ys, color=:red, linestyle=:dash, label="analytical")
axislegend(ax1, position=:rb)
display(fig)
save("stokes2d_poiseuille_profile.png", fig)

# Error metrics on the profile
profile_err = ux_profile .- ux_analytical
ℓ2_profile = sqrt(sum(abs2, profile_err) / length(profile_err))
ℓinf_profile = maximum(abs, profile_err)
println("Profile L2 error = ", ℓ2_profile, ", Linf = ", ℓinf_profile)

# Reshape to field matrices for heatmaps
Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))
P  = reshape(pω,  (length(xp), length(yp)))

# Additional field plots: u_x, u_y, p
fig2 = Figure(resolution=(1200, 360))
ax2 = Axis(fig2[1,1], xlabel="x", ylabel="y", title="u_x field")
hm1 = heatmap!(ax2, xs, ys, Ux'; colormap=:viridis)
Colorbar(fig2[1,2], hm1)

ax3 = Axis(fig2[1,3], xlabel="x", ylabel="y", title="u_y field")
hm2 = heatmap!(ax3, xs, ys, Uy'; colormap=:balance)
Colorbar(fig2[1,4], hm2)

fig3 = Figure(resolution=(600, 360))
ax4 = Axis(fig3[1,1], xlabel="x", ylabel="y", title="pressure field p")
hm3 = heatmap!(ax4, xp, yp, P'; colormap=:plasma)
Colorbar(fig3[1,2], hm3)

display(fig2)
display(fig3)
save("stokes2d_poiseuille_fields.png", fig2)
save("stokes2d_poiseuille_pressure.png", fig3)

# Global sanity checks
Uy_max = maximum(abs, Uy)
P_std = std(P)
println("Sanity: max |u_y| = ", Uy_max, ", std(p) = ", P_std)
