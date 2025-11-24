using Penguin
using CairoMakie
using LinearAlgebra

"""
Steady Stokes solution of the lid-driven cavity in 2D. The square cavity has
no-slip walls except for the top lid, which moves rightward with unit
velocity. We display speed, pressure, streamlines, and velocity magnitude
contours.
"""

###########
# Geometry
###########
nx, ny = 128, 128
Lx, Ly = 1.0, 1.0
x0, y0 = 0.0, 0.0

body = (x, y, _=0) -> -1.0  # entire domain is fluid

###########
# Meshes
###########
mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

###########
# Capacities & operators
###########
capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
U_lid = 1.0

ux_left   = Dirichlet((x, y) -> 0.0)
ux_right  = Dirichlet((x, y) -> 0.0)
ux_bottom = Dirichlet((x, y) -> 0.0)
ux_top    = Dirichlet((x, y) -> U_lid)

uy_zero = Dirichlet((x, y) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right, :bottom=>ux_bottom, :top=>ux_top
))
bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero
))
pressure_gauge = PinPressureGauge()

u_bc = Dirichlet(0.0)

###########
# Physics
###########
μ = 1.0
ρ = 1.0
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
# Solve
###########
nu = prod(operator_ux.size)
np = prod(operator_p.size)
x0_vec = zeros(4*nu + np)

solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, u_bc; x0=x0_vec)
solve_StokesMono!(solver; method=Base.:\)

println("Lid-driven cavity solved. Unknowns = ", length(solver.x))

uωx = solver.x[1:nu]
uωy = solver.x[2nu+1:3nu]
pω  = solver.x[4nu+1:end]
println("Sanity: max |u_y| = ", maximum(abs, uωy))

###########
# Visualisation
###########
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
Xp = mesh_p.nodes[1];  Yp = mesh_p.nodes[2]
Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))
P  = reshape(pω, (length(Xp), length(Yp)))

speed = sqrt.(Ux.^2 .+ Uy.^2)

nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))
velocity_field(x, y) = Point2f(Ux[nearest_index(xs, x), nearest_index(ys, y)],
                               Uy[nearest_index(xs, x), nearest_index(ys, y)])

fig = Figure(resolution=(1200, 700))

ax_speed = Axis(fig[1,1], xlabel="x", ylabel="y", title="Speed magnitude")
hm_speed = heatmap!(ax_speed, xs, ys, speed; colormap=:viridis)
Colorbar(fig[1,2], hm_speed)

ax_pressure = Axis(fig[1,3], xlabel="x", ylabel="y", title="Pressure field")
hm_pressure = heatmap!(ax_pressure, Xp, Yp, P; colormap=:balance)
Colorbar(fig[1,4], hm_pressure)

ax_stream = Axis(fig[2,1], xlabel="x", ylabel="y", title="Velocity streamlines")
streamplot!(ax_stream, velocity_field, xs[1]..xs[end], ys[1]..ys[end]; colormap=:thermal)

ax_contours = Axis(fig[2,3], xlabel="x", ylabel="y", title="Velocity magnitude contours")
contour!(ax_contours, xs, ys, speed; levels=20, color=:navy)

display(fig)

###########
# Centerline profiles
###########
icol = nearest_index(xs, x0 + Lx/2)
row  = nearest_index(ys, y0 + Ly/2)
u_center_vertical = Ux[icol, :]
v_center_horizontal = Uy[:, row]

fig_profiles = Figure(resolution=(800, 350))
ax_vert = Axis(fig_profiles[1,1], xlabel="u_x", ylabel="y",
               title="Vertical centerline u_x(x=0.5)")
lines!(ax_vert, u_center_vertical, ys)

ax_horiz = Axis(fig_profiles[1,2], xlabel="x", ylabel="u_y",
                title="Horizontal centerline u_y(y=0.5)")
lines!(ax_horiz, xs, v_center_horizontal)

display(fig_profiles)
save("stokes2d_lid_driven_cavity.png", fig)
save("stokes2d_lidcavity_profile.png", fig_profiles)
