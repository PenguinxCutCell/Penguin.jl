using Penguin
using CairoMakie
using LinearAlgebra

"""
Unsteady Stokes solution of the lid-driven cavity in 2D. We start from rest,
advance with a Crank–Nicolson scheme, and visualise the final velocity/pressure
fields together with simple diagnostics.
"""

###########
# Geometry
###########
nx, ny = 64, 64
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

ux_left   = Dirichlet((x, y, t) -> 0.0)
ux_right  = Dirichlet((x, y, t) -> 0.0)
ux_bottom = Dirichlet((x, y, t) -> 0.0)
ux_top    = Dirichlet((x, y, t) -> U_lid)

uy_zero = Dirichlet((x, y, t) -> 0.0)

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
# Time integration setup
###########
Δt = 0.01
T_end = 1.0
scheme = :CN

nu = prod(operator_ux.size)
np = prod(operator_p.size)
x0_vec = zeros(4*nu + np)

solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, u_bc; x0=x0_vec)

println("Running unsteady lid-driven cavity with Δt=$(Δt), T_end=$(T_end), scheme=$(scheme)")
times, states = solve_StokesMono_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=scheme, method=Base.:\)

println("Completed steps = ", length(times)-1)

final_state = states[end]

uωx = final_state[1:nu]
uωy = final_state[2nu+1:3nu]
pω  = final_state[4nu+1:end]
println("Sanity: max |u_y| = ", maximum(abs, uωy))

###########
# Visualisation of final state
###########
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
Xp = mesh_p.nodes[1];  Yp = mesh_p.nodes[2]
Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))
P  = reshape(pω,  (length(Xp), length(Yp)))

speed = sqrt.(Ux.^2 .+ Uy.^2)

nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))
velocity_field(x, y) = Point2f(Ux[nearest_index(xs, x), nearest_index(ys, y)],
                               Uy[nearest_index(xs, x), nearest_index(ys, y)])

fig = Figure(resolution=(1200, 700))
ax_speed = Axis(fig[1,1], xlabel="x", ylabel="y", title="Speed magnitude (t = $(times[end]))")
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
save("stokes2d_lid_cavity_unsteady_fields.png", fig)

###########
# Centerline profiles (final time)
###########
icol = nearest_index(xs, x0 + Lx/2)
row  = nearest_index(ys, y0 + Ly/2)
u_center_vertical = Ux[icol, :]
v_center_horizontal = Uy[:, row]

fig_profiles = Figure(resolution=(800, 350))
ax_vert = Axis(fig_profiles[1,1], xlabel="u_x", ylabel="y",
               title="Vertical centerline u_x (final time)")
lines!(ax_vert, u_center_vertical, ys)

ax_horiz = Axis(fig_profiles[1,2], xlabel="x", ylabel="u_y",
                title="Horizontal centerline u_y (final time)")
lines!(ax_horiz, xs, v_center_horizontal)

display(fig_profiles)
save("stokes2d_lid_cavity_unsteady_profiles.png", fig_profiles)

###########
# Simple diagnostics
###########
function kinetic_energy(uωx, uωy, Vx, Vy, ρ)
    kx = sum((uωx.^2) .* diag(Vx))
    ky = sum((uωy.^2) .* diag(Vy))
    return 0.5 * ρ * (kx + ky)
end

energies = map(states) do state
    ux = state[1:nu]
    uy = state[2nu+1:3nu]
    kinetic_energy(ux, uy, operator_ux.V, operator_uy.V, ρ)
end

fig_energy = Figure(resolution=(700, 300))
ax_energy = Axis(fig_energy[1,1], xlabel="time", ylabel="0.5 ρ ||u||²", title="Kinetic energy history")
lines!(ax_energy, times, energies)
display(fig_energy)
save("stokes2d_lid_cavity_unsteady_energy.png", fig_energy)

println("Final kinetic energy = ", last(energies))
