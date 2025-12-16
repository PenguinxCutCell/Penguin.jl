using Penguin
using CairoMakie

"""
Laminar channel flow advanced with the incremental projection Navier–Stokes
solver. Parabolic inflow is imposed on the left boundary, no-slip on walls,
and a zero-gradient outflow on the right.
"""

###########
# Geometry
###########
nx, ny = 96, 48
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, 0.0

mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

###########
# Capacities & operators
###########
full_body = (x, y, _=0.0) -> -1.0

capacity_ux = Capacity(full_body, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(full_body, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(full_body, mesh_p; compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
Umax = 1.0
parabolic = (x, y, t=0.0) -> begin
    ξ = (y - (y0 + Ly / 2)) / (Ly / 2)
    Umax * (1 - ξ^2)
end

ux_left   = Dirichlet((x, y, t=0.0) -> parabolic(x, y, t))
ux_right  = Outflow()
ux_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
ux_top    = Dirichlet((x, y, t=0.0) -> 0.0)

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left,
    :right=>ux_right,
    :bottom=>ux_bottom,
    :top=>ux_top,
))
bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero
))
pressure_gauge = MeanPressureGauge()

bc_cut = Dirichlet(0.0)

###########
# Physics
###########
μ = 1.0
ρ = 1.0
Re = ρ * Umax * Ly / μ
println("Reynolds number (based on channel height) = $(Re)")

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

solver = NavierStokesProjectionMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut; x0=x0_vec)

Δt = 0.00001
T_end = 0.25

println("Running projection Navier–Stokes simulation...")
times, histories = solve_NavierStokesProjection_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=:BE)

println("Simulation finished. Stored states = ", length(histories))
println("Final time step = ", times[end])

###########
# Post-processing
###########
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
uωx = solver.x[1:nu_x]
Ux = reshape(uωx, (length(xs), length(ys)))

ix_mid = Int(clamp(round(length(xs) / 2), 1, length(xs)))
ux_profile = Ux[ix_mid, :]

analytic_profile = [parabolic(0.0, y) for y in ys]

fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1], xlabel="y", ylabel="u_x", title="Channel centerline profile at x ≈ L/2")
lines!(ax, ys, ux_profile; label="projection solver")
lines!(ax, ys, analytic_profile; linestyle=:dash, color=:red, label="parabolic inflow")
axislegend(ax, position=:rb)

save("channel_flow_projection.png", fig)
println("Saved centerline profile to channel_flow_projection.png")
