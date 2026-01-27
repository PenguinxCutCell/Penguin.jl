using Penguin
using CairoMakie
using LinearAlgebra

"""
Steady Stokes flow (Poiseuille-like) past a circular obstacle embedded in a
channel. The circle is described by a level-set, so capacities exercise the
cut-cell path. The cut/interface BC enforces u^γ = 0.
"""

###########
# Geometry
###########
nx, ny = 32, 16
channel_length = 4.0
channel_height = 1.0
x0, y0 = -0.5, -0.5

circle_center = (0.5, 0.0)
circle_radius = 0.2

circle_body = (x, y, _=0) -> circle_radius - sqrt((x - circle_center[1])^2 + (y - circle_center[2])^2)

###########
# Meshes
###########
mesh_p  = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0 - 0.5*dy))

###########
# Capacities & operators (cut cell aware)
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
parabolic = (x, y) -> begin
    ξ = (y - (y0 + channel_height/2)) / (channel_height/2)
    Umax * (1 - ξ^2)
end

ux_left   = Dirichlet((x, y) -> parabolic(x, y))
ux_right  = Outflow()
ux_bottom = Dirichlet((x, y) -> 0.0)
ux_top    = Dirichlet((x, y) -> 0.0)

uy_zero = Dirichlet((x, y) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right, :bottom=>ux_bottom, :top=>ux_top
))
bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero
))
pressure_gauge = PinPressureGauge()

u_bc = Dirichlet(0.0)  # enforce u^γ = 0 on the interface

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

println("Stokes flow around circle solved. Unknowns = ", length(solver.x))

uωx = solver.x[1:nu]
uωy = solver.x[2nu+1:3nu]
pω  = solver.x[4nu+1:end]
println("Sanity: max |u_y| = ", maximum(abs, uωy))

###########
# Heatmap visualization of velocity components
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

ax_speed = Axis(fig[1,1], xlabel="x", ylabel="y", title="Speed magnitude", aspect=DataAspect())
hm_speed = heatmap!(ax_speed, xs, ys, speed; colormap=:plasma)
contour!(ax_speed, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig[1,2], hm_speed)

ax_pressure = Axis(fig[1,3], xlabel="x", ylabel="y", title="Pressure field", aspect=DataAspect())
hm_pressure = heatmap!(ax_pressure, Xp, Yp, P; colormap=:balance)
contour!(ax_pressure, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig[1,4], hm_pressure)

ax_stream = Axis(fig[2,1], xlabel="x", ylabel="y", title="Velocity streamlines", aspect=DataAspect())
streamplot!(ax_stream, velocity_field, xs[1]..xs[end], ys[1]..ys[end]; colormap=:thermal)
contour!(ax_stream, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:red, linewidth=2)

ax_contours = Axis(fig[2,3], xlabel="x", ylabel="y", title="Velocity magnitude contours", aspect=DataAspect())
contour!(ax_contours, xs, ys, speed; levels=10, color=:navy)
contour!(ax_contours, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:black, linewidth=2)

display(fig)
save("stokes2d_circle_flow.png", fig)

n_p = prod(operator_p.size)
n_ux = prod(operator_ux.size)
n_uy = prod(operator_uy.size)

uωx = solver.x[1:n_ux]
uγx = solver.x[n_ux+1:2n_ux]
uωy = solver.x[2n_ux+1:3n_uy]
uγy = solver.x[3n_uy+1:4n_uy]
pω  = solver.x[4n_uy+1:end]

# Discrete volume interface contribution : ∫_{Γ} μ ∇u dS ⋅ ex
Int_Shear_x = operator_ux.H' * operator_ux.Wꜝ * operator_ux.G * uωx +
               operator_ux.H' * operator_ux.Wꜝ * operator_ux.H * uγx

# Discrete volume interface contribution : ∫_{Γ} μ ∇u dS ⋅ ey
Int_Shear_y = operator_uy.H' * operator_uy.Wꜝ * operator_uy.G * uωy +
               operator_uy.H' * operator_uy.Wꜝ * operator_uy.H * uγy

# Discrete volume interface contribution : ∫_{Γ} μ ∇u^T dS ⋅ ex
Int_Shear_x_T = operator_uy.H' * operator_uy.Wꜝ * operator_uy.G * uωx +
                 operator_uy.H' * operator_uy.Wꜝ * operator_uy.H * uγx

# Discrete volume interface contribution : ∫_{Γ} μ ∇u^T dS ⋅ ey
Int_Shear_y_T = operator_ux.H' * operator_ux.Wꜝ * operator_ux.G * uωy +
                 operator_ux.H' * operator_ux.Wꜝ * operator_ux.H * uγy

# Discrete volume interface contribution : ∫_{Γ} -p n dS
# Pressure has no interface DOF (no pγ), only pω
# Following nusselt_profile pattern: H' * Wꜝ * G * field
Int_Pressure_x = -operator_ux.H' * operator_ux.Wꜝ * operator_p.G[1:n_ux, :] * pω
Int_Pressure_y = -operator_uy.H' * operator_uy.Wꜝ * operator_p.G[n_ux+1:n_ux+n_uy, :] * pω

Γ_x = diag(capacity_ux.Γ)
Γ_y = diag(capacity_uy.Γ)
Γ_p = diag(capacity_p.Γ)

# Already integrated over the interface length
shear_x_nds = Int_Shear_x
shear_x_T_nds = Int_Shear_x_T
shear_y_nds = Int_Shear_y
shear_y_T_nds = Int_Shear_y_T
pressure_x_nds = Int_Pressure_x
pressure_y_nds = Int_Pressure_y


diameter = 2 * circle_radius

force_diag = compute_navierstokes_force_diagnostics(solver)
body_force = navierstokes_reaction_force_components(force_diag; acting_on=:body)
pressure_body = -force_diag.integrated_pressure
viscous_body = -force_diag.integrated_viscous
coeffs = drag_lift_coefficients(force_diag; ρ=ρ, U_ref=Umax, length_ref=diameter, acting_on=:body)
println("Integrated forces on the cylinder (body reaction):")
println("  Fx = $(round(body_force[1]; sigdigits=6)) (pressure=$(round(pressure_body[1]; sigdigits=6)), viscous=$(round(viscous_body[1]; sigdigits=6)))")
println("  Fy = $(round(body_force[2]; sigdigits=6)) (pressure=$(round(pressure_body[2]; sigdigits=6)), viscous=$(round(viscous_body[2]; sigdigits=6)))")
println("  Drag coefficient Cd = $(round(coeffs.Cd; sigdigits=6)), lift coefficient Cl = $(round(coeffs.Cl; sigdigits=6))")

pressure_trace = pressure_trace_on_cut(solver; center=circle_center)

