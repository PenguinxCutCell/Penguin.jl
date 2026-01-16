using Penguin
using Printf

"""
Steady 2D Navier–Stokes flow past a circular cylinder, with drag/lift
computed from `compute_navierstokes_force_diagnostics`.

Set `Cd_ref`/`Cl_ref` below if you want to compare against a theoretical
or reference value.
"""

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Re = 20.0
U_ref = 1.0
diameter = 1.0
radius = diameter / 2

ρ = 1.0
μ = ρ * U_ref * diameter / Re

# Optional reference values for comparison
Cd_ref = nothing  # e.g. 1.4
Cl_ref = 0.0

# ---------------------------------------------------------------------------
# Domain and geometry
# ---------------------------------------------------------------------------
Lx, Ly = 10.0, 10.0
x0, y0 = -Lx / 2, -Ly / 2
cylinder_center = (0.0, 0.0)

circle_body = (x, y, _=0.0) -> radius - hypot(x - cylinder_center[1], y - cylinder_center[2])

nx, ny = 256, 128  # number of pressure cells in each direction
mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))

# Staggered velocity grids
_dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
_dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * _dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * _dy))

# ---------------------------------------------------------------------------
# Capacities and operators
# ---------------------------------------------------------------------------
capacity_ux = Capacity(circle_body, mesh_ux)
capacity_uy = Capacity(circle_body, mesh_uy)
capacity_p  = Capacity(circle_body, mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# ---------------------------------------------------------------------------
# Boundary conditions (uniform inflow, open outflow)
# ---------------------------------------------------------------------------
ux_left   = Dirichlet((x, y, t=0.0) -> U_ref)
ux_right  = Outflow()
ux_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
ux_top    = Dirichlet((x, y, t=0.0) -> 0.0)

uy_left   = Dirichlet((x, y, t=0.0) -> 0.0)
uy_right  = Outflow()
uy_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
uy_top    = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left => ux_left,
    :right => ux_right,
    :bottom => ux_bottom,
    :top => ux_top
))

bc_uy = BorderConditions(Dict(
    :left => uy_left,
    :right => uy_right,
    :bottom => uy_bottom,
    :top => uy_top
))

pressure_gauge = PinPressureGauge()
interface_bc = Dirichlet(0.0)

# ---------------------------------------------------------------------------
# Physics and solver setup
# ---------------------------------------------------------------------------
f_u = (x, y, z=0.0, t=0.0) -> 0.0
f_p = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, f_u, f_p)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)

x0_vec = ones(2 * (nu_x + nu_y) + np)

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)

println("Steady Navier–Stokes around a cylinder")
println(@sprintf("Re=%.3f, grid=%dx%d", Re, nx, ny))

iters, final_res = solve_NavierStokesMono_steady!(solver; nlsolve_method=:picard, tol=1e-9, maxiter=60, relaxation=1.0)

#println(@sprintf("Converged in %d iterations, residual=%.3e", iters, final_res))

# ---------------------------------------------------------------------------
# Force diagnostics
# ---------------------------------------------------------------------------
force_diag = compute_navierstokes_force_diagnostics(solver)
body_force = navierstokes_reaction_force_components(force_diag; acting_on=:body)
coeffs = drag_lift_coefficients(force_diag; ρ=ρ, U_ref=U_ref, length_ref=diameter, acting_on=:body)

pressure_body = -force_diag.integrated_pressure
viscous_body = -force_diag.integrated_viscous

println("Integrated forces on the cylinder (body reaction):")
println(@sprintf("  Fx = %.6e (pressure=%.6e, viscous=%.6e)", body_force[1], pressure_body[1], viscous_body[1]))
println(@sprintf("  Fy = %.6e (pressure=%.6e, viscous=%.6e)", body_force[2], pressure_body[2], viscous_body[2]))
println(@sprintf("  Cd = %.6f, Cl = %.6f", coeffs.Cd, coeffs.Cl))

if Cd_ref !== nothing
    cd_err = abs(coeffs.Cd - Cd_ref) / max(abs(Cd_ref), eps())
    println(@sprintf("  Cd reference = %.6f, relative error = %.3e", Cd_ref, cd_err))
end

if Cl_ref != 0.0
    cl_err = abs(coeffs.Cl - Cl_ref) / max(abs(Cl_ref), eps())
    println(@sprintf("  Cl reference = %.6f, relative error = %.3e", Cl_ref, cl_err))
end
