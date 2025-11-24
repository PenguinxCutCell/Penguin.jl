using Penguin
using LinearAlgebra
using LinearSolve

"""
Steady 3D Navier–Stokes channel driven by a parabolic inflow.
Outlet uses an outflow condition; remaining walls are no-slip.
"""

###########
# Geometry / Meshes
###########
Nx, Ny, Nz = 16, 16, 12
Lx, Ly, Lz = 2.0, 1.0, 0.5
x0, y0, z0 = 0.0, -Ly/2, -Lz/2

body = (x, y, z, _t=0.0) -> -1.0

mesh_p  = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0, z0))
Δx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
Δy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
Δz = mesh_p.nodes[3][2] - mesh_p.nodes[3][1]
mesh_ux = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0 - 0.5*Δx, y0, z0))
mesh_uy = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0 - 0.5*Δy, z0))
mesh_uz = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0, z0 - 0.5*Δz))

cap_ux = Capacity(body, mesh_ux)
cap_uy = Capacity(body, mesh_uy)
cap_uz = Capacity(body, mesh_uz)
cap_p  = Capacity(body, mesh_p)

op_ux = DiffusionOps(cap_ux)
op_uy = DiffusionOps(cap_uy)
op_uz = DiffusionOps(cap_uz)
op_p  = DiffusionOps(cap_p)

###########
# Boundary conditions
###########
Umax = 1.0
parabolic = (x,y,z) -> begin
    ξ = (y - (y0 + Ly/2)) / (Ly/2)
    η = (z - (z0 + Lz/2)) / (Lz/2)
    Umax * (1 - ξ^2) * (1 - η^2)
end

bc_ux = BorderConditions(Dict(
    :left     => Dirichlet(parabolic),
    :right    => Outflow(),
    :bottom   => Dirichlet(0.0),
    :top      => Dirichlet(0.0),
    :front    => Dirichlet(0.0),
    :back     => Dirichlet(0.0)
))

no_slip = Dirichlet(0.0)
bc_uy = BorderConditions(Dict(
    :left=>no_slip, :right=>no_slip,
    :bottom=>no_slip, :top=>no_slip,
    :front=>no_slip, :back=>no_slip
))
bc_uz = BorderConditions(Dict(
    :left=>no_slip, :right=>no_slip,
    :bottom=>no_slip, :top=>no_slip,
    :front=>no_slip, :back=>no_slip
))

pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)

###########
# Physics
###########
μ = 1.0
ρ = 1.0
fᵤ = (x,y,z) -> 0.0
fₚ = (x,y,z) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy, mesh_uz),
              (cap_ux, cap_uy, cap_uz),
              (op_ux, op_uy, op_uz),
              mesh_p,
              cap_p,
              op_p,
              μ, ρ, fᵤ, fₚ)

nu_x = prod(op_ux.size); nu_y = prod(op_uy.size); nu_z = prod(op_uz.size)
np = prod(op_p.size)
xlen = 2 * (nu_x + nu_y + nu_z) + np
x0_vec = zeros(xlen)

solver = NavierStokesMono(fluid, (bc_ux, bc_uy, bc_uz), pressure_gauge, bc_cut; x0=x0_vec)
solve_NavierStokesMono_steady!(solver; nlsolve_method=:picard, relaxation=0.8,
                               tol=1e-6, maxiter=20, method=Base.:\)

println("3D Navier–Stokes channel solved. Unknowns = ", length(solver.x))

off_uωx = 0
off_uγx = nu_x
off_uωy = 2 * nu_x
off_uγy = 2 * nu_x + nu_y
off_uωz = 2 * nu_x + 2 * nu_y
off_uγz = 2 * nu_x + 2 * nu_y + nu_z
off_p   = 2 * (nu_x + nu_y + nu_z)

Ux = reshape(solver.x[off_uωx+1:off_uωx+nu_x],
             (length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2]), length(mesh_ux.nodes[3])))
Uy = reshape(solver.x[off_uωy+1:off_uωy+nu_y],
             (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2]), length(mesh_uy.nodes[3])))
Uz = reshape(solver.x[off_uωz+1:off_uωz+nu_z],
             (length(mesh_uz.nodes[1]), length(mesh_uz.nodes[2]), length(mesh_uz.nodes[3])))

speed = sqrt.(Ux.^2 .+ Uy.^2 .+ Uz.^2)
println("Velocity magnitude range: ", minimum(speed), " .. ", maximum(speed))
