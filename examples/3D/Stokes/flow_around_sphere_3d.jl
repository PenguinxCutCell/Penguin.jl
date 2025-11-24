using Penguin
using CairoMakie
using LinearAlgebra
using LinearSolve
using WriteVTK

"""
Steady Stokes flow past a spherical obstacle embedded in a 3D channel.
The sphere is represented implicitly via a level-set function feeding the
cut-cell capacity construction. The interface (cut) boundary is given a
homogeneous Dirichlet velocity (no-slip) enforcing u^γ = 0.

Domain: rectangular box of length Lx and square cross-section Ly×Lz.
Inlet/Outlet: parabolic profile in x-component, zero for y,z.
Walls: no-slip.
Sphere: centered, radius R.
"""

###########
# Geometry
###########
# Grid & domain (center sphere inside channel)
Nx, Ny, Nz = 32, 32, 16
Lx, Ly, Lz = 4.0, 1.0, 1.0
x0, y0, z0 = -Lx/2, -Ly/2, -Lz/2   # domain centered at origin

sphere_center = (0.0, 0.0, 0.0)
R = 0.2
###########
sphere_body = (x, y, z, _t=0.0) -> -(sqrt((x - sphere_center[1])^2 +
                             (y - sphere_center[2])^2 +
                             (z - sphere_center[3])^2) - R)

###########
# Meshes (staggered velocity, collocated pressure)
###########
mesh_p  = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0, z0))
Δx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
Δy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
Δz = mesh_p.nodes[3][2] - mesh_p.nodes[3][1]
mesh_ux = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0 - 0.5*Δx, y0, z0))
mesh_uy = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0 - 0.5*Δy, z0))
mesh_uz = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0, z0 - 0.5*Δz))

###########
# Capacities & Operators
###########
println("capacity ux...")
cap_ux = Capacity(sphere_body, mesh_ux; method="ImplicitIntegration")
println("capacity uy...")
cap_uy = Capacity(sphere_body, mesh_uy; method="ImplicitIntegration")
println("capacity uz...")
cap_uz = Capacity(sphere_body, mesh_uz; method="ImplicitIntegration")
println("capacity p...")
cap_p  = Capacity(sphere_body, mesh_p; method="ImplicitIntegration")

op_ux = DiffusionOps(cap_ux)
op_uy = DiffusionOps(cap_uy)
op_uz = DiffusionOps(cap_uz)
op_p  = DiffusionOps(cap_p)

###########
# Boundary Conditions
###########
Umax = 1.0
# Parabolic profile in y,z at inlet/outlet for u_x; zero elsewhere
parabolic = (x,y,z) -> begin
    ξ = (y - (y0 + Ly/2)) / (Ly/2)
    η = (z - (z0 + Lz/2)) / (Lz/2)
    Umax * (1 - ξ^2) * (1 - η^2)
end

# For ux component
ux_left   = Dirichlet((x,y,z) -> parabolic(x,y,z))
ux_right  = Outflow()
ux_bottom = Dirichlet(0.0)
ux_top    = Dirichlet(0.0)
ux_front  = Dirichlet(0.0)
ux_back   = Dirichlet(0.0)

# For uy component (all walls no-slip, inlet/outlet zero cross-flow)
uy_zero = Dirichlet(0.0)

# For uz component
uz_zero = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right,
    :bottom=>ux_bottom, :top=>ux_top,
    :front=>ux_front, :back=>ux_back
))

bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero,
    :bottom=>uy_zero, :top=>uy_zero,
    :front=>uy_zero, :back=>uy_zero
))

bc_uz = BorderConditions(Dict(
    :left=>uz_zero, :right=>uz_zero,
    :bottom=>uz_zero, :top=>uz_zero,
    :front=>uz_zero, :back=>uz_zero
))

pressure_gauge = PinPressureGauge()

# Cut-cell interface (sphere) -> enforce u^γ = 0
bc_cut = Dirichlet(0.0)

###########
# Physics
###########
μ = 1.0; ρ = 1.0
fᵤ = (x,y,z) -> 0.0
fₚ = (x,y,z) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy, mesh_uz),
              (cap_ux, cap_uy, cap_uz),
              (op_ux, op_uy, op_uz),
              mesh_p,
              cap_p,
              op_p,
              μ, ρ, fᵤ, fₚ)

###########
# Solve (steady)
###########
nu_x = prod(op_ux.size); nu_y = prod(op_uy.size); nu_z = prod(op_uz.size)
np = prod(op_p.size)
# state length = 2*(nu_x+nu_y+nu_z)+np
xlen = 2*(nu_x+nu_y+nu_z) + np
x0_vec = zeros(xlen)

solver = StokesMono(fluid, (bc_ux, bc_uy, bc_uz), pressure_gauge, bc_cut; x0=x0_vec)
solve_StokesMono!(solver; algorithm=UMFPACKFactorization())
println("3D Stokes flow around sphere solved. Unknowns = ", length(solver.x))

# Extract fields
uωx = solver.x[1:nu_x]
uωy = solver.x[2*nu_x + nu_y + 1 - (nu_y):2*nu_x + nu_y]  # We'll reshape systematically below
# Better: reconstruct via offsets
off_uωx = 0
off_uγx = nu_x
off_uωy = 2*nu_x
off_uγy = 2*nu_x + nu_y
off_uωz = 2*nu_x + 2*nu_y
off_uγz = 2*nu_x + 2*nu_y + nu_z
off_p   = 2*(nu_x + nu_y + nu_z)

Ux = reshape(solver.x[off_uωx+1:off_uωx+nu_x],
             (length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2]), length(mesh_ux.nodes[3])))
Uy = reshape(solver.x[off_uωy+1:off_uωy+nu_y],
             (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2]), length(mesh_uy.nodes[3])))
Uz = reshape(solver.x[off_uωz+1:off_uωz+nu_z],
             (length(mesh_uz.nodes[1]), length(mesh_uz.nodes[2]), length(mesh_uz.nodes[3])))
P  = reshape(solver.x[off_p+1:off_p+np],
             (length(mesh_p.nodes[1]), length(mesh_p.nodes[2]), length(mesh_p.nodes[3])))

println("Velocity magnitude stats: min=", minimum(sqrt.(Ux.^2 .+ Uy.^2 .+ Uz.^2)),
        " max=", maximum(sqrt.(Ux.^2 .+ Uy.^2 .+ Uz.^2)))

###########
# Simple slice visualization
###########
xs = mesh_p.nodes[1]; ys = mesh_p.nodes[2]; zs = mesh_p.nodes[3]
mid_z_index = Int(clamp(round(length(zs)/2), 1, length(zs)))
Ux_slice = Ux[:,:,mid_z_index]
Uy_slice = Uy[:,:,mid_z_index]
Uz_slice = Uz[:,:,mid_z_index]
P_slice  = P[:,:,mid_z_index]

speed_slice = sqrt.(Ux_slice.^2 .+ Uy_slice.^2 .+ Uz_slice.^2)

fig = Figure(resolution=(1200,600))
ax_speed = Axis(fig[1,1], title="|u| slice (z=mid)", xlabel="x", ylabel="y")
hm = heatmap!(ax_speed, mesh_ux.nodes[1], mesh_ux.nodes[2], speed_slice; colormap=:plasma)
Colorbar(fig[1,2], hm)

ax_pressure = Axis(fig[1,3], title="Pressure slice (z=mid)", xlabel="x", ylabel="y")
hm2 = heatmap!(ax_pressure, xs, ys, P_slice; colormap=:balance)
Colorbar(fig[1,4], hm2)

save("stokes3d_sphere_slice.png", fig)
println("Saved stokes3d_sphere_slice.png")

###########
# Export to VTK for ParaView inspection
###########
Ux_cut = reshape(solver.x[off_uγx+1:off_uγx+nu_x], size(Ux))
Uy_cut = reshape(solver.x[off_uγy+1:off_uγy+nu_y], size(Uy))
Uz_cut = reshape(solver.x[off_uγz+1:off_uγz+nu_z], size(Uz))

vtk_grid("stokes3d_velocity_x", mesh_ux.nodes[1], mesh_ux.nodes[2], mesh_ux.nodes[3]) do vtk
    vtk["uωx"] = Ux
    vtk["uγx"] = Ux_cut
end

vtk_grid("stokes3d_velocity_y", mesh_uy.nodes[1], mesh_uy.nodes[2], mesh_uy.nodes[3]) do vtk
    vtk["uωy"] = Uy
    vtk["uγy"] = Uy_cut
end

vtk_grid("stokes3d_velocity_z", mesh_uz.nodes[1], mesh_uz.nodes[2], mesh_uz.nodes[3]) do vtk
    vtk["uωz"] = Uz
    vtk["uγz"] = Uz_cut
end

vtk_grid("stokes3d_pressure", mesh_p.nodes[1], mesh_p.nodes[2], mesh_p.nodes[3]) do vtk
    vtk["pressure"] = P
end

println("VTK files written: stokes3d_velocity_x/yz.vtr and stokes3d_pressure.vtr (open in ParaView).")
