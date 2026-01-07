using Penguin
using CairoMakie
using LinearAlgebra
using Printf
using DelimitedFiles
using IterativeSolvers

"""
Stokes flow around a sphere in a large box with uniform far-field velocity.
This adapts the Lamb analytical solution for comparison along a 45-degree ray
in the x-y plane (z = 0).

Note: to better approximate an unbounded domain, increase L and resolution.
"""

# ---------------------------------------------------------------------------
# Geometry and meshes
# ---------------------------------------------------------------------------
Nx, Ny, Nz = 12, 12, 12
L = 24.0
x0 = -L / 2
y0 = -L / 2
z0 = -L / 2

R = 1.0
sphere_center = (0.0, 0.0, 0.0)
sphere_body = (x, y, z, _t=0.0) -> R - sqrt((x - sphere_center[1])^2 +
                                            (y - sphere_center[2])^2 +
                                            (z - sphere_center[3])^2)

mesh_p = Penguin.Mesh((Nx, Ny, Nz), (L, L, L), (x0, y0, z0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
dz = mesh_p.nodes[3][2] - mesh_p.nodes[3][1]
mesh_ux = Penguin.Mesh((Nx, Ny, Nz), (L, L, L), (x0 - 0.5 * dx, y0, z0))
mesh_uy = Penguin.Mesh((Nx, Ny, Nz), (L, L, L), (x0, y0 - 0.5 * dy, z0))
mesh_uz = Penguin.Mesh((Nx, Ny, Nz), (L, L, L), (x0, y0, z0 - 0.5 * dz))

# ---------------------------------------------------------------------------
# Capacities/operators
# ---------------------------------------------------------------------------
cap_ux = Capacity(sphere_body, mesh_ux; method="VOFI", integration_method=:vofijul)
cap_uy = Capacity(sphere_body, mesh_uy; method="VOFI", integration_method=:vofijul)
cap_uz = Capacity(sphere_body, mesh_uz; method="VOFI", integration_method=:vofijul)
cap_p  = Capacity(sphere_body, mesh_p;  method="VOFI", integration_method=:vofijul)

op_ux = DiffusionOps(cap_ux)
op_uy = DiffusionOps(cap_uy)
op_uz = DiffusionOps(cap_uz)
op_p  = DiffusionOps(cap_p)

nu_x = prod(op_ux.size)
nu_y = prod(op_uy.size)
nu_z = prod(op_uz.size)
np = prod(op_p.size)

# ---------------------------------------------------------------------------
# Boundary conditions (uniform far-field velocity)
# ---------------------------------------------------------------------------
U_inf = 1.0
ux_far = Dirichlet((x, y, z) -> U_inf)
uy_far = Dirichlet(0.0)
uz_far = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_far, :right=>ux_far,
    :bottom=>ux_far, :top=>ux_far,
    :front=>ux_far, :back=>ux_far
))

bc_uy = BorderConditions(Dict(
    :left=>uy_far, :right=>uy_far,
    :bottom=>uy_far, :top=>uy_far,
    :front=>uy_far, :back=>uy_far
))

bc_uz = BorderConditions(Dict(
    :left=>uz_far, :right=>uz_far,
    :bottom=>uz_far, :top=>uz_far,
    :front=>uz_far, :back=>uz_far
))

pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)  # no-slip on sphere

# ---------------------------------------------------------------------------
# Physics and solver
# ---------------------------------------------------------------------------
mu = 1.0
rho = 1.0
f_u = (x, y, z) -> 0.0
f_p = (x, y, z) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy, mesh_uz),
              (cap_ux, cap_uy, cap_uz),
              (op_ux, op_uy, op_uz),
              mesh_p,
              cap_p,
              op_p,
              mu, rho, f_u, f_p)

xlen = 2 * (nu_x + nu_y + nu_z) + np
x0_vec = zeros(xlen)


solver = StokesMono(fluid, (bc_ux, bc_uy, bc_uz), pressure_gauge, bc_cut; x0=x0_vec)
println("solving Stokes flow around sphere...")
solve_StokesMono!(solver; method=IterativeSolvers.bicgstabl)

# ---------------------------------------------------------------------------
# Extract fields
# ---------------------------------------------------------------------------
off_uox = 0
off_ugx = nu_x
off_uoy = 2 * nu_x
off_ugy = 2 * nu_x + nu_y
off_uoz = 2 * nu_x + 2 * nu_y
off_ugz = 2 * nu_x + 2 * nu_y + nu_z
off_p   = 2 * (nu_x + nu_y + nu_z)

Ux = reshape(solver.x[off_uox+1:off_uox+nu_x],
             (length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2]), length(mesh_ux.nodes[3])))
Uy = reshape(solver.x[off_uoy+1:off_uoy+nu_y],
             (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2]), length(mesh_uy.nodes[3])))
Uz = reshape(solver.x[off_uoz+1:off_uoz+nu_z],
             (length(mesh_uz.nodes[1]), length(mesh_uz.nodes[2]), length(mesh_uz.nodes[3])))
P  = reshape(solver.x[off_p+1:off_p+np],
             (length(mesh_p.nodes[1]), length(mesh_p.nodes[2]), length(mesh_p.nodes[3])))

# ---------------------------------------------------------------------------
# Slice visualization (z = 0 plane)
# ---------------------------------------------------------------------------
xs = mesh_p.nodes[1]
ys = mesh_p.nodes[2]
zs = mesh_p.nodes[3]
mid_z = Int(clamp(round(length(zs) / 2), 1, length(zs)))

Ux_slice = Ux[:, :, mid_z]
Uy_slice = Uy[:, :, mid_z]
Uz_slice = Uz[:, :, mid_z]
P_slice  = P[:, :, mid_z]
speed_slice = sqrt.(Ux_slice.^2 .+ Uy_slice.^2 .+ Uz_slice.^2)

theta_vals = range(0.0, 2.0 * pi; length=200)
circle_x = R .* cos.(theta_vals)
circle_y = R .* sin.(theta_vals)

fig = Figure(resolution=(1200, 520))
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Speed slice (z=0)")
hm1 = heatmap!(ax1, mesh_ux.nodes[1], mesh_ux.nodes[2], speed_slice; colormap=:plasma)
lines!(ax1, circle_x, circle_y; color=:white, linewidth=2)
Colorbar(fig[1, 2], hm1, label="|u|")

ax2 = Axis(fig[1, 3], xlabel="x", ylabel="y", title="Pressure slice (z=0)")
hm2 = heatmap!(ax2, xs, ys, P_slice; colormap=:balance)
lines!(ax2, circle_x, circle_y; color=:white, linewidth=2)
Colorbar(fig[1, 4], hm2, label="p")

save("stokes3d_lamb_sphere_slice.png", fig)
println("Saved stokes3d_lamb_sphere_slice.png")

# ---------------------------------------------------------------------------
# Analytical comparison along a 45-degree ray in the x-y plane
# ---------------------------------------------------------------------------
function trilinear_interp(nodes, field, x, y, z)
    xs, ys, zs = nodes
    nx, ny, nz = length(xs), length(ys), length(zs)

    ix = clamp(searchsortedlast(xs, x), 1, nx - 1)
    iy = clamp(searchsortedlast(ys, y), 1, ny - 1)
    iz = clamp(searchsortedlast(zs, z), 1, nz - 1)

    x1, x2 = xs[ix], xs[ix + 1]
    y1, y2 = ys[iy], ys[iy + 1]
    z1, z2 = zs[iz], zs[iz + 1]
    tx = (x2 == x1) ? 0.0 : (x - x1) / (x2 - x1)
    ty = (y2 == y1) ? 0.0 : (y - y1) / (y2 - y1)
    tz = (z2 == z1) ? 0.0 : (z - z1) / (z2 - z1)

    c000 = field[ix,   iy,   iz  ]
    c100 = field[ix+1, iy,   iz  ]
    c010 = field[ix,   iy+1, iz  ]
    c110 = field[ix+1, iy+1, iz  ]
    c001 = field[ix,   iy,   iz+1]
    c101 = field[ix+1, iy,   iz+1]
    c011 = field[ix,   iy+1, iz+1]
    c111 = field[ix+1, iy+1, iz+1]

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    return c0 * (1 - tz) + c1 * tz
end

vrad_exact(r, theta) = cos(theta) * (1.0 + 0.5 / r^3 - 1.5 / r)
vtheta_exact(r, theta) = -sin(theta) * (1.0 - 0.25 / r^3 - 0.75 / r)

alpha = 45.0 * pi / 180.0
r_samples = range(1.2 * R, stop=4.0 * R, length=100)
vr_num = similar(r_samples)
vt_num = similar(r_samples)
vr_ana = similar(r_samples)
vt_ana = similar(r_samples)

for (i, r) in enumerate(r_samples)
    x = r * cos(alpha)
    y = r * sin(alpha)
    z = 0.0

    ux = trilinear_interp(mesh_ux.nodes, Ux, x, y, z)
    uy = trilinear_interp(mesh_uy.nodes, Uy, x, y, z)

    vr_num[i] = ux * cos(alpha) + uy * sin(alpha)
    vt_num[i] = -ux * sin(alpha) + uy * cos(alpha)

    vr_ana[i] = vrad_exact(r, alpha)
    vt_ana[i] = vtheta_exact(r, alpha)
end

profile_data = hcat(r_samples, vr_num, vt_num, vr_ana, vt_ana)
writedlm("stokes3d_lamb_sphere_profile.csv", profile_data, ',')
println("Saved stokes3d_lamb_sphere_profile.csv")

fig_prof = Figure(resolution=(900, 400))
axr = Axis(fig_prof[1, 1], xlabel="r", ylabel="v_r", title="Radial velocity (alpha=45 deg)")
lines!(axr, r_samples, vr_num; label="numerical", linewidth=2)
lines!(axr, r_samples, vr_ana; label="analytical", linestyle=:dash, linewidth=2)
axislegend(axr, position=:rb)

axt = Axis(fig_prof[1, 2], xlabel="r", ylabel="v_theta", title="Tangential velocity (alpha=45 deg)")
lines!(axt, r_samples, vt_num; label="numerical", linewidth=2)
lines!(axt, r_samples, vt_ana; label="analytical", linestyle=:dash, linewidth=2)
axislegend(axt, position=:rb)

save("stokes3d_lamb_sphere_profiles.png", fig_prof)
println("Saved stokes3d_lamb_sphere_profiles.png")
