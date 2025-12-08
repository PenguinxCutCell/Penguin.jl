using Penguin
using CairoMakie
using LinearAlgebra
using LinearSolve

"""
3D two-layer Poiseuille flow for the diphasic Stokes solver.

Channel: [0, Lx] × [0, Ly] × [0, Lz] with interface at y = Ly/2.
Phase 1 occupies the lower half, phase 2 the upper half, both driven by the
same imposed pressure gradient; no-slip on top/bottom/front/back walls.
"""

###########
# Grids
###########
Nx, Ny, Nz = 16, 16, 8
Lx, Ly, Lz = 20.0, 1.0, 1.0
x0, y0, z0 = 0.0, 0.0, 0.0
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

# Pressure grid (cell-centered) and staggered velocity grids
mesh_p  = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0, z0))
mesh_ux = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0 - 0.5 * dx, y0, z0))
mesh_uy = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0 - 0.5 * dy, z0))
mesh_uz = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0, z0 - 0.5 * dz))

###########
# Capacities and operators (per phase)
###########
y_mid = Ly / 2
body_lower = (x, y, z, _=0.0) -> y - y_mid       # negative in lower half
body_upper = (x, y, z, _=0.0) -> y_mid - y       # negative in upper half

cap_ux_a = Capacity(body_lower, mesh_ux; compute_centroids=false)
cap_uy_a = Capacity(body_lower, mesh_uy; compute_centroids=false)
cap_uz_a = Capacity(body_lower, mesh_uz; compute_centroids=false)
cap_p_a  = Capacity(body_lower, mesh_p;  compute_centroids=false)

cap_ux_b = Capacity(body_upper, mesh_ux; compute_centroids=false)
cap_uy_b = Capacity(body_upper, mesh_uy; compute_centroids=false)
cap_uz_b = Capacity(body_upper, mesh_uz; compute_centroids=false)
cap_p_b  = Capacity(body_upper, mesh_p;  compute_centroids=false)

op_ux_a = DiffusionOps(cap_ux_a); op_uy_a = DiffusionOps(cap_uy_a); op_uz_a = DiffusionOps(cap_uz_a); op_p_a = DiffusionOps(cap_p_a)
op_ux_b = DiffusionOps(cap_ux_b); op_uy_b = DiffusionOps(cap_uy_b); op_uz_b = DiffusionOps(cap_uz_b); op_p_b = DiffusionOps(cap_p_b)

###########
# Boundary conditions
###########
Umax = 1.0
parabola = (x, y, z) -> 4 * Umax * (y - y0) * (Ly - (y - y0)) / (Ly^2)

ux_wall = Dirichlet((x, y, z) -> 0.0)
bc_ux = BorderConditions(Dict(
    :bottom => ux_wall, :top => ux_wall,
    :front => ux_wall,  :back => ux_wall,
))

uy_zero = Dirichlet((x, y, z) -> 0.0)
bc_uy = BorderConditions(Dict(
    :bottom => uy_zero, :top => uy_zero,
    :left => uy_zero, :right => uy_zero,
    :front => uy_zero, :back => uy_zero
))

uz_zero = Dirichlet((x, y, z) -> 0.0)
bc_uz = BorderConditions(Dict(
    :bottom => uz_zero, :top => uz_zero,
    :left => uz_zero, :right => uz_zero,
    :front => uz_zero, :back => uz_zero
))

bc_ux_a, bc_ux_b = bc_ux, bc_ux
bc_uy_a, bc_uy_b = bc_uy, bc_uy
bc_uz_a, bc_uz_b = bc_uz, bc_uz

pressure_gauges = (PinPressureGauge(), PinPressureGauge())

###########
# Sources and material
###########
gradP = -0.21                # imposed pressure gradient (dp/dx)
fᵤ = (x, y, z) -> -gradP
fₚ = (x, y, z) -> 0.0
μ_a, μ_b = 1.0, 10.0
ρ_a, ρ_b = 1.0, 1.0

fluid_a = Fluid((mesh_ux, mesh_uy, mesh_uz),
                (cap_ux_a, cap_uy_a, cap_uz_a),
                (op_ux_a, op_uy_a, op_uz_a),
                mesh_p,
                cap_p_a,
                op_p_a,
                μ_a, ρ_a, fᵤ, fₚ)

fluid_b = Fluid((mesh_ux, mesh_uy, mesh_uz),
                (cap_ux_b, cap_uy_b, cap_uz_b),
                (op_ux_b, op_uy_b, op_uz_b),
                mesh_p,
                cap_p_b,
                op_p_b,
                μ_b, ρ_b, fᵤ, fₚ)

###########
# Solve
###########
ic_x = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))
ic_y = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))
ic_z = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))

solver = StokesDiph(fluid_a, fluid_b,
                    (bc_ux_a, bc_uy_a, bc_uz_a),
                    (bc_ux_b, bc_uy_b, bc_uz_b),
                    (ic_x, ic_y, ic_z),
                    pressure_gauges)
solve_StokesDiph!(solver; algorithm=UMFPACKFactorization())

println("3D diphasic Poiseuille solved. Unknowns = ", length(solver.x))

###########
# Extract fields
###########
nu_x = prod(op_ux_a.size); nu_y = prod(op_uy_a.size); nu_z = prod(op_uz_a.size)
np_a = prod(op_p_a.size);  np_b = prod(op_p_b.size)
sum_nu = nu_x + nu_y + nu_z
off_p1 = 2 * sum_nu
off_phase2 = off_p1 + np_a

u1ωx = solver.x[1:nu_x];                            u1γx = solver.x[nu_x+1:2nu_x]
u1ωy = solver.x[2nu_x+1:2nu_x+nu_y];                u1γy = solver.x[2nu_x+nu_y+1:2nu_x+2nu_y]
u1ωz = solver.x[2nu_x+2nu_y+1:2nu_x+2nu_y+nu_z];    u1γz = solver.x[2nu_x+2nu_y+nu_z+1:2nu_x+2nu_y+2nu_z]
p1   = solver.x[off_p1+1:off_p1+np_a]

u2ωx = solver.x[off_phase2+1:off_phase2+nu_x];                         u2γx = solver.x[off_phase2+nu_x+1:off_phase2+2nu_x]
u2ωy = solver.x[off_phase2+2nu_x+1:off_phase2+2nu_x+nu_y];             u2γy = solver.x[off_phase2+2nu_x+nu_y+1:off_phase2+2nu_x+2nu_y]
u2ωz = solver.x[off_phase2+2nu_x+2nu_y+1:off_phase2+2nu_x+2nu_y+nu_z]; u2γz = solver.x[off_phase2+2nu_x+2nu_y+nu_z+1:off_phase2+2nu_x+2nu_y+2nu_z]
p2   = solver.x[off_phase2+2 * sum_nu + 1:off_phase2+2 * sum_nu + np_b]

xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]; zs = mesh_ux.nodes[3]
xp = mesh_p.nodes[1];  yp = mesh_p.nodes[2];  zp = mesh_p.nodes[3]

Ux1 = reshape(u1ωx, (length(xs), length(ys), length(zs)))
Ux2 = reshape(u2ωx, (length(xs), length(ys), length(zs)))
Uy1 = reshape(u1ωy, (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2]), length(mesh_uy.nodes[3])))
Uy2 = reshape(u2ωy, (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2]), length(mesh_uy.nodes[3])))
Uz1 = reshape(u1ωz, (length(mesh_uz.nodes[1]), length(mesh_uz.nodes[2]), length(mesh_uz.nodes[3])))
Uz2 = reshape(u2ωz, (length(mesh_uz.nodes[1]), length(mesh_uz.nodes[2]), length(mesh_uz.nodes[3])))
P1  = reshape(p1,   (length(xp), length(yp), length(zp)))
P2  = reshape(p2,   (length(xp), length(yp), length(zp)))

Ux1g = reshape(u1γx, size(Ux1)); Ux2g = reshape(u2γx, size(Ux2))
Uy1g = reshape(u1γy, size(Uy1)); Uy2g = reshape(u2γy, size(Uy2))
Uz1g = reshape(u1γz, size(Uz1)); Uz2g = reshape(u2γz, size(Uz2))

###########
# Analytical profiles (depend only on y)
###########
h1 = y_mid - y0          # lower layer thickness
h2 = (y0 + Ly) - y_mid   # upper layer thickness
μ1, μ2 = μ_a, μ_b
G = gradP
den_common = h1 * μ2 + h2 * μ1
den1 = 2 * μ1 * den_common
den2 = 2 * μ2 * den_common
A = h1^2 * μ2 - h2^2 * μ1
B = -G * h1 * h2 * (h1 + h2) / (2 * den_common)

function u_analytical_phase1(y)
    y1 = y - y_mid  # map global y into [-h1, 0]
    return (G / den1) * (den_common * y1^2 + A * y1 - h1 * h2 * μ1 * (h1 + h2))
end

function u_analytical_phase2(y)
    y2 = y - y_mid  # map global y into [0, h2]
    return (G / den2) * (den_common * y2^2 + A * y2 - h1 * h2 * μ2 * (h1 + h2))
end

ux_target = similar(ys)
for (j, y) in enumerate(ys)
    ux_target[j] = y <= y_mid ? u_analytical_phase1(y) : u_analytical_phase2(y)
end

# Analytical pressure profile
p_target = gradP .* xp .+ 0.0

###########
# Mid-channel profile (x,z centered)
###########
ix = Int(cld(length(xs), 2))
iz = Int(cld(length(zs), 2))
ux_profile = similar(ys)
for (j, y) in enumerate(ys)
    ux_profile[j] = y <= y_mid ? Ux1[ix, j, iz] : Ux2[ix, j, iz]
end

fig = Figure(resolution=(900, 420))
ax1 = Axis(fig[1, 1], xlabel="u_x", ylabel="y", title="Mid-channel profile (x,z centered)")
lines!(ax1, ux_profile, ys, label="numerical")
lines!(ax1, ux_target, ys, color=:red, linestyle=:dash, label="analytical")
axislegend(ax1, position=:rb)

###########
# Profiles at various x (z fixed at mid-plane)
###########
fig_multi = Figure(resolution=(900, 420))
ax_multi = Axis(fig_multi[1, 1], xlabel="u_x", ylabel="y", title="Profiles at various x (z mid)")
n_profiles = 4
x_indices = round.(Int, LinRange(1, length(xs), n_profiles))
for ixp in x_indices
    ux_profile_x = similar(ys)
    for (j, y) in enumerate(ys)
        ux_profile_x[j] = y <= y_mid ? Ux1[ixp, j, iz] : Ux2[ixp, j, iz]
    end
    lines!(ax_multi, ux_profile_x, ys, label="x=$(round(xs[ixp]; digits=2))")
end
lines!(ax_multi, ux_target, ys, color=:black, linestyle=:dash, label="analytical")
axislegend(ax_multi, position=:rb)
display(fig_multi)

###########
# Field plots (z mid-plane slices)
###########
Ux1_slice = Ux1[:, :, iz]
Ux2_slice = Ux2[:, :, iz]
P1_slice  = P1[:, :, iz]
P2_slice  = P2[:, :, iz]

fig2 = Figure(resolution=(1200, 500))
ax2 = Axis(fig2[1, 1], xlabel="x", ylabel="y", title="u_x phase 1 (z mid)")
hm1 = heatmap!(ax2, xs, ys, Ux1_slice'; colormap=:viridis)
Colorbar(fig2[1, 2], hm1)

ax3 = Axis(fig2[1, 3], xlabel="x", ylabel="y", title="u_x phase 2 (z mid)")
hm2 = heatmap!(ax3, xs, ys, Ux2_slice'; colormap=:viridis)
Colorbar(fig2[1, 4], hm2)

fig4 = Figure(resolution=(1200, 500))
ax6 = Axis(fig4[1, 1], xlabel="x", ylabel="y", title="Pressure phase 1 (z mid)")
hm5 = heatmap!(ax6, xp, yp, P1_slice'; colormap=:viridis)
Colorbar(fig4[1, 2], hm5)
ax7 = Axis(fig4[1, 3], xlabel="x", ylabel="y", title="Pressure phase 2 (z mid)")
hm6 = heatmap!(ax7, xp, yp, P2_slice'; colormap=:viridis)
Colorbar(fig4[1, 4], hm6)

display(fig)
display(fig2)
display(fig4)
save("stokes_diph_poiseuille_3d_profiles.png", fig)
save("stokes_diph_poiseuille_3d_velocity.png", fig2)
save("stokes_diph_poiseuille_3d_pressure.png", fig4)

println("Midline Linf error (u_x vs analytical): ",
        maximum(abs, ux_profile[2:end-2] .- ux_target[2:end-2]))

