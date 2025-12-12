using Penguin
using CairoMakie
using LinearAlgebra
using LinearSolve

"""
Two-layer Poiseuille flow for the diphasic Stokes solver.

Channel: [0, Lx] × [0, Ly] with interface at y = Ly/2.
Phase 1 occupies the lower half, phase 2 the upper half, both driven by the
same imposed parabolic profile on the left/right boundaries; no-slip on top/bottom.
"""

###########
# Grids
###########
nx, ny = 64, 64
Lx, Ly = 20.0, 1.0
x0, y0 = 0.0, 0.0
dx, dy = Lx/nx, Ly/ny

# Pressure grid (cell-centered) and staggered velocity grids
mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

###########
# Capacities and operators (per phase)
###########
y_mid = Ly / 2
body_lower = (x, y, _=0.0) -> y - y_mid      # negative in lower half
body_upper = (x, y, _=0.0) -> y_mid - y      # negative in upper half

cap_ux_a = Capacity(body_lower, mesh_ux;compute_centroids=false); cap_uy_a = Capacity(body_lower, mesh_uy;compute_centroids=false); cap_p_a = Capacity(body_lower, mesh_p;compute_centroids=false)
cap_ux_b = Capacity(body_upper, mesh_ux;compute_centroids=false); cap_uy_b = Capacity(body_upper, mesh_uy;compute_centroids=false); cap_p_b = Capacity(body_upper, mesh_p;compute_centroids=false)

op_ux_a = DiffusionOps(cap_ux_a); op_uy_a = DiffusionOps(cap_uy_a); op_p_a = DiffusionOps(cap_p_a)
op_ux_b = DiffusionOps(cap_ux_b); op_uy_b = DiffusionOps(cap_uy_b); op_p_b = DiffusionOps(cap_p_b)

###########
# Boundary conditions
###########
Umax = 1.0
parabola = (x, y) -> 4Umax * (y - y0) * (Ly - (y - y0)) / (Ly^2)

ux_wall  = Dirichlet((x, y)->0.0)

bc_ux = BorderConditions(Dict(
    :bottom=>ux_wall, :top=>ux_wall, 
))

uy_zero = Dirichlet((x, y)->0.0)
bc_uy = BorderConditions(Dict(
    :bottom=>uy_zero, :top=>uy_zero, :left=>uy_zero, :right=>uy_zero
))

bc_ux_a, bc_ux_b = bc_ux, bc_ux
bc_uy_a, bc_uy_b = bc_uy, bc_uy

pressure_gauges = (PinPressureGauge(), PinPressureGauge())

###########
# Sources and material
###########
gradP = -0.21                # imposed pressure gradient (dp/dx)
fᵤ = (x, y, z=0.0) -> -gradP
fₚ = (x, y, z=0.0) -> 0.0
μ_a, μ_b = 1.0, 10.0
ρ_a, ρ_b = 1.0, 1.0

fluid_a = Fluid((mesh_ux, mesh_uy),
                (cap_ux_a, cap_uy_a),
                (op_ux_a, op_uy_a),
                mesh_p,
                cap_p_a,
                op_p_a,
                μ_a, ρ_a, fᵤ, fₚ)

fluid_b = Fluid((mesh_ux, mesh_uy),
                (cap_ux_b, cap_uy_b),
                (op_ux_b, op_uy_b),
                mesh_p,
                cap_p_b,
                op_p_b,
                μ_b, ρ_b, fᵤ, fₚ)

###########
# Solve
###########
ic_x = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))
ic_y = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))

solver = StokesDiph(fluid_a, fluid_b, (bc_ux_a, bc_uy_a), (bc_ux_b, bc_uy_b), (ic_x, ic_y), pressure_gauges)
solve_StokesDiph!(solver; algorithm=UMFPACKFactorization())

println("Diphasic Poiseuille solved. Unknowns = ", length(solver.x))

###########
# Extract fields
###########
nu_x = prod(op_ux_a.size); nu_y = prod(op_uy_a.size)
np_a = prod(op_p_a.size); np_b = prod(op_p_b.size)
sum_nu = nu_x + nu_y
off_p1 = 2 * sum_nu
off_phase2 = off_p1 + np_a

u1ωx = solver.x[1:nu_x];         u1γx = solver.x[nu_x+1:2nu_x]
u1ωy = solver.x[2nu_x+1:2nu_x+nu_y]; u1γy = solver.x[2nu_x+nu_y+1:2nu_x+2nu_y]
p1   = solver.x[off_p1+1:off_p1+np_a]

u2ωx = solver.x[off_phase2+1:off_phase2+nu_x];           u2γx = solver.x[off_phase2+nu_x+1:off_phase2+2nu_x]
u2ωy = solver.x[off_phase2+2nu_x+1:off_phase2+2nu_x+nu_y]; u2γy = solver.x[off_phase2+2nu_x+nu_y+1:off_phase2+2nu_x+2nu_y]
p2   = solver.x[off_phase2+2*(nu_x+nu_y)+1:off_phase2+2*(nu_x+nu_y)+np_b]

xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
xp = mesh_p.nodes[1];  yp = mesh_p.nodes[2]

LIu = LinearIndices((length(xs), length(ys)))
LIp = LinearIndices((length(xp), length(yp)))

Ux1 = reshape(u1ωx, (length(xs), length(ys)))
Ux2 = reshape(u2ωx, (length(xs), length(ys)))
Uy1 = reshape(u1ωy, (length(xs), length(ys)))
Uy2 = reshape(u2ωy, (length(xs), length(ys)))
P1  = reshape(p1,   (length(xp), length(yp)))
P2  = reshape(p2,   (length(xp), length(yp)))
Ux1g = reshape(u1γx, (length(xs), length(ys)))
Ux2g = reshape(u2γx, (length(xs), length(ys)))
Uy1g = reshape(u1γy, (length(xs), length(ys)))
Uy2g = reshape(u2γy, (length(xs), length(ys)))

# Mask empty regions for each phase
mask_ux_a = reshape(cap_ux_a.cell_types, size(Ux1))
mask_ux_b = reshape(cap_ux_b.cell_types, size(Ux2))
#Ux1[mask_ux_a .== 0] .= NaN
#Ux2[mask_ux_b .== 0] .= NaN

# Analytical two-layer Poiseuille profile (global y in [0, Ly], interface at y_mid)
h1 = y_mid - y0          # lower layer thickness
h2 = (y0 + Ly) - y_mid   # upper layer thickness
μ1, μ2 = μ_a, μ_b
G = gradP               # dp/dx
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
# Mid-channel profile
###########
icol = Int(cld(length(xs), 2))
ux_profile = similar(ys)
for (j, y) in enumerate(ys)
    ux_profile[j] = y <= y_mid ? Ux1[icol, j] : Ux2[icol, j]
end

fig = Figure(resolution=(900, 420))
ax1 = Axis(fig[1, 1], xlabel="u_x", ylabel="y", title="Mid-channel profile")
lines!(ax1, ux_profile, ys, label="numerical")
lines!(ax1, ux_target, ys, color=:red, linestyle=:dash, label="analytical")
axislegend(ax1, position=:rb)

###########
# Multiple Mid-channel profiles
###########
fig_multi = Figure(resolution=(900, 420))
ax_multi = Axis(fig_multi[1, 1], xlabel="u_x", ylabel="y", title="Mid-channel profiles at various x")
n_profiles = 5
x_indices = round.(Int, LinRange(1, length(xs), n_profiles))
for ix in x_indices
    ux_profile_x = similar(ys)
    for (j, y) in enumerate(ys)
        ux_profile_x[j] = y <= y_mid ? Ux1[ix, j] : Ux2[ix, j]
    end
    lines!(ax_multi, ux_profile_x, ys, label="x=$(round(xs[ix], digits=2))")
end
lines!(ax_multi, ux_target, ys, color=:black, linestyle=:dash, label="analytical")
axislegend(ax_multi, position=:rb)
display(fig_multi)

###########
# Field plots
###########
fig2 = Figure(resolution=(1200, 500))
ax2 = Axis(fig2[1, 1], xlabel="x", ylabel="y", title="u_x phase 1")
hm1 = heatmap!(ax2, xs, ys, Ux1'; colormap=:viridis)
Colorbar(fig2[1, 2], hm1)

ax3 = Axis(fig2[1, 3], xlabel="x", ylabel="y", title="u_x phase 2")
hm2 = heatmap!(ax3, xs, ys, Ux2'; colormap=:viridis)
Colorbar(fig2[1, 4], hm2)

display(fig)
display(fig2)
save("stokes_diph_poiseuille_profiles.png", fig)
save("stokes_diph_poiseuille_velocity.png", fig2)

println("Profile Linf error (u_x vs analytical): ",
        maximum(abs, ux_profile[2:end-2] .- ux_target[2:end-2]))

# L2 norm volume integrated error
l2_error_sq = 0.0
for i in 1:length(xs), j in 1:length(ys)
    y = ys[j]
    if y <= y_mid
        u_num = Ux1[i, j]
        u_ana = u_analytical_phase1(y)
    else
        u_num = Ux2[i, j]
        u_ana = u_analytical_phase2(y)
    end
    global l2_error_sq += (u_num - u_ana)^2 * dx * dy
end
l2_error = sqrt(l2_error_sq)
println("L2 norm volume integrated error: ", l2_error)

fig3 = Figure(resolution=(900, 400))
ax4 = Axis(fig3[1, 1], xlabel="x", ylabel="y", title="u_γ,x phase 1")
hm3 = heatmap!(ax4, xs, ys, Ux1g'; colormap=:viridis)
Colorbar(fig3[1, 2], hm3)

ax5 = Axis(fig3[1, 3], xlabel="x", ylabel="y", title="u_γ,x phase 2")
hm4 = heatmap!(ax5, xs, ys, Ux2g'; colormap=:viridis)
Colorbar(fig3[1, 4], hm4)
save("stokes_diph_poiseuille_gamma_velocity.png", fig3)
println("Saved stokes_diph_poiseuille_gamma_velocity.png")

fig4 = Figure(resolution=(900, 400))
ax6 = Axis(fig4[1, 1], xlabel="x", ylabel="y", title="Pressure phase 1")
hm5 = heatmap!(ax6, xp, yp, P1'; colormap=:viridis)
Colorbar(fig4[1, 2], hm5)
ax7 = Axis(fig4[1, 3], xlabel="x", ylabel="y", title="Pressure phase 2")
hm6 = heatmap!(ax7, xp, yp, P2'; colormap=:viridis)
Colorbar(fig4[1, 4], hm6)
save("stokes_diph_poiseuille_pressure.png", fig4)
println("Saved stokes_diph_poiseuille_pressure.png")
