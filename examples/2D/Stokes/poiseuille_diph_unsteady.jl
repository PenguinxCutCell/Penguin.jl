using Penguin
using CairoMakie
using LinearAlgebra
using LinearSolve

"""
Unsteady two-layer Poiseuille flow for the diphasic Stokes solver.

Channel: [0, Lx] × [0, Ly] with interface at y = Ly/2.
Phase 1 occupies the lower half, phase 2 the upper half.
The flow starts from rest and is driven by a constant pressure gradient; the
parabolic steady profile progressively appears.
"""

###########
# Grids
###########
nx, ny = 32, 32
Lx, Ly = 20.0, 1.0
x0, y0 = 0.0, 0.0
dx, dy = Lx / nx, Ly / ny

mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

###########
# Capacities and operators (per phase)
###########
y_mid = Ly / 2
body_lower = (x, y, _=0.0) -> y - y_mid      # negative in lower half
body_upper = (x, y, _=0.0) -> y_mid - y      # negative in upper half

cap_ux_a = Capacity(body_lower, mesh_ux; compute_centroids=false)
cap_uy_a = Capacity(body_lower, mesh_uy; compute_centroids=false)
cap_p_a  = Capacity(body_lower, mesh_p;  compute_centroids=false)

cap_ux_b = Capacity(body_upper, mesh_ux; compute_centroids=false)
cap_uy_b = Capacity(body_upper, mesh_uy; compute_centroids=false)
cap_p_b  = Capacity(body_upper, mesh_p;  compute_centroids=false)

op_ux_a = DiffusionOps(cap_ux_a); op_uy_a = DiffusionOps(cap_uy_a); op_p_a = DiffusionOps(cap_p_a)
op_ux_b = DiffusionOps(cap_ux_b); op_uy_b = DiffusionOps(cap_uy_b); op_p_b = DiffusionOps(cap_p_b)

###########
# Boundary conditions
###########
Umax = 1.0
parabola = (x, y) -> 4 * Umax * (y - y0) * (Ly - (y - y0)) / (Ly^2)

ux_wall  = Dirichlet((x, y, t) -> 0.0)
bc_ux = BorderConditions(Dict(
    :bottom => ux_wall, :top => ux_wall,
))

uy_zero = Dirichlet((x, y, t) -> 0.0)
bc_uy = BorderConditions(Dict(
    :bottom => uy_zero, :top => uy_zero, :left => uy_zero, :right => uy_zero
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
# Solve (unsteady from rest)
###########
ic_x = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))
ic_y = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))

solver = StokesDiph(fluid_a, fluid_b,
                    (bc_ux_a, bc_uy_a),
                    (bc_ux_b, bc_uy_b),
                    (ic_x, ic_y),
                    pressure_gauges)

Δt = 0.05
T_end = 2.0
times, histories = solve_StokesDiph_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=:CN,
                                               algorithm=UMFPACKFactorization(), store_states=true)
println("Unsteady diphasic Poiseuille completed. Steps = ", length(times) - 1)

###########
# Extract fields (final state)
###########
state = histories[end]
nu_x = prod(op_ux_a.size); nu_y = prod(op_uy_a.size)
np_a = prod(op_p_a.size); np_b = prod(op_p_b.size)
sum_nu = nu_x + nu_y
off_p1 = 2 * sum_nu
off_phase2 = off_p1 + np_a

u1ωx = state[1:nu_x];         u1γx = state[nu_x+1:2nu_x]
u1ωy = state[2nu_x+1:2nu_x+nu_y]; u1γy = state[2nu_x+nu_y+1:2nu_x+2nu_y]
p1   = state[off_p1+1:off_p1+np_a]

u2ωx = state[off_phase2+1:off_phase2+nu_x];           u2γx = state[off_phase2+nu_x+1:off_phase2+2nu_x]
u2ωy = state[off_phase2+2nu_x+1:off_phase2+2nu_x+nu_y]; u2γy = state[off_phase2+2nu_x+nu_y+1:off_phase2+2nu_x+2nu_y]
p2   = state[off_phase2+2 * (nu_x + nu_y) + 1:off_phase2+2 * (nu_x + nu_y) + np_b]

xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
xp = mesh_p.nodes[1];  yp = mesh_p.nodes[2]

Ux1 = reshape(u1ωx, (length(xs), length(ys)))
Ux2 = reshape(u2ωx, (length(xs), length(ys)))
Uy1 = reshape(u1ωy, (length(xs), length(ys)))
Uy2 = reshape(u2ωy, (length(xs), length(ys)))
P1  = reshape(p1,   (length(xp), length(yp)))
P2  = reshape(p2,   (length(xp), length(yp)))

###########
# Analytical two-layer Poiseuille profile
###########
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

###########
# Profiles through time
###########
icol = Int(cld(length(xs), 2))
function profile_at_state(state_vec)
    ux1 = state_vec[1:nu_x]
    ux2 = state_vec[off_phase2+1:off_phase2+nu_x]
    Ux1_mid = reshape(ux1, (length(xs), length(ys)))
    Ux2_mid = reshape(ux2, (length(xs), length(ys)))
    prof = similar(ys)
    for (j, y) in enumerate(ys)
        prof[j] = y <= y_mid ? Ux1_mid[icol, j] : Ux2_mid[icol, j]
    end
    return prof
end

final_profile = profile_at_state(state)

fig = Figure(resolution=(900, 420))
ax1 = Axis(fig[1, 1], xlabel="u_x", ylabel="y", title="Mid-channel profile (final)")
lines!(ax1, final_profile, ys, label="numerical")
lines!(ax1, ux_target, ys, color=:red, linestyle=:dash, label="analytical")
axislegend(ax1, position=:rb)

fig_multi = Figure(resolution=(900, 420))
ax_multi = Axis(fig_multi[1, 1], xlabel="u_x", ylabel="y", title="Profiles over time (x mid)")
sample_ids = round.(Int, LinRange(1, length(histories), 5))
for idx in sample_ids
    prof = profile_at_state(histories[idx])
    lines!(ax_multi, prof, ys, label="t=$(round(times[idx]; digits=2))")
end
lines!(ax_multi, ux_target, ys, color=:black, linestyle=:dash, label="analytical")
axislegend(ax_multi, position=:rb)
display(fig_multi)

###########
# Field plots at final time
###########
fig2 = Figure(resolution=(1200, 500))
ax2 = Axis(fig2[1, 1], xlabel="x", ylabel="y", title="u_x phase 1 (final)")
hm1 = heatmap!(ax2, xs, ys, Ux1'; colormap=:viridis)
Colorbar(fig2[1, 2], hm1)

ax3 = Axis(fig2[1, 3], xlabel="x", ylabel="y", title="u_x phase 2 (final)")
hm2 = heatmap!(ax3, xs, ys, Ux2'; colormap=:viridis)
Colorbar(fig2[1, 4], hm2)

display(fig)
display(fig2)
save("stokes_diph_poiseuille_unsteady_profiles.png", fig)
save("stokes_diph_poiseuille_unsteady_fields.png", fig2)
