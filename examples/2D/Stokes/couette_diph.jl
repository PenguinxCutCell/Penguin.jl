using Penguin
using CairoMakie
using LinearSolve

"""
Two-layer Couette flow for the diphasic Stokes solver.

Domain: [0, Lx] × [0, Ly] with interface at y = Ly/2.
Bottom wall moves at U_bottom, top wall at U_top. No pressure gradient.
Velocity continuity and shear continuity enforced at the interface.
"""

###########
# Grids
###########
nx, ny = 64, 64
Lx, Ly = 10.0, 1.0
x0, y0 = 0.0, 0.0
dx, dy = Lx/nx, Ly/ny

mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

###########
# Capacities and operators
###########
y_mid = Ly / 2
body_lower = (x, y, _=0.0) -> y - y_mid
body_upper = (x, y, _=0.0) -> y_mid - y

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
U_bottom = 0.0
U_top    = 1.0

bc_ux = BorderConditions(Dict(
    :bottom => Dirichlet((x, y)->U_bottom),
    :top    => Dirichlet((x, y)->U_top),
))

bc_uy = BorderConditions(Dict(
    :bottom => Dirichlet((x, y)->0.0),
    :top    => Dirichlet((x, y)->0.0),
    :left   => Dirichlet((x, y)->0.0),
    :right  => Dirichlet((x, y)->0.0),
))

bc_ux_a, bc_ux_b = bc_ux, bc_ux
bc_uy_a, bc_uy_b = bc_uy, bc_uy

pressure_gauges = (PinPressureGauge(), PinPressureGauge())

###########
# Sources and material
###########
fᵤ = (x, y, z=0.0) -> 0.0
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
# Interface conditions
###########
ic_x = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))  # shear continuity, velocity continuity
ic_y = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))

###########
# Solve
###########
solver = StokesDiph(fluid_a, fluid_b, (bc_ux_a, bc_uy_a), (bc_ux_b, bc_uy_b), (ic_x, ic_y), pressure_gauges)
solve_StokesDiph!(solver; method=Base.:\)

println("Diphasic Couette solved. Unknowns = ", length(solver.x))

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

# Analytical two-layer Couette profile (Ub at y=-h1, Ut at y=h2, interface y=0)
h1 = y_mid - y0
h2 = (y0 + Ly) - y_mid
μ1, μ2 = μ_a, μ_b
Ub, Ut = U_bottom, U_top
a1 = (Ut - Ub) / (h1 + (μ1/μ2) * h2)
a2 = (μ1/μ2) * a1
uI = Ub + a1 * h1  # interface velocity

function u_couette_phase1(y)
    yloc = y - y_mid  # [-h1, 0]
    return Ub + a1 * (yloc + h1)
end

function u_couette_phase2(y)
    yloc = y - y_mid  # [0, h2]
    return Ut - a2 * (h2 - yloc)
end

ux_target = similar(ys)
for (j, y) in enumerate(ys)
    ux_target[j] = y <= y_mid ? u_couette_phase1(y) : u_couette_phase2(y)
end

###########
# Mid-channel profile
###########
icol = Int(cld(length(xs), 2))
ux_profile = similar(ys)
for (j, y) in enumerate(ys)
    ux_profile[j] = y <= y_mid ? Ux1[icol, j] : Ux2[icol, j]
end

ux_profile = ux_profile[1:end-1]  # remove top boundary point
ux_target  = ux_target[1:end-1]   # remove top boundary point
ys_profile = ys[1:end-1]         # remove top boundary point

fig = Figure(resolution=(900, 420))
ax1 = Axis(fig[1, 1], xlabel="u_x", ylabel="y", title="Mid-channel profile (Couette)")
lines!(ax1, ux_profile, ys_profile, label="numerical")
lines!(ax1, ux_target, ys_profile, color=:red, linestyle=:dash, label="analytical")
axislegend(ax1, position=:rb)

println("Profile Linf error (u_x vs analytical): ",
        maximum(abs, ux_profile .- ux_target))

display(fig)
save("stokes_diph_couette_profile.png", fig)