using Penguin
using CairoMakie
using LinearAlgebra
using LinearSolve

"""
Two-layer Couette–Poiseuille flow for the diphasic Stokes solver.

Domain: [0, Lx] × [0, H] with interface at y = d.
Bottom wall at y=0 with velocity U0, top wall at y=H with velocity UH.
Constant pressure gradient in x-direction: dp/dx = -G.
Velocity and shear continuity enforced at the interface.
"""

###########
# Grids
###########
nx, ny = 64, 64
Lx, H = 1.0, 1.0
x0, y0 = 0.0, 0.0
dx, dy = Lx/nx, H/ny

mesh_p  = Penguin.Mesh((nx, ny), (Lx, H), (x0, y0))
mesh_ux = Penguin.Mesh((nx, ny), (Lx, H), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, H), (x0, y0 - 0.5*dy))

###########
# Capacities and operators
###########
d = H / 2  # interface location
body_lower = (x, y, _=0.0) -> y - d
body_upper = (x, y, _=0.0) -> d - y

cap_ux_a = Capacity(body_lower, mesh_ux; compute_centroids=false)
cap_uy_a = Capacity(body_lower, mesh_uy; compute_centroids=false)
cap_p_a  = Capacity(body_lower, mesh_p;  compute_centroids=false)
cap_ux_b = Capacity(body_upper, mesh_ux; compute_centroids=false)
cap_uy_b = Capacity(body_upper, mesh_uy; compute_centroids=false)
cap_p_b  = Capacity(body_upper, mesh_p;  compute_centroids=false)

op_ux_a = DiffusionOps(cap_ux_a); op_uy_a = DiffusionOps(cap_uy_a); op_p_a = DiffusionOps(cap_p_a)
op_ux_b = DiffusionOps(cap_ux_b); op_uy_b = DiffusionOps(cap_uy_b); op_p_b = DiffusionOps(cap_p_b)

###########
# Physical parameters
###########
U0 = 0.0    # bottom wall velocity
UH = 1.0    # top wall velocity
G  = 1.0    # pressure gradient magnitude (dp/dx = -G)
μ_a, μ_b = 1.0, 10.0  # viscosities (phase 1 = lower, phase 2 = upper)
ρ_a, ρ_b = 1.0, 1.0

###########
# Boundary conditions
###########
bc_ux = BorderConditions(Dict(
    :bottom => Dirichlet((x, y)->U0),
    :top    => Dirichlet((x, y)->UH),
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
fᵤ = (x, y, z=0.0) -> G   # body force equivalent to pressure gradient dp/dx = -G
fₚ = (x, y, z=0.0) -> 0.0

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

println("Diphasic Couette–Poiseuille solved. Unknowns = ", length(solver.x))

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

###########
# Analytical Couette–Poiseuille solution
###########
# Phase 1 (0 ≤ y ≤ d): u1(y) = (G/(2*mu1))*y^2 + A1*y + B1
# Phase 2 (d ≤ y ≤ H): u2(y) = (G/(2*mu2))*y^2 + A2*y + B2
mu1, mu2 = μ_a, μ_b

B1 = U0
A1 = (1/H) * (UH - U0 + (G/2) * ((H^2 - d^2)/mu2 + d^2*(1/mu1 - 1/mu2)))
A2 = (mu1/mu2)*A1 + d*(G/mu2 - G/mu1)
B2 = U0 - d*((G/(2*mu1))*d + A1) + d*((G/(2*mu2))*d + A2)

function u_analytical_phase1(y)
    return (G/(2*mu1))*y^2 + A1*y + B1
end

function u_analytical_phase2(y)
    return (G/(2*mu2))*y^2 + A2*y + B2
end

ux_target = similar(ys)
for (j, y) in enumerate(ys)
    ux_target[j] = y <= d ? u_analytical_phase1(y) : u_analytical_phase2(y)
end

###########
# Mid-channel profile
###########
icol = Int(cld(length(xs), 2))
ux_profile = similar(ys)
for (j, y) in enumerate(ys)
    ux_profile[j] = y <= d ? Ux1[icol, j] : Ux2[icol, j]
end

ux_profile = ux_profile[1:end-1]  # remove top boundary point
ux_target_plot = ux_target[1:end-1]
ys_profile = ys[1:end-1]

fig = Figure(resolution=(900, 420))
ax1 = Axis(fig[1, 1], xlabel="u_x", ylabel="y", title="Mid-channel profile (Couette–Poiseuille)")
lines!(ax1, ux_profile, ys_profile, label="numerical")
lines!(ax1, ux_target_plot, ys_profile, color=:red, linestyle=:dash, label="analytical")
axislegend(ax1, position=:rb)

println("Profile Linf error (u_x vs analytical): ",
        maximum(abs, ux_profile .- ux_target_plot))

display(fig)
save("stokes_diph_couette_poiseuille_profile.png", fig)
