using Penguin
using CairoMakie
using LinearAlgebra
using SpecialFunctions
using Printf

"""
Lamb-Chaplygin dipole impinging on a circular cylinder (curved boundary test).
The setup mirrors the Basilisk example `sandbox/Antoonvh/test_curved_boundaries.c`
and tracks the enstrophy evolution for an embedded no-slip body.
"""

###########
# Domain and meshes
###########
Lx = 25.0
Ly = 25.0
nx = 48
ny = 48
mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (-0.5 * dx, 0.0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (0.0, -0.5 * dy))

###########
# Curved body: cylinder centered at (xo, yo_c) with radius 1
###########
xo = 12.1
yo_c = 5.0
radius = 1.0
cylinder_levelset = (x, y, _=0.0) -> radius - hypot(x - xo, y - yo_c) # positive inside solid

capacity_ux = Capacity(cylinder_levelset, mesh_ux; compute_centroids=true)
capacity_uy = Capacity(cylinder_levelset, mesh_uy; compute_centroids=true)
capacity_p  = Capacity(cylinder_levelset, mesh_p;  compute_centroids=true)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)

mask_x = reshape(diag(capacity_ux.W[1]), (length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2])))
mask_y = reshape(diag(capacity_uy.W[2]), (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2])))

###########
# Boundary conditions (no-slip body, outflow on box)
###########
outflow = Outflow(0.0)
wall = Dirichlet(0.0)
bc_ux = BorderConditions(Dict(:left=>wall, :right=>wall, :bottom=>outflow, :top=>outflow))
bc_uy = BorderConditions(Dict(:left=>wall, :right=>wall, :bottom=>outflow, :top=>outflow))
pressure_gauge = PinPressureGauge()
body_bc = Dirichlet(0.0) # no-slip on the cylinder

###########
# Lamb–Chaplygin dipole initial condition
###########
const k_root = 3.8317
dipole_center = (xo, 10.0) # 5 units above cylinder center, matching Basilisk layout
U0 = 1.0

function lamb_dipole_velocity(x, y; center=dipole_center, R=1.0, U=U0)
    ξ = x - center[1]
    η = y - center[2]
    r = hypot(ξ, η)
    if r < 1e-12
        return (0.0, 0.0)
    end
    sθ = η / r
    cθ = ξ / r

    if r < R
        arg = k_root * r / R
        base = -2 * besselj1(arg) / (k_root * besselj0(k_root)) + r / R
        dbase_dr = (-2 / (besselj0(k_root) * R)) * (besselj0(arg) - besselj1(arg) / arg) + 1 / R
        u_r = (U / r) * base * cθ
        u_θ = -U * dbase_dr * sθ
    else
        coef = U * (R^2 / r^2)
        u_r = coef * cθ
        u_θ = coef * sθ
    end

    u_x = u_r * cθ - u_θ * sθ
    u_y = u_r * sθ + u_θ * cθ
    return (u_x, u_y)
end

function initial_state_from_dipole(mesh_ux, mesh_uy; center=dipole_center, R=1.0, U=U0)
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    uomega_x0 = Float64[lamb_dipole_velocity(x, y; center=center, R=R, U=U)[1] for x in xs_ux, y in ys_ux] .* mask_x
    uomega_y0 = Float64[lamb_dipole_velocity(x, y; center=center, R=R, U=U)[2] for x in xs_uy, y in ys_uy] .* mask_y

    ugamma_x0 = copy(uomega_x0)
    ugamma_y0 = copy(uomega_y0)

    return vec(uomega_x0), vec(ugamma_x0), vec(uomega_y0), vec(ugamma_y0)
end

###########
# Diagnostics: vorticity and enstrophy (cut-cell circulation)
###########
function unpack_velocity_fields(state, nu_x, nu_y, mesh_ux, mesh_uy)
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    uωx = state[1:nu_x]
    uωy = state[2nu_x + 1:2nu_x + nu_y]

    Ux = reshape(uωx, (length(xs_ux), length(ys_ux)))
    Uy = reshape(uωy, (length(xs_uy), length(ys_uy)))
    return Ux, Uy
end

function enstrophy_circulation(fluid, state)
    omega = circulation_vorticity(fluid, state)
    V = reshape(diag(fluid.capacity_p.V), size(omega))
    return sum(V .* omega.^2)
end

###########
# Solver setup and run
###########
rho = 1.0
Re = 500.0                     # matches Basilisk mu = 1/500
mu = rho * U0 * radius / Re
dt = 0.001
T_end = 0.1

f_u = (x, y, z=0.0, t=0.0) -> 0.0
f_p = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              mu, rho, f_u, f_p)

uomega_x0, ugamma_x0, uomega_y0, ugamma_y0 = initial_state_from_dipole(mesh_ux, mesh_uy)
x0_vec = vcat(uomega_x0, ugamma_x0, uomega_y0, ugamma_y0, zeros(Float64, np))

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, body_bc; x0=x0_vec)
println(@sprintf("Vortex–cylinder collision: Re = %.0f, mu = %.4g, dt = %.4f", Re, mu, dt))

times, states = solve_NavierStokesMono_unsteady!(solver; Δt=dt, T_end=T_end, scheme=:CN)
println("Completed $(length(times) - 1) steps; final time = $(times[end])")

###########
# Post-processing: enstrophy and final snapshots
###########
ens = [enstrophy_circulation(fluid, states[i]) for i in 1:length(states)]

open("enstrophy_vortex_cylinder.csv", "w") do io
    println(io, "time,enstrophy")
    for (t, e) in zip(times, ens)
        println(io, @sprintf("%.6f,%.8e", t, e))
    end
end
println("Saved enstrophy time series to enstrophy_vortex_cylinder.csv")

Ux, Uy = unpack_velocity_fields(states[end], nu_x, nu_y, mesh_ux, mesh_uy)
omega_final = circulation_vorticity(fluid, states[end])
speed_final = sqrt.(Ux.^2 .+ Uy.^2)

xs_p, ys_p = mesh_p.centers
xs_ux, ys_ux = mesh_ux.nodes
θ = range(0, 2π, length=180)
circle_x = xo .+ radius .* cos.(θ)
circle_y = yo_c .+ radius .* sin.(θ)

fig = Figure(resolution=(1200, 520))
ax_speed = Axis(fig[1, 1], xlabel="x", ylabel="y", title="|u| at t=$(round(times[end]; digits=2))")
hm_speed = heatmap!(ax_speed, xs_ux, ys_ux, speed_final; colormap=:plasma)
lines!(ax_speed, circle_x, circle_y; color=:white, linewidth=2)
Colorbar(fig[1, 2], hm_speed, label="|u|")

ax_vort = Axis(fig[1, 3], xlabel="x", ylabel="y", title="Vorticity at t=$(round(times[end]; digits=2))")
hm_vort = heatmap!(ax_vort, xs_p, ys_p, omega_final; colormap=:curl)
lines!(ax_vort, circle_x, circle_y; color=:white, linewidth=2)
Colorbar(fig[1, 4], hm_vort, label="ω")

save("vortex_cylinder_collision.png", fig)
println("Saved snapshot to vortex_cylinder_collision.png")
