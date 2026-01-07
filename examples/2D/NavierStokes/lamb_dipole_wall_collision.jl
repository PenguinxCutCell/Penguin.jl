using Penguin
using CairoMakie
using LinearAlgebra
using Printf
using SpecialFunctions

"""
Dipolar (Lamb-Chaplygin) vortex propagating toward a no-slip wall, inspired by
the Basilisk example `sandbox/Antoonvh/lamb-dipole.c`. The vortex core has
radius `R=1` and travels with speed `U0`; viscosity sets the Reynolds number
`Re = U0 * R / nu`. Three cases are run: Re = 750, 1500, and 3000.
"""

###########
# Domain and discretization
###########
Lx = 15.0
Ly = 15.0
nx = 64
ny = 64
x0 = 0.0
y0 = 0.0

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

###########
# Geometry and capacities (no internal body)
###########
fluid_everywhere = (x, y, _t=0.0) -> -1.0
capacity_ux = Capacity(fluid_everywhere, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(fluid_everywhere, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(fluid_everywhere, mesh_p;  compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)

###########
# Boundary conditions
###########
wall_bc = Dirichlet(0.0)      # no-slip on the left wall
outflow_bc = Outflow(0.0)     # weakly enforcing zero normal gradient elsewhere

bc_ux = BorderConditions(Dict(
    :left=>wall_bc, :right=>wall_bc, :bottom=>outflow_bc, :top=>outflow_bc
))
bc_uy = BorderConditions(Dict(
    :left=>wall_bc, :right=>wall_bc, :bottom=>outflow_bc, :top=>outflow_bc
))

pressure_gauge = PinPressureGauge()
interface_bc = Dirichlet(0.0)

###########
# Lamb–Chaplygin dipole velocity field
###########
const k_root = 3.8317 # first zero of J1; defines Lamb–Chaplygin core

function lamb_dipole_velocity(x, y; center=(4.7, 7.6), R=1.0, U0=1.0)
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

        u_r = (U0 / r) * base * cθ
        u_θ = -U0 * dbase_dr * sθ
    else
        coef = U0 * (R^2 / r^2)
        u_r = coef * cθ
        u_θ = coef * sθ
    end

    u_x = u_r * cθ - u_θ * sθ
    u_y = u_r * sθ + u_θ * cθ
    return (u_x, u_y)
end

function initial_state_from_dipole(mesh_ux, mesh_uy; center, R, U0)
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    uomega_x0 = Float64[lamb_dipole_velocity(x, y; center=center, R=R, U0=U0)[1] for x in xs_ux, y in ys_ux]
    uomega_y0 = Float64[lamb_dipole_velocity(x, y; center=center, R=R, U0=U0)[2] for x in xs_uy, y in ys_uy]

    ugamma_x0 = copy(uomega_x0)
    ugamma_y0 = copy(uomega_y0)

    return vec(uomega_x0), vec(ugamma_x0), vec(uomega_y0), vec(ugamma_y0)
end

###########
# Helpers: vorticity and plotting
###########
function compute_vorticity(Ux, Uy, dx, dy)
    nx_u, ny_u = size(Ux)
    omega = zeros(nx_u, ny_u)
    @inbounds for j in 2:ny_u-1
        for i in 2:nx_u-1
            dUy_dx = (Uy[i + 1, j] - Uy[i - 1, j]) / (2 * dx)
            dUx_dy = (Ux[i, j + 1] - Ux[i, j - 1]) / (2 * dy)
            omega[i, j] = dUy_dx - dUx_dy
        end
    end
    omega[1, :] .= omega[2, :]
    omega[end, :] .= omega[end - 1, :]
    omega[:, 1] .= omega[:, 2]
    omega[:, end] .= omega[:, end - 1]
    return omega
end

function plot_snapshot(fluid, state, mesh_ux, mesh_uy, Re; outfile_prefix="lamb_dipole")
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes
    dx = xs_ux[2] - xs_ux[1]
    dy = ys_ux[2] - ys_ux[1]

    Ux = reshape(state[1:nu_x], (length(xs_ux), length(ys_ux)))
    Uy = reshape(state[2nu_x + 1:2nu_x + nu_y], (length(xs_uy), length(ys_uy)))
    speed = sqrt.(Ux.^2 .+ Uy.^2)
    omega = compute_vorticity(Ux, Uy, dx, dy)

    fig = Figure(resolution=(1200, 520))
    ax_speed = Axis(fig[1, 1], xlabel="x", ylabel="y", title="|u|, Re=$(Int(round(Re)))")
    hm_speed = heatmap!(ax_speed, xs_ux, ys_ux, speed; colormap=:plasma)
    Colorbar(fig[1, 2], hm_speed, label="|u|")

    ax_omega = Axis(fig[1, 3], xlabel="x", ylabel="y", title="Vorticity, Re=$(Int(round(Re)))")
    hm_omega = heatmap!(ax_omega, xs_ux, ys_ux, omega; colormap=:curl)
    Colorbar(fig[1, 4], hm_omega, label="ω")

    outfile = @sprintf("%s_Re%d.png", outfile_prefix, Int(round(Re)))
    save(outfile, fig)
    println("Saved snapshot to $(outfile)")
end

###########
# Time stepping
###########
rho = 1.0
U0 = 1.0
R = 1.0
Re_values = [750.0, 1500.0, 3000.0]
dt = 0.01
T_end = 1.0      # Basilisk used 12; reduce here for a quicker turnaround

uomega_x0, ugamma_x0, uomega_y0, ugamma_y0 = initial_state_from_dipole(mesh_ux, mesh_uy; center=(4.7, 7.6), R=R, U0=U0)

for Re in Re_values
    mu = rho * U0 * R / Re
    println(@sprintf("Running Lamb dipole against a wall: Re = %.0f (mu = %.4g)", Re, mu))

    f_u = (x, y, z=0.0, t=0.0) -> 0.0
    f_p = (x, y, z=0.0, t=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (capacity_ux, capacity_uy),
                  (operator_ux, operator_uy),
                  mesh_p,
                  capacity_p,
                  operator_p,
                  mu, rho, f_u, f_p)

    x0_vec = vcat(uomega_x0, ugamma_x0, uomega_y0, ugamma_y0, zeros(Float64, np))
    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)

    times, states = solve_NavierStokesMono_unsteady!(solver; Δt=dt, T_end=T_end, scheme=:CN)
    println(@sprintf("Re = %.0f completed in %d steps (final t = %.3f)", Re, length(times) - 1, times[end]))

    plot_snapshot(fluid, states[end], mesh_ux, mesh_uy, Re; outfile_prefix="lamb_dipole_wall")
end

println("Lamb–Chaplygin dipole runs finished.")
