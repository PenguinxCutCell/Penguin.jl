using Penguin
using CairoMakie
using LinearAlgebra
using SpecialFunctions
using Printf

"""
Dipolar vortex approaching an opening between two peninsulas (Basilisk rebound2.c).
Embedded walls are built from two horizontal slabs with rounded tips, leaving
an opening at x = 0. Vorticity dynamics are saved as a movie.
"""

# -------------------------
# Domain and meshes
# -------------------------
L = 30.0
nx = 48
ny = 48
mesh_p = Penguin.Mesh((nx, ny), (L, L), (-L/2, -L/2))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (L, L), (-L/2 - 0.5dx, -L/2))
mesh_uy = Penguin.Mesh((nx, ny), (L, L), (-L/2, -L/2 - 0.5dy))

# -------------------------
# Embedded boundary: two peninsulas with rounded tips
# -------------------------
xpl, ypl, wpl = 1.2, 0.12, 0.2
function peninsula_levelset(x, y, _=0.0)
    ax = abs(x)
    if ax < xpl
        # rounded tip centered at (±xpl, ypl)
        return sqrt((ax - xpl)^2 + (y - ypl)^2) - wpl
    else
        # horizontal slab of half-width wpl around ypl
        return abs(y - ypl) - wpl
    end
end

capacity_ux = Capacity(peninsula_levelset, mesh_ux; compute_centroids=true, method="VOFI", integration_method=:vofijul)
capacity_uy = Capacity(peninsula_levelset, mesh_uy; compute_centroids=true, method="VOFI", integration_method=:vofijul)
capacity_p  = Capacity(peninsula_levelset, mesh_p;  compute_centroids=true, method="VOFI", integration_method=:vofijul)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)

# -------------------------
# Boundary conditions
# -------------------------
outflow = Outflow(0.0)
wall = Dirichlet(0.0)
bc_ux = BorderConditions(Dict(:left=>wall, :right=>wall, :bottom=>outflow, :top=>outflow))
bc_uy = BorderConditions(Dict(:left=>wall, :right=>wall, :bottom=>outflow, :top=>outflow))
interface_bc = Dirichlet(0.0) # no-slip on the embedded boundary
pressure_gauge = PinPressureGauge()

# -------------------------
# Lamb–Chaplygin dipole initial condition
# -------------------------
const k_root = 3.8317
dipole_center = (0.0, -5.0) # start below the opening
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

function initial_state(mesh_ux, mesh_uy)
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    uomega_x0 = Float64[
        peninsula_levelset(x, y) > 0 ? 0.0 : lamb_dipole_velocity(x, y)[1]
        for x in xs_ux, y in ys_ux
    ]
    uomega_y0 = Float64[
        peninsula_levelset(x, y) > 0 ? 0.0 : lamb_dipole_velocity(x, y)[2]
        for x in xs_uy, y in ys_uy
    ]
    ugamma_x0 = copy(uomega_x0)
    ugamma_y0 = copy(uomega_y0)
    return vec(uomega_x0), vec(ugamma_x0), vec(uomega_y0), vec(ugamma_y0)
end

# -------------------------
# Helpers
# -------------------------
function unpack_velocity_fields(state, nu_x, nu_y, mesh_ux, mesh_uy)
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes
    uωx = state[1:nu_x]
    uωy = state[2nu_x + 1:2nu_x + nu_y]
    Ux = reshape(uωx, (length(xs_ux), length(ys_ux)))
    Uy = reshape(uωy, (length(xs_uy), length(ys_uy)))
    return Ux, Uy
end

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

# -------------------------
# Physics and solver
# -------------------------
rho = 1.0
Re = 3000.0
mu = rho * U0 * 1.0 / Re
dt = 0.01
T_end = 1.0

f_u = (x, y, z=0.0, t=0.0) -> 0.0
f_p = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              mu, rho, f_u, f_p)

uomega_x0, ugamma_x0, uomega_y0, ugamma_y0 = initial_state(mesh_ux, mesh_uy)
x0_vec = vcat(uomega_x0, ugamma_x0, uomega_y0, ugamma_y0, zeros(Float64, np))

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)
println(@sprintf("Vortex rebound/opening: Re=%.0f, mu=%.4g, dt=%.4f", Re, mu, dt))

times, states = solve_NavierStokesMono_unsteady!(solver; Δt=dt, T_end=T_end, scheme=:CN)
println("Completed $(length(times)-1) steps, final time = $(times[end])")

# -------------------------
# Output: vorticity snapshots and animation
# -------------------------
xp, yp = mesh_p.nodes
θ = range(0, 2π, length=200)

function opening_outline()
    xs = range(-L/2, L/2, length=400)
    ys = range(-L/2, L/2, length=400)
    [peninsula_levelset(x, y) for x in xs, y in ys]
end

# Animation
frames = min(120, length(states))
frame_idx = round.(Int, range(1, length(states), length=frames))
vort_samples = Float64[]
for idx in frame_idx
    Ux, Uy = unpack_velocity_fields(states[idx], nu_x, nu_y, mesh_ux, mesh_uy)
    omega = compute_vorticity(Ux, Uy, dx, dy)
    append!(vort_samples, omega[:])
end
vmax = maximum(abs, vort_samples); vmax = vmax == 0 ? 1e-6 : vmax

fig = Figure(resolution=(900, 800))
ax = Axis(fig[1, 1], xlabel="x", ylabel="y", aspect=DataAspect(),
          title="Vorticity")
vort_obs = Observable(zeros(length(xp), length(yp)))
hm = heatmap!(ax, xp, yp, vort_obs; colormap=:coolwarm, colorrange=(-vmax, vmax))
Colorbar(fig[1, 2], hm, label="ω")

outfile = "vortex_rebound_vorticity.mp4"
record(fig, outfile, 1:frames; framerate=12) do f
    idx = frame_idx[f]
    Ux, Uy = unpack_velocity_fields(states[idx], nu_x, nu_y, mesh_ux, mesh_uy)
    omega = compute_vorticity(Ux, Uy, dx, dy)
    vort_obs[] = omega
    ax.title = @sprintf("Vorticity, t = %.2f", times[idx])
end
println("Saved animation to $(outfile)")

# Final snapshot
Ux_final, Uy_final = unpack_velocity_fields(states[end], nu_x, nu_y, mesh_ux, mesh_uy)
omega_final = compute_vorticity(Ux_final, Uy_final, dx, dy)
Ux_final = reshape(solver.x[1:nu_x], (length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2])))
Uy_final = reshape(solver.x[2nu_x + 1:2nu_x + nu_y], (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2])))
speed_final = sqrt.(Ux_final.^2 .+ Uy_final.^2)

fig2 = Figure(resolution=(1200, 520))
ax_speed = Axis(fig2[1, 1], xlabel="x", ylabel="y", title="|u| at t=$(round(times[end]; digits=2))")
hm_speed = heatmap!(ax_speed, mesh_ux.nodes[1], mesh_ux.nodes[2], speed_final; colormap=:plasma)
Colorbar(fig2[1, 2], hm_speed, label="|u|")

ax_vort = Axis(fig2[1, 3], xlabel="x", ylabel="y", title="Vorticity at t=$(round(times[end]; digits=2))")
hm_vort = heatmap!(ax_vort, xp, yp, omega_final; colormap=:curl)
Colorbar(fig2[1, 4], hm_vort, label="ω")

save("vortex_rebound_opening.png", fig2)
println("Saved snapshot to vortex_rebound_opening.png")
