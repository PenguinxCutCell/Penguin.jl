using Penguin
using CairoMakie
using IterativeSolvers

"""
Doubly-periodic shear layer instability (adapted from Basilisk doublep.c).
Two tanh shear layers in a unit square are perturbed by a small transverse
velocity to trigger roll-up of Kelvin-Helmholtz vortices.

Reference: Hokpunna & Manhart (2010), J. Comput. Phys. 229(20):7545-7570.
"""

# ---------------------------------------------------------------------------
# Domain and discretization
# ---------------------------------------------------------------------------
Lx = 1.0
Ly = 1.0
nx = 128
ny = 128

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (-0.5 * dx, 0.0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (0.0, -0.5 * dy))

# ---------------------------------------------------------------------------
# Capacities/operators (fully filled domain)
# ---------------------------------------------------------------------------
solid_indicator = (x, y, _=0.0) -> -1.0
capacity_ux = Capacity(solid_indicator, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(solid_indicator, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(solid_indicator, mesh_p;  compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)

# ---------------------------------------------------------------------------
# Boundary conditions (periodic in both directions)
# ---------------------------------------------------------------------------
periodic_bc = BorderConditions(Dict(
    :left=>Periodic(), :right=>Periodic(),
    :bottom=>Periodic(), :top=>Periodic()
))
bc_ux = periodic_bc
bc_uy = periodic_bc
pressure_gauge = PinPressureGauge()
interface_bc = Dirichlet(0.0)

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
rho = 1.0
Re = 1e4
mu = rho / Re
f_u = (x, y, z=0.0, t=0.0) -> 0.0
f_p = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              mu, rho, f_u, f_p)

# ---------------------------------------------------------------------------
# Initial conditions (double shear layer)
# ---------------------------------------------------------------------------
sigma = 30.0
epss = 0.05

shear_profile(y) = y <= 0.5 ? tanh(sigma * (y - 0.25)) : tanh(sigma * (0.75 - y))
perturbation(x) = epss * sin(2.0 * pi * x)

function initial_state(mesh_ux, mesh_uy)
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    uomega_x0 = Matrix{Float64}(undef, length(xs_ux), length(ys_ux))
    uomega_y0 = Matrix{Float64}(undef, length(xs_uy), length(ys_uy))

    @inbounds for (j, y) in enumerate(ys_ux)
        val = shear_profile(y)
        for i in eachindex(xs_ux)
            uomega_x0[i, j] = val
        end
    end

    @inbounds for (i, x) in enumerate(xs_uy)
        val = perturbation(x)
        for j in eachindex(ys_uy)
            uomega_y0[i, j] = val
        end
    end

    ugamma_x0 = copy(uomega_x0)
    ugamma_y0 = copy(uomega_y0)
    return vec(uomega_x0), vec(ugamma_x0), vec(uomega_y0), vec(ugamma_y0)
end

uomega_x0, ugamma_x0, uomega_y0, ugamma_y0 = initial_state(mesh_ux, mesh_uy)
x0_vec = vcat(uomega_x0, ugamma_x0, uomega_y0, ugamma_y0, zeros(Float64, np))

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function unpack_velocity_fields(state, nu_x, nu_y, mesh_ux, mesh_uy)
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes
    uomega_x = state[1:nu_x]
    uomega_y = state[2 * nu_x + 1:2 * nu_x + nu_y]
    Ux = reshape(uomega_x, (length(xs_ux), length(ys_ux)))
    Uy = reshape(uomega_y, (length(xs_uy), length(ys_uy)))
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

# ---------------------------------------------------------------------------
# Time integration (store states, then sample for visualization)
# ---------------------------------------------------------------------------
dt = 0.0025
T_end = 1.2
frame_dt = 0.025
scheme = :CN
make_animation = true

println("[Shear Layer] Re=$(Re), nx=$(nx), ny=$(ny), dt=$(dt), T_end=$(T_end)")

times, states = solve_NavierStokesMono_unsteady!(solver; method=IterativeSolvers.gmres ,Δt=dt, T_end=T_end, scheme=scheme)

xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]
frame_times = Float64[]
vort_frames = Vector{Matrix{Float64}}()

global next_frame = 0.0
for (ti, st) in zip(times, states)
    if ti >= next_frame - 1e-12
        Ux, Uy = unpack_velocity_fields(st, nu_x, nu_y, mesh_ux, mesh_uy)
        omega = compute_vorticity(Ux, Uy, dx, dy)
        push!(vort_frames, omega)
        push!(frame_times, ti)
        global next_frame += frame_dt
    end
end

global vmax = 0.0
for omega in vort_frames
    global vmax = max(vmax, maximum(abs, omega))
end
vmax = vmax == 0.0 ? 1e-6 : vmax

# ---------------------------------------------------------------------------
# Visualisation (final snapshot and optional animation)
# ---------------------------------------------------------------------------
omega_final = vort_frames[end]
levels = range(-0.8 * vmax, 0.8 * vmax, length=13)

fig = Figure(resolution=(800, 700))
ax = Axis(fig[1, 1], xlabel="x", ylabel="y",
          title="Vorticity at t=$(round(frame_times[end]; digits=3))",
          aspect=DataAspect())
hm = heatmap!(ax, xs, ys, omega_final; colormap=:coolwarm, colorrange=(-vmax, vmax))
contour!(ax, xs, ys, omega_final; levels=levels, color=:black, linewidth=1)
Colorbar(fig[1, 2], hm, label="vorticity")
save("navierstokes2d_doubly_periodic_shear_layer.png", fig)
display(fig)

println("[Shear Layer] snapshot saved to navierstokes2d_doubly_periodic_shear_layer.png")

if make_animation && length(vort_frames) > 1
    fig_anim = Figure(resolution=(800, 700))
    ax_anim = Axis(fig_anim[1, 1], xlabel="x", ylabel="y",
                   title="Vorticity",
                   aspect=DataAspect())
    vort_obs = Observable(vort_frames[1])
    hm_anim = heatmap!(ax_anim, xs, ys, vort_obs; colormap=:coolwarm, colorrange=(-vmax, vmax))
    Colorbar(fig_anim[1, 2], hm_anim, label="vorticity")

    outfile = "navierstokes2d_doubly_periodic_shear_layer.mp4"
    record(fig_anim, outfile, 1:length(vort_frames); framerate=12) do k
        vort_obs[] = vort_frames[k]
        ax_anim.title = "Vorticity, t=$(round(frame_times[k]; digits=3))"
    end
    println("[Shear Layer] animation saved to $(outfile)")
end
