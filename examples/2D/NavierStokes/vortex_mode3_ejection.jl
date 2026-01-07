using Penguin
using CairoMakie
using LinearAlgebra
using SpecialFunctions
using Printf

"""
Vortex ejection from a mode-3 instability (Kizner et al., 2013).
No-slip unit cylinder at the origin; initial velocity follows the
piecewise potential/vortical profile with an m=3 perturbation.
Outputs an animation of the vorticity field.
"""

###########
# Domain and meshes
###########
L = 40.0
nx = 128          # ~10 cells per radius; raise to improve resolution
ny = 128
mesh_p = Penguin.Mesh((nx, ny), (L, L), (-L / 2, -L / 2))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (L, L), (-L / 2 - 0.5 * dx, -L / 2))
mesh_uy = Penguin.Mesh((nx, ny), (L, L), (-L / 2, -L / 2 - 0.5 * dy))

###########
# Cylinder geometry (radius = 1)
###########
radius = 1.0
cylinder_levelset = (x, y, _=0.0) -> radius - hypot(x, y)

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
# Boundary conditions
###########
outflow = Outflow(0.0)
wall = Dirichlet(0.0)
bc_ux = BorderConditions(Dict(:left=>wall, :right=>wall, :bottom=>outflow, :top=>outflow))
bc_uy = BorderConditions(Dict(:left=>wall, :right=>wall, :bottom=>outflow, :top=>outflow))
body_bc = Dirichlet(0.0) # no-slip on cylinder
pressure_gauge = PinPressureGauge()

###########
# Mode-3 perturbed vortex profile (Kizner)
###########
a1 = 1.5
b1 = 2.25
P = 0.005
m = 3.0
gamma = (a1^2 - 1.0) / (b1^2 - a1^2)

radp_factor = 1 / sqrt(1 + 0.5 * P^2)

function radial_velocity(r, theta_p)
    r1 = (1 + P * sin(theta_p)) * r * radp_factor
    if 0.9 < r1 < a1
        return r1 - 1 / r1
    elseif a1 <= r1 <= b1
        return -gamma * r1 + ((1 + gamma) * a1^2 - 1.0) / r1
    else
        return 0.0
    end
end

function initial_state(mesh_ux, mesh_uy)
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    uomega_x0 = Matrix{Float64}(undef, length(xs_ux), length(ys_ux))
    uomega_y0 = Matrix{Float64}(undef, length(xs_uy), length(ys_uy))

    @inbounds for (i, x) in enumerate(xs_ux)
        for (j, y) in enumerate(ys_ux)
            r2 = x^2 + y^2
            if r2 < radius^2
                uomega_x0[i, j] = 0.0
                continue
            end
            r = sqrt(r2)
            theta_p = m * asin(clamp(x / r, -1.0, 1.0))
            vr = radial_velocity(r, theta_p)
            if abs(r2) < 1e-14
                uomega_x0[i, j] = 0.0
            else
                uomega_x0[i, j] = 0.5 * vr * r * (-y) / r2
            end
        end
    end

    @inbounds for (i, x) in enumerate(xs_uy)
        for (j, y) in enumerate(ys_uy)
            r2 = x^2 + y^2
            if r2 < radius^2
                uomega_y0[i, j] = 0.0
                continue
            end
            r = sqrt(r2)
            theta_p = m * asin(clamp(x / r, -1.0, 1.0))
            vr = radial_velocity(r, theta_p)
            if abs(r2) < 1e-14
                uomega_y0[i, j] = 0.0
            else
                uomega_y0[i, j] = 0.5 * vr * r * x / r2
            end
        end
    end

    uomega_x0 .*= mask_x
    uomega_y0 .*= mask_y
    ugamma_x0 = copy(uomega_x0)
    ugamma_y0 = copy(uomega_y0)
    return vec(uomega_x0), vec(ugamma_x0), vec(uomega_y0), vec(ugamma_y0)
end

###########
# Helpers
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

###########
# Physics and solver
###########
rho = 1.0
Re = 3000.0
mu = rho * 1.0 * radius / Re
dt = 0.0015
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

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, body_bc; x0=x0_vec)
println(@sprintf("Mode-3 vortex ejection: Re=%.0f, mu=%.4g, dt=%.4f", Re, mu, dt))

times, states = solve_NavierStokesMono_unsteady!(solver; Δt=dt, T_end=T_end, scheme=:BE)
println("Completed $(length(times) - 1) steps, final time = $(times[end])")

###########
# Animation (vorticity field)
###########
xp, yp = mesh_p.centers
θ = range(0, 2π, length=200)
circle_x = radius .* cos.(θ)
circle_y = radius .* sin.(θ)

frames = min(120, length(states))
frame_idx = round.(Int, range(1, length(states), length=frames))

# Precompute global color range for smoother animation
vort_samples = Float64[]
for idx in frame_idx
    Ux, Uy = unpack_velocity_fields(states[idx], nu_x, nu_y, mesh_ux, mesh_uy)
    omega = compute_vorticity(Ux, Uy, dx, dy)
    append!(vort_samples, omega[:])
end
vmax = maximum(abs, vort_samples)
vmax = vmax == 0 ? 1e-6 : vmax

fig = Figure(resolution=(900, 800))
ax = Axis(fig[1, 1], xlabel="x", ylabel="y", aspect=DataAspect(),
          title="Vorticity")
vort_obs = Observable(zeros(length(xp), length(yp)))
hm = heatmap!(ax, xp, yp, vort_obs; colormap=:coolwarm, colorrange=(-vmax, vmax))
lines!(ax, circle_x, circle_y; color=:black, linewidth=2)
Colorbar(fig[1, 2], hm, label="ω")

outfile = "vortex_mode3_vorticity.mp4"
record(fig, outfile, 1:frames; framerate=12) do f
    idx = frame_idx[f]
    Ux, Uy = unpack_velocity_fields(states[idx], nu_x, nu_y, mesh_ux, mesh_uy)
    omega = compute_vorticity(Ux, Uy, dx, dy)
    vort_obs[] = omega
    ax.title = @sprintf("Vorticity, t = %.2f", times[idx])
end
println("Saved animation to $(outfile)")
