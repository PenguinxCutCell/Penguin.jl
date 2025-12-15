using Penguin
using CairoMakie
using StaticArrays
using IterativeSolvers

# Free fall of a dense rigid cylinder in air.
# The cylinder starts at rest and falls under gravity inside a rectangular cavity.
# Outputs: field snapshots (u_x, u_y, |u|, vorticity) and body kinematics (Yc, Uc_y vs. time).

############
# Geometry and body definition
############
nx, ny = 80, 80
Lx, Ly = 0.1, 0.2           # cavity width/height [m]
x0, y0 = 0.0, 0.0           # bottom-left corner

radius = 0.0125
center0 = SVector(0.05, 0.15)
velocity0 = SVector(0.0, 0.0)
body_shape = (x, y, c) -> radius - sqrt((x - c[1])^2 + (y - c[2])^2)

############
# Meshes
############
mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

############
# Capacities/operators at t=0 (body at center0)
############
capacity_ux = Capacity((x, y, _=0.0) -> body_shape(x, y, center0), mesh_ux)
capacity_uy = Capacity((x, y, _=0.0) -> body_shape(x, y, center0), mesh_uy)
capacity_p = Capacity((x, y, _=0.0) -> body_shape(x, y, center0), mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p = DiffusionOps(capacity_p)

############
# Boundary conditions:
# - Slip on lateral walls: u dot n = 0 (Dirichlet on u_x), tangential stress-free (Outflow on u_y)
# - Top/bottom: no penetration (Dirichlet on u_y), shear-free tangential flow (Outflow on u_x)
############
ux_slip = Dirichlet(0.0)
uy_shear_free = Outflow()

bc_ux = BorderConditions(Dict(
    :left => ux_slip, :right => ux_slip,
    :bottom => Outflow(), :top => Outflow()
))
bc_uy = BorderConditions(Dict(
    :left => uy_shear_free, :right => uy_shear_free,
    :bottom => Dirichlet(0.0), :top => Dirichlet(0.0)
))

pressure_gauge = PinPressureGauge()

# Cut boundary placeholder (overwritten each step in the FSI loop)
bc_cut_init = (Dirichlet(0.0), Dirichlet(0.0))

############
# Physics
############
mu_air = 1.8e-5
rho_air = 1.2
gravity = -9.81
f_u = (x, y, z=0.0, t=0.0) -> 0.0
f_p = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              mu_air, rho_air, f_u, f_p)

############
# Solver setup
############
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
Ntot = 2 * (nu_x + nu_y) + np
x0_vec = zeros(Ntot)

scheme = :BE
dt = 5.0e-4
T_end = 0.15

stokes_solver = MovingStokesUnsteadyMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut_init;
                                         scheme=scheme, x0=x0_vec)

# Rigid-body properties (dense cylinder)
rho_cylinder = 7800.0
area = pi * radius^2
mass = rho_cylinder * area
gravity_force = (t, c, v) -> SVector(0.0, mass * gravity)
fsi = MovingStokesFSI2D(stokes_solver, body_shape, mass, center0, velocity0)

println("Running free-fall cylinder case:")
println("  radius=$(radius), mass=$(mass), dt=$(dt), T_end=$(T_end)")

times, states, centers, velocities, forces = solve_MovingStokesFSI2D!(fsi, mesh_p,
                                                                      dt, 0.0, T_end,
                                                                      (bc_ux, bc_uy);
                                                                      scheme=scheme,
                                                                      method=IterativeSolvers.gmres,
                                                                      geometry_method="VOFI",
                                                                      integration_method=:vofijul,
                                                                      compute_centroids=true,
                                                                      external_force=gravity_force)

println("Completed $(length(times) - 1) steps; final center = ", centers[end], ", velocity = ", velocities[end])

############
# Helpers
############
function unpack_velocity_fields(state, nu_x, nu_y, mesh_ux, mesh_uy)
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    uomx = state[1:nu_x]
    uomy = state[2nu_x + 1:2nu_x + nu_y]

    Ux = reshape(uomx, (length(xs_ux), length(ys_ux)))
    Uy = reshape(uomy, (length(xs_uy), length(ys_uy)))
    return Ux, Uy
end

function compute_vorticity(Ux, Uy, dx, dy)
    nx_u, ny_u = size(Ux)
    omega = zeros(nx_u, ny_u)
    @inbounds for j in 2:ny_u-1
        for i in 2:nx_u-1
            dUy_dx = (Uy[i + 1, j] - Uy[i - 1, j]) / (2dx)
            dUx_dy = (Ux[i, j + 1] - Ux[i, j - 1]) / (2dy)
            omega[i, j] = dUy_dx - dUx_dy
        end
    end
    omega[1, :] .= omega[2, :]
    omega[end, :] .= omega[end - 1, :]
    omega[:, 1] .= omega[:, 2]
    omega[:, end] .= omega[:, end - 1]
    return omega
end

############
# Visualization: field snapshots
############
function plot_flow_fields(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y, dx, dy;
                          frame::Int=length(times), outfile::String="falling_cylinder_free_fall_fields.png")
    t = times[frame]
    state = states[frame]
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    Ux, Uy = unpack_velocity_fields(state, nu_x, nu_y, mesh_ux, mesh_uy)
    speed = sqrt.(Ux.^2 .+ Uy.^2)
    vorticity = compute_vorticity(Ux, Uy, dx, dy)

    cx = [c[1] for c in centers]
    cy = [c[2] for c in centers]
    theta = range(0, 2pi, length=180)
    circle_x = cx[frame] .+ radius .* cos.(theta)
    circle_y = cy[frame] .+ radius .* sin.(theta)

    fig = Figure(size=(1100, 900))
    ax_ux = Axis(fig[1, 1], xlabel="x [m]", ylabel="y [m]", title="u_x at t=$(round(t, digits=3))", aspect=DataAspect())
    hm_ux = heatmap!(ax_ux, xs_ux, ys_ux, Ux; colormap=:balance)
    lines!(ax_ux, circle_x, circle_y, color=:white, linewidth=2)
    Colorbar(fig[1, 2], hm_ux, label="u_x [m/s]")

    ax_uy = Axis(fig[2, 1], xlabel="x [m]", ylabel="y [m]", title="u_y at t=$(round(t, digits=3))", aspect=DataAspect())
    hm_uy = heatmap!(ax_uy, xs_uy, ys_uy, Uy; colormap=:balance)
    lines!(ax_uy, circle_x, circle_y, color=:white, linewidth=2)
    Colorbar(fig[2, 2], hm_uy, label="u_y [m/s]")

    ax_speed = Axis(fig[3, 1], xlabel="x [m]", ylabel="y [m]", title="|u| at t=$(round(t, digits=3))", aspect=DataAspect())
    hm_speed = heatmap!(ax_speed, xs_ux, ys_ux, speed; colormap=:plasma)
    lines!(ax_speed, circle_x, circle_y, color=:white, linewidth=2)
    Colorbar(fig[3, 2], hm_speed, label="|u| [m/s]")

    ax_vort = Axis(fig[4, 1], xlabel="x [m]", ylabel="y [m]", title="Vorticity at t=$(round(t, digits=3))", aspect=DataAspect())
    hm_vort = heatmap!(ax_vort, xs_ux, ys_ux, vorticity; colormap=:curl)
    lines!(ax_vort, circle_x, circle_y, color=:white, linewidth=2)
    Colorbar(fig[4, 2], hm_vort, label="omega [1/s]")

    save(outfile, fig)
    println("Saved flow-field snapshots to $(outfile)")
    return fig
end

############
# Body kinematics: Yc and Uc_y vs time
############
function plot_body_kinematics(times, centers, velocities, gravity, center0;
                              outfile::String="falling_cylinder_free_fall_kinematics.png")
    cy = [c[2] for c in centers]
    vy = [v[2] for v in velocities]

    t0 = times[1]
    tau = times .- t0
    y_exact = center0[2] .+ 0.5 .* gravity .* (tau .^ 2)
    vy_exact = gravity .* tau

    fig = Figure(size=(900, 420))

    ax_y = Axis(fig[1, 1], xlabel="time [s]", ylabel="Yc [m]", title="Center of mass Yc")
    scatter!(ax_y, times, cy, color=:blue, label="numerical")
    lines!(ax_y, times, y_exact, color=:black, label="free-fall exact")
    axislegend(ax_y; position=:lb)

    ax_vy = Axis(fig[1, 2], xlabel="time [s]", ylabel="Uc_y [m/s]", title="Vertical velocity Uc_y")
    scatter!(ax_vy, times, vy, color=:red, label="numerical")
    lines!(ax_vy, times, vy_exact, color=:black, label="Uc_y = g*t")
    axislegend(ax_vy; position=:lb)

    save(outfile, fig)
    println("Saved kinematics plot to $(outfile)")
    return fig
end

############
# Run post-processing
############
plot_flow_fields(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y, dx, dy;
                 frame=length(times))
plot_body_kinematics(times, centers, velocities, gravity, center0)
