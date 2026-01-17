using Penguin
using CairoMakie
using GeometryBasics
using StaticArrays
using IterativeSolvers
using LinearAlgebra
using CSV
using DataFrames

# Dense rigid cylinder settling toward a no-slip bottom wall.
# The bottom uses u = 0 Dirichlet for both components (instead of the outflow
# condition in the free-fall case). The cylinder starts low in the cavity so
# the wall-induced slowdown is visible.

############
# Geometry and body definition
############
nx, ny = 64,64
Lx, Ly = 0.1, 0.2           # cavity width/height [m]
x0, y0 = 0.0, 0.0           # bottom-left corner

radius = 0.0125
center0 = SVector(0.05, 0.05)  # start close to the bottom wall
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
# - Top: shear-free tangential flow (Outflow on u_x), no-penetration (Dirichlet on u_y)
# - Bottom: no-slip (Dirichlet on both u_x and u_y)
############
ux_slip = Dirichlet(0.0)
uy_shear_free = Outflow()
ux_bottom_no_slip = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(
    :left => ux_slip, :right => ux_slip,
    :bottom => ux_bottom_no_slip, :top => ux_bottom_no_slip
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
mu_air = 1.0
rho_air = 1.0
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
T_end = 0.08

stokes_solver = MovingStokesUnsteadyMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut_init;
                                         scheme=scheme, x0=x0_vec)

# Rigid-body properties (dense cylinder)
rho_cylinder = 78.0
area = pi * radius^2
mass = rho_cylinder * area
gravity_force = (t, c, v) -> SVector(0.0, mass * gravity)
fsi = MovingStokesFSI2D(stokes_solver, body_shape, mass, center0, velocity0)

println("Running sedimenting cylinder with bottom no-slip wall:")
println("  radius=$(radius), mass=$(mass), dt=$(dt), T_end=$(T_end)")

times, states, centers, velocities, forces = solve_MovingStokesFSI2D!(fsi, mesh_p,
                                                                      dt, 0.0, T_end,
                                                                      (bc_ux, bc_uy);
                                                                      scheme=scheme,
                                                                      method=Base.:\,
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

############
# Visualization: |u| and streamlines side by side
############
function plot_speed_streamlines(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y;
                                frame::Int=length(times),
                                outfile::String="falling_cylinder_bottom_wall_speed_streamlines.png")
    t = times[frame]
    state = states[frame]
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    Ux, Uy = unpack_velocity_fields(state, nu_x, nu_y, mesh_ux, mesh_uy)
    speed = sqrt.(Ux.^2 .+ Uy.^2)

    cx = [c[1] for c in centers]
    cy = [c[2] for c in centers]
    theta = range(0, 2pi, length=180)
    circle_x = cx[frame] .+ radius .* cos.(theta)
    circle_y = cy[frame] .+ radius .* sin.(theta)

    nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))
    velocity_field(x, y) = Point2f(Ux[nearest_index(xs_ux, x), nearest_index(ys_ux, y)],
                                   Uy[nearest_index(xs_uy, x), nearest_index(ys_uy, y)])

    fig = Figure(size=(1100, 450))

    ax_speed = Axis(fig[1, 1], xlabel="x [m]", ylabel="y [m]",
                    title="|u| at t=$(round(t, digits=3))", aspect=DataAspect())
    hm_speed = heatmap!(ax_speed, xs_ux, ys_ux, speed; colormap=:plasma)
    lines!(ax_speed, circle_x, circle_y, color=:white, linewidth=2)
    lines!(ax_speed, [xs_ux[1], xs_ux[end]], [y0, y0]; color=:gray, linestyle=:dash, linewidth=1)
    Colorbar(fig[1, 2], hm_speed, label="|u| [m/s]")

    ax_stream = Axis(fig[1, 3], xlabel="x [m]", ylabel="y [m]",
                     title="Streamlines at t=$(round(t, digits=3))", aspect=DataAspect())
    streamplot!(ax_stream, velocity_field, xs_ux[1]..xs_ux[end], ys_ux[1]..ys_ux[end];
                density=1.0, color=(p) -> norm(p))
    lines!(ax_stream, circle_x, circle_y, color=:red, linewidth=2)
    lines!(ax_stream, [xs_ux[1], xs_ux[end]], [y0, y0]; color=:black, linestyle=:dash, linewidth=1)

    save(outfile, fig)
    println("Saved |u| and streamlines side-by-side plot to $(outfile)")
    return fig
end

############
# Body kinematics and wall approach
############
function plot_gap_and_velocity(times, centers, velocities, radius, y_bottom;
                               outfile::String="falling_cylinder_bottom_wall_velocity.png")
    gaps = [c[2] - radius - y_bottom for c in centers]
    vy = [v[2] for v in velocities]

    fig = Figure(size=(1200, 420))

    ax_gap = Axis(fig[1, 1], xlabel="time [s]", ylabel="gap to bottom [m]",
                  title="Wall gap over time")
    lines!(ax_gap, times, gaps, color=:black)

    ax_vy_time = Axis(fig[1, 2], xlabel="time [s]", ylabel="U_c,y [m/s]",
                      title="Vertical velocity (downwards negative)")
    lines!(ax_vy_time, times, vy, color=:blue)
    scatter!(ax_vy_time, times, vy; markersize=3, color=:blue)

    ax_vy_gap = Axis(fig[1, 3], xlabel="gap to bottom [m]", ylabel="U_c,y [m/s]",
                     title="Vertical velocity vs. wall gap")
    lines!(ax_vy_gap, gaps, vy, color=:red)

    save(outfile, fig)
    println("Saved gap/velocity plot to $(outfile)")
    return fig
end

function summarize_velocity_vs_gap(times, centers, velocities, radius, y_bottom;
                                   levels::Tuple=(0.03, 0.02, 0.01, 0.005))
    gaps = [c[2] - radius - y_bottom for c in centers]
    vy = [v[2] for v in velocities]

    println("Velocity trend while approaching the bottom wall (negative = downward):")
    for level in levels
        idx = findlast(>=(level - eps()), gaps)
        if idx !== nothing
            println("  gap≈$(round(gaps[idx]; digits=4)) m at t=$(round(times[idx]; digits=4)) s -> Uc_y=$(round(vy[idx]; digits=4)) m/s")
        end
    end
    idx_min_gap = argmin(gaps)
    println("  Minimum gap = $(round(gaps[idx_min_gap]; digits=5)) m at t=$(round(times[idx_min_gap]; digits=4)) s with Uc_y=$(round(vy[idx_min_gap]; digits=4)) m/s")
end

############
# Save kinematics and post-process
############
gap_to_wall = [c[2] - radius - y0 for c in centers]
df_kinematics = DataFrame(time=times,
                          center_x=[c[1] for c in centers],
                          center_y=[c[2] for c in centers],
                          velocity_x=[v[1] for v in velocities],
                          velocity_y=[v[2] for v in velocities],
                          gap_to_bottom=gap_to_wall)
CSV.write("falling_cylinder_bottom_wall_kinematics_$(nx).csv", df_kinematics)
println("Saved body kinematics data to falling_cylinder_bottom_wall_kinematics_$(nx).csv")

plot_speed_streamlines(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y;
                       frame=length(times))
plot_gap_and_velocity(times, centers, velocities, radius, y0)
summarize_velocity_vs_gap(times, centers, velocities, radius, y0)

