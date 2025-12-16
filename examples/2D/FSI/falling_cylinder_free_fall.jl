using Penguin
using CairoMakie
using GeometryBasics
using StaticArrays
using IterativeSolvers
using LinearAlgebra

# Relative L2 error helper
Err2_rel(q_ref, q) = sqrt(sum(abs2, q_ref .- q) / max(sum(abs2, q_ref), eps(Float64)))

# Free fall of a dense rigid cylinder in air.
# The cylinder starts at rest and falls under gravity inside a rectangular cavity.
# Outputs: field snapshots (u_x, u_y, |u|, vorticity) and body kinematics (Yc, Uc_y vs. time).

############
# Geometry and body definition
############
nx, ny = 256, 256
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
T_end = 0.1

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
# Visualization: |u| and vorticity side by side
############
function plot_speed_vorticity(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y, dx, dy;
                              frame::Int=length(times),
                              outfile::String="falling_cylinder_free_fall_speed_vorticity.png")
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

    fig = Figure(size=(1100, 450))

    ax_speed = Axis(fig[1, 1], xlabel="x [m]", ylabel="y [m]",
                    title="|u| at t=$(round(t, digits=3))", aspect=DataAspect())
    hm_speed = heatmap!(ax_speed, xs_ux, ys_ux, speed; colormap=:plasma)
    lines!(ax_speed, circle_x, circle_y, color=:white, linewidth=2)
    Colorbar(fig[1, 2], hm_speed, label="|u| [m/s]")

    ax_vort = Axis(fig[1, 3], xlabel="x [m]", ylabel="y [m]",
                   title="Vorticity at t=$(round(t, digits=3))", aspect=DataAspect())
    hm_vort = heatmap!(ax_vort, xs_ux, ys_ux, vorticity; colormap=:curl)
    lines!(ax_vort, circle_x, circle_y, color=:white, linewidth=2)
    Colorbar(fig[1, 4], hm_vort, label="ω [1/s]")

    save(outfile, fig)
    println("Saved |u| and vorticity side-by-side plot to $(outfile)")
    return fig
end

############
# Visualization: |u| and streamlines side by side
############
function plot_speed_streamlines(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y;
                                frame::Int=length(times),
                                outfile::String="falling_cylinder_free_fall_speed_streamlines.png")
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
    Colorbar(fig[1, 2], hm_speed, label="|u| [m/s]")

    ax_stream = Axis(fig[1, 3], xlabel="x [m]", ylabel="y [m]",
                     title="Streamlines at t=$(round(t, digits=3))", aspect=DataAspect())
    streamplot!(ax_stream, velocity_field, xs_ux[1]..xs_ux[end], ys_ux[1]..ys_ux[end];
                density=1.0, color=(p) -> norm(p))
    lines!(ax_stream, circle_x, circle_y, color=:red, linewidth=2)

    save(outfile, fig)
    println("Saved |u| and streamlines side-by-side plot to $(outfile)")
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

    err_y = Err2_rel(y_exact, cy)
    err_vy = Err2_rel(vy_exact, vy)

    fig = Figure(size=(900, 420))

    ax_y = Axis(fig[1, 1], xlabel="time [s]", ylabel="Yc [m]",
                title="Center of mass Yc (Err2_rel=$(round(err_y; digits=3)))")
    scatter!(ax_y, times, cy, color=:blue, label="numerical")
    lines!(ax_y, times, y_exact, color=:black, label="free-fall exact")
    axislegend(ax_y; position=:lb)

    ax_vy = Axis(fig[1, 2], xlabel="time [s]", ylabel="Uc_y [m/s]",
                 title="Vertical velocity Uc_y (Err2_rel=$(round(err_vy; digits=3)))")
    scatter!(ax_vy, times, vy, color=:red, label="numerical")
    lines!(ax_vy, times, vy_exact, color=:black, label="Uc_y = g*t")
    axislegend(ax_vy; position=:lb)

    save(outfile, fig)
    println("Saved kinematics plot to $(outfile)")
    println("Err2_rel(Yc) = $(err_y), Err2_rel(Uc_y) = $(err_vy)")
    return fig
end

############
# Streamlines at final state
############
function plot_streamlines(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y;
                          frame::Int=length(times),
                          outfile::String="falling_circle_streamlines.png")
    t = times[frame]
    state = states[frame]

    uωx = state[1:nu_x]
    uωy = state[2nu_x + 1:2nu_x + nu_y]

    xs = mesh_ux.nodes[1]
    ys = mesh_ux.nodes[2]

    Ux = reshape(uωx, (length(xs), length(ys)))
    Uy = reshape(uωy, (length(xs), length(ys)))

    nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))
    velocity_field(x, y) = Point2f(Ux[nearest_index(xs, x), nearest_index(ys, y)],
                                   Uy[nearest_index(xs, x), nearest_index(ys, y)])

    cx = [c[1] for c in centers]
    cy = [c[2] for c in centers]

    fig_stream = Figure(resolution=(800, 600))
    ax_s = Axis(fig_stream[1, 1], xlabel="x", ylabel="y", title="Streamlines at t=$(round(t, digits=3))")

    # Use a callable color function (Makie expects a function), map every point to 0.0 and use a gray colormap
    streamplot!(ax_s, velocity_field, xs[1]..xs[end], ys[1]..ys[end]; density=2.0, color=(p) -> norm(p))
    # Outline current circle position
    theta = range(0, 2pi, length=180)
    circle_x = cx[frame] .+ radius .* cos.(theta)
    circle_y = cy[frame] .+ radius .* sin.(theta)
    lines!(ax_s, circle_x, circle_y, color=:red, linewidth=2)

    save(outfile, fig_stream)
    println("Saved streamline plot to $(outfile)")
    return fig_stream
end

plot_streamlines(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y)

############
# Animation: |u| and streamlines side by side
############
function animate_speed_streamlines(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y;
                                   n_frames::Int=80, framerate::Int=10,
                                   outfile::String="falling_cylinder_speed_streamlines.gif")
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    n_frames = min(n_frames, length(states))
    frame_indices = round.(Int, range(1, length(states), length=n_frames))

    state_to_fields = function (state)
        uomx = state[1:nu_x]
        uomy = state[2nu_x + 1:2nu_x + nu_y]
        Ux = reshape(uomx, (length(xs_ux), length(ys_ux)))
        Uy = reshape(uomy, (length(xs_uy), length(ys_uy)))
        speed = sqrt.(Ux.^2 .+ Uy.^2)
        return speed, Ux, Uy
    end

    extrema_pairs = map(states) do st
        s, _, _ = state_to_fields(st)
        return (minimum(s), maximum(s))
    end
    global_min = 0.0
    global_max = 1.0

    speed0, Ux0, Uy0 = state_to_fields(states[frame_indices[1]])
    speed_obs = Observable(speed0)

    theta = range(0, 2pi, length=180)
    cx_all = [c[1] for c in centers]
    cy_all = [c[2] for c in centers]
    circle_x_obs = Observable(cx_all[frame_indices[1]] .+ radius .* cos.(theta))
    circle_y_obs = Observable(cy_all[frame_indices[1]] .+ radius .* sin.(theta))
    title_obs = Observable("Velocity magnitude and streamlines at t = $(round(times[frame_indices[1]]; digits=3))")

    fig_anim = Figure(resolution=(1100, 520))

    ax_speed = Axis(fig_anim[1, 1], xlabel="x [m]", ylabel="y [m]",
                    title=title_obs, aspect=DataAspect())
    hm_anim = heatmap!(ax_speed, xs_ux, ys_ux, speed_obs; colormap=:plasma, colorrange=(global_min, global_max))
    lines!(ax_speed, circle_x_obs, circle_y_obs, color=:white, linewidth=2)
    Colorbar(fig_anim[1, 2], hm_anim, label="|u|")

    # make the streamlines axis use the same x/y limits as the heatmap so it's not too wide
    ax_stream = Axis(fig_anim[1, 3],
                     xlabel="x [m]", ylabel="y [m]",
                     title="", aspect=DataAspect(),
                     limits=(xs_ux[1], xs_ux[end], ys_ux[1], ys_ux[end]))


    nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))

    record(fig_anim, outfile, 1:n_frames; framerate=framerate) do frame
        empty!(ax_stream)
        idx = frame_indices[frame]
        speed, Ux, Uy = state_to_fields(states[idx])
        speed_obs[] = speed
        circle_x_obs[] = cx_all[idx] .+ radius .* cos.(theta)
        circle_y_obs[] = cy_all[idx] .+ radius .* sin.(theta)
        t = times[idx]
        title_obs[] = "Velocity magnitude and streamlines at t = $(round(t; digits=3))"

        # rebuild streamlines for this frame
        velocity_field(x, y) = Point2f(Ux[nearest_index(xs_ux, x), nearest_index(ys_ux, y)],
                                       Uy[nearest_index(xs_uy, x), nearest_index(ys_uy, y)])
        streamplot!(ax_stream, velocity_field, xs_ux[1]..xs_ux[end], ys_ux[1]..ys_ux[end];
                    density=1.0, color=(p) -> norm(p))
        lines!(ax_stream, circle_x_obs[], circle_y_obs[], color=:red, linewidth=2)
    end


    println("Animation saved as $(outfile)")
    return outfile
end

# Save to CSV the body kinematics data
using CSV
using DataFrames
df_kinematics = DataFrame(time=times,
                             center_x=[c[1] for c in centers],
                             center_y=[c[2] for c in centers],
                             velocity_x=[v[1] for v in velocities],
                             velocity_y=[v[2] for v in velocities])
CSV.write("falling_cylinder_free_fall_kinematics_$(nx).csv", df_kinematics)
println("Saved body kinematics data to falling_cylinder_free_fall_kinematics_$(nx).csv")


############
# Run post-processing
############
plot_flow_fields(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y, dx, dy;
                 frame=length(times))
plot_speed_vorticity(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y, dx, dy;
                     frame=length(times))
plot_speed_streamlines(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y;
                       frame=length(times))
plot_body_kinematics(times, centers, velocities, gravity, center0)
animate_speed_streamlines(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y)
