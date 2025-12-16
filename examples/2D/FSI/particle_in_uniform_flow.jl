using Penguin
using CairoMakie
using StaticArrays
using IterativeSolvers

# Rigid particle convected by a uniform Stokes flow (explicit FSI coupling).
# The particle is a translating circle; the fluid drag accelerates it until the
# relative slip decreases.

############
# Geometry and body definition
############
nx, ny = 35, 35
Lx, Ly = 8.0, 2.0
x0, y0 = -Lx / 2, -Ly / 2

radius = 0.25
center0 = SVector(-2.0, 0.0)
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
# Capacities/operators at t=0 (particle at center0)
############
capacity_ux = Capacity((x, y, _=0.0) -> body_shape(x, y, center0), mesh_ux; method="VOFI", integration_method=:vofijul)
capacity_uy = Capacity((x, y, _=0.0) -> body_shape(x, y, center0), mesh_uy; method="VOFI", integration_method=:vofijul)
capacity_p = Capacity((x, y, _=0.0) -> body_shape(x, y, center0), mesh_p; method="VOFI", integration_method=:vofijul)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p = DiffusionOps(capacity_p)

############
# Boundary conditions: uniform inflow, outflow at right, no-slip walls
############
Umax = 1.0
parabola = (x, y) -> 4Umax * (y - y0) * (Ly - (y - y0)) / (Ly^2)
ux_left = Dirichlet((x, y, _) -> parabola(x, y))
ux_right = Outflow()
ux_bottom = Dirichlet(0.0)
ux_top = Dirichlet(0.0)

uy_zero = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(
    :left => ux_left, :right => ux_right,
    :bottom => ux_bottom, :top => ux_top
))
bc_uy = BorderConditions(Dict(
    :left => uy_zero, :right => uy_zero,
    :bottom => uy_zero, :top => uy_zero
))

pressure_gauge = PinPressureGauge()

# Cut boundary placeholder (overwritten each step inside FSI loop)
bc_cut_init = (Dirichlet(0.0), Dirichlet(0.0))

############
# Physics
############
mu = 1.0
rho = 1.0
f_u = (x, y, z=0.0, t=0.0) -> 0.0
f_p = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              mu, rho, f_u, f_p)

############
# Solver setup
############
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
Ntot = 2 * (nu_x + nu_y) + np
x0_vec = zeros(Ntot)

scheme = :BE
dt = 0.01
T_end = 0.5

stokes_solver = MovingStokesUnsteadyMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut_init;
                                         scheme=scheme, x0=x0_vec)

# Rigid-body properties
mass = 2.0
fsi = MovingStokesFSI2D(stokes_solver, body_shape, mass, center0, velocity0)

U_in = 1.0
println("Running particle-in-uniform-flow case:")
println("  U_in=$(U_in), radius=$(radius), mass=$(mass), dt=$(dt), T_end=$(T_end)")

times, states, centers, velocities, forces = solve_MovingStokesFSI2D!(fsi, mesh_p,
                                                                      dt, 0.0, T_end,
                                                                      (bc_ux, bc_uy);
                                                                      scheme=scheme,
                                                                      method=IterativeSolvers.gmres,
                                                                      geometry_method="VOFI",
                                                                      integration_method=:vofijul,
                                                                      compute_centroids=true)

println("Completed $(length(times) - 1) steps; final center = ", centers[end], ", velocity = ", velocities[end])

############
# Visualization
############
function visualize_particle(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y;
                            frame=length(times))
    t = times[frame]
    state = states[frame]

    uwx = state[1:nu_x]
    uwy = state[2nu_x + 1:2nu_x + nu_y]

    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes
    Ux = reshape(uwx, (length(xs_ux), length(ys_ux)))
    Uy = reshape(uwy, (length(xs_uy), length(ys_uy)))
    speed = sqrt.(Ux.^2 .+ Uy.^2)

    cx = [c[1] for c in centers]
    cy = [c[2] for c in centers]

    fig = Figure(size=(900, 400))

    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="t = $(round(t, digits=3))",
              aspect=DataAspect())
    hm = heatmap!(ax, xs_ux, ys_ux, speed; colormap=:viridis)
    lines!(ax, cx, cy, color=:white, linewidth=2, label="trajectory")
    scatter!(ax, [cx[end]], [cy[end]], color=:red, markersize=10, label="particle")

    theta = range(0, 2pi, length=100)
    circle_x = cx[end] .+ radius .* cos.(theta)
    circle_y = cy[end] .+ radius .* sin.(theta)
    lines!(ax, circle_x, circle_y, color=:white, linewidth=2)

    axislegend(ax; position=:lt)
    Colorbar(fig[1, 2], hm, label="|u|")

    ax2 = Axis(fig[2, 1], xlabel="time", ylabel="center x", title="Particle drift")
    lines!(ax2, times, cx, color=:black)
    return fig
end

fig = visualize_particle(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y)
save("particle_uniform_flow.png", fig)
println("Saved visualization to particle_uniform_flow.png")

############
# Animation
############
function animate_particle(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y;
                          n_frames::Int=80, framerate::Int=10, outfile::String="particle_uniform_flow.gif")
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    # Subsample frames to keep file size reasonable
    n_frames = min(n_frames, length(states))
    frame_indices = round.(Int, range(1, length(states), length=n_frames))

    # Helper: reshape state -> speed field
    state_to_speed = function (state)
        uomx = state[1:nu_x]
        uomy = state[2nu_x + 1:2nu_x + nu_y]
        Ux = reshape(uomx, (length(xs_ux), length(ys_ux)))
        Uy = reshape(uomy, (length(xs_uy), length(ys_uy)))
        return sqrt.(Ux.^2 .+ Uy.^2)
    end

    # Fix colorbar across frames
    extrema_pairs = map(states) do st
        s = state_to_speed(st)
        return (minimum(s), maximum(s))
    end
    global_min = minimum(first, extrema_pairs)
    global_max = maximum(last, extrema_pairs)

    # Observables for animation
    speed_obs = Observable(state_to_speed(states[frame_indices[1]]))
    title_obs = Observable("Velocity magnitude at t = $(round(times[frame_indices[1]]; digits=3))")

    theta = range(0, 2pi, length=120)
    cx_all = [c[1] for c in centers]
    cy_all = [c[2] for c in centers]
    circle_x_obs = Observable(cx_all[frame_indices[1]] .+ radius .* cos.(theta))
    circle_y_obs = Observable(cy_all[frame_indices[1]] .+ radius .* sin.(theta))
    traj_x_obs = Observable([cx_all[1:frame_indices[1]]; fill(NaN, length(cx_all) - frame_indices[1])])
    traj_y_obs = Observable([cy_all[1:frame_indices[1]]; fill(NaN, length(cy_all) - frame_indices[1])])

    fig_anim = Figure(resolution=(900, 520))
    ax_anim = Axis(fig_anim[1, 1], xlabel="x", ylabel="y", title=title_obs, aspect=DataAspect())

    hm_anim = heatmap!(ax_anim, xs_ux, ys_ux, speed_obs; colormap=:plasma, colorrange=(global_min, global_max))
    lines!(ax_anim, traj_x_obs, traj_y_obs, color=:white, linewidth=2)
    lines!(ax_anim, circle_x_obs, circle_y_obs, color=:white, linewidth=2)
    scatter!(ax_anim, [cx_all[frame_indices[1]]], [cy_all[frame_indices[1]]], color=:red, markersize=10)
    Colorbar(fig_anim[1, 2], hm_anim, label="Velocity magnitude")

    record(fig_anim, outfile, 1:n_frames; framerate=framerate) do frame
        idx = frame_indices[frame]
        speed_obs[] = state_to_speed(states[idx])
        t = times[idx]
        title_obs[] = "Velocity magnitude at t = $(round(t; digits=3))"

        circle_x_obs[] = cx_all[idx] .+ radius .* cos.(theta)
        circle_y_obs[] = cy_all[idx] .+ radius .* sin.(theta)
        traj_x_obs[] = [cx_all[1:idx]; fill(NaN, length(cx_all) - idx)]
        traj_y_obs[] = [cy_all[1:idx]; fill(NaN, length(cy_all) - idx)]
    end

    println("Animation saved as $(outfile)")
    return outfile
end

animate_particle(times, states, centers, mesh_ux, mesh_uy, radius, nu_x, nu_y)
