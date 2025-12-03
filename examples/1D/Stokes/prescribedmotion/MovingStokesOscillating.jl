using Penguin
using IterativeSolvers
using CairoMakie

# 1D oscillating piston-type motion solved with the moving Stokes solver

# Spatial meshes (pressure cell centered, velocity staggered)
nx = 80
Lx = 1.0
x0 = 0.0
mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))
dx = Lx / nx
mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5 * dx,))

# Body motion: interface oscillates around the mid-point
ξ₀ = 0.5 * Lx
ampl = 0.15 * Lx
freq = 2π
body = (x, t, _=0.0) -> x - (ξ₀ + ampl * sin(freq * t))

# Time window
Δt = 0.001
Tstart = 0.0
Tend = 0.1

# Initial space-time meshes/capacities/operators
STmesh_u = Penguin.SpaceTimeMesh(mesh_u, [Tstart, Tstart + Δt], tag=mesh_u.tag)
STmesh_p = Penguin.SpaceTimeMesh(mesh_p, [Tstart, Tstart + Δt], tag=mesh_p.tag)
capacity_u = Capacity(body, STmesh_u; compute_centroids=true)
capacity_p = Capacity(body, STmesh_p; compute_centroids=true)
operator_u = DiffusionOps(capacity_u)
operator_p = DiffusionOps(capacity_p)

# Physical data
μ = 1.0
ρ = 1.0
fᵤ = (x, y=0.0, z=0.0, t=0.0) -> 0.0 #5.0 * sin(π * (x - x0) / Lx) * cos(freq * t)
fₚ = (x, y=0.0, z=0.0, t=0.0) -> 0.0

fluid = Fluid(STmesh_u, capacity_u, operator_u,
              STmesh_p, capacity_p, operator_p,
              μ, ρ, fᵤ, fₚ)

# Boundary conditions (Dirichlet on the outer walls)
bc_velocity = BorderConditions(Dict(:bottom => Dirichlet(0.0),
                                    :top    => Dirichlet(0.0)))
u_body = (x, t) -> ampl * freq * cos(freq * t)
bc_cut = Dirichlet((x, t, _=0) -> u_body(x, t))
pressure_gauge = PinPressureGauge()

# Build and solve
solver = MovingStokesMono(fluid, bc_velocity, bc_cut;
                          pressure_gauge=pressure_gauge,
                          scheme="BE", t_eval=Tstart + Δt,
                          x0=zeros(2 * (nx + 1) + (nx + 1)))

times = solve_MovingStokesMono!(solver, fluid, body,
                                mesh_u, mesh_p,
                                bc_velocity, bc_cut,
                                Δt, Tstart, Tend;
                                pressure_gauge=pressure_gauge,
                                scheme="BE",
                                method=Base.:\)

println("Stored states: ", length(solver.states))

# Simple visualization of the last velocity profile
state = solver.states[end]
nu = nx + 1
uω = state[1:nu]
positions = mesh_u.nodes[1][1:nu]

fig = Figure(resolution=(700, 400))
ax = Axis(fig[1, 1], xlabel="x", ylabel="u", title="Oscillating 1D moving Stokes (last state)")
lines!(ax, positions, uω, color=:steelblue, linewidth=2)
display(fig)

function create_moving_stokes_animation(solver, mesh, times; filename="moving_stokes.gif")
    xs = mesh.nodes[1][1:length(mesh.nodes[1])]
    nu = length(xs)
    vel_obs = Observable(solver.states[1][1:nu])
    interface_pos = ξ₀ + ampl * sin(freq * times[1])
    interface_x = Observable([interface_pos, interface_pos])

    fig = Figure(resolution=(800, 400))
    ax = Axis(fig[1, 1], xlabel="x", ylabel="u", title="Velocity and moving interface")
    lines!(ax, xs, vel_obs, color=:steelblue, linewidth=2)
    lines!(ax, interface_x, [-1.0, 1.0], color=:red, linewidth=2, linestyle=:dash)

    record(fig, filename, 1:length(solver.states)) do i
        vel_obs[] = solver.states[i][1:nu]
        interface_pos = ξ₀ + ampl * sin(freq * times[min(i, length(times))])
        interface_x[] = [interface_pos, interface_pos]
    end
    println("Saved animation to $(filename)")
end

create_moving_stokes_animation(solver, mesh_u, times, filename="moving_stokes.gif")
