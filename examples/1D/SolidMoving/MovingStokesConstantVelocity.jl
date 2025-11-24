using Penguin
using IterativeSolvers
using CairoMakie

# 1D constant-speed moving interface for the prescribed-motion Stokes solver

nx = 60
Lx = 1.0
x0 = 0.0

mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))
dx = Lx / nx
mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5 * dx,))

ξ₀ = 0.25 * Lx
Vint = 0.2 * Lx  # constant interface speed

body = (x, t, _=0.0) -> x - (ξ₀ + Vint * t)
u_body = (x, t) -> Vint

Δt = 0.01
Tstart = 0.0
Tend = 1.0

STmesh_u = Penguin.SpaceTimeMesh(mesh_u, [Tstart, Tstart + Δt], tag=mesh_u.tag)
STmesh_p = Penguin.SpaceTimeMesh(mesh_p, [Tstart, Tstart + Δt], tag=mesh_p.tag)
capacity_u = Capacity(body, STmesh_u; compute_centroids=true)
capacity_p = Capacity(body, STmesh_p; compute_centroids=true)
operator_u = DiffusionOps(capacity_u)
operator_p = DiffusionOps(capacity_p)

μ = 1.0
ρ = 1.0
fᵤ = (x, y=0.0, z=0.0, t=0.0) -> 0.0
fₚ = (x, y=0.0, z=0.0, t=0.0) -> 0.0

bc_velocity = BorderConditions(Dict(:bottom => Dirichlet(0.0),
                                    :top    => Dirichlet(0.0)))
bc_cut = Dirichlet((x, t, _=0.0) -> u_body(x, t))
pressure_gauge = PinPressureGauge()

fluid = Fluid(STmesh_u, capacity_u, operator_u,
              STmesh_p, capacity_p, operator_p,
              μ, ρ, fᵤ, fₚ)

solver = MovingStokesMono(fluid, bc_velocity, bc_cut;
                          pressure_gauge=pressure_gauge,
                          scheme="BE", t_eval=Tstart + Δt)

times = solve_MovingStokesMono!(solver, fluid, body,
                                mesh_u, mesh_p,
                                bc_velocity, bc_cut,
                                Δt, Tstart, Tend;
                                pressure_gauge=pressure_gauge,
                                scheme="BE",
                                method=gmres)

nu = nx + 1
x_nodes = mesh_u.nodes[1][1:nu]

fig = Figure(resolution=(700, 400))
ax = Axis(fig[1, 1], xlabel="x", ylabel="u", title="Constant-speed interface Stokes")
for (i, t) in enumerate(range(Tstart, Tend, length=4))
    idx = clamp(round(Int, i * length(solver.states) / 4), 1, length(solver.states))
    uω = solver.states[idx][1:nu]
    lines!(ax, x_nodes, uω, label="t=$(round(times[min(idx, length(times))], digits=3))")
end
axislegend(ax, position=:rb)
display(fig)

function create_constant_velocity_animation(solver, mesh, times; filename="moving_stokes_constant_velocity.gif")
    nu = length(mesh.nodes[1])
    x_nodes = mesh.nodes[1][1:nu]
    vel_obs = Observable(solver.states[1][1:nu])

    fig = Figure(resolution=(700, 400))
    ax = Axis(fig[1, 1], xlabel="x", ylabel="u", title="Constant-speed interface")
        lines!(ax, x_nodes, vel_obs, color=:steelblue, linewidth=2)

    record(fig, filename, 1:length(solver.states)) do i
        vel_obs[] = solver.states[i][1:nu]
        ax.title = "t=$(round(times[min(i, length(times))], digits=3))"
    end
    println("Saved animation to $(filename)")
end

create_constant_velocity_animation(solver, mesh_u, times, filename="moving_stokes_constant_velocity.gif")

# Animation gif
