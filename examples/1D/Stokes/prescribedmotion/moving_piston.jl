using Penguin
using IterativeSolvers

# 1D prescribed-motion Stokes sanity test (oscillating interface)

nx = 64
Lx = 1.0
x0 = 0.0
mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))
dx = Lx / nx
mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5 * dx,))

ξ₀ = 0.5 * Lx
ampl = 0.1 * Lx
freq = 2π
body = (x, t, _=0.0) -> x - (ξ₀ + ampl * sin(freq * t))
u_body = (x, t) -> ampl * freq * cos(freq * t)

Δt = 0.01
Tstart = 0.0
Tend = 0.05

STmesh_u = Penguin.SpaceTimeMesh(mesh_u, [Tstart, Tstart + Δt])
STmesh_p = Penguin.SpaceTimeMesh(mesh_p, [Tstart, Tstart + Δt])
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
bc_cut = Dirichlet((x, t, _=0) -> u_body(x, t))
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

println("Max velocity over run: ", maximum(abs, vcat(solver.states...)))
println("Recorded time samples: ", times)
