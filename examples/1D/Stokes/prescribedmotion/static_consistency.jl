using Penguin
using IterativeSolvers
using LinearAlgebra
# Static limit check: moving solver with zero motion vs steady Stokes

nx = 40
Lx = 1.0
x0 = 0.0
mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))
dx = Lx / nx
mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5 * dx,))

body = (x, t=0.0, _=0.0) -> x - 0.5 * Lx
Δt = 0.01

STmesh_u = Penguin.SpaceTimeMesh(mesh_u, [0.0, Δt])
STmesh_p = Penguin.SpaceTimeMesh(mesh_p, [0.0, Δt])
capacity_u = Capacity(body, STmesh_u; compute_centroids=true)
capacity_p = Capacity(body, STmesh_p; compute_centroids=true)
operator_u = DiffusionOps(capacity_u)
operator_p = DiffusionOps(capacity_p)

μ = 1.0
ρ = 1.0
fᵤ = (x, y=0.0, z=0.0) -> sin(π * x / Lx)
fₚ = (x, y=0.0, z=0.0) -> 0.0

bc_velocity = BorderConditions(Dict(:bottom => Dirichlet(0.0),
                                    :top    => Dirichlet(0.0)))
bc_cut = Dirichlet(0.0)
pressure_gauge = PinPressureGauge()

fluid_ST = Fluid(STmesh_u, capacity_u, operator_u,
                 STmesh_p, capacity_p, operator_p,
                 μ, ρ, fᵤ, fₚ)

solver_mov = MovingStokesMono(fluid_ST, bc_velocity, bc_cut;
                              pressure_gauge=pressure_gauge,
                              scheme="BE",
                              t_eval=Δt)

solve_MovingStokesMono!(solver_mov, fluid_ST, body,
                        mesh_u, mesh_p,
                        bc_velocity, bc_cut,
                        Δt, 0.0, 0.0;
                        pressure_gauge=pressure_gauge,
                        scheme="BE",
                        method=gmres)

capacity_u_static = Capacity(body, mesh_u; compute_centroids=true)
capacity_p_static = Capacity(body, mesh_p; compute_centroids=true)
operator_u_static = DiffusionOps(capacity_u_static)
operator_p_static = DiffusionOps(capacity_p_static)
fluid_static = Fluid(mesh_u, capacity_u_static, operator_u_static,
                     mesh_p, capacity_p_static, operator_p_static,
                     μ, ρ, fᵤ, fₚ)

solver_static = StokesMono(fluid_static, bc_velocity, pressure_gauge, bc_cut)
solve_StokesMono!(solver_static; method=gmres)

diff = solver_mov.states[end][1:(nx+1)] .- solver_static.x[1:(nx+1)]
println("Static limit residual norm = ", norm(diff))
