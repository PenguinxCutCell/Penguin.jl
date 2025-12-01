using Penguin
using IterativeSolvers
using VTLInputs
using VTKOutputs

#=
    PoissonFrance2D.jl

Solve a steady 2D diffusion (Poisson) problem inside the geometry described by
`cube.stl`. The STL is converted into a signed distance function
(SDF) with VTLInputs (restricted to the z = 0 plane) and used as the level-set
representation for cut-cell capacities.
=#

# Path to the STL model (shared with the 3D example)
stl_path = normpath(joinpath(@__DIR__, "cube.stl"))
isfile(stl_path) || error("Missing STL file at $(stl_path).")

# Signed distance function coming from the STL (restricted to 2D)
sdf2d = compute_stl_sdf(stl_path; dims=2)
level_set = (x, y, _t = 0.0) -> -sdf2d(x, y)

# Mesh covering the STL extent with a small padding
x0, y0 = -1.0, -1.0
lx, ly = 4.0, 4.0
nx, ny = 60, 60
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Build capacity and differential operators from the STL-based level set
capacity = Capacity(level_set, mesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
operator = DiffusionOps(capacity)

# Boundary conditions: exterior box at 1.0, STL interface pinned to 0.0
bc_outer = Dirichlet(1.0)
bc_interface = Dirichlet(0.0)
bc_b = BorderConditions(Dict(
    :left => bc_outer,
    :right => bc_outer,
    :top => bc_outer,
    :bottom => bc_outer,
))

# Source term and diffusivity
source = (x, y, _t = 0.0) -> 1.0
diffusivity = (x, y, _t = 0.0) -> 1.0
phase = Phase(capacity, operator, source, diffusivity)

# Steady diffusion solver
solver = DiffusionSteadyMono(phase, bc_b, bc_interface)
solve_DiffusionSteadyMono!(solver; method=Base.:\)

# Export numerical solution
write_vtk(joinpath(@__DIR__, "poisson_france_2d"), mesh, solver)

plot_solution(solver, mesh, level_set, capacity)

println("2D Poisson problem on France STL solved. Unknowns = $(length(solver.x))")
