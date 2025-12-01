using Penguin
using IterativeSolvers
using VTLInputs
using VTKOutputs

#= 
	PoissonSTL.jl

Solve a steady 3D diffusion (Poisson) problem inside the geometry described by the
`france-outline-1000.stl` surface. The STL is converted into a signed distance
function (SDF) with VTLInputs, then used directly as the level-set description
when building cut-cell capacities.
=#

# Path to the STL model
stl_path = joinpath(@__DIR__, "france-outline-1000.stl")
isfile(stl_path) || error("Missing STL file at $(stl_path).")

# Signed distance function coming from the STL
sdf = compute_stl_sdf(stl_path)
level_set = (x, y, z, _t = 0.0) -> -sdf(x, y, z)

# Mesh covering the STL extent with a small padding
x0, y0, z0 = -8.0, -8.0, -0.5
lx, ly, lz = 16.5, 16.5, 1.0
nx, ny, nz = 80, 80, 16
mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))

# Build capacity and differential operators from the STL-based level set
capacity = Capacity(level_set, mesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
operator = DiffusionOps(capacity)

# Boundary conditions: exterior box at 1.0, STL surface pinned to 0.0
bc_outer = Dirichlet(1.0)
bc_interface = Dirichlet(0.0)
bc_b = BorderConditions(Dict(
))

# Source term and diffusivity
source = (x, y, z, _t = 0.0) -> 1.0
diffusivity = (x, y, z, _t = 0.0) -> 1.0
phase = Phase(capacity, operator, source, diffusivity)

# Steady diffusion solver
solver = DiffusionSteadyMono(phase, bc_b, bc_interface)
solve_DiffusionSteadyMono!(solver; method=Base.:\)

# Export numerical solution
write_vtk(joinpath(@__DIR__, "poisson_france_stl"), mesh, solver)

# Optional visualization when running in a graphical session
#plot_solution(solver, mesh, level_set, capacity)

println("Poisson problem on France STL solved. Unknowns = $(length(solver.x))")
