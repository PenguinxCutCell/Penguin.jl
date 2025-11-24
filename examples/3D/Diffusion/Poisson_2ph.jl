using Penguin
using IterativeSolvers

### 3D Test Case : Diphasic Steady Diffusion Equation inside a Sphere
# Define the mesh
nx, ny, nz = 20, 20, 20
lx, ly, lz = 4., 4., 4.
x0, y0, z0 = 0., 0., 0.
domain = ((x0, lx), (y0, ly), (z0, lz))
mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))

# Define the body
radius, center = ly/4, (lx/2, ly/2, lz/2) #.+ (0.01, 0.01, 0.01)
sphere = (x,y,z)->(sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - radius)
sphere_c = (x,y,z)->-(sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - radius)

# Define the capacity
@time capacity = Capacity(sphere, mesh)
@time capacity_c = Capacity(sphere_c, mesh)

# Define the operators
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc_i = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc, :front => bc, :back => bc))

ic = InterfaceConditions(ScalarJump(1.0, 0.5, 0.0), FluxJump(1.0, 1.0, 0.0))

# Define the source term
f1 = (x,y,z)->1.0
f2 = (x,y,z)->1.0

D1, D2 = (x,y,z)->1.0, (x,y,z)->1.0

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, D1)
Fluide_2 = Phase(capacity_c, operator_c, f2, D2)

# Define the solver
@time solver = DiffusionSteadyDiph(Fluide_1, Fluide_2, bc_b, ic)

# Solve the problem
@time solve_DiffusionSteadyDiph!(solver; method=Base.:\)

# Plot the solution usign Makie
#plot_solution(solver, mesh, sphere, capacity)