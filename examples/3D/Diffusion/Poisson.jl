using Penguin
using IterativeSolvers

### 3D Test Case : Monophasic Steady Diffusion Equation inside a Disk
# Define the mesh
nx, ny, nz = 30, 30, 30
lx, ly, lz = 4., 4., 4.
x0, y0, z0 = 0., 0., 0.
domain = ((x0, lx), (y0, ly), (z0, lz))
mesh = Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))

# Define the body
radius, center = ly/3, (lx/2, ly/2, lz/2) #.+ (0.01, 0.01, 0.01)
circle = (x, y, z) -> -(sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - radius)

# Define the capacity
capacity = Capacity(circle, mesh)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(1.0)
bc_i = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc, :front => bc, :back => bc))

# Define the source term
h = (x,y,z)-> 1.0 #sin(x)*cos(10*y)
Dif = (x,y,z)-> 1.0

Fluide = Phase(capacity, operator, h, Dif)

# Define the solver
solver = DiffusionSteadyMono(Fluide, bc_b, bc_i)

# Solve the problem
solve_DiffusionSteadyMono!(solver; method=Base.:\)

# Write the solution to a VTK file
write_vtk("poisson_3d", mesh, solver)

# Plot the solution
#plot_solution(solver, mesh, circle, capacity)
