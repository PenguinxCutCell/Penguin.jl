using Penguin
using IterativeSolvers

### 3D Test Case : Monophasic Steady Diffusion Equation inside a Sphere
# Define the mesh
nx, ny, nz = 40, 40, 40
lx, ly, lz = 4., 4., 4.
x0, y0, z0 = 0., 0., 0.
domain = ((x0, lx), (y0, ly), (z0, lz))
mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))

# Define the body
radius, center = ly/4, (lx/2, ly/2, lz/2) .+ (0.01, 0.01, 0.01)
sphere = (x,y,z)->(sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - radius)

# Define the capacity
capacity = Capacity(sphere, mesh)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(10.0)
bc_i = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc, :front => bc, :back => bc))

# Define the source term
f = (x,y,z,t)->1.0
D = (x,y,z)->1.0

# Define the phase
Fluide = Phase(capacity, operator, f, D)

# Initial condition
u0ₒ = zeros((nx+1)*(ny+1)*(nz+1))
u0ᵧ = ones((nx+1)*(ny+1)*(nz+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
Δt = 0.01
Tend = 0.1
solver = DiffusionUnsteadyMono(Fluide, bc_b, bc_i, Δt, u0, "BE")

# Solve the problem
solve_DiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, bc, "BE"; method=IterativeSolvers.bicgstabl, abstol=1e-15, verbose=false)

# Write the solution to a VTK file
#write_vtk("heat_3d", mesh, solver)

# Plot the solution
#plot_solution(solver, mesh, sphere, capacity; state_i=5)

# Animation
#animate_solution(solver, mesh, sphere)
