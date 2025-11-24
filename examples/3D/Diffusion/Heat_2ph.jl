using Penguin
using IterativeSolvers

### 3D Test Case : Diphasic Unsteady Diffusion Equation inside a Sphere
# Define the mesh
nx, ny, nz = 40, 40, 40
lx, ly, lz = 4., 4., 4.
x0, y0, z0 = 0., 0., 0.
domain = ((x0, lx), (y0, ly), (z0, lz))
mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))

# Define the body
radius, center = ly/4, (lx/2, ly/2, lz/2) #.+ (0.01, 0.01, 0.01)
sphere = (x,y,z)->(sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - radius)
sphere_c = (x,y,z)->-(sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - radius)

# Define the capacity
capacity = Capacity(sphere, mesh)
capacity_c = Capacity(sphere_c, mesh)

# Define the operators
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(1.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc, :front => bc, :back => bc))

ic = InterfaceConditions(ScalarJump(1.0, 2.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Define the source term
f1 = (x,y,z,t)->0.0
f2 = (x,y,z,t)->0.0

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, (x,y,z)->1.0)
Fluide_2 = Phase(capacity_c, operator_c, f2, (x,y,z)->1.0)

# Initial condition
u0ₒ1 = ones((nx+1)*(ny+1)*(nz+1))
u0ᵧ1 = ones((nx+1)*(ny+1)*(nz+1))
u0ₒ2 = zeros((nx+1)*(ny+1)*(nz+1))
u0ᵧ2 = zeros((nx+1)*(ny+1)*(nz+1))
u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Define the solver
Δt = 0.01
Tend = 0.2
solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "BE")

# Solve the problem
solve_DiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, Δt, Tend, bc_b, ic, "BE"; method=Base.:\)

# Write the solution to a VTK file
#write_vtk("heat_3d", mesh, solver)

# Plot the solution
plot_solution(solver, mesh, sphere, capacity; state_i=10)