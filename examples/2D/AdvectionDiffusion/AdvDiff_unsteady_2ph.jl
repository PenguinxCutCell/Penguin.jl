using Penguin
using IterativeSolvers

### 2D Test Case : Diphasic Unsteady Advection-Diffusion Equation inside a Disk
# Define the mesh
nx, ny = 40, 40
lx, ly = 8., 8.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/8, (lx/2, ly/2) #.+ (0.1, 0.1)
circle = (x,y,_=0)->sqrt((x-center[1])^2 + (y-center[2])^2) - radius
circle_c = (x,y,_=0)->-(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# Define the capacity
capacity = Capacity(circle, mesh)
capacity_c = Capacity(circle_c, mesh)

# Initialize the velocity field with a rotating field
uₒx, uₒy = initialize_rotating_velocity_field(nx, ny, lx, ly, x0, y0, 3.0)
uₒ = (uₒx, uₒy)

# For boundary velocities, if they are zero:
uᵧ = zeros(2 * (nx + 1) * (ny + 1))


# Define the operators
operator = ConvectionOps(capacity, uₒ, uᵧ)
operator_c = ConvectionOps(capacity_c, uₒ, uᵧ)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc))

ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Define the source term
f1 = (x,y,z,t)->0.0
f2 = (x,y,z,t)->0.0

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, (x,y,_)->1.0)
Fluide_2 = Phase(capacity_c, operator_c, f2, (x,y,_)->1.0)

# Initial condition
u0ₒ1 = ones((nx+1)*(ny+1))
u0ᵧ1 = ones((nx+1)*(ny+1))
u0ₒ2 = zeros((nx+1)*(ny+1))
u0ᵧ2 = zeros((nx+1)*(ny+1))
u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Define the solver
Δt = 0.1
Tend = 1.0
solver = AdvectionDiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "CN")

# Solve the problem
solve_AdvectionDiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, Δt, Tend, bc_b, ic, "CN"; method=Base.:\)

# Write the solution to a VTK file
write_vtk("solution", mesh, solver)

# Plot the solution
plot_solution(solver, mesh, circle, capacity)

# Animation
#animate_solution(solver, mesh, circle)