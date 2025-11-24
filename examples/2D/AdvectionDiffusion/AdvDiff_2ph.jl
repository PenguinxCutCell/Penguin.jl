using Penguin
using IterativeSolvers

### 2D Test Case : Diphasic Steady Advection-Diffusion Equation inside a Disk
# Define the mesh
nx, ny = 40, 40
lx, ly = 4., 4.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/4, (lx/2, ly/2) .+ (0.1, 0.1)
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
f1 = (x,y,_)->1.0 #cos(x)*sin(10*y)
f2 = (x,y,_)->0.0 #cos(x)*sin(10*y)

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, (x,y,_)->1.0)
Fluide_2 = Phase(capacity_c, operator_c, f2, (x,y,_)->1.0)

# Define the solver
solver = AdvectionDiffusionSteadyDiph(Fluide_1, Fluide_2, bc_b, ic)

# Solve the problem
solve_AdvectionDiffusionSteadyDiph!(solver; method=Base.:\)

# Plot the solution usign Makie
plot_solution(solver, mesh, circle, capacity)