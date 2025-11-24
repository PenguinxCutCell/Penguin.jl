using Penguin
using IterativeSolvers

### 2D Test Case : Monophasic Steady Advection-Diffusion Equation inside a Disk
# Define the mesh
nx, ny = 40, 40
lx, ly = 4., 4.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/4, (lx/2, ly/2) .+ (0.01, 0.01)
circle = (x,y,_=0)->-(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# Define the capacity
capacity = Capacity(circle, mesh)

# Initialize the velocity field with a rotating field
uₒx, uₒy = initialize_rotating_velocity_field(nx, ny, lx, ly, x0, y0, 3.0)
uₒ = (uₒx, uₒy)

# For boundary velocities, if they are zero:
uᵧ = zeros(2 * (nx + 1) * (ny + 1))

# Define the operators
operator = ConvectionOps(capacity, uₒ, uᵧ)

# Define the boundary conditions
ic = Dirichlet(1.0)
bc = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc))

# Define the source term
f = (x,y,_)-> 0.0 #sin(x)*cos(10*y)

Fluide = Phase(capacity, operator, f, (x,y,_)->0.0)

# Define the solver
solver = AdvectionDiffusionSteadyMono(Fluide, bc_b, ic)

# Solve the problem
solve_AdvectionDiffusionSteadyMono!(solver; method=Base.:\)

# Plot the solution
plot_solution(solver, mesh, circle, capacity)