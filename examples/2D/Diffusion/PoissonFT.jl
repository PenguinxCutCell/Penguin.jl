using Penguin
using FrontCutTracking

# Poisson equation inside a disk using Front Tracking
# Define the mesh
nx, ny = 40, 40
lx, ly = 4., 4.
x0, y0 = 0., 0.
mesh = Mesh((nx, ny), (lx, ly), (x0, y0))

# Create a front tracker for a circle centered at (2,2) with radius 1
front = FrontTracker()
create_circle!(front, 2.0, 2.0, 1.0, 500)  # Use 500 points for accurate circle

# Compute capacity using front tracking
capacity = Capacity(front, mesh)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(1.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))

# Define the source term and coefficients
f(x,y,_=0) = 4.0
D(x,y,_=0) = 1.0

# Define the Fluid
Fluide = Phase(capacity, operator, f, D)

# Define the solver
solver = DiffusionSteadyMono(Fluide, bc_b, bc)

# Solve the system
solve_DiffusionSteadyMono!(solver; method=Base.:\)

# For visualization, we need a level set function
# We can use the SDF from the front tracker
LS(x,y,_=0) = sdf(front, x, y)

# Plot the solution
plot_solution(solver, mesh, LS, capacity)

# Analytical solution
u_analytic(x,y) = 1.0 - (x-2)^2 - (y-2)^2

# Compute the error
u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytic, solver, capacity, 2, true)

println("Global error: ", global_err)
println("Error in full fluid cells: ", full_err)
println("Error in cut cells: ", cut_err)