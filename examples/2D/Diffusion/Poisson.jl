using Penguin
using LinearSolve

# Poisson equation inside a disk
# Define the mesh
nx, ny = 320, 320
lx, ly = 4., 4.
x0, y0 = 0., 0.
mesh = Mesh((nx, ny), (lx, ly), (x0, y0))

LS(x,y,_=0) = (sqrt((x-2)^2 + (y-2)^2) - 1.0) 

# Define the capacity
capacity = Capacity(LS, mesh, method="VOFI") # or capacity = Capacity(LS, mesh, method="ImplicitIntegration")

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
solve_DiffusionSteadyMono!(solver, algorithm=KrylovJL_GMRES(), log=true)

# Plot the solution
plot_solution(solver, mesh, LS, capacity)

# Analytical solution
u_analytic(x,y) = 1.0 - (x-2)^2 - (y-2)^2

# Compute the error
u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytic, solver, capacity, 2, false)