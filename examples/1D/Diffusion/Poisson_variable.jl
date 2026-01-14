using Penguin
using IterativeSolvers
using LinearAlgebra, SparseArrays

### 1D Test Case : Monophasic Steady Diffusion with Variable Coefficient
# Define the mesh
nx = 50
lx = 1.0
x0 = 0.0
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
center = 0.5
radius = 0.1
body = (x, _=0) -> sqrt((x-center)^2) - radius

# Define the capacity
capacity = Capacity(body, mesh)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc1, :bottom => bc1))

# Define the source term and diffusion coefficient
g = (x, y, _=0) -> x
k = (x, y, _=0) -> 1.0 + 0.5 * sin(2.0 * pi * x)

Fluide = Phase(capacity, operator, g, k)

# Define the solver
solver = DiffusionSteadyMonoVariable(Fluide, bc_b, bc)

# Solve the problem
solve_DiffusionSteadyMono!(solver; method=Base.:\)

# Plot the solution
plot_solution(solver, mesh, body, capacity)
