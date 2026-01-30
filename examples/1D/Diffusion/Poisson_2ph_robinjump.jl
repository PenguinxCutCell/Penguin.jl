using Penguin
using IterativeSolvers

### 1D Test Case : Diphasic Steady Diffusion Equation (Robin jump at interface)
# Define the mesh
nx = 40
lx = 1.0
x0 = 0.0
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body (interface at x = pos)
pos = 0.5
body = (x, _=0) -> x - pos
body_c = (x, _=0) -> pos - x

# Define the capacity
capacity = Capacity(body, mesh)
capacity_c = Capacity(body_c, mesh)

# Define the operators
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc_left = Dirichlet(1.0)
bc_right = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc_left, :bottom => bc_right))

# Interface conditions (Robin jump + flux jump)
k1,k2 = 1.0, 1.0
α = 1.0
β = 0.0
gγ = 0.0
bc_i = RobinJump(α, β, gγ)
bc_f = FluxJump(k1, k2, 0.0)

# Define the source term
f1 = (x, y, _=0) -> 0.0
f2 = (x, y, _=0) -> 0.0

# Diffusion coefficients
D1 = (x, y, _=0) -> k1
D2 = (x, y, _=0) -> k2

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, D1)
Fluide_2 = Phase(capacity_c, operator_c, f2, D2)

# Define the solver (Robin jump overload)
solver = DiffusionSteadyDiph(Fluide_1, Fluide_2, bc_b, bc_i, bc_f)

# Solve the problem
solve_DiffusionSteadyDiph!(solver; method=Base.:\)

# Plot the solution
plot_solution(solver, mesh, body, capacity)

