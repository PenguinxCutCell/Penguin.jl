using Penguin
using IterativeSolvers
using LinearAlgebra, SparseArrays

### 1D Test Case : Monophasic Steady Diffusion Equation
# Define the mesh
nx = 320
lx = 1.
x0 = 0.
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
center = 0.5
radius = 0.1
body = (x, _=0) -> sqrt((x-center)^2) - radius

# Define the capacity
capacity = Capacity(body, mesh)

# Define the operators
operator = DiffusionOps(capacity)

# Redefine W and V : Rebuild the operator
operator = DiffusionOps(capacity)

#volume_redefinition!(capacity, operator)
operator = DiffusionOps(capacity)
# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc1, :bottom => bc1))

# Define the source term
g = (x, y, _=0) -> x
a = (x, y, _=0) -> 1.0

Fluide = Phase(capacity, operator, g, a)

# Define the solver
solver = DiffusionSteadyMono(Fluide, bc_b, bc)

# Solve the problem
solve_DiffusionSteadyMono!(solver; method=Base.:\)

# Plot the solution
plot_solution(solver, mesh, body, capacity)

# Analytical solution
a, b = center - radius, center + radius
u_analytical = (x) -> - (x-center)^3/6 - (center*(x-center)^2)/2 + radius^2/6 * (x-center) + center*radius^2/2

u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytical, solver, capacity, 2)

# Plot the error
err = u_ana - u_num
err[capacity.cell_types .== 0] .= NaN
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="Log10 of the error", title="Monophasic Steady Diffusion Equation")
scatter!(ax, log10.(abs.(err)), label="Log10 of the error")
display(fig)