using Penguin
using IterativeSolvers
using LinearAlgebra, SparseArrays

### 1D Test Case : Monophasic Steady Diffusion Equation
# Define the mesh
nx = 10
lx = 1.
x0 = 0.
Δx = lx / nx
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
center = 0.5
radius = 0.1
body = (x, _=0) -> -1.0

# Define the capacity
capacity = Capacity(body, mesh; compute_centroids=false)

# Define the operators
operator = DiffusionOps(capacity)

# Redefine W and V : Rebuild the operator
#operator = DiffusionOps(capacity)

#volume_redefinition!(capacity, operator)
#operator = DiffusionOps(capacity)
# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(1.0), :bottom => Dirichlet(0.0)))

# Define the source term
g = (x, y, _=0) -> 0.0
a = (x, y, _=0) -> 1.0

Fluide = Phase(capacity, operator, g, a)

# Define the solver
solver = DiffusionSteadyMono(Fluide, bc_b, bc)

# Solve the problem
solve_DiffusionSteadyMono!(solver; method=Base.:\)

# Plot the solution
#plot_solution(solver, mesh, body, capacity)

# Analytical solution
a, b = center - radius, center + radius
u_analytical = (x) -> (x-lx/nx)/(lx - lx/nx)

u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytical, solver, capacity, 2)

# Plot the numerical and analytical solutions
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="u",)
scatter!(ax, mesh.centers[1], u_num[1:end-1], label="Numerical Solution")
lines!(ax, mesh.centers[1], u_ana[1:end-1], label="Analytical Solution (from check_convergence)")
lines!(ax, mesh.centers[1], u_analytical.(range(x0, lx, length=nx)), linestyle=:dash, color=:red, label="Analytical Solution (function)")
Legend(fig[1, 2], ax; orientation = :vertical)
display(fig)  

# Plot the error
err = u_ana - u_num
err[capacity.cell_types .== 0] .= NaN
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="Log10 of the error", title="Monophasic Steady Diffusion Equation")
scatter!(ax, log10.(abs.(err)), label="Log10 of the error")
display(fig)