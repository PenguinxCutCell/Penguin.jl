using Penguin
using IterativeSolvers

### 1D Test Case : Monophasic Unsteady Diffusion Equation 
# Define the mesh
nx = 80
lx = 10.0
x0 = 0.0
domain=((x0,lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
center, radius = 0.25, 1.0
body = (x, _=0) -> -(x - center)

# Define the capacity
capacity = Capacity(body, mesh)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc0 = Robin(1.0,1.0,0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(1.0), :bottom => Dirichlet(0.0)))

# Define the source term
f = (x,y,z,t)->0.0
D = (x,y,z)->5.0
# Define the phase
Fluide = Phase(capacity, operator, f, D)

# Initial condition
u0ₒ = ones(nx+1)
u0ᵧ = ones(nx+1)
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
Δt = 0.5 * (lx/nx)^2
Tend = 1.0
solver = DiffusionUnsteadyMono(Fluide, bc_b, bc0, Δt, u0, "CN")

# Solve the problem
solve_DiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, bc0, "CN"; method=Base.:\)

# Write the solution to a VTK file
#write_vtk("heat_1d", mesh, solver)

# Plot the solution
plot_solution(solver, mesh, body, capacity; state_i=10)

# Animation
#animate_solution(solver, mesh, body)

# Analytical solution
using SpecialFunctions

function analytical(x; t=Tend)
    a=5.0
    k=1.0
    x = x - center
    return 1.0*(erf(x/(2*sqrt(a*t))) + exp(k*x+a*k^2*t)*erfc(x/(2*sqrt(a*t)) + k*sqrt(a*t)))
end

u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(analytical, solver, capacity, 2)


x = range(x0, stop=lx, length=nx+1)
y = analytical.(x)

y[capacity.cell_types .!= 1] .= NaN

y_num = solver.states[end][1:nx+1]
y_num[capacity.cell_types .!= 1] .= NaN

using CairoMakie
fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)")
lines!(ax, x, y, color = :blue, label = "Analytical solution")
scatter!(ax, x, y_num, color = :red, label = "Numerical solution")
axislegend(ax, position = :rb)
display(fig)


# Plot analytical solution for different time
using CairoMakie

fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)")
for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y = analytical.(x, t=t)
    y[capacity.cell_types .!= 1] .= NaN
    lines!(ax, x, y, label = "t = $t")
end
axislegend(ax, position = :rb)
display(fig)
