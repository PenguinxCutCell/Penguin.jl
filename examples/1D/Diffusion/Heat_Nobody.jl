using Penguin
using IterativeSolvers

### 2D Test Case : Monophasic Unsteady Diffusion Equation with a NoBody struct
# Define the mesh
nx = 40
lx = 4.
x0 = 0.
domain = ((x0,lx),)
mesh = Penguin.Mesh((nx, ), (lx, ), (x0, ))

# Define the body : NoBody
circle = (x, _=0) -> -1.0 

# Define the capacity
capacity = Capacity(circle, mesh)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions 
bc = Dirichlet(10.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc1, :bottom => bc))

# Define the source term
f = (x,y,z,t)->1.0
D = (x,y,z)->1.0

# Define the phase
Fluide = Phase(capacity, operator, f, D)

# Initial condition
u0ₒ = zeros((nx+1))
u0ᵧ = ones((nx+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
Δt = 0.01
Tend = 1.0
solver = DiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, "CN")

# Solve the problem
solve_DiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, bc, "CN"; method=Base.:\)

# Write the solution to a VTK file
#write_vtk("heat", mesh, solver)

# Plot the solution
#plot_solution(solver, mesh, circle, capacity; state_i=10)

# Animation
animate_solution(solver, mesh, circle)
