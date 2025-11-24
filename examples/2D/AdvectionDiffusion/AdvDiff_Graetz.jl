using Penguin
using IterativeSolvers
using LinearAlgebra

### 2D Test Case : Monophasic Unsteady Advection-Diffusion Equation inside a Disk
# Define the mesh
nx, ny = 160, 160
lx, ly = 16., 16.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body : Planar
radius, center = ly/8, (lx/2, ly/2) .+ (0.01, 0.01)
circle = (x,y,_=0)->(y - center[2])

# Define the capacity
capacity = Capacity(circle, mesh)

# Initialize the velocity field with a Poiseuille flow
uₒx, uₒy = initialize_poiseuille_velocity_field(nx, ny, lx, ly, x0, y0)
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
f = (x,y,z,t)-> 0.0 #sin(x)*cos(10*y)

Fluide = Phase(capacity, operator, f, (x,y,_)->1.0)

# Initial condition
# Analytical solution T (x, t) = 10/(4D (t + 1/2)) exp(− ‖x−xc(t)‖^2/(4D(t+ 1/2)
T0ₒ = zeros((nx+1)*(ny+1))
T0ᵧ = zeros((nx+1)*(ny+1))

# Combine the temperature arrays if needed
T0 = vcat(T0ₒ, T0ᵧ)

# Define the solver
Δt = 0.01
Tend = 2.0
solver = AdvectionDiffusionUnsteadyMono(Fluide, bc_b, ic, Δt, T0, "CN")

# Solve the problem
solve_AdvectionDiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, ic,"CN"; method=Base.:\)

# Plot the solution
#plot_solution(solver, mesh, circle, capacity)

# Write the solution to a VTK file
write_vtk("advdiff", mesh, solver)

# Animation
#animate_solution(solver, mesh, circle)