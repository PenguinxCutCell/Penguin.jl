using Penguin
using IterativeSolvers
using LinearAlgebra

### 2D Test Case : Monophasic Unsteady Advection-Diffusion Equation inside a Disk
# Define the mesh
nx, ny = 80, 80
lx, ly = 16., 16.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/1, (lx/2, ly/2) .+ (0.01, 0.01)
circle = (x,y,_=0)->(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

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
ic = Dirichlet(0.0)
bc = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc))

# Define the source term
f = (x, y, z, t) -> begin
    r = sqrt((x - lx/2)^2 + (y - lx/4)^2)
    if r <= 0.4
        return 1.0
    else
        return 0.0
    end
end


Fluide = Phase(capacity, operator, f, (x,y,_)->0.0)

# Initial condition
# Analytical solution T (x, t) = 10/(4D (t + 1/2)) exp(− ‖x−xc(t)‖^2/(4D(t+ 1/2)
T0ₒ = zeros((nx+1)*(ny+1))
T0ᵧ = zeros((nx+1)*(ny+1))
x, y = mesh.centers[1], mesh.centers[2]

#initialize_temperature_circle!(T0ₒ, T0ᵧ, x, y, center, lx/30, 1.0, nx, ny)

# Combine the temperature arrays if needed
T0 = vcat(T0ₒ, T0ᵧ)

# Define the solver
Δt = 0.01
Tend = 1.0
solver = AdvectionDiffusionUnsteadyMono(Fluide, bc_b, ic, Δt, T0, "CN")

# Solve the problem
solve_AdvectionDiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, ic, "CN"; method=Base.:\)
#solve_AdvectionDiffusionUnsteadyMono!(solver, Fluide, T0, Δt, Tend, bc_b, ic; method=IterativeSolvers.bicgstabl, verbose=false)

# Plot the solution
#plot_solution(solver, mesh, circle, capacity)

# Write the solution to a VTK file
write_vtk("advdiff", mesh, solver)

#plot_profile(solver, mesh; x=lx/2.01)

# Animation
#animate_solution(solver, mesh, circle)