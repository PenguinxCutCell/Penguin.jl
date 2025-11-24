using Penguin
using IterativeSolvers, SpecialFunctions
using Roots


### 2D Test Case : Diphasic Unsteady Diffusion Equation inside a moving Disk
# Define the mesh
nx, ny = 40, 40
lx, ly = 4., 4.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
# Define parameters for the moving circle : Low frequency circular motion
ω = 0.3
circle_center(t) = [2.0 + 0.5*cos(ω*t), 2.0 + 0.5*sin(ω*t)]
circle_radius(t) = 1.0 + 0.2*sin(ω*t)

# Circle velocity and radius derivative
center_velocity(t) = [-0.5*ω*sin(ω*t), 0.5*ω*cos(ω*t)]
radius_derivative(t) = 0.2*ω*cos(ω*t)

# Level set function S(x,t) = ||x - c(t)||^2 - R(t)^2
# body > 0 inside the circle, < 0 outside
function S_func(x, y, t)
    c = circle_center(t)
    R = circle_radius(t)
    return (x - c[1])^2 + (y - c[2])^2 - R^2
end

# Level set functions: negative inside the circle (Ω⁻), positive outside (Ω⁺)
body(x, y, t) = S_func(x, y, t)
body_c(x, y, t) = -S_func(x, y, t)

# Define the Space-Time mesh
Δt = 0.01
Tend = 0.5
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)
capacity_c = Capacity(body_c, STmesh)

# Define the operators
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => Neumann(0.0), :bottom => Neumann(0.0)))
ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))


# Define the source term
f1 = (x,y,z,t)->0.0
f2 = (x,y,z,t)->0.0

K1= (x,y,z)->1.0
K2= (x,y,z)->1.0

# Define the phase
Fluide1 = Phase(capacity, operator, f1, K1)
Fluide2 = Phase(capacity_c, operator_c, f2, K2)

# Initial condition
u0ₒ1 = ones((nx+1)*(ny+1))
u0ᵧ1 = ones((nx+1)*(ny+1))
u0ₒ2 = zeros((nx+1)*(ny+1))
u0ᵧ2 = zeros((nx+1)*(ny+1))
u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Define the solver
solver = MovingDiffusionUnsteadyDiph(Fluide1, Fluide2, bc_b, ic, Δt, u0, mesh, "BE")

# Solve the problem
solve_MovingDiffusionUnsteadyDiph!(solver, Fluide1, Fluide2, body, body_c, Δt, Tend, bc_b, ic, mesh, "BE"; method=Base.:\)

# Animation
animate_solution(solver, mesh, body)
