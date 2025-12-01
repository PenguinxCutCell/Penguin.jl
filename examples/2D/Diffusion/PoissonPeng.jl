using Penguin
using LinearSolve

# Poisson equation inside a disk
# Define the mesh
nx, ny = 320, 320
lx, ly = 4., 4.
x0, y0 = -2., -2.
mesh = Mesh((nx, ny), (lx, ly), (x0, y0))

# Simple 2D penguin SDF: φ(x,y) < 0 inside the penguin, > 0 outside.
# Coordinates: penguin standing on "ground" near y = 0, centered at x = 0.

# --- basic SDF primitives -----------------------------------------------------

@inline sdCircle(x, y, cx, cy, R) =
    hypot(x - cx, y - cy) - R

@inline function sdCapsule(x, y, x1, y1, x2, y2, r)
    # distance to line segment [A,B] thickened by radius r
    px, py = x - x1, y - y1
    bx, by = x2 - x1, y2 - y1
    h = clamp((px*bx + py*by) / (bx*bx + by*by), 0.0, 1.0)
    dx = px - bx*h
    dy = py - by*h
    return hypot(dx, dy) - r
end

@inline sdf_union(a, b) = min(a, b)

# --- penguin SDF --------------------------------------------------------------

function penguin(x::Real, y::Real)
    x = float(x)
    y = float(y)

    # Body: union of a few circles (back, belly, head, tail, feet)
    body_back  = sdCircle(x, y, -0.15, 0.35, 0.60)  # big back
    body_front = sdCircle(x, y,  0.25, 0.25, 0.50)  # belly/front
    head       = sdCircle(x, y,  0.30, 0.87, 0.22)  # head
    tail       = sdCircle(x, y, -0.65, 0.20, 0.18)  # tail bump
    feet       = sdCircle(x, y,  0.20, -0.30, 0.25) # feet lump

    body = sdf_union(body_back, body_front)
    body = sdf_union(body, head)
    body = sdf_union(body, tail)
    body = sdf_union(body, feet)

    # Flipper: capsule along the side
    flipper = sdCapsule(x, y,
                        0.05, 0.25,   # start point
                        0.65,-0.05,   # end point
                        0.07)         # radius

    body = sdf_union(body, flipper)

    # Beak: small capsule in front of head
    beak = sdCapsule(x, y,
                     0.48, 0.83,
                     0.78, 0.80,
                     0.04)

    φ = sdf_union(body, beak)

    return φ          # φ(x,y) < 0 inside penguin
end



LS(x,y,_=0) = penguin(x, y)

# Define the capacity
capacity = Capacity(LS, mesh, method="VOFI", integration_method=:vofijul, compute_centroids=false) # or capacity = Capacity(LS, mesh, method="ImplicitIntegration")

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
solve_DiffusionSteadyMono!(solver, algorithm=UMFPACKFactorization(), log=true)

# Plot the solution
plot_solution(solver, mesh, LS, capacity)

# Analytical solution
u_analytic(x,y) = 1.0 - (x-2)^2 - (y-2)^2

# Compute the error
u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytic, solver, capacity, 2, false)