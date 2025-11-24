# 2D Darcy Flow

This example solves a Darcy flow equation with a circular interface in a 4×4 domain, using fixed pressure conditions on two sides. The code creates a mesh, sets up boundary conditions, and solves for both pressure and velocity.

```julia
using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CairoMakie

# 1) Define the 2D Mesh
nx, ny = 80, 80
lx, ly = 4.0, 4.0
x0, y0 = 0.0, 0.0
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# 2) Define a Circular Interface
radius, center = ly/4, (lx/2, ly/2) .+ (0.01, 0.01)
circle = (x, y, _=0) -> -(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# 3) Capacity & Operator
capacity = Capacity(circle, mesh)
operator = DiffusionOps(capacity)

# 4) Boundary Conditions (Pressure on top/bottom, none on other sides)
bc_10 = Dirichlet(10.0)
bc_20 = Dirichlet(20.0)
bc_p  = BorderConditions(Dict(:top => bc_10, :bottom => bc_20))
ic    = Neumann(0.0)   # Interface condition

# 5) Define Source & Phase
f_zero = (x, y, z)->0.0
K      = (x, y, z)->1.0
Fluide = Phase(capacity, operator, f_zero, K)

# 6) Create Darcy Solver & Solve
solver = DarcyFlow(Fluide, bc_p, ic)
solve_DarcyFlow!(solver; method=Base.:\)

# 7) Compute Velocity Field
u = solve_darcy_velocity(solver, Fluide)

# 8) Plot Velocity (Optional)
xrange = range(x0, stop=lx, length=nx+1)
yrange = range(y0, stop=ly, length=ny+1)
ux, uy = u[1:div(end,2)], u[div(end,2)+1:end]
ux = reshape(ux, (nx+1, ny+1))
uy = reshape(uy, (nx+1, ny+1))
mag = sqrt.(ux.^2 .+ uy.^2)

fig = Figure(size=(800, 800))
ax  = Axis(fig[1, 1], backgroundcolor=:white, xlabel="x", ylabel="y")
ar  = arrows!(ax, xrange, yrange, ux, uy, arrowsize=10, arrowcolor=mag, linecolor=mag, lengthscale=0.02)
Colorbar(fig[1, 2], label="|u|", ar)
display(fig)
```

**Summary of Steps:**
1. Build a uniform 2D mesh.  
2. Define a circular interface and set up a capacity.  
3. Assign boundary conditions for fixed pressure on the top and bottom boundaries.  
4. Create a Darcy solver, solve for pressure, and compute the velocity field.  
5. (Optional) Visualize the velocity field using Makie.

![](assets/darcy/velocity_arrow.png)
