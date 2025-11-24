# Phase Construction

The `Phase` struct in Penguin.jl represents a phase for numerical modeling. It ties together capacity (defining geometry and volume parameters), operator objects (for diffusion, convection, etc.), a source term, and a diffusion coefficient function. Below is a typical usage example in a 2D Poisson problem.

```julia
using Penguin

# 1) Define the mesh
nx, ny = 5, 5
lx, ly = 4., 4.
x0, y0 = 0., 0.
mesh = Mesh((nx, ny), (lx, ly), (x0, y0))

# 2) Define the body via signed distance function
LS(x, y, _=0) = sqrt(x^2 + y^2) - 0.5

# 3) Build capacity
capacity = Capacity(LS, mesh, method="VOFI")

# 4) Create operators
operators = DiffusionOps(capacity)

# 5) Set boundary conditions
bc_0 = Dirichlet(0.0)
bc_1 = Dirichlet(1.0)
bcs = BorderConditions(Dict(:left => bc_1, :right => bc_1, :top => bc_1, :bottom => bc_1))

# 6) Define source term and diffusion coefficient
f(x, y) = 0.0
D(x, y) = 1.0

# 7) Assemble Phase
phase = Phase(capacity, operators, f, D)
```

### Fields
- **capacity::AbstractCapacity**  
  Identifies the geometric and volumetric properties required for the phase.  
- **operator::AbstractOperators**  
  Contains discretized operators (e.g., diffusion, convection) used in PDE solves.  
- **source::Function**  
  Represents an external source or sink (e.g., heat source, chemical reaction).  
- **Diffusion_coeff::Function**  
  A function returning the diffusion coefficient, possibly space-/time-dependent.

By combining all these elements, `Phase` provides a convenient way to encapsulate the problem’s computational details for a specific phase or region in the domain.