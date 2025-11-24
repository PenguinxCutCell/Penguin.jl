# 1D Diffusion - Prescribed Motion

Below is an example illustrating how to solve a 1D diffusion problem on a moving domain using a space–time mesh. The domain’s motion is prescribed by a two‑argument signed distance function, `body(x, t)`. In this case, the boundary moves over time, and a new `SpaceTimeMesh` is rebuilt at each time step to reflect this motion.

---

```julia
using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CairoMakie

# 1) Define the spatial mesh
nx = 40
lx = 1.0
x0 = 0.0
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# 2) Define the moving interface as a two‑argument signed distance function
#    body(x,t) ~ (x - xf - c*√t). 
#    Negative inside, zero on the interface, positive outside.
xf = 0.01 * lx
c  = 1.0
body = (x, t, _=0) -> (x - xf - c * sqrt(t))

# 3) Build a space–time mesh
Δt   = 0.01
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt])

# 4) Create Capacity & Diffusion Operations
capacity = Capacity(body, STmesh)
operator = DiffusionOps(capacity)

# 5) Boundary & Initial Conditions
bc_b = BorderConditions(Dict(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
bc   = Dirichlet(0.0)

u0_bulk = zeros(nx + 1)
u0_bc   = zeros(nx + 1)
u0      = vcat(u0_bulk, u0_bc)

# 6) Define source & diffusion coefficient
f = (x,y,z,t)-> 0.0
K = (x,y,z)-> 1.0
Fluide = Phase(capacity, operator, f, K)

# 7) Setup and Solve
solver = MovingDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")
solve_MovingDiffusionUnsteadyMono!(solver, Fluide, body, Δt, Tend, bc_b, bc, mesh, "BE"; method=Base.:\)

# 8) Plot or Animate Results
plot_solution(solver, mesh, body, capacity; state_i=1)
animate_solution(solver, mesh, body)
```

**Key Points:**
1. The function `body(x, t)` defines the interface location over time.  
2. A space–time mesh is instantiated at each time step to reflect boundary motion.  
3. The solution vector is split into bulk values and boundary values.  
4. Both boundary conditions and diffusion operators are updated as the body moves, ensuring a consistent time evolution.

![](assets/solidmoving1D/comp_analytical_stef_1d.png)
