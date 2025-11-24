# Available Solvers

Penguin.jl provides a suite of solvers for steady and unsteady diffusion, advection–diffusion, and Darcy flow in both monophasic and multiphasic (diphasic) contexts. The table below summarizes each solver, its exported functions, and the governing PDEs it addresses.

| Solver Type                                | PDE Form (Representative)                                                                                                                                                  | Solver Constructors / Methods                          |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| **Diffusion (Steady, Monophasic)**        | Steady: ∇·(D∇u) = f                                                                                                                                                       | `DiffusionSteadyMono`, `solve_DiffusionSteadyMono!`   |
| **Diffusion (Steady, Diphasic)**          | Steady with phase interface: ∇·(D₁∇u₁) = f₁ in Ω₁, ∇·(D₂∇u₂) = f₂ in Ω₂, plus interface conditions                                                                        | `DiffusionSteadyDiph`, `solve_DiffusionSteadyDiph!`   |
| **Diffusion (Unsteady, Monophasic)**      | ∂u/∂t = ∇·(D∇u) + f                                                                                                                                                       | `DiffusionUnsteadyMono`, `solve_DiffusionUnsteadyMono!` |
| **Diffusion (Unsteady, Diphasic)**        | ∂uᵢ/∂t = ∇·(Dᵢ∇uᵢ) + fᵢ in each phase domain, with jump conditions for flux and scalar across the interface                                                                | `DiffusionUnsteadyDiph`, `solve_DiffusionUnsteadyDiph!` |
| **Advection–Diffusion (Steady, Mono)**    | Steady: v·∇u - ∇·(D∇u) = f                                                                                                                                                 | `AdvectionDiffusionSteadyMono`, `solve_AdvectionDiffusionSteadyMono!` |
| **Advection–Diffusion (Steady, Diph)**    | Steady with two phases: vᵢ·∇uᵢ - ∇·(Dᵢ∇uᵢ) = fᵢ, plus interface conditions                                                                                                 | `AdvectionDiffusionSteadyDiph`, `solve_AdvectionDiffusionSteadyDiph!` |
| **Advection–Diffusion (Unsteady, Mono)**  | ∂u/∂t + v·∇u - ∇·(D∇u) = f                                                                                                                                                 | `AdvectionDiffusionUnsteadyMono`, `solve_AdvectionDiffusionUnsteadyMono!` |
| **Advection–Diffusion (Unsteady, Diph)**  | ∂uᵢ/∂t + vᵢ·∇uᵢ - ∇·(Dᵢ∇uᵢ) = fᵢ in each phase, with jump & flux interface conditions                                                                                     | `AdvectionDiffusionUnsteadyDiph`, `solve_AdvectionDiffusionUnsteadyDiph!` |
| **Darcy Flow (Steady)**                   | -∇·(K∇p) = f;  u = -K∇p  (Darcy’s law)                                                                                                                                      | `DarcyFlow`, `solve_DarcyFlow!`, `solve_darcy_velocity` |
| **Darcy Flow (Unsteady)**                 | ∂(φp)/∂t = ∇·(K∇p) + f, for porosity φ                                                                                                                                     | `DarcyFlowUnsteady`, `solve_DarcyFlowUnsteady!`       |
| **Stokes (Steady, Monophasic)**           | -∇·(2μD(u)) + ∇p = f,  ∇·u = 0  (pure Stokes, no v·∇u convective term)   | `StokesMono`, `solve_StokesMono!` |
| **Stokes (Unsteady, Monophasic)**         | ρ ∂u/∂t - ∇·(2μD(u)) + ∇p = f,  ∇·u = 0  (pure Stokes, no v·∇u convective term) | `StokesMono`, `solve_StokesMono_unsteady!` |
| **Moving Diffusion (Unsteady, Mono)**     | ∂u/∂t = ∇·(D∇u) in a domain moving in time, e.g., boundary moves with a prescribed velocity                                                                                | `MovingDiffusionUnsteadyMono`, `solve_MovingDiffusionUnsteadyMono!` |
| **Moving Diffusion (Unsteady, Diph)**     | Same as above with two phases, each having its own domain/capacity that evolves in time                                                                                    | `MovingDiffusionUnsteadyDiph`, `solve_MovingDiffusionUnsteadyDiph!` |
| **Moving Liquid Diffusion (Unsteady, Mono)** | ∂u/∂t = ∂/∂x(D∂u/∂x)+f in a moving domain where the moving interface is computed via a Stefan condition (non-prescribed motion)  | `MovingLiquidDiffusionUnsteadyMono`, `solve_MovingLiquidDiffusionUnsteadyMono!` |
| **Moving Liquid Diffusion (Unsteady, Diph)** | Same as above with two phases, each having its own domain/capacity that evolves in time                                                                                    | `MovingLiquidDiffusionUnsteadyDiph`, `solve_MovingLiquidDiffusionUnsteadyDiph!` |
---

## Example Usage

Below is a typical workflow using one of the solvers, such as the unsteady monophasic diffusion solver:

```julia
using Penguin
using IterativeSolvers
using WriteVTK
using CairoMakie

# 1) Define Mesh
nx, ny = 80, 80
lx, ly = 8.0, 8.0
x0, y0 = 0.0, 0.0
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# 2) Define the Interface (Circle)
radius, center = ly/4, (lx/2, ly/2)
circle  = (x, y, _=0) ->  sqrt((x-center[1])^2 + (y-center[2])^2) - radius
circle_c = (x, y, _=0) -> -circle(x, y)

# 3) Create Capacities & Operators
capacity  = Capacity(circle,  mesh)
capacity_c = Capacity(circle_c, mesh)
operator   = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# 4) Boundary & Interface Conditions
bc    = Robin(1.0,1.0,1.0)
bc_b  = BorderConditions(Dict(:left => bc, :right => bc, :top => bc, :bottom => bc))
ic    = InterfaceConditions(ScalarJump(1.0, 2.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# 5) Define Phases
f = (x, y, z, t)->0.0
Fluide_1 = Phase(capacity, operator, f, (x, y, z)->1.0)
Fluide_2 = Phase(capacity_c, operator_c, f, (x, y, z)->1.0)

# 6) Set Initial Condition
u0ₒ1 = ones((nx+1)*(ny+1))
u0ᵧ1 = ones((nx+1)*(ny+1))
u0ₒ2 = zeros((nx+1)*(ny+1))
u0ᵧ2 = ones((nx+1)*(ny+1))
u0   = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# 7) Time Setup & Solver Creation
Δt   = 0.01
Tend = 1.0
solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "BE")

# 8) Solve
solve_DiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, Δt, Tend, bc_b, ic, "CN"; method=Base.:\)

# 9) Visualization
plot_solution(solver, mesh, circle, capacity, state_i=101)
```

1. **Mesh**: A uniform mesh is created in 2D.  
2. **Boundary Conditions**: A Robin boundary  
3. **Solver**: Selected from the table above. Choose time scheme (“CN” for Crank–Nicolson, for example).  
4. **Postprocessing**: Visualize results, export VTK, or analyze data (e.g., compute fluxes).

---

## Additional Notes

- Each solver has variants for single-phase (Mono) or two-phase (Diph) problems.  
- Steady solvers do not require a time-step parameter (Δt).  
- For moving geometry, a [SpaceTimeMesh] and updated [Capacity] are used each time step.  
- The PDE forms indicated above are representative; actual solver usage can be further customized by specifying different boundary conditions, interface conditions, or source terms.