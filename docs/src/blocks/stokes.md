# Stokes Solver Overview

Penguin provides prototype fully coupled incompressible Stokes solvers for mono- and diphasic problems on Cartesian grids with embedded boundaries:

- `StokesMono` – single fluid, velocity components on (optionally) staggered grids and pressure on a collocated grid.
- `StokesDiph` – two-fluid extension with distinct phase material properties and interface boundary conditions.

Both support steady assembly and an unsteady implicit $\theta$-scheme (`:BE` / `:CN`) for
\[
 \rho \frac{\partial u}{\partial t} - \nabla\cdot(2\mu D(u)) + \nabla p = f, \qquad \nabla\cdot u = 0.
\]
Currently the nonlinear advection term is omitted (pure Stokes). A pressure gauge DOF is fixed automatically.

## Quick Usage (Monophasic 2D)

```julia
using Penguin

# Grids (staggered velocity, collocated pressure)
mesh_p  = Mesh((nx, ny), (Lx, Ly), (x0, y0))
mesh_ux = Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

body = (x,y,_=0)->-1.0
cap_ux = Capacity(body, mesh_ux); cap_uy = Capacity(body, mesh_uy); cap_p = Capacity(body, mesh_p)
op_ux = DiffusionOps(cap_ux); op_uy = DiffusionOps(cap_uy); op_p = DiffusionOps(cap_p)

μ = 1.0; ρ = 1.0; fᵤ = (x,y,z=0)->0.0; f_p = (x,y,z=0)->0.0
fluid = Fluid((mesh_ux, mesh_uy), (cap_ux, cap_uy), (op_ux, op_uy), mesh_p, cap_p, op_p, μ, ρ, fᵤ, f_p)

bc_ux = BorderConditions(Dict(:left=>Dirichlet(0.0), :right=>Dirichlet(0.0), :bottom=>Dirichlet(0.0), :top=>Dirichlet(1.0)))
bc_uy = BorderConditions(Dict(:left=>Dirichlet(0.0), :right=>Dirichlet(0.0), :bottom=>Dirichlet(0.0), :top=>Dirichlet(0.0)))

pressure_gauge = PinPressureGauge()   # or MeanPressureGauge() for zero-mean pressure
bc_cut = Dirichlet(0.0)  # placeholder cut boundary (no internal interface)

solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut)
solve_StokesMono_unsteady!(solver; Δt=Δt, T_end=Tend, scheme=:CN)
```

## Diphasic Variant

For `StokesDiph`, construct two `Fluid` objects (A,B) with their own properties `(μ_A, ρ_A)` and `(μ_B, ρ_B)` over the same geometrical region but different signed-distance capacities. Provide interface conditions through `bc_cut` and call `StokesDiph(fluid_a, fluid_b, (bc_uxA, bc_uyA), (bc_uxB, bc_uyB), pressure_gauge, bc_cut)` then `solve_StokesDiph!` / unsteady variant.

## Limitations / Roadmap

- Preconditioning & iterative solvers WIP (current examples use direct solve or simple reduction).
- Pressure accuracy in manufactured solutions that assume NS pressure requires added body force or advection term.

See `benchmark/TaylorGreen.jl` for a detailed convergence test and the README section for further instructions.
