# Penguin

[![Build Status](https://github.com/PenguinxCutCell/Penguin.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PenguinxCutCell/Penguin.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/PenguinxCutCell/Penguin.jl/graph/badge.svg?token=YQUDHCTHI7)](https://codecov.io/gh/PenguinxCutCell/Penguin.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://penguinxcutcell.github.io/Penguin.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://penguinxcutcell.github.io/Penguin.jl/dev)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

Cut-cell finite-volume solvers for heat and mass transfer with moving interfaces, written in Julia.

Penguin implements cut-cell/embedded-boundary discretizations, and a set of steady/unsteady solvers for diffusion, advection–diffusion, Darcy flow, Navier-Stokes and Stefan-type phase change. It provides plotting/animation helpers.

## Features

- Scalar diffusion and advection–diffusion (steady/unsteady)
- Diphasic/monophasic problems with interface capacities and jumps
- Darcy flow (steady and unsteady) / Stokes flow (fully coupled, unsteady θ-scheme) / Navier-Stokes (fully coupled, unsteady θ-scheme)
- Solid- or liquid-moving configurations; prescribed or free interface motion
- Boundary conditions: Dirichlet, Neumann, Robin, Periodic
- Front utilities: signed distance functions, lagrangian front tracking
- Visualization (CairoMakie)

See `src/Penguin.jl` for the main API exports and `examples/` for runnable scripts organized by dimension and physics.

## Installation

Requires Julia ≥ 1.0.

```julia
] add https://github.com/PenguinxCutCell/Penguin.jl
```

Optionally, instantiate this repo to get exact dependencies used for the examples:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick Start (2D heat diffusion)

This minimal example mirrors `examples/2D/Diffusion/Heat.jl` and solves a monophasic unsteady diffusion problem in a square with a circular embedded boundary.

```julia
using Penguin

# Mesh and domain
nx, ny = 80, 80
lx, ly = 4.0, 4.0
x0, y0 = 0.0, 0.0
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Body (level set of a circle)
radius, center = ly/4, (lx/2, ly/2)
circle = (x,y,_=0)->(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# Capacities and operators
capacity = Capacity(circle, mesh)
operator = DiffusionOps(capacity)

# Boundaries and phase
bc_interior = Dirichlet((x,y,z,t)->sin(π*x) * sin(π*y))
bc_zero = Dirichlet(0.0)
bc_border = BorderConditions(Dict(:left=>bc_zero, :right=>bc_zero, :top=>bc_zero, :bottom=>bc_zero))
f = (x,y,z,t)->0.0
D = (x,y,z)->1.0
fluid = Phase(capacity, operator, f, D)

# Initial condition and time step
u0 = vcat(zeros((nx+1)*(ny+1)), ones((nx+1)*(ny+1)))
Δt = 0.25 * (lx/nx)^2
Tend = 0.01

# Solver and run
solver = DiffusionUnsteadyMono(fluid, bc_border, bc_interior, Δt, u0, "BE")
solve_DiffusionUnsteadyMono!(solver, fluid, Δt, Tend, bc_border, bc_interior, "BE")

# Plot or export
plot_solution(solver, mesh, circle, capacity; state_i=1)
# write_vtk("heat", mesh, solver)
```

## Examples

- 1D: `examples/1D/{Diffusion, AdvectionDiffusion, LiquidMoving, SolidMoving, Concentration, BinaryMelting, NavierStokes}`
- 2D: `examples/2D/{Diffusion, AdvectionDiffusion, Darcy, LiquidMoving, SolidMoving, StefanFT}`
- 3D: `examples/3D/{Diffusion, AdvectionDiffusion, Darcy, SolidMoving}`

Each folder contains self-contained scripts demonstrating setup, solve, and visualization.

## Status and Roadmap

 Implemented
- Diffusion and advection–diffusion (steady/unsteady)
- Darcy flow (steady/unsteady)
- Diphasic capacities and interface jump conditions
- Prescribed interface motion
- 1D/2D non‑prescribed interface motion (front tracking)
- Fully coupled Stokes/ Navier-Stokes flow solver for monophasic/diphasic problems 
- Streamfunction–vorticity formulation (2D)

In development
- Phase change with Navier-Stokes coupling
- Preconditioning and Domain Decomposition
- Multi-species transport

## Citing

If this package contributes to your work, please cite the repository. A formal citation entry will be added alongside future publications.

## License

This project is licensed under the terms of the LICENSE file.
