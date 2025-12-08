# Moving Stokes 1D/3D Implementation

This directory contains examples of the Moving Stokes solver for various dimensions.

## Overview

The Moving Stokes solver handles unsteady Stokes flow with prescribed moving geometries. The solver has been implemented for 1D, 2D, and 3D problems.

## Examples

### 1D Example
- **File**: `oscillating_interface.jl`
- **Description**: An oscillating interface in 1D that moves sinusoidally
- **Features**:
  - Oscillating boundary with prescribed velocity
  - Backward Euler time integration
  - Visualization of velocity field over time
  - Animation generation

### 2D Example (Existing)
Located in `examples/2D/SolidMovingStokes/`
- **Files**: `oscillating_circle.jl`, `MovingStokesOscillatingCylinder.jl`
- **Description**: Oscillating circle/cylinder in 2D flow
- **Features**: Full 2D Stokes flow around moving body

### 3D Example
- **File**: `oscillating_sphere.jl`
- **Description**: An oscillating sphere in 3D that moves vertically
- **Features**:
  - Oscillating sphere with prescribed velocity
  - Backward Euler time integration
  - Visualization of velocity field slices
  - **Note**: This is computationally expensive; use small grid sizes for testing

## Implementation Details

The implementation extends the existing 2D Moving Stokes solver to support 1D and 3D:

### Source Code
- **File**: `src/prescribedmotionsolver/stokes.jl`
- **Added Functions**:
  - `stokes1D_moving_blocks`: Build operator blocks for 1D
  - `stokes3D_moving_blocks`: Build operator blocks for 3D
  - `assemble_stokes1D_moving!`: Assemble 1D system matrix
  - `assemble_stokes3D_moving!`: Assemble 3D system matrix
  - `solve_MovingStokesUnsteadyMono!` for 1D and 3D: Solve functions with proper dispatch

### Key Features
- Uses SpaceTimeMesh for handling moving boundaries
- Supports Backward Euler (BE) and Crank-Nicolson (CN) time integration schemes
- Handles cut-cell boundary conditions
- Staggered grid for velocity components

## Usage

```julia
using Penguin

# Define moving body (example for 1D)
body = (x, t) -> x_interface(t) - x

# Create solver
solver = MovingStokesUnsteadyMono(fluid, bc_u, pressure_gauge, bc_cut; scheme=:BE)

# Solve
times, states = solve_MovingStokesUnsteadyMono!(solver, body, mesh,
                                                 Δt, T_start, T_end,
                                                 bc_u, bc_cut;
                                                 scheme=:BE,
                                                 geometry_method="VOFI")
```

## Notes

- The 2D solver was not modified and works as before
- The 1D solver is suitable for quick testing and verification
- The 3D solver is computationally intensive and should be run with small grid sizes
- All solvers support the same interface and time integration schemes
