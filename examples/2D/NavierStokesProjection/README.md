# Navier-Stokes Projection Method Examples

This directory contains examples demonstrating the three projection methods for solving 2D incompressible Navier-Stokes equations implemented in Penguin.jl.

## Projection Methods

### 1. Chorin/Temam (Original, Non-Incremental)

The original projection method by Chorin and Temam:

**Algorithm:**
1. Compute intermediate velocity u* without pressure:
   ```
   (u* - u^n)/Δt = ν∇²u* + explicit_convection + f
   ```

2. Solve Poisson equation for pressure:
   ```
   ∇²p^{n+1} = (1/Δt)∇·u*
   ```

3. Correct velocity:
   ```
   u^{n+1} = u* - Δt∇p^{n+1}
   ```

**Characteristics:**
- First-order accurate in time for pressure
- Simple to implement
- May have pressure boundary layer issues

### 2. Incremental Pressure-Correction (Godunov / Van Kan)

Uses pressure splitting: p^{n+1} = p^n + ϕ^{n+1}

**Algorithm:**
1. Momentum step with old pressure:
   ```
   (u* - u^n)/Δt = ν∇²u* - ∇p^n + explicit_convection + f
   ```

2. Solve Poisson for pressure increment:
   ```
   ∇²ϕ^{n+1} = (1/Δt)∇·u*
   ```

3. Correct velocity and update pressure:
   ```
   u^{n+1} = u* - Δt∇ϕ^{n+1}
   p^{n+1} = p^n + ϕ^{n+1}
   ```

**Characteristics:**
- Second-order accurate in time
- Better pressure accuracy than Chorin/Temam
- Still has some numerical boundary layer effects

### 3. Rotational Incremental Pressure-Correction ⭐

The most accurate projection method with rotational correction:

**Algorithm:**
1. Momentum step with old pressure (same as incremental)

2. Solve Poisson for pressure increment (same as incremental)

3. Rotational correction:
   ```
   ∇p^{n+1} = ∇(p^n + ϕ^{n+1}) - ν∇(∇·u*)
   u^{n+1} = u* - Δt∇p^{n+1}
   ```

**Characteristics:**
- Most accurate method
- Eliminates numerical boundary layers
- Recommended for high-fidelity simulations

## Convection Treatment

All methods use **Adams-Bashforth 2** for explicit convection:
- First step: Forward Euler
- Subsequent steps: AB2 = (3/2)*conv^n - (1/2)*conv^{n-1}

This provides second-order accuracy for the explicit nonlinear terms.

## Running the Examples

### Comparison Example

```julia
julia> include("projection_methods_comparison.jl")
```

This example:
- Sets up a lid-driven cavity problem
- Solves with all three projection methods
- Compares accuracy and performance
- Generates visualization comparing pressure fields

## Implementation Details

The projection methods are implemented in `/home/runner/work/Penguin.jl/Penguin.jl/src/solver/navierstokesproj.jl` with:

- Enum `ProjectionMethod` with values: `ChorinTemam`, `IncrementalPC`, `RotationalPC`
- Struct `NavierStokesProj2D` containing solver state
- Functions for each projection step (intermediate velocity, Poisson solve, correction)

Key features:
- Uses existing Penguin.jl infrastructure for operators and boundary conditions
- Integrates with the skew-symmetric convection operators from NavierStokesMono
- Supports all standard boundary conditions (Dirichlet, Neumann, Periodic, etc.)
- Pressure gauge constraints for well-posedness

## References

1. Chorin, A. J. (1968). "Numerical solution of the Navier-Stokes equations"
2. Temam, R. (1969). "Sur l'approximation de la solution des équations de Navier-Stokes"
3. Van Kan, J. (1986). "A second-order accurate pressure-correction scheme"
4. Guermond, J. L., Minev, P., & Shen, J. (2006). "An overview of projection methods"
