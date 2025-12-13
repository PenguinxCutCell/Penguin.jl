# Oscillating Thermal Circle Example

This example demonstrates the **MovingStokesScalarCoupler** solver, which couples:
- Moving Stokes equations (low Reynolds number fluid flow)
- Scalar transport equation (heat/temperature diffusion and advection)

## Problem Description

A hot circular object oscillates vertically in a quiescent fluid:
- The circle has radius R = 0.4 and oscillates with amplitude A = 0.4
- The circle surface is maintained at a constant temperature T_body = 1.0
- The far-field fluid is at ambient temperature T_ambient = 0.0
- Heat diffuses from the circle into the fluid with thermal diffusivity κ = 0.1
- The temperature field is advected by the Stokes velocity field

### Physics

1. **Moving Stokes Flow**: The oscillating circle creates a flow field in the surrounding fluid
2. **Heat Transfer**: The hot circle releases heat into the fluid
3. **Advection-Diffusion**: The temperature is transported by the fluid velocity and diffuses

### Coupling Strategy

This example uses **passive coupling**:
- The Stokes equations are solved first to get the velocity field
- The temperature is then advected by this velocity field
- The temperature does not feedback to the momentum equations (buoyancy β = 0)

For buoyancy-driven flows, set β > 0 to enable thermal feedback.

## Files

- `oscillating_thermal_circle.jl`: Main simulation script

## Running the Example

```julia
using Penguin
include("oscillating_thermal_circle.jl")
```

This will:
1. Run a coupled Stokes-heat simulation
2. Generate visualization `oscillating_thermal_circle.png` showing velocity and temperature
3. Create an animation `oscillating_thermal_circle.gif` of the evolution
4. Print diagnostics to the console

## Output

- **oscillating_thermal_circle.png**: Side-by-side plots of velocity and temperature at different times
- **oscillating_thermal_circle.gif**: Animation showing the evolution of velocity and temperature fields

## Parameters

You can modify the following parameters in the script:

- `nx, ny`: Grid resolution (default: 24×24)
- `radius`: Circle radius (default: 0.4)
- `A_osc`: Oscillation amplitude (default: 0.4)
- `ω_osc`: Oscillation frequency (default: 2π, one period per unit time)
- `κ`: Thermal diffusivity (default: 0.1)
- `T_body`: Circle surface temperature (default: 1.0)
- `β`: Thermal expansion coefficient for buoyancy (default: 0.0)
- `Δt`: Time step (default: 0.025)
- `T_end`: Simulation end time (default: 0.2)

## Implementation Details

The **MovingStokesScalarCoupler** solver is implemented in:
`src/prescribedmotionsolver/stokes_scalar_coupling.jl`

It combines:
- `MovingStokesUnsteadyMono`: For the moving Stokes problem with prescribed geometry motion
- Scalar transport equations: For advection-diffusion of temperature
- Time integration using SpaceTimeMesh for moving boundaries

The solver handles:
- Cut-cell methods for the moving geometry
- Staggered grid for velocity components
- Time integration with Backward Euler (BE) or Crank-Nicolson (CN) schemes
- Optional buoyancy coupling
