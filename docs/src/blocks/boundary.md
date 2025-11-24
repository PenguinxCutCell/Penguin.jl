# Boundary and Interface Conditions

This page explains how to set up boundary conditions in Penguin.jl for both single-phase and two-phase problems.  
Boundary conditions typically apply to the domain edges, while interface conditions apply to the boundary between two phases.

---

## One-Phase Boundary Conditions

In single-phase problems, boundaries often enforce Dirichlet, Neumann, or Robin conditions:

### Dirichlet

Enforces a value for the variable (e.g., temperature, concentration) at the boundary.  
- Constant or function-based definition.

```julia
bc = Dirichlet(1.0)                   # Constant Dirichlet
bc = Dirichlet(x -> sin(x))           # Function of space
bc = Dirichlet((x, t) -> sin(x)*cos(t)) # Function of space & time
```

### Neumann

Enforces a flux condition (e.g., derivative normal to the boundary).  
- Constant or function-based definition.

```julia
bc = Neumann(1.0)                     # Constant Neumann
bc = Neumann(x -> sin(x))             # Function of space
```

### Robin

Combines both Dirichlet- and Neumann-like components (e.g., α·T + β·∇T·n = g).  
- Constant or function-based definition.

```julia
bc = Robin(1.0, 1.0, 1.0)             # α = 1.0, β = 1.0, value = 1.0
bc = Robin(x -> sin(x), 1.0, 1.0)     # α depends on x, β = 1.0, value = 1.0
```

### Periodic

Specifies that the domain edges are identified with each other (no jump across boundary).  
```julia
bc = Periodic()
```

### Specifying borders

Use the `BorderConditions` struct to assign boundary types to each side of the domain.
```julia
border_dict = Dict(
    :left => Dirichlet(0.0),
    :right => Neumann(0.0),
    :top => Robin(1.0, 1.0, 0.0),
    :bottom => Periodic()
)
bcs = BorderConditions(border_dict)
```

---

## Two-Phase Interface Conditions

For two-phase (e.g., diphasic) problems, interface boundary conditions define how variables and fluxes behave across the interface. Penguin.jl provides two main types:

### ScalarJump

Specifies a jump in the variable itself across the interface (e.g., [[αT]] = constant).  
- Constant or function-based definition.

```julia
# α₁, α₂ define how each phase’s variable contributes to the jump
bc = ScalarJump(0.0, 1.0, 0.0)          # α₁=0.0, α₂=1.0, value=0.0
bc = ScalarJump(x -> sin(x), 1.0, 0.0)  # α₁ depends on x, α₂=1.0
```

### FluxJump

Specifies a jump in flux across the interface (e.g., [[β∇T.n]] = constant).  
- Constant or function-based definition.

```julia
# β₁, β₂ define how each phase’s flux contributes
bc = FluxJump(0.0, 1.0, 0.0)            # β₁=0.0, β₂=1.0
bc = FluxJump(x -> sin(x), 1.0, 0.0)    # β₁ depends on x, β₂=1.0
```

### Specifying interface conditions

Use the `InterfaceConditions` struct for two-phase scenarios:
```julia
interface_bc = InterfaceConditions(
    ScalarJump(0.0, 1.0, 0.0),
    FluxJump(1.0, 1.0, 0.0)
)
```
Here, `:scalar` and `:flux` jump conditions govern how the variable and its flux connect across the interface.

---

## Summary

- **Single-phase** domains typically use `Dirichlet`, `Neumann`, `Robin`, or `Periodic` boundary conditions, assigned via `BorderConditions`.  
- **Two-phase** problems add interface conditions (`ScalarJump` and `FluxJump`) controlling how variables or fluxes behave across phase boundaries.  
