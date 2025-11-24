# Building Operators

Penguin.jl provides various operators for discretizing and solving PDE-like problems over a mesh. This includes diffusion operators, convection operators, and utility functions for gradients (∇) and divergences (∇₋).

## DiffusionOps

`DiffusionOps` constructs matrices based on a given `Capacity`. These matrices encode diffusion-like terms used for solving PDEs.

### Example

```julia
using Penguin

nx, ny = 50, 50
lx, ly = 4., 4.
x0, y0 = 0., 0.
mesh = Mesh((nx, ny), (lx, ly), (x0, y0))

Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
LS(x, y, _=0) = sqrt(x^2 + y^2) - 0.5

capacity = Capacity(LS, mesh, method="VOFI")
operators = DiffusionOps(capacity)

# Compute the gradient of a scalar field
grad = ∇(operators, ones(2*nx*ny))
@test grad == zeros(2*nx*ny)  # Example test checking a uniform field

# Compute the divergence of a vector field
div = ∇₋(operators, ones(2*nx*ny), ones(2*nx*ny))
```

- The function ∇(operators, p) computes a discrete gradient of the scalar field p.
- The function ∇₋(operators, qω, qγ) computes a discrete divergence of vector fields qω and qγ.
- These operations rely on the sparse matrices stored in the `DiffusionOps` struct.

## ConvectionOps

`ConvectionOps` generates additional operators needed for convection terms, given a `Capacity` and velocities. Its fields handle computations of fluxes and boundary terms for convective processes.

### Example

```julia
using Penguin

nx = 160
lx = 4.
x0 = 0.
mesh = Mesh((nx,), (lx,), (x0,))
Φ(X) = sqrt(X[1]^2) - 0.5
LS(x, _=0) = sqrt(x^2) - 0.5

capacity = Capacity(LS, mesh, method="VOFI")

# Bulk velocity and interface velocity
uₒ = (ones(nx),)
uᵧ = zeros(nx)

operators = ConvectionOps(capacity, uₒ, uᵧ)

# Example check of an operator product
@test size(operators.G' * operators.Wꜝ * operators.G) == (nx, nx)
@test operators.size == (nx,)
```

- `Wꜝ`, `G`, and other fields are built from the capacity and user-defined velocities.  
- Each dimension in `ConvectionOps` has a corresponding set of sparse matrices for convection computations.

## Summary

1. **DiffusionOps**: Manages diffusion-like matrices (G, H, Wꜝ, etc.).  
2. **ConvectionOps**: Manages convection-like matrices (C, K, G, H, etc.).  
3. **∇** and **∇₋**: Provide discrete gradient and divergence operators, respectively.  

These components form the foundation for assembling PDE systems in Penguin.jl.
Future Operators are in construction ...