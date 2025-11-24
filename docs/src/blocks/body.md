# Defining a Body

A body can be represented with a signed distance function (SDF). Penguin.jl supports non-vectorized SDFs. 

## Example

```julia
using Penguin

# 1) Non-vectorized version
LS(x, y, _=0) = sqrt(x^2 + y^2) - 0.5 

# Create a mesh
nx, ny = 160, 160
lx, ly = 4., 4.
x0, y0 = 0., 0.
mesh = Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body using either approach
# Option A: VOFI
capacity = Capacity(LS, mesh, method="VOFI")

# Option B: ImplicitIntegration
capacity = Capacity(LS, mesh, method="ImplicitIntegration")
```

- **LS(x, y, _=0)**: A non-vectorized function that can be easily used by VOFI and ImplicitIntegration methods.


With these definitions, you can build simulation code that distinguishes inside vs. outside regions of a body.