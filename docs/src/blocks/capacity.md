# Assemble the Capacity

The `Capacity` struct stores all the capacities of a system in N dimensions. It computes various geometric and physical properties (like boundary integrals, volumetric quantities, and interface centroids) based on a user-defined signed distance function (SDF) and a `Mesh`. 

## Defining a Capacity

You create a `Capacity` by providing:
- A body function (SDF) capable of indicating whether points lie inside or outside the body.  
- A `Mesh` corresponding to the computational domain.  
- A `method` to compute the capacity. Currently, "VOFI" and "ImplicitIntegration" are fully implemented.

Example:
```julia
capacity = Capacity(body_function, mesh, method="VOFI")
```

## 1D Capacity Example

```julia
nx = 160
lx = 4.
x0 = 0.
mesh = Mesh((nx,), (lx,), (x0,))
    
# Define a signed distance function (SDF) for a 1D body
LS(x, _=0) = sqrt(x^2) - 0.5
    
capacity = Capacity(LS, mesh, method="VOFI")
```

In 1D, `capacity.A` and `capacity.B` are tuples of one sparse matrix each, while `capacity.W` holds staggered volume matrices.

## 2D Capacity Example

```julia
nx, ny = 160, 160
lx, ly = 4., 4.
x0, y0 = 0., 0.
mesh = Mesh((nx, ny), (lx, ly), (x0, y0))

# Define a signed distance function (SDF) for a 2D body
LS(x, y, _=0) = sqrt(x^2 + y^2) - 0.5
    
capacity = Capacity(LS, mesh, method="ImplicitIntegration")
```
In 2D, `Capacity` manages two matrices for each dimension (e.g., `Ax` and `Ay`).

## Fields in Capacity

- **A** (NTuple of Sparse Matrices): Stores A-capacity (faces surfaces) in each dimension.  
- **B** (NTuple of Sparse Matrices): Stores B-capacity (centered surfaces) in each dimension.
- **V** (Sparse Matrix): Holds volume capacity.  
- **W** (NTuple of Sparse Matrices): Represents staggered volumes in each dimension.  
- **C_ω** (Vector of SVector{N,Float64}): Centroids of each cell.  
- **C_γ** (Vector of SVector{N,Float64}): Centroids of the interface (boundary).  
- **Γ** (Sparse Matrix): Stores interface measures.  
- **cell_types** (Vector{Float64}): Classification of cells (e.g. full/cut/empty).  
- **mesh** (AbstractMesh): The mesh on which capacity is calculated.  
- **body** (Function): Your signed distance function.

## Summary

`Capacity` provides a way to compute and store geometric and physical properties of a body (defined by an SDF) on a given mesh. VOFI is the primary method for calculating capacity, capturing key information like cell volumes, boundary integrals, and interface positions.