# Mesh Usage

This page explains how to construct a `Mesh` in **Penguin.jl** when you specify the number of cells along each dimension, the overall domain lengths, and the origin coordinates. This form of initialization automatically computes the cell centers, nodes, and border cells.

## Defining a Mesh with Counts and Domain Dimensions

Consider a 2D mesh defined as follows:

```julia
using Penguin

# Define the mesh parameters
nx, ny = 5, 5         # Number of cells in x and y directions
lx, ly = 4.0, 4.0     # Total domain lengths in x and y
x0, y0 = 0.0, 0.0     # Origin (starting coordinates)

# Create the mesh
mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
```

In this setup:
- **nx, ny**: Specify the resolution (number of cells) along each axis.
- **lx, ly**: Define the overall lengths of the domain in the x and y directions.
- **x0, y0**: Set the starting coordinate for the mesh, which is useful for offsetting the grid.

## How It Works

Using this constructor, `Penguin.jl` automatically performs the following:
1. **Nodes**: Computes the boundaries of each cell.
2. **Centers**: Determines the position of each cell center.
3. **Tagging**: Adds metadata (such as border cell indices) to help identify cells at the domain boundaries.

## Viewing Mesh Properties

After creating your mesh, you can explore its properties:

```julia
# Print the computed centers and number of cells
println("Cell centers: ", mesh.centers)
println("Total number of cells: ", nC(mesh))
```

You can also retrieve border cells, which are defined as those at the edges of the domain:

```julia
borders = mesh.tag.border_cells
println("Number of border cells: ", length(borders))
```

## Summary

By defining a mesh with cell counts and domain dimensions as shown:
- You easily construct and configure the grid.
- The `Mesh` constructor automatically generates necessary geometric data.
- You’re equipped to further use the mesh in simulations or data analysis.

This approach is especially useful when you want a quick setup using a uniform grid without manually precomputing coordinate vectors.  

## Defining a Space–Time Mesh

Besides a purely spatial mesh, *Penguin.jl* also provides a `SpaceTimeMesh` that incorporates time as an additional dimension. This is particularly useful when your domain changes in time, such as when boundaries move according to a prescribed interface function. For instance:

```julia
# Define the spatial mesh
nx = 40
lx = 1.0
x0 = 0.0
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define a moving boundary (body) as a function of x and t
xf = 0.01 * lx
c = 1.0
body = (x, t, _=0) -> (x - xf - c * sqrt(t))

# Define the Space-Time mesh
Δt   = 0.01
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt])
```

Here, the `body` function captures how the interface changes over time, and `SpaceTimeMesh` is rebuilt at each time interval `Δt` to track this motion. This approach makes it straightforward to handle time‐evolving geometries within your simulations.