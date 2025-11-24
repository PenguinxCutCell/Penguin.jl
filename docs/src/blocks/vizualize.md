# Visualize

The `visualize` module in Penguin.jl provides tools for creating visual representations of data. This module is designed to help users easily generate plots, charts, and other visualizations to better understand their data.


## Usage

Below are some examples of how to use the `visualize` module:

### Example 1: Plotting Solutions

```julia
using Penguin

# Define a solver, mesh, body function, and capacity
mesh = ... # Your mesh object
body = ... # Your body function
capacity = ... # Your capacity object
solver = ... # Your solver object

# Plot the solution
plot_solution(solver, mesh, body, capacity)
```

### Example 2: Animation Solutions

This function triggers an animation to visually represent the evolution of the computed solution over a given mesh and body
parameters.

```julia
using Penguin

# Define a solver, mesh, body function, and capacity
mesh = ... # Your mesh object
body = ... # Your body function
capacity = ... # Your capacity object
solver = ... # Your solver object

# Plot the solution
animate_solution(solver, mesh, body)
```