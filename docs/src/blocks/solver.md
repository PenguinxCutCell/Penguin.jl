# Using Solver

This section describes how to create and solve a steady-state, monophasic diffusion problem in Penguin.jl.

## Defining the Fluid (Phase)

Before setting up the solver, define a `Phase` that encapsulates the mesh, operators, source function, and diffusion coefficient:

```julia
Fluide = Phase(capacity, operator, f, D)
```
- `capacity`: Contains geometry and capacity (e.g. volume) details.  
- `operator`: Typically a `DiffusionOps` object that provides discrete operators.  
- `f`: Source term function (e.g., heat generation).  
- `D`: Diffusion coefficient function or constant.

## Constructing the Solver

Use the `DiffusionSteadyMono` function to assemble a solver for the steady, monophasic diffusion problem:

```julia
solver = DiffusionSteadyMono(Fluide, bc_b, bc)
```
- `Fluide`: The `Phase` defined above.  
- `bc_b`: A `BorderConditions` object, assigning Dirichlet/Neumann/Robin conditions on the domain borders.  
- `bc`: An internal boundary condition object (if needed).

This step:  
1. Creates a `Solver` object.  
2. Builds the system matrix `A` and right-hand side vector `b`.  
3. Applies boundary conditions to ensure the system reflects the specified physics.

## Solving the System

You can solve the system using either the default iterative/direct solvers or with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) algorithms.

### Default (IterativeSolvers/Base)

```julia
solve_DiffusionSteadyMono!(solver)
```

### With LinearSolve.jl

To use a LinearSolve.jl algorithm, pass the `algorithm` keyword argument:

```julia
using LinearSolve
solve_DiffusionSteadyMono!(solver; algorithm=KrylovJL_GMRES())
```
or for a direct solver:
```julia
solve_DiffusionSteadyMono!(solver; algorithm=UMFPACKFactorization())
```

This routine:  
1. Checks that the solver is properly initialized.  
2. Invokes the internal `solve_system!` to compute the field values, using the specified algorithm if provided.  
3. Stores the solution vector in `solver.x`.

## Accessing the Solution

The final solution is split in two parts:  
- `uo` is the solution in the cells.  
- `ug` is the additional solution component (e.g., interface or boundary-related) if used in your system.

After the system is solved, retrieve them as:

```julia
uo = solver.x[1:end÷2]
ug = solver.x[end÷2+1:end]
```

## Summary

1. **Define Phase**: Collect capacity, operators, source function, and diffusion coefficient.  
2. **Instantiate Solver**: Call `DiffusionSteadyMono` to prepare the linear system and boundary conditions.  
3. **Solve**: Use `solve_DiffusionSteadyMono!` to fill `solver.x` with the steady-state solution. Optionally, use the `algorithm` keyword for LinearSolve.jl support.  
4. **Extract Results**: Split `solver.x` into desired components for post-