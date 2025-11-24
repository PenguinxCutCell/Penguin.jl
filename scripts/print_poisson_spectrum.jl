#!/usr/bin/env julia

using Penguin
using LinearAlgebra
using SparseArrays

# Minimal sweep: meshes and diffusivity ratios
meshes = [8, 16, 32]
ratios = [1.0, 10.0, 100.0]

# Geometry
lx = 1.0
x0 = 0.0
center = 0.5
radius = 0.21
body = (x, _=0) -> sqrt((x-center)^2) - radius

# Source
g = (x, y, _=0) -> x

# Boundary conditions
bc_val = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc_val, :bottom => bc_val))
bc_i = Dirichlet(0.0)

println("scheme,nx,D,lambda_min,lambda_max,cond2,rows,cols,nnz")
for nx in meshes
    # Mesh and capacity
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    capacity = Capacity(body, mesh)
    for D in ratios
        # Diffusion coefficient
        a = (x, y, _=0) -> D
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, g, a)
        solver = DiffusionSteadyMono(phase, bc_b, bc_i)

        # Trim zero rows/cols then compute eigen extrema
        A = solver.A; b = solver.b
        Ad = Matrix(A)
        evals = eigvals(Ad)
        λmin = minimum(evals)
        λmax = maximum(evals)
        cond2 = λmax / λmin
        println(join([
            "poisson", string(nx), string(D), string(λmin), string(λmax), string(cond2),
            string(size(A,1)), string(size(A,2)), string(nnz(A))
        ], ","))
    end
end
