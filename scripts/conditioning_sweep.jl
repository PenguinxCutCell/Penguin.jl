#!/usr/bin/env julia

using Penguin
using LinearAlgebra
using SparseArrays

# Sweep parameters
ratios = [1.0, 10.0, 100.0]         # D2/D1
meshes = [8, 16, 32]                 # number of cells (nx)

# Problem setup (mirrors examples/1D/Diffusion/Heat_2ph.jl)
lx = 8.0
x0 = 0.0
xint = 4.05

# Interface body definitions
body   = (x, _=0) -> (x - xint)
body_c = (x, _=0) -> -(x - xint)

# Boundary and interface conditions (as in example)
bc1 = Dirichlet(0.0)
bc0 = Dirichlet(1.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc0, :bottom => bc1))
ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Zero source terms
f1 = (x,y,z,t)->0.0
f2 = (x,y,z,t)->0.0

"""
Build the cut-cell (diphasic) system matrix A for given nx and diffusivity ratio r = D2/D1.
Returns a trimmed sparse A (zero rows/cols removed), along with λmin, λmax, cond2.
"""
function cutcell_stats(nx::Int, ratio::Float64)
    # Mesh and capacities
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    capacity   = Capacity(body,  mesh)
    capacity_c = Capacity(body_c, mesh)

    # Operators
    operator   = DiffusionOps(capacity)
    operator_c = DiffusionOps(capacity_c)

    # Diffusivities
    D1val = 1.0
    D2val = ratio
    D1 = (x,y,z)->D1val
    D2 = (x,y,z)->D2val

    # Phases
    Fluide_1 = Phase(capacity,   operator,   f1, D1)
    Fluide_2 = Phase(capacity_c, operator_c, f2, D2)

    # Initial condition vectors (as in example)
    u0ₒ1 = zeros(nx+1)
    u0ᵧ1 = zeros(nx+1)
    u0ₒ2 = ones(nx+1)
    u0ᵧ2 = ones(nx+1)
    u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

    # Time stepping (small dt, one step is enough to assemble A)
    Δt = 0.5 * (lx/nx)^2
    Tend = Δt
    solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "CN")

    # Run a single step to ensure A is built
    solve_DiffusionUnsteadyDiph!(
        solver, Fluide_1, Fluide_2,
        Δt, Tend, bc_b, ic, "CN";
        method = Base.:\
    )

    # Extract and trim A
    A = solver.A
    b = solver.b
    Atrim, _, _, _ = remove_zero_rows_cols!(copy(A), copy(b))

    # Convert to dense symmetric for eigen analysis (SPD expected)
    Ad = Matrix(Atrim)
    evals = eigvals(Ad)
    evals = real.(evals)  # should be real
    λmin = minimum(evals)
    λmax = maximum(evals)
    cond2 = cond(Ad, 2)
    return Atrim, λmin, λmax, cond2
end

"""
Build a standard 1D finite-volume diffusion matrix (Dirichlet at x=0 and x=L),
with piecewise constant D: D=D1 for x<xint, D=D2 for x>xint, uniform grid nx.
Returns dense symmetric A, along with λmin, λmax, cond2.
"""
function fv_stats(nx::Int, ratio::Float64)
    D1 = 1.0
    D2 = ratio
    L = lx
    dx = L / nx
    # We use cell-centered FV with N = nx unknowns.
    N = nx

    # Face transmissibilities T_{i+1/2} = D_face/dx
    # Faces i+1/2 for i=0..N; boundary faces at 1/2 and N+1/2.
    faces = collect(0:N) # index i for face i+1/2
    T = zeros(Float64, N+1)
    # Position of face i+1/2
    x_face = (i)-> (i+0.5)*dx
    for i in faces
        xF = x_face(i)
        if isapprox(xF, xint; atol=1e-12)
            # Interface face: harmonic average
            Df = 2.0 / (1.0/D1 + 1.0/D2)
            T[i+1] = Df / dx
        elseif xF < xint
            T[i+1] = D1 / dx
        else
            T[i+1] = D2 / dx
        end
    end

    # Assemble symmetric tridiagonal matrix A (Dirichlet BCs incorporated on diagonal)
    A = zeros(Float64, N, N)
    for i in 1:N
        Tm = T[i]     # T_{i-1/2}
        Tp = T[i+1]   # T_{i+1/2}
        A[i,i] = Tm + Tp
        if i > 1
            A[i,i-1] = -Tm
            A[i-1,i] = -Tm
        end
        if i < N
            A[i,i+1] = -Tp
            A[i+1,i] = -Tp
        end
    end

    Ad = Symmetric(A)
    evals = eigvals(Ad)
    λmin = minimum(evals)
    λmax = maximum(evals)
    cond2 = λmax / λmin
    return A, λmin, λmax, cond2
end

function main()
    # Output CSV
    outpath = joinpath(@__DIR__, "conditioning_sweep.csv")
    open(outpath, "w") do io
        println(io, "scheme,nx,ratio,lambda_min,lambda_max,cond2,rows,cols,nnz")
        for nx in meshes
            for ratio in ratios
                # Cut-cell
                Atrim, λmin_c, λmax_c, cond2_c = cutcell_stats(nx, ratio)
                println(io, join([
                    "cutcell",
                    string(nx),
                    string(ratio),
                    string(λmin_c),
                    string(λmax_c),
                    string(cond2_c),
                    string(size(Atrim,1)),
                    string(size(Atrim,2)),
                    string(nnz(Atrim))
                ], ","))

                # Standard FV
                A_fv, λmin_fv, λmax_fv, cond2_fv = fv_stats(nx, ratio)
                println(io, join([
                    "fv",
                    string(nx),
                    string(ratio),
                    string(λmin_fv),
                    string(λmax_fv),
                    string(cond2_fv),
                    string(size(A_fv,1)),
                    string(size(A_fv,2)),
                    string(nnz(sparse(A_fv)))
                ], ","))
            end
        end
    end

    println("CSV written to: ", outpath)
end

main()


