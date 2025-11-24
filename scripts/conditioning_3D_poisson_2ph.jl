#!/usr/bin/env julia

using Penguin
using LinearAlgebra
using SparseArrays

# 3D diphasic Poisson (steady diffusion) conditioning sweep
# Warning: full dense eigen-decomposition scales poorly; keep meshes modest.

# Parameters to sweep
meshes = [(4,4,4), (8,8,8), (16,16,16)]  # (nx,ny,nz)
ratios = [1.0, 10.0, 100.0]   # D2/D1

# Domain and body (sphere in box)
lx, ly, lz = 4.0, 4.0, 4.0
x0, y0, z0 = 0.0, 0.0, 0.0
radius, center = ly/4, (lx/2, ly/2, lz/2)
sphere(x,y,z) = sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - radius
sphere_c(x,y,z) = -sphere(x,y,z)

# Boundary and interface conditions
bc0 = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
  :left=>bc0, :right=>bc0, :top=>bc0, :bottom=>bc0, :front=>bc0, :back=>bc0
))
ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Sources
f1 = (x,y,z)->0.0
f2 = (x,y,z)->0.0

"""
Build cut-cell (diphasic) matrix for 3D Poisson at given resolution and ratio.
Returns trimmed A, λmin, λmax, cond2.
"""
function cutcell_stats_3d(nx::Int, ny::Int, nz::Int, ratio::Float64)
    mesh = Penguin.Mesh((nx,ny,nz), (lx,ly,lz), (x0,y0,z0))
    capacity   = Capacity(sphere,   mesh)
    capacity_c = Capacity(sphere_c, mesh)

    operator   = DiffusionOps(capacity)
    operator_c = DiffusionOps(capacity_c)

    D1val = 1.0
    D2val = ratio
    D1 = (x,y,z)->D1val
    D2 = (x,y,z)->D2val

    phase1 = Phase(capacity,   operator,   f1, D1)
    phase2 = Phase(capacity_c, operator_c, f2, D2)

    solver = DiffusionSteadyDiph(phase1, phase2, bc_b, ic)

    # Extract and trim A
    A = solver.A
    b = solver.b
    Atrim, _, _, _ = remove_zero_rows_cols!(copy(A), copy(b))

    # Dense symmetric for eigen analysis
    Ad = Matrix(Atrim)
    evals = eigvals(Ad)
    evals = real.(evals)  # should be real
    λmin = minimum(evals)
    λmax = maximum(evals)
    cond2 = λmax / λmin
    return Atrim, λmin, λmax, cond2
end

"""
Assemble a standard 7-point FV Poisson matrix on a uniform grid with Dirichlet=0 on all faces.
Piecewise constant D: D=D1 inside sphere, D=D2 outside. Interface handled via harmonic averaging
between neighboring cell-centered diffusivities.
Returns A, λmin, λmax, cond2.
"""
function fv3d_stats(nx::Int, ny::Int, nz::Int, ratio::Float64)
    Lx, Ly, Lz = lx, ly, lz
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz

    # Index mapping for (i,j,k) -> linear
    lin(i,j,k) = (k-1)*nx*ny + (j-1)*nx + i
    N = nx*ny*nz

    # Cell centers
    xc(i) = x0 + (i-0.5)*dx
    yc(j) = y0 + (j-0.5)*dy
    zc(k) = z0 + (k-0.5)*dz

    # Cell-centered D
    D1, D2 = 1.0, ratio
    Dcell = Array{Float64}(undef, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        x, y, z = xc(i), yc(j), zc(k)
        Dcell[i,j,k] = sphere(x,y,z) <= 0 ? D1 : D2
    end

    # Helper: harmonic average
    harm(a,b) = 2.0 / (1.0/a + 1.0/b)

    # Assemble 7-point stencil with transmissibilities
    A = spzeros(Float64, N, N)
    for k in 1:nz, j in 1:ny, i in 1:nx
        p = lin(i,j,k)
        diag_val = 0.0

        # x- neighbor
        if i > 1
            dface = harm(Dcell[i-1,j,k], Dcell[i,j,k])
            T = dface / dx^2
            q = lin(i-1,j,k)
            A[p,q] -= T; A[q,p] -= T
            diag_val += T
        else
            dface = Dcell[i,j,k]
            T = dface / dx^2
            diag_val += T
        end

        # x+ neighbor
        if i < nx
            dface = harm(Dcell[i,j,k], Dcell[i+1,j,k])
            T = dface / dx^2
            q = lin(i+1,j,k)
            A[p,q] -= T; A[q,p] -= T
            diag_val += T
        else
            dface = Dcell[i,j,k]
            T = dface / dx^2
            diag_val += T
        end

        # y- neighbor
        if j > 1
            dface = harm(Dcell[i,j-1,k], Dcell[i,j,k])
            T = dface / dy^2
            q = lin(i,j-1,k)
            A[p,q] -= T; A[q,p] -= T
            diag_val += T
        else
            dface = Dcell[i,j,k]
            T = dface / dy^2
            diag_val += T
        end

        # y+ neighbor
        if j < ny
            dface = harm(Dcell[i,j,k], Dcell[i,j+1,k])
            T = dface / dy^2
            q = lin(i,j+1,k)
            A[p,q] -= T; A[q,p] -= T
            diag_val += T
        else
            dface = Dcell[i,j,k]
            T = dface / dy^2
            diag_val += T
        end

        # z- neighbor
        if k > 1
            dface = harm(Dcell[i,j,k-1], Dcell[i,j,k])
            T = dface / dz^2
            q = lin(i,j,k-1)
            A[p,q] -= T; A[q,p] -= T
            diag_val += T
        else
            dface = Dcell[i,j,k]
            T = dface / dz^2
            diag_val += T
        end

        # z+ neighbor
        if k < nz
            dface = harm(Dcell[i,j,k], Dcell[i,j,k+1])
            T = dface / dz^2
            q = lin(i,j,k+1)
            A[p,q] -= T; A[q,p] -= T
            diag_val += T
        else
            dface = Dcell[i,j,k]
            T = dface / dz^2
            diag_val += T
        end

        A[p,p] += diag_val
    end

    Ad = Matrix(A)
    evals = eigvals(Ad)
    evals = real.(evals)  # should be real
    λmin = minimum(evals)
    λmax = maximum(evals)
    cond2 = λmax / λmin
    return A, λmin, λmax, cond2
end

function main()
    outpath = joinpath(@__DIR__, "conditioning_3D_poisson_2ph.csv")
    open(outpath, "w") do io
        println(io, "scheme,nx,ny,nz,ratio,lambda_min,lambda_max,cond2,rows,cols,nnz")
        for (nx,ny,nz) in meshes
            for ratio in ratios
                # Cut-cell diphasic
                Atrim, λmin_c, λmax_c, cond2_c = cutcell_stats_3d(nx,ny,nz,ratio)
                println(io, join([
                    "cutcell",
                    string(nx), string(ny), string(nz), string(ratio),
                    string(λmin_c), string(λmax_c), string(cond2_c),
                    string(size(Atrim,1)), string(size(Atrim,2)), string(nnz(Atrim))
                ], ","))

                # Standard FV
                A_fv, λmin_fv, λmax_fv, cond2_fv = fv3d_stats(nx,ny,nz,ratio)
                println(io, join([
                    "fv",
                    string(nx), string(ny), string(nz), string(ratio),
                    string(λmin_fv), string(λmax_fv), string(cond2_fv),
                    string(size(A_fv,1)), string(size(A_fv,2)), string(nnz(A_fv))
                ], ","))
            end
        end
    end
    println("CSV written to: ", outpath)
end

main()
