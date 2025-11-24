#!/usr/bin/env julia

using Penguin
using LinearAlgebra
using SparseArrays
using CairoMakie

# Mesh sizes to sweep
meshes = [8, 16, 32]

# Domain and geometry (from examples/1D/Diffusion/Poisson.jl)
lx = 1.0
x0 = 0.0
center = 0.5
radius = 0.21
body = (x, _=0) -> sqrt((x-center)^2) - radius

# Boundary conditions: Dirichlet 0 on both ends
bc_left  = Dirichlet(0.0)
bc_right = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(0.0)))

# Internal boundary (interface) condition: Dirichlet 0 on the embedded boundary
bc_i = Dirichlet(0.0)

# Source and diffusion coefficient (constant)
g = (x, y, _=0) -> x
a = (x, y, _=0) -> 1.0

"""
Build the cut-cell monophasic steady diffusion (Poisson) matrix for nx,
trim zero rows/cols, and return λmin, λmax, cond2.
"""
function cutcell_stats_poisson(nx::Int)
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    capacity = Capacity(body, mesh)
    operator = DiffusionOps(capacity)
    Fluide   = Phase(capacity, operator, g, a)

    solver = DiffusionSteadyMono(Fluide, bc_b, bc_i)
    # Ensure A and b are finalized (constructor already builds them; solve is safe)
    #solve_DiffusionSteadyMono!(solver; method=Base.:\)

    A = solver.A
    b = solver.b
    Atrim, _, _, _ = remove_zero_rows_cols!(copy(A), copy(b))

    Ad = Matrix(Atrim)
    ev = eigen(Ad)
    evals = ev.values
    imin = argmin(evals)
    vmin = ev.vectors[:, imin]
    λmin = evals[imin]
    λmax = maximum(evals)
    cond2 = λmax / λmin
    return Atrim, λmin, λmax, cond2, evals, vmin
end

"""
Standard 1D finite-volume Poisson matrix on [0,L] with Dirichlet 0 at both ends,
constant diffusion a=1. Returns eigen extrema and cond2.
"""
function fv_stats_poisson(nx::Int)
    L = lx
    dx = L / nx
    N = nx
    T = 1.0 / dx  # face transmissibility with D=1

    A = zeros(Float64, N, N)
    for i in 1:N
        A[i,i] = 2T
        if i > 1
            A[i,i-1] = -T
            A[i-1,i] = -T
        end
        if i < N
            A[i,i+1] = -T
            A[i+1,i] = -T
        end
    end

    Ad = Symmetric(A)
    ev = eigen(Ad)
    evals = ev.values
    imin = argmin(evals)
    vmin = ev.vectors[:, imin]
    λmin = evals[imin]
    λmax = maximum(evals)
    cond2 = λmax / λmin
    return A, λmin, λmax, cond2, evals, vmin
end

function main()
    outpath = joinpath(@__DIR__, "conditioning_poisson.csv")
    open(outpath, "w") do io
        println(io, "scheme,nx,lambda_min,lambda_max,cond2,rows,cols,nnz")
        for nx in meshes
            # Cut-cell Poisson
            Atrim, λmin_c, λmax_c, cond2_c, evals_c, vmin_c = cutcell_stats_poisson(nx)
            println(io, join([
                "cutcell",
                string(nx),
                string(λmin_c),
                string(λmax_c),
                string(cond2_c),
                string(size(Atrim,1)),
                string(size(Atrim,2)),
                string(nnz(Atrim))
            ], ","))

            # Standard FV Poisson
            A_fv, λmin_fv, λmax_fv, cond2_fv, evals_fv, vmin_fv = fv_stats_poisson(nx)
            println(io, join([
                "fv",
                string(nx),
                string(λmin_fv),
                string(λmax_fv),
                string(cond2_fv),
                string(size(A_fv,1)),
                string(size(A_fv,2)),
                string(nnz(sparse(A_fv)))
            ], ","))

            # Plot eigenvalues (log scale) for this nx
            fig = Figure(size = (800, 450))
            ax = Axis(fig[1,1];
                      xlabel = "Eigenvalue index",
                      ylabel = "Eigenvalue",
                      #yscale = log10,
                      title = "Poisson eigenvalues (nx=$(nx))")

            idx_c = 1:length(evals_c)
            idx_f = 1:length(evals_fv)
            scatter!(ax, idx_c, sort(evals_c); color = :dodgerblue, markersize = 6, label = "cutcell")
            scatter!(ax, idx_f, sort(evals_fv); color = :orange, markersize = 5, label = "fv")
            axislegend(ax, position = :rb)

            # Save figure next to CSV
            outpng = joinpath(@__DIR__, "conditioning_poisson_eigs_nx$(nx).png")
            save(outpng, fig)
            display(fig)

            # Plot eigenvector associated with the minimum eigenvalue (potential negative mode)
            fig_v = Figure(size = (900, 420))
            axv1 = Axis(fig_v[1,1]; xlabel = "Index", ylabel = "v_min (normalized)", title = "Cutcell v_min (nx=$(nx))")
            axv2 = Axis(fig_v[1,2]; xlabel = "Index", ylabel = "v_min (normalized)", title = "FV v_min (nx=$(nx))")
            vcn = vmin_c ./ maximum(abs.(vmin_c))
            vfn = vmin_fv ./ maximum(abs.(vmin_fv))
            lines!(axv1, 1:length(vcn), vcn; color=:dodgerblue)
            scatter!(axv1, 1:length(vcn), vcn; color=:dodgerblue, markersize=4)
            lines!(axv2, 1:length(vfn), vfn; color=:orange)
            scatter!(axv2, 1:length(vfn), vfn; color=:orange, markersize=4)
            outpng_v = joinpath(@__DIR__, "conditioning_poisson_vmin_nx$(nx).png")
            save(outpng_v, fig_v)
            display(fig_v)

            # Plot sparsity patterns (spy) for cutcell (trimmed) and FV
            fig_spy = Figure(size = (900, 420))
            ax1 = Axis(fig_spy[1,1]; xlabel = "Column", ylabel = "Row", title = "Cutcell A (trimmed)", yreversed=true)
            ax2 = Axis(fig_spy[1,2]; xlabel = "Column", ylabel = "Row", title = "FV A", yreversed=true)
            r_c, c_c, _ = findnz(sparse(Atrim))
            scatter!(ax1, c_c, r_c; markersize=3, color=:black)
            r_f, c_f, _ = findnz(sparse(A_fv))
            scatter!(ax2, c_f, r_f; markersize=3, color=:black)
            outpng_spy = joinpath(@__DIR__, "conditioning_poisson_spy_nx$(nx).png")
            save(outpng_spy, fig_spy)
            display(fig_spy)
        end
    end

    println("CSV written to: ", outpath)
end

main()
