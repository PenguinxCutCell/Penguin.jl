using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using SpecialFunctions
using Roots

function run_mesh_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    norm
)

    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]

    for (nx, ny) in zip(nx_list, ny_list)
        # Build mesh
        x0, y0 = 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

        # Define the body
        circle = (x,y,_=0) -> (sqrt((x-center[1])^2 + (y-center[2])^2) - radius)
            
        # Define capacity/operator
        capacity = Capacity(circle, mesh)
        operator = DiffusionOps(capacity)

        # BC + solver
        bc_boundary = Dirichlet(1.0)
        bc0 = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(
            :left   => bc0,
            :right  => bc0,
            :top    => bc0,
            :bottom => bc0
        ))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        u0ₒ = zeros((nx+1)*(ny+1))
        u0ᵧ = ones((nx+1)*(ny+1))
        u0 = vcat(u0ₒ, u0ᵧ)

        Δt = 0.5*(lx/nx)^2
        Tend = 0.1

        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "BE") # Start by a backward Euler scheme to prevent oscillation due to CN scheme

        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, "BE"; method=Base.:\)

        # Compute errors
        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm)

        # Representative mesh size ~ 1 / min(nx, ny)
        push!(h_vals, 1.0 / min(nx, ny))

        push!(err_vals,       global_err)
        push!(err_full_vals,  full_err)
        push!(err_cut_vals,   cut_err)
        push!(err_empty_vals, empty_err)
    end

    # Model for curve_fit
    function fit_model(x, p)
        p[1] .* x .+ p[2]
    end

    # Fit each on log scale: log(err) = p*log(h) + c
    log_h = log.(h_vals)

    function do_fit(log_err)
        fit_result = curve_fit(fit_model, log_h, log_err, [-1.0, 0.0])
        return fit_result.param[1], fit_result.param[2]  # (p_est, c_est)
    end

    p_global, _ = do_fit(log.(err_vals))
    p_full,   _ = do_fit(log.(err_full_vals))
    p_cut,    _ = do_fit(log.(err_cut_vals))

    # Round
    p_global = round(p_global, digits=2)
    p_full   = round(p_full, digits=2)
    p_cut    = round(p_cut, digits=2)

    println("Estimated order of convergence (global) = ", p_global)
    println("Estimated order of convergence (full)   = ", p_full)
    println("Estimated order of convergence (cut)    = ", p_cut)

    # Plot in log-log scale
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "h",
        ylabel = "L$norm error",
        title  = "Convergence plot",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(10),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(10),
    )

    scatter!(ax, h_vals, err_vals,       label="Global error ($p_global)", markersize=12)
    lines!(ax, h_vals, err_vals,         color=:black)
    scatter!(ax, h_vals, err_full_vals,  label="Full error ($p_full)",   markersize=12)
    lines!(ax, h_vals, err_full_vals,    color=:black)
    scatter!(ax, h_vals, err_cut_vals,   label="Cut error ($p_cut)",     markersize=12)
    lines!(ax, h_vals, err_cut_vals,     color=:black)

    lines!(ax, h_vals, 10.0*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax, h_vals, 1.0*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    axislegend(ax, position=:rb)
    display(fig)

    return (
        h_vals,
        err_vals,
        err_full_vals,
        err_cut_vals,
        err_empty_vals,
        p_global,
        p_full,
        p_cut,
    )
end

# Example usage:
nx_list = [20, 40, 80, 160, 320]
ny_list = [20, 40, 80, 160, 320]
radius, center = 1.0, (2.01, 2.01)
function radial_heat_(x, y)
    t=0.1
    R=1.0

    function j0_zeros(N; guess_shift=0.25)
        zs = zeros(Float64, N)
        # The m-th zero of J₀ is *roughly* near (m - guess_shift)*π for large m.
        # We'll bracket around that approximate location and refine via find_zero.
        for m in 1:N
            # approximate location
            x_left  = (m - guess_shift - 0.5)*pi
            x_right = (m - guess_shift + 0.5)*pi
            # ensure left>0
            x_left = max(x_left, 1e-6)
            
            # We'll use bisection or Brent's method from Roots.jl
            αm = find_zero(besselj0, (x_left, x_right))
            zs[m] = αm
        end
        return zs
    end

    alphas = j0_zeros(100)
    N=length(alphas)
    r = sqrt((x - center[1])^2 + (y - center[2])^2)
    if r >= R
        # Not physically in the domain, so return NaN or handle as you wish.
        return NaN
    end
    
    # If in the disk: sum the series
    s = 0.0
    for m in 1:N
        αm = alphas[m]
        s += exp(-αm^2 * t) * besselj0(αm * (r / R)) / (αm * besselj1(αm))
    end
    return 1.0 - 2.0*s
end


run_mesh_convergence(nx_list, ny_list, radius, center, radial_heat_, norm=Inf)