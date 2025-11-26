using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using SpecialFunctions
using Roots

"""
    run_time_convergence(dt_list, radius, center, u_analytical; nx=40, ny=40, lx=4.0, ly=4.0, norm=2, Tend=0.1)

Perform a temporal convergence study by varying the time step Δt while keeping the spatial mesh fixed.

# Arguments
- `dt_list::Vector{Float64}`: List of time step values.
- `radius::Float64`: Radius used in the body (here a circle function).
- `center::Tuple{Float64,Float64}`: Center of the circle.
- `u_analytical::Function`: Function that computes the analytical solution.
- `nx, ny`: Mesh resolution parameters.
- `lx, ly`: Domain dimensions.
- `norm`: The error norm (e.g. 2).
- `Tend`: Final time for the simulation.
"""
function run_time_convergence(
    dt_list::Vector{Float64},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    nx::Int=40,
    ny::Int=40,
    lx::Float64=4.0,
    ly::Float64=4.0,
    norm=2,
    Tend::Float64=0.1
)
    dt_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]

    # Build the spatial mesh (fixed resolution)
    x0, y0 = 0.0, 0.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

    # Define the body as a circle (static in space)
    circle = (x, y, _=0) -> (sqrt((x - center[1])^2 + (y - center[2])^2) - radius)
    capacity = Capacity(circle, mesh)
    operator = DiffusionOps(capacity)

    # Define boundary conditions
    bc_boundary = Dirichlet(1.0)
    bc0 = Dirichlet(0.0)
    bc_b = BorderConditions(Dict(
        :left   => bc0,
        :right  => bc0,
        :top    => bc0,
        :bottom => bc0
    ))
    phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

    # Define the initial condition (uniform temperature)
    u0ₒ = zeros((nx+1)*(ny+1))
    u0ᵧ = ones((nx+1)*(ny+1))
    u0 = vcat(u0ₒ, u0ᵧ)

    # Loop over dt_list for temporal convergence
    for Δt in dt_list
        println("Running simulation with Δt = $Δt")
        # Use the same final time for all experiments.
        Tend_local = Tend

        # Create the solver; here we start with a backward Euler for stability.
        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "BE") # Start by a backward Euler scheme to prevent oscillation due to CN scheme
        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, "BE"; method=Base.:\)

        # Compute errors with respect to the analytical solution.
        # The function check_convergence is assumed to return (u_ana, u_num, global_err, full_err, cut_err, empty_err)
        u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytical, solver, capacity, norm)
        push!(dt_vals, Δt)
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
    end

    # Define a linear model for the logarithms, i.e. log(err) = p * log(Δt) + c
    fit_model(x, p) = p[1] .* x .+ p[2]
    log_dt = log.(dt_vals)
    fit_result = curve_fit(fit_model, log_dt, log.(err_vals), [-1.0, 0.0])
    fit_result_cut = curve_fit(fit_model, log_dt, log.(err_cut_vals), [-1.0, 0.0])
    #fit_result_empty = curve_fit(fit_model, log_dt, log.(err_empty_vals), [-1.0, 0.0])
    fit_result_full = curve_fit(fit_model, log_dt, log.(err_full_vals), [-1.0, 0.0])
    p_est = round(fit_result.param[1], digits=2)
    p_est_cut = round(fit_result_cut.param[1], digits=2)
    #p_est_empty = round(fit_result_empty.param[1], digits=2)
    p_est_full = round(fit_result_full.param[1], digits=2)
    println("Estimated temporal order of convergence (global) = ", p_est)
    println("Estimated temporal order of convergence (cut) = ", p_est_cut)
    #println("Estimated temporal order of convergence (empty) = ", p_est_empty)
    println("Estimated temporal order of convergence (full) = ", p_est_full)

    # Plot convergence in log–log scale
    fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel = "Δt",
        ylabel = "Global error (L$norm)",
        xscale = log10,
        yscale = log10,
        title  = "Temporal Convergence (order $p_est)"
    )
    scatter!(ax, dt_vals, err_vals, markersize = 10, label="Global error")
    scatter!(ax, dt_vals, err_full_vals, markersize = 10, label="Full error")
    scatter!(ax, dt_vals, err_cut_vals, markersize = 10, label="Cut error")
    #scatter!(ax, dt_vals, err_empty_vals, markersize = 10, label="Empty error")
    lines!(ax, dt_vals, err_vals, color=:black)
    lines!(ax, dt_vals, err_full_vals, color=:black)
    lines!(ax, dt_vals, err_cut_vals, color=:black)
    #lines!(ax, dt_vals, err_empty_vals, color=:black)

    lines!(ax, dt_vals, 10.0*dt_vals.^2.0, label="O(Δt²)", color=:black, linestyle=:dash)
    lines!(ax, dt_vals, 1.0*dt_vals.^1.0, label="O(Δt¹)", color=:black, linestyle=:dashdot)
    axislegend(ax, position=:rb)
    display(fig)

    return dt_vals, err_vals, p_est, err_full_vals, p_est_full, err_cut_vals, p_est_cut#, err_empty_vals, p_est_empty
end

# Example usage:
# Define time step list for temporal refinement
dt_list = [0.1, 0.05, 0.025, 0.0125, 0.00625]
radius, center = 1.0, (2.01, 2.01)

# Define an analytical solution function.
# Here we use a placeholder function; in a realistic example, replace this with the proper solution.
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


run_time_convergence(dt_list, radius, center, radial_heat_; nx=640, ny=640, Tend=0.1)