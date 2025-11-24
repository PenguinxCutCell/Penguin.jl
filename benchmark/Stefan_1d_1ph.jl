using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using SpecialFunctions  # For erf function
using Roots  # For finding λ
using CSV, DataFrames
using Dates
using Printf
using Statistics

"""
    find_lambda(Stefan_number)

Find the dimensionless parameter λ for the one-phase Stefan problem by solving:
λ*exp(λ²)*erf(λ) - Stefan_number/sqrt(π) = 0
"""
function find_lambda(Stefan_number)
    f = (λ) -> λ*exp(λ^2)*erf(λ) - Stefan_number/sqrt(π)
    lambda = find_zero(f, 0.1)
    return lambda
end

"""
    analytical_T(x, t, T₀, k, lambda)

Analytical temperature distribution for the one-phase Stefan problem:
T(x,t) = T₀ - T₀/erf(λ) * erf(x/(2*sqrt(k*t)))
"""
function analytical_T(x, t, T₀, k, lambda)
    if t <= 0
        # Handle t=0 case
        return T₀
    end
    return T₀ - T₀/erf(lambda) * erf(x/(2*sqrt(k*t)))
end

"""
    analytical_gradT(x, t, T₀, k, lambda)

Analytical temperature gradient for the one-phase Stefan problem:
∇T(x,t) = -T₀/(sqrt(t)) * exp(-x²/(4*t)) / (sqrt(π) * erf(λ))
"""
function analytical_gradT(x, t, T₀, k, lambda)
    if t <= 0
        # Handle t=0 case
        return 0.0
    end
    return -T₀/(sqrt(t)) * exp(-x^2/(4*k*t)) / (sqrt(π) * erf(lambda))
end

"""
    interface_position(t, k, lambda)

Calculate the interface position at time t:
s(t) = 2*λ*sqrt(k*t)
"""
function interface_position(t, k, lambda)
    return 2 * lambda * sqrt(k * t)
end

"""
    interface_velocity(t, k, lambda)

Calculate the interface velocity at time t:
v(t) = λ*sqrt(k/t)
"""
function interface_velocity(t, k, lambda)
    if t <= 0
        # Handle t=0 case
        return Inf
    end
    return lambda * sqrt(k / t)
end

"""
    run_stefan_1d_mesh_convergence(nx_list, T₀, k, Stefan_number; kwargs...)

Run mesh convergence study for the 1D one-phase Stefan problem.
"""
function run_stefan_1d_mesh_convergence(
    nx_list::Vector{Int},
    T₀::Float64,         # Initial temperature
    k::Float64,          # Thermal diffusivity
    Stefan_number::Float64;  # Stefan number
    lx::Float64=2.0,     # Domain length
    x0::Float64=0.0,     # Domain start
    Tstart::Float64=0.0, # Start time (to avoid singularity at t=0)
    Tend::Float64=0.1,   # Final simulation time
    norm::Real=2,        # Norm for error calculation
    relative::Bool=true, # Use relative error
    output_dir::String="stefan_1d_convergence_results"
)
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Create a timestamp for this run
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    run_dir = joinpath(output_dir, timestamp)
    mkpath(run_dir)
    
    # Find lambda (analytical solution parameter)
    lambda = find_lambda(Stefan_number)
    println("Analytical solution parameter λ = $lambda")
    
    # Calculate final interface position
    xint_final = interface_position(Tend, k, lambda)
    println("Analytical interface position at t=$Tend: x = $xint_final")
    
    # Initialize storage arrays
    h_vals = Float64[]
    dt_vals = Float64[]        # To track dt values
    err_vals = Float64[]       # Global errors
    err_full_vals = Float64[]  # Full cell errors
    err_cut_vals = Float64[]   # Cut cell errors
    pos_error_vals = Float64[] # Interface position error
    
    # Save test configuration
    config_df = DataFrame(
        parameter = ["T₀", "k", "Stefan_number", "lx", "x0", "Tend", "lambda", "xint_final", "norm", "relative"],
        value = [T₀, k, Stefan_number, lx, x0, Tend, lambda, xint_final, norm, relative]
    )
    CSV.write(joinpath(run_dir, "config.csv"), config_df)
    
    # For each mesh resolution
    for nx in nx_list
        println("\n===== Testing mesh size nx = $nx =====")
        
        # Build mesh
        mesh = Penguin.Mesh((nx,), (lx,), (x0,))
        
        # Set initial interface position (we'll start at t=dt to avoid t=0)
        Δt = 0.5 * (lx/nx)^2   # Time step based on mesh size (CFL condition)
        push!(dt_vals, Δt)
        xint_init = interface_position(Tstart, k, lambda)
        
        # Define the body (level set function)
        body_func = (x, t, _=0) -> (x - xint_init)
        
        # Define the Space-Time mesh
        STmesh = Penguin.SpaceTimeMesh(mesh, [Δt, 2Δt], tag=mesh.tag)
        
        # Define the capacity
        capacity = Capacity(body_func, STmesh)
        
        # Define the operator
        operator = DiffusionOps(capacity)
        
        # Define the boundary conditions
        bc0 = Dirichlet(T₀)  # Left boundary (fixed temperature)
        bc1 = Dirichlet(0.0) # Right boundary/interface (melting point)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:bottom => bc0, :top => bc1))
        
        # Interface condition with Stefan condition
        L = 1.0      # Latent heat
        rho = 1.0    # Density
        stef_cond = InterfaceConditions(nothing, FluxJump(k, 0.0, rho*L))
        
        # Define the source term (no source for basic Stefan problem)
        f = (x,y,z,t) -> 0.0
        
        # Define diffusion coefficient
        K_func = (x,y,z) -> k
        
        # Define the phase
        Liquid = Phase(capacity, operator, f, K_func)
        
        # Initial condition based on analytical solution
        u0ₒ = zeros(nx+1)  # Bulk field
        u0ᵧ = zeros(nx+1)  # Interface field
        
        # Fill with initial analytical solution at t=Tstart
        for i in 1:nx+1
            x = mesh.nodes[1][i]
            if x < xint_init
                u0ₒ[i] = analytical_T(x, Tstart, T₀, k, lambda)
                u0ᵧ[i] = analytical_T(x, Tstart, T₀, k, lambda)
            end
        end
        
        u0 = vcat(u0ₒ, u0ᵧ)
        
        # Define the solver
        solver = MovingLiquidDiffusionUnsteadyMono(Liquid, bc_b, bc1, Δt, u0, mesh, "BE")
        
        # Newton parameters
        max_iter = 20
        tol = 1e-12
        reltol = 1e-12
        α = 1.0
        Newton_params = (max_iter, tol, reltol, α)
        
        # Solve the problem
        println("  Solving problem...")
        solver, residuals, xint_history, _ = solve_MovingLiquidDiffusionUnsteadyMono!(
            solver, Liquid, xint_init, Δt, Tstart, Tend, bc_b, bc1, stef_cond, mesh, "BE";
            Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\
        )
        
        # Calculate final interface position from numerical solution
        xint_num_final = xint_history[end]
        pos_error = abs(xint_num_final - xint_final)
        push!(pos_error_vals, pos_error)
        
        # Save interface position history
        interface_df = DataFrame(
            time = range(Tstart, Tend, length=length(xint_history)),
            position = xint_history,
            analytical = [interface_position(t, k, lambda) for t in range(Tstart, Tend, length=length(xint_history))]
        )
        interface_file = joinpath(run_dir, @sprintf("interface_nx_%04d.csv", nx))
        CSV.write(interface_file, interface_df)
        
        # Define analytical solution function for error calculation
        function T_analytical(x)
            if x < xint_final
                return analytical_T(x, Tend, T₀, k, lambda)
            else
                return 0.0  # Outside liquid domain
            end
        end
        
        # Compute errors
        println("  Computing errors...")
        body_tend = (x, _=0) -> (x - xint_final)
        capacity_tend = Capacity(body_tend, mesh)
        
        # Check convergence using the analytical solution
        (u_ana, u_num, global_err, full_err, cut_err, empty_err) =
            check_convergence(T_analytical, solver, capacity_tend, norm, relative)
        
        # Store mesh size and errors
        push!(h_vals, lx / nx)
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        
        # Save individual test result to CSV
        test_df = DataFrame(
            mesh_size = lx / nx,
            nx = nx,
            dt = Δt,
            global_error = global_err,
            full_error = full_err,
            cut_error = cut_err,
            empty_error = empty_err,
            position_error = pos_error
        )
        
        # Use formatted nx in filename for easier reading
        test_filename = @sprintf("mesh_%04d.csv", nx)
        test_filepath = joinpath(run_dir, test_filename)
        CSV.write(test_filepath, test_df)
        println("  Saved result to $(test_filepath)")
    end
    
    # Fit convergence rates
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
    
    # Calculate rates based on last 3 points
    function compute_last3_rate(h_data, err_data)
        # Get last 3 points (or fewer if not enough data)
        n = length(h_data)
        idx_start = max(1, n-2)
        last_h = h_data[idx_start:n]
        last_err = err_data[idx_start:n]
        
        # Compute rate using linear regression on log-log data
        log_h = log.(last_h)
        log_err = log.(last_err)
        
        if length(log_h) > 1
            h_mean = mean(log_h)
            err_mean = mean(log_err)
            numerator = sum((log_h .- h_mean) .* (log_err .- err_mean))
            denominator = sum((log_h .- h_mean).^2)
            rate = numerator / denominator
            return round(rate, digits=2)
        else
            return 0.0
        end
    end
    
    # Fit convergence rates
    p_global, _ = do_fit(log.(err_vals))
    p_full, _ = do_fit(log.(err_full_vals))
    p_cut, _ = do_fit(log.(err_cut_vals))
    p_pos, _ = do_fit(log.(pos_error_vals))
    
    # Also calculate last 3 point rates
    last3_p_global = compute_last3_rate(h_vals, err_vals)
    last3_p_full = compute_last3_rate(h_vals, err_full_vals)
    last3_p_cut = compute_last3_rate(h_vals, err_cut_vals)
    last3_p_pos = compute_last3_rate(h_vals, pos_error_vals)
    
    # Round for display
    p_global = round(p_global, digits=2)
    p_full = round(p_full, digits=2)
    p_cut = round(p_cut, digits=2)
    p_pos = round(p_pos, digits=2)
    
    # Print convergence rates
    println("\n===== Convergence Rates =====")
    println("Global error: p = $p_global (all), p = $last3_p_global (last 3)")
    println("Full cells:   p = $p_full (all), p = $last3_p_full (last 3)")
    println("Cut cells:    p = $p_cut (all), p = $last3_p_cut (last 3)")
    println("Position:     p = $p_pos (all), p = $last3_p_pos (last 3)")

    # Save final summary results to CSV
    df = DataFrame(
        mesh_size = h_vals,
        nx = nx_list,
        dt = dt_vals,
        global_error = err_vals,
        full_error = err_full_vals,
        cut_error = err_cut_vals,
        position_error = pos_error_vals
    )
    
    metadata = DataFrame(
        parameter = ["p_global", "p_full", "p_cut", "p_pos", 
                    "last3_p_global", "last3_p_full", "last3_p_cut", "last3_p_pos"],
        value = [p_global, p_full, p_cut, p_pos,
                last3_p_global, last3_p_full, last3_p_cut, last3_p_pos]
    )
    
    # Write final results
    summary_file = joinpath(run_dir, "summary.csv")
    rates_file = joinpath(run_dir, "convergence_rates.csv")
    CSV.write(summary_file, df)
    CSV.write(rates_file, metadata)
    
    println("Results saved to $(run_dir)")
    println("  Summary: $(summary_file)")
    println("  Convergence rates: $(rates_file)")
    
    return (
        h_vals,
        err_vals,
        err_full_vals,
        err_cut_vals,
        pos_error_vals,
        p_global,
        p_full,
        p_cut,
        p_pos,
        last3_p_global,
        last3_p_full,
        last3_p_cut,
        last3_p_pos,
        run_dir
    )
end

"""
    plot_stefan_convergence_results(h_vals, err_vals, err_full_vals, err_cut_vals, 
                                   pos_error_vals, p_rates, last3_rates; style=:temperature, save_dir=".")

Create publication-quality convergence plots for Stefan problem.
Style can be :temperature (temperature error) or :position (interface position error).
"""
function plot_stefan_convergence_results(
    h_vals::Vector{Float64},
    err_vals::Vector{Float64},
    err_full_vals::Vector{Float64},
    err_cut_vals::Vector{Float64},
    pos_error_vals::Vector{Float64},
    p_rates::Tuple{Float64, Float64, Float64, Float64},
    last3_rates::Tuple{Float64, Float64, Float64, Float64};
    style::Symbol=:temperature,
    save_dir::String="."
)
    # Create directory if it doesn't exist
    mkpath(save_dir)
    
    # Scientific color palette (colorblind-friendly)
    colors = [:darkblue, :darkred, :darkgreen, :purple]
    symbols = [:circle, :rect, :diamond, :star5]
    
    # Function to compute power law curve from order and data
    function power_fit(h, p, h_data, err_data)
        # Get last 3 points (or fewer if not enough data)
        n = length(h_data)
        idx_start = max(1, n-2)
        last_three_h = h_data[idx_start:n]
        last_three_err = err_data[idx_start:n]
        
        # Compute C based on the finest grid point (last point)
        C = last_three_err[end] / (last_three_h[end]^p)
        return C .* h.^p
    end
    
    # Unpack rates
    p_global, p_full, p_cut, p_pos = p_rates
    last3_p_global, last3_p_full, last3_p_cut, last3_p_pos = last3_rates
    
    # Generate fine mesh for fitted curves
    h_fine = 10 .^ range(log10(minimum(h_vals)), log10(maximum(h_vals)), length=100)
    
    if style == :temperature
        # Figure for temperature field errors
        fig = Figure(resolution=(800, 600), fontsize=14)
        
        ax = Axis(
            fig[1, 1],
            xlabel = "Mesh size (h)",
            ylabel = "Relative Error (L₂ norm)",
            title  = "Stefan Problem - Temperature Field Convergence",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticks = IntervalsBetween(10),
            xticks = LogTicks(WilkinsonTicks(5))
        )

        # Plot data points and fitted curve
        scatter!(ax, h_vals, err_vals, color=colors[1], marker=symbols[1],
                markersize=10, label="Global error (p = $(last3_p_global))")
        lines!(ax, h_fine, power_fit(h_fine, last3_p_global, h_vals, err_vals), 
              color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax, h_vals, err_full_vals, color=colors[2], marker=symbols[2],
                markersize=10, label="Full error (p = $(last3_p_full))")
        lines!(ax, h_fine, power_fit(h_fine, last3_p_full, h_vals, err_full_vals), 
              color=colors[2], linestyle=:dash, linewidth=2)
        
        scatter!(ax, h_vals, err_cut_vals, color=colors[3], marker=symbols[3],
                markersize=10, label="Cut error (p = $(last3_p_cut))")
        lines!(ax, h_fine, power_fit(h_fine, last3_p_cut, h_vals, err_cut_vals), 
              color=colors[3], linestyle=:dash, linewidth=2)
        
        # Add reference slopes
        h_ref = h_vals[end]
        err_ref = err_vals[end] * 0.5
        
        lines!(ax, h_fine, err_ref * (h_fine/h_ref).^2, color=:black, linestyle=:dot, 
              linewidth=1.5, label="O(h²)")
        lines!(ax, h_fine, err_ref * (h_fine/h_ref), color=:black, linestyle=:dashdot, 
              linewidth=1.5, label="O(h)")
        
        # Add legend
        axislegend(ax, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        
        # Save the figure
        save(joinpath(save_dir, "stefan_temperature_convergence.pdf"), fig, pt_per_unit=1)
        save(joinpath(save_dir, "stefan_temperature_convergence.png"), fig, px_per_unit=4)
        
        return fig
        
    else # style == :position
        # Figure for interface position error
        fig = Figure(resolution=(800, 600), fontsize=14)
        
        ax = Axis(
            fig[1, 1],
            xlabel = "Mesh size (h)",
            ylabel = "Absolute Error",
            title  = "Stefan Problem - Interface Position Convergence",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticks = IntervalsBetween(10),
            xticks = LogTicks(WilkinsonTicks(5))
        )

        # Plot position error data and fitted curve
        scatter!(ax, h_vals, pos_error_vals, color=colors[4], marker=symbols[4],
                markersize=12, label="Interface Position (p = $(last3_p_pos))")
        lines!(ax, h_fine, power_fit(h_fine, last3_p_pos, h_vals, pos_error_vals), 
              color=colors[4], linestyle=:dash, linewidth=2)
        
        # Add reference slopes
        h_ref = h_vals[end]
        err_ref = pos_error_vals[end] * 0.5
        
        lines!(ax, h_fine, err_ref * (h_fine/h_ref).^2, color=:black, linestyle=:dot, 
              linewidth=1.5, label="O(h²)")
        lines!(ax, h_fine, err_ref * (h_fine/h_ref), color=:black, linestyle=:dashdot, 
              linewidth=1.5, label="O(h)")
        
        # Add legend
        axislegend(ax, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        
        # Save the figure
        save(joinpath(save_dir, "stefan_position_convergence.pdf"), fig, pt_per_unit=1)
        save(joinpath(save_dir, "stefan_position_convergence.png"), fig, px_per_unit=4)
        
        return fig
    end
end

"""
    plot_stefan_temperature_profile(nx, T₀, k, lambda, t_final, run_dir; save_dir=".")

Plot temperature profile for the finest mesh solution compared with analytical solution.
"""
function plot_stefan_temperature_profile(
    nx::Int,
    T₀::Float64,
    k::Float64,
    lambda::Float64,
    t_final::Float64,
    run_dir::String;
    save_dir::String="."
)
    # Load the interface position data
    interface_file = joinpath(run_dir, @sprintf("interface_nx_%04d.csv", nx))
    if !isfile(interface_file)
        error("Interface position file not found: $interface_file")
    end
    
    df = CSV.read(interface_file, DataFrame)
    
    # Get final interface position
    xint_num = df.position[end]
    xint_ana = df.analytical[end]
    
    # Load the configuration
    config_file = joinpath(run_dir, "config.csv")
    if !isfile(config_file)
        error("Configuration file not found: $config_file")
    end
    
    config = CSV.read(config_file, DataFrame)
    lx = config[config.parameter .== "lx", :value][1]
    x0 = config[config.parameter .== "x0", :value][1]
    
    # Create spatial grid
    x_grid = range(x0, stop=lx, length=1000)
    
    # Calculate analytical solution
    u_analytical = zeros(length(x_grid))
    for (i, x) in enumerate(x_grid)
        if x < xint_ana
            u_analytical[i] = analytical_T(x, t_final, T₀, k, lambda)
        else
            u_analytical[i] = 0.0  # Outside liquid domain
        end
    end
    
    # Create figure
    fig = Figure(resolution=(900, 600), fontsize=14)
    
    ax = Axis(
        fig[1, 1],
        xlabel = "Position (x)",
        ylabel = "Temperature",
        title = "Stefan Problem Temperature Profile at t = $t_final"
    )
    
    # Plot analytical solution
    lines!(ax, x_grid, u_analytical, color=:red, linewidth=3, 
           label="Analytical Solution")
    
    # Mark analytical interface position
    vlines!(ax, xint_ana, color=:red, linewidth=2, linestyle=:dash,
            label="Analytical Interface")
    
    # Mark numerical interface position
    vlines!(ax, xint_num, color=:blue, linewidth=2, linestyle=:dash,
            label="Numerical Interface")
    
    # Add position error annotation
    pos_err = abs(xint_num - xint_ana)
    text!(ax, "Interface Position Error: $(round(pos_err, digits=6))",
         position = (0.5*lx, 0.1*T₀),
         fontsize = 14,
         color = :black)
    
    # Add legend
    axislegend(ax, position=:best, framevisible=true, backgroundcolor=(:white, 0.7))
    
    # Save the figure
    mkpath(save_dir)
    save(joinpath(save_dir, "stefan_temperature_profile.pdf"), fig, pt_per_unit=1)
    save(joinpath(save_dir, "stefan_temperature_profile.png"), fig, px_per_unit=4)
    
    return fig
end

"""
    plot_stefan_interface_position(nx, k, lambda, run_dir; save_dir=".")

Plot interface position evolution over time compared with analytical solution.
"""
function plot_stefan_interface_position(
    nx::Int,
    k::Float64,
    lambda::Float64,
    run_dir::String;
    save_dir::String="."
)
    # Load the interface position data
    interface_file = joinpath(run_dir, @sprintf("interface_nx_%04d.csv", nx))
    if !isfile(interface_file)
        error("Interface position file not found: $interface_file")
    end
    
    df = CSV.read(interface_file, DataFrame)
    
    # Create figure
    fig = Figure(resolution=(900, 600), fontsize=14)
    
    ax = Axis(
        fig[1, 1],
        xlabel = "Time",
        ylabel = "Interface Position",
        title = "Stefan Problem Interface Evolution"
    )
    
    # Plot numerical and analytical positions
    lines!(ax, df.time, df.position, color=:blue, linewidth=3, 
           label="Numerical Solution")
    lines!(ax, df.time, df.analytical, color=:red, linewidth=2, linestyle=:dash,
           label="Analytical: 2λ√(kt)")
    
    # Calculate error metrics
    abs_errors = abs.(df.position .- df.analytical)
    max_error = maximum(abs_errors)
    mean_error = mean(abs_errors)
    
    # Add error information to plot
    text!(ax, "Maximum error: $(round(max_error, digits=6))\nMean error: $(round(mean_error, digits=6))",
         position = (0.6*maximum(df.time), 0.2*maximum(df.position)),
         fontsize = 14,
         color = :black)
    
    # Add legend
    axislegend(ax, position=:best, framevisible=true, backgroundcolor=(:white, 0.7))
    
    # Save the figure
    mkpath(save_dir)
    save(joinpath(save_dir, "stefan_interface_evolution.pdf"), fig, pt_per_unit=1)
    save(joinpath(save_dir, "stefan_interface_evolution.png"), fig, px_per_unit=4)
    
    return fig
end

"""
    plot_stefan_results(nx_list, T₀, k, Stefan_number, run_dir; save_dir=".")

Create a comprehensive set of plots for the Stefan problem results.
"""
function plot_stefan_results(
    nx_list::Vector{Int},
    T₀::Float64,
    k::Float64,
    Stefan_number::Float64,
    run_dir::String;
    save_dir::String="."
)
    # Create output directory
    mkpath(save_dir)
    
    # Load summary and rates
    summary_file = joinpath(run_dir, "summary.csv")
    rates_file = joinpath(run_dir, "convergence_rates.csv")
    
    if !isfile(summary_file) || !isfile(rates_file)
        error("Missing summary or rates file in $run_dir")
    end
    
    df = CSV.read(summary_file, DataFrame)
    rates = CSV.read(rates_file, DataFrame)
    
    # Extract data
    h_vals = df.mesh_size
    err_vals = df.global_error
    err_full_vals = df.full_error
    err_cut_vals = df.cut_error
    pos_error_vals = df.position_error
    
    # Get convergence rates
    p_global = rates[rates.parameter .== "p_global", :value][1]
    p_full = rates[rates.parameter .== "p_full", :value][1]
    p_cut = rates[rates.parameter .== "p_cut", :value][1]
    p_pos = rates[rates.parameter .== "p_pos", :value][1]
    
    last3_p_global = rates[rates.parameter .== "last3_p_global", :value][1]
    last3_p_full = rates[rates.parameter .== "last3_p_full", :value][1]
    last3_p_cut = rates[rates.parameter .== "last3_p_cut", :value][1]
    last3_p_pos = rates[rates.parameter .== "last3_p_pos", :value][1]
    
    # Calculate lambda
    lambda = find_lambda(Stefan_number)
    
    # Create plots
    println("Creating temperature field convergence plot...")
    fig_temp = plot_stefan_convergence_results(
        h_vals, 
        err_vals, 
        err_full_vals, 
        err_cut_vals,
        pos_error_vals,
        (p_global, p_full, p_cut, p_pos),
        (last3_p_global, last3_p_full, last3_p_cut, last3_p_pos),
        style=:temperature,
        save_dir=save_dir
    )
    
    println("Creating interface position convergence plot...")
    fig_pos = plot_stefan_convergence_results(
        h_vals, 
        err_vals, 
        err_full_vals, 
        err_cut_vals,
        pos_error_vals,
        (p_global, p_full, p_cut, p_pos),
        (last3_p_global, last3_p_full, last3_p_cut, last3_p_pos),
        style=:position,
        save_dir=save_dir
    )
    
    # Use the finest mesh for temperature and interface plots
    nx_finest = maximum(nx_list)
    t_final = df.dt[1] # Use the first dt value as the final time (they should all be equal for convergence study)
    
    println("Creating temperature profile plot...")
    fig_profile = plot_stefan_temperature_profile(
        nx_finest,
        T₀,
        k,
        lambda,
        t_final,
        run_dir,
        save_dir=save_dir
    )
    
    println("Creating interface position plot...")
    fig_interface = plot_stefan_interface_position(
        nx_finest,
        k,
        lambda,
        run_dir,
        save_dir=save_dir
    )
    
    return (fig_temp, fig_pos, fig_profile, fig_interface)
end

# Example usage
function run_stefan_benchmark()
    # Define test parameters
    nx_list = [20, 40, 80, 160]  # Mesh sizes for convergence study
    T₀ = 1.0             # Initial temperature
    k = 1.0              # Thermal diffusivity
    Stefan_number = 1.0  # Stefan number
    
    # Calculate lambda and expected interface position
    lambda = find_lambda(Stefan_number)
    t_start = 0.01
    t_final = 0.1
    xint_final = interface_position(t_final, k, lambda)
    
    # Run convergence tests
    println("Running Stefan problem mesh convergence...")
    println("Analytical solution: λ = $lambda")
    println("Expected interface position at t=$t_final: x = $xint_final")
    
    results = run_stefan_1d_mesh_convergence(
        nx_list,
        T₀,
        k,
        Stefan_number,
        lx=10.0*xint_final,  # Domain size based on expected interface position
        x0=0.1,
        Tstart=t_start,
        Tend=t_final,
        norm=2,
        relative=false
    )
    
    # Create publication-quality plots
    h_vals, err_vals, err_full_vals, err_cut_vals, pos_error_vals,
    p_global, p_full, p_cut, p_pos,
    last3_p_global, last3_p_full, last3_p_cut, last3_p_pos, run_dir = results
    
    plots = plot_stefan_results(
        nx_list,
        T₀,
        k,
        Stefan_number,
        run_dir
    )
    
    return results, plots
end

# Run the benchmark when this file is executed directly
results, plots = run_stefan_benchmark()
