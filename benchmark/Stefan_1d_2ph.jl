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
    find_lambda(SteL, SteS, alphaL, alphaS)

Find the dimensionless parameter λ for the two-phase Stefan problem by solving:
λ√π = SteL/(exp(λ²)erf(λ)) - SteS/(√(αL/αS)exp(αL/αS λ²)erf(√(αL/αS)λ))
"""
function find_lambda(SteL, SteS, alphaL, alphaS)
    ratio = sqrt(alphaL/alphaS)
    
    function f(lambda)
        term1 = SteL/(exp(lambda^2)*erf(lambda))
        term2 = SteS/(ratio*exp((alphaL/alphaS)*lambda^2)*erf(ratio*lambda))
        return lambda*sqrt(π) - term1 + term2
    end
    
    # Find the root (use a reasonable range for lambda)
    lambda = find_zero(f, (0.01, 5.0))
    return lambda
end

"""
    analytical_u_L(x, t, uL, lambda, alphaL)

Analytical solution for the liquid phase:
uL(x,t) = uL - uL (erf(x/(2√(αL t)))/erf(λ))
"""
function analytical_u_L(x, t, uL, lambda, alphaL)
    if t <= 0
        # Handle t=0 case
        return uL
    end
    return uL - uL * (erf(x/(2*sqrt(alphaL*t)))/erf(lambda))
end

"""
    analytical_u_S(x, t, uS, lambda, alphaL, alphaS)

Analytical solution for the solid phase:
uS(x,t) = uS - uS (erf(x/(2√(αS t)))/erf(√(αL/αS) λ))
"""
function analytical_u_S(x, t, uS, lambda, alphaL, alphaS)
    if t <= 0
        # Handle t=0 case
        return uS
    end
    ratio = sqrt(alphaL/alphaS)
    return uS - uS * (erf(x/(2*sqrt(alphaS*t)))/erf(ratio*lambda))
end

"""
    interface_position(t, lambda, alphaL)

Calculate the interface position at time t:
x_interface(t) = 2λ√(αL t)
"""
function interface_position(t, lambda, alphaL)
    return 2 * lambda * sqrt(alphaL * t)
end

"""
    run_stefan_1d_mesh_convergence(nx_list, uL, uS, alphaL, alphaS, SteL, SteS; kwargs...)

Run mesh convergence study for the 1D two-phase Stefan problem.
"""
function run_stefan_1d_mesh_convergence(
    nx_list::Vector{Int},
    uL::Float64,      # Temperature in liquid region
    uS::Float64,      # Temperature in solid region
    alphaL::Float64,  # Thermal diffusivity of liquid
    alphaS::Float64,  # Thermal diffusivity of solid
    SteL::Float64,    # Stefan number for liquid
    SteS::Float64;    # Stefan number for solid
    lx::Float64=2.0,  # Domain length
    x0::Float64=0.0,  # Domain start
    xint_init::Float64=nothing,  # Initial interface position (default to 0.05*lx if nothing)
    Tstart::Float64=nothing,  # Start time (calculated from xint_init if nothing)
    Tend::Float64=0.1,       # Final simulation time
    norm::Real=2,            # Norm for error calculation
    relative::Bool=true,     # Use relative error
    npts::Int=3,             # Number of points for fitting convergence rate
    output_dir::String="stefan_1d_convergence_results"
)
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Create a timestamp for this run
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    run_dir = joinpath(output_dir, timestamp)
    mkpath(run_dir)
    
    # Find lambda (analytical solution parameter)
    lambda = find_lambda(SteL, SteS, alphaL, alphaS)
    println("Analytical solution parameter λ = $lambda")
    
    # Set default initial interface position if not provided
    if isnothing(xint_init)
        xint_init = 0.05 * lx
    end
    println("Initial interface position: x = $xint_init")
    
    # Calculate the initial time corresponding to this interface position
    # From xint_init = 2*lambda*sqrt(alphaL*t_init) -> solve for t_init
    if isnothing(Tstart)
        Tstart = (xint_init / (2*lambda*sqrt(alphaL)))^2
    end
    println("Initial time t_init = $Tstart for interface at x = $xint_init")
    
    # Calculate final interface position
    xint_final = interface_position(Tend, lambda, alphaL)
    println("Analytical interface position at t=$Tend: x = $xint_final")
    
    # Initialize storage arrays
    h_vals = Float64[]
    dt_vals = Float64[] # To track dt values
    
    # Global errors (all cells)
    err1_vals = Float64[]  # Liquid phase
    err2_vals = Float64[]  # Solid phase
    err_combined_vals = Float64[]  # Combined phases
    
    # Full cell errors
    err1_full_vals = Float64[]  # Liquid phase
    err2_full_vals = Float64[]  # Solid phase
    err_full_combined_vals = Float64[]  # Combined phases
    
    # Cut cell errors
    err1_cut_vals = Float64[]  # Liquid phase
    err2_cut_vals = Float64[]  # Solid phase
    err_cut_combined_vals = Float64[]  # Combined phases
    
    # Interface position error
    pos_error_vals = Float64[]

    # Save test configuration
    config_df = DataFrame(
        parameter = ["uL", "uS", "alphaL", "alphaS", "SteL", "SteS", "lx", "x0", 
                     "xint_init", "Tstart", "Tend", "lambda", "xint_final", "norm", "relative"],
        value = [uL, uS, alphaL, alphaS, SteL, SteS, lx, x0, 
                 xint_init, Tstart, Tend, lambda, xint_final, norm, relative]
    )
    CSV.write(joinpath(run_dir, "config.csv"), config_df)


    # For each mesh resolution
    for nx in nx_list
        println("\n===== Testing mesh size nx = $nx =====")
        
    # Build mesh
        mesh = Penguin.Mesh((nx,), (lx,), (x0,))
        
        # Define the body (level set function)
        body_func = (x, t, _=0) -> (x - xint_init)
        body_c_func = (x, t, _=0) -> -(x - xint_init)
        
        # Define the Space-Time mesh
        Δt = 0.5 * (lx/nx)^2 / max(alphaL, alphaS)
        push!(dt_vals, Δt)
        STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
        
        # Define the capacity
        capacity = Capacity(body_func, STmesh)
        capacity_c = Capacity(body_c_func, STmesh)
        
        # Define the operators
        operator = DiffusionOps(capacity)
        operator_c = DiffusionOps(capacity_c)
        
        # Define the boundary conditions
        bc1 = Dirichlet(uL)  # Left boundary (liquid)
        bc0 = Dirichlet(uS)  # Right boundary (solid)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:bottom => bc1, :top => bc0))
        
        # Interface conditions with Stefan condition
        kL = alphaL  # Thermal conductivity of liquid
        kS = alphaS  # Thermal conductivity of solid
        L = 1.0      # Latent heat
        rho = 1.0    # Density
        
        ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(kL, kS, rho*L))
        
        # Define the source term (no source for basic Stefan problem)
        f1 = (x,y,z,t)->0.0
        f2 = (x,y,z,t)->0.0
        
        # Define diffusion coefficients
        K1_func = (x,y,z)->alphaL
        K2_func = (x,y,z)->alphaS
        
        # Define the phases
        Liquid = Phase(capacity, operator, f1, K1_func)
        Solid = Phase(capacity_c, operator_c, f2, K2_func)
        
        # Initial condition based on analytical solution
        u0ₒ1 = zeros(nx+1)  # Bulk field - Liquid
        u0ᵧ1 = zeros(nx+1)  # Interface field - Liquid
        u0ₒ2 = zeros(nx+1)  # Bulk field - Solid
        u0ᵧ2 = zeros(nx+1)  # Interface field - Solid
        
        # Fill with initial analytical solution
        for i in 1:nx+1
            x = mesh.nodes[1][i]
            if x < xint_init
                u0ₒ1[i] = analytical_u_L(x, Tstart, uL, lambda, alphaL)
                u0ᵧ1[i] = analytical_u_L(x, Tstart, uL, lambda, alphaL)
            else
                u0ₒ2[i] = analytical_u_S(x - xint_init, Tstart, uS, lambda, alphaL, alphaS)
                u0ᵧ2[i] = analytical_u_S(x - xint_init, Tstart, uS, lambda, alphaL, alphaS)
            end
        end
        
        u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)
        
        # Define the solver
        solver = MovingLiquidDiffusionUnsteadyDiph(Liquid, Solid, bc_b, ic, Δt, u0, mesh, "BE")
        
        # Newton parameters
        max_iter = 100
        tol = 1e-8
        reltol = 1e-8
        α = 1.0
        Newton_params = (max_iter, tol, reltol, α)
        
        # Solve the problem
        println("  Solving problem...")
        solver, residuals, xint_history = solve_MovingLiquidDiffusionUnsteadyDiph!(
            solver, Liquid, Solid, xint_init, Δt, Tstart, Tend, bc_b, ic, mesh, "BE";
            Newton_params=Newton_params, method=Base.:\
        )
        
        # Calculate final interface position from numerical solution
        xint_num_final = xint_history[end]
        pos_error = abs(xint_num_final - xint_final)
        push!(pos_error_vals, pos_error)
        
        # Save interface position history
        interface_df = DataFrame(
            time = range(0, Tend, length=length(xint_history)),
            position = xint_history,
            analytical = [interface_position(t, lambda, alphaL) for t in range(0, Tend, length=length(xint_history))]
        )
        interface_file = joinpath(run_dir, @sprintf("interface_nx_%04d.csv", nx))
        CSV.write(interface_file, interface_df)
        
        # Define analytical solution functions for error calculation
        function u1_analytical(x)
            if x < xint_num_final
                return analytical_u_L(x, Tend, uL, lambda, alphaL)
            else
                return 0.0  # Outside liquid domain
            end
        end
        
        function u2_analytical(x)
            if x >= xint_num_final
                return analytical_u_S(x - xint_num_final, Tend, uS, lambda, alphaL, alphaS)
            else
                return 0.0  # Outside solid domain
            end
        end
        
        # Compute errors
        println("  Computing errors...")
        body_tend = (x, _=0) -> (x - xint_num_final)
        body_c_tend = (x, _=0) -> -(x - xint_num_final)
        
        capacity_tend = Capacity(body_tend, mesh)
        capacity_c_tend = Capacity(body_c_tend, mesh)
        
        (ana_sols, num_sols, global_errs, full_errs, cut_errs, empty_errs) = 
            check_convergence_diph(u1_analytical, u2_analytical, solver, capacity_tend, capacity_c_tend, norm, relative)
        
        # Unpack error values
        (err1, err2, err_combined) = global_errs
        (err1_full, err2_full, err_full_combined) = full_errs
        (err1_cut, err2_cut, err_cut_combined) = cut_errs
        
        # Store mesh size and errors
        push!(h_vals, lx / nx)
        
        # Store global errors
        push!(err1_vals, err1)
        push!(err2_vals, err2)
        push!(err_combined_vals, err_combined)
        
        # Store full cell errors
        push!(err1_full_vals, err1_full)
        push!(err2_full_vals, err2_full)
        push!(err_full_combined_vals, err_full_combined)
        
        # Store cut cell errors
        push!(err1_cut_vals, err1_cut)
        push!(err2_cut_vals, err2_cut)
        push!(err_cut_combined_vals, err_cut_combined)
        
        # Save individual test result to CSV
        test_df = DataFrame(
            mesh_size = lx / nx,
            nx = nx,
            dt = Δt,
            phase1_error = err1,
            phase2_error = err2,
            combined_error = err_combined,
            phase1_full_error = err1_full,
            phase2_full_error = err2_full,
            combined_full_error = err_full_combined,
            phase1_cut_error = err1_cut,
            phase2_cut_error = err2_cut,
            combined_cut_error = err_cut_combined,
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

    function do_fit(log_err, use_last_n=npts)
        # Use only the last n points
        n = min(use_last_n, length(log_h))
        idx = length(log_h) - n + 1 : length(log_h)
        
        # Fit using only those points
        fit_result = curve_fit(fit_model, log_h[idx], log_err[idx], [-1.0, 0.0])
        return fit_result.param[1], fit_result.param[2]  # (p_est, c_est)
    end
    
    # Fit convergence rates for each phase and cell type (using last npts points)
    println("\nFitting convergence rates...")
    p1_global, _ = do_fit(log.(err1_vals))
    p2_global, _ = do_fit(log.(err2_vals))
    p_combined, _ = do_fit(log.(err_combined_vals))
    
    p1_full, _ = do_fit(log.(err1_full_vals))
    p2_full, _ = do_fit(log.(err2_full_vals))
    p_full_combined, _ = do_fit(log.(err_full_combined_vals))
    
    p1_cut, _ = do_fit(log.(err1_cut_vals))
    p2_cut, _ = do_fit(log.(err2_cut_vals))
    p_cut_combined, _ = do_fit(log.(err_cut_combined_vals))
    
    pos_rate, _ = do_fit(log.(pos_error_vals))
    
    # Round for display
    p1_global = round(p1_global, digits=2)
    p2_global = round(p2_global, digits=2)
    p_combined = round(p_combined, digits=2)
    
    p1_full = round(p1_full, digits=2)
    p2_full = round(p2_full, digits=2)
    p_full_combined = round(p_full_combined, digits=2)
    
    p1_cut = round(p1_cut, digits=2)
    p2_cut = round(p2_cut, digits=2)
    p_cut_combined = round(p_cut_combined, digits=2)
    
    pos_rate = round(pos_rate, digits=2)
    
    # Print convergence rates
    println("\n===== Convergence Rates =====")
    println("\n--- Global Errors (All Cells) ---")
    println("Liquid phase: p = $p1_global (last $npts)")
    println("Solid phase: p = $p2_global (last $npts)")
    println("Combined: p = $p_combined (last $npts)")
    
    println("\n--- Full Cell Errors ---")
    println("Liquid phase: p = $p1_full (last $npts)")
    println("Solid phase: p = $p2_full (last $npts)")
    println("Combined: p = $p_full_combined (last $npts)")
    
    println("\n--- Cut Cell Errors ---")
    println("Liquid phase: p = $p1_cut (last $npts)")
    println("Solid phase: p = $p2_cut (last $npts)")
    println("Combined: p = $p_cut_combined (last $npts)")
    
    println("\n--- Interface Position Error ---")
    println("Position: p = $pos_rate (last $npts)")

    # Save final summary results to CSV
    df = DataFrame(
        mesh_size = h_vals,
        nx = nx_list,
        dt = dt_vals,
        liquid_error = err1_vals,
        solid_error = err2_vals,
        combined_error = err_combined_vals,
        liquid_full_error = err1_full_vals,
        solid_full_error = err2_full_vals,
        combined_full_error = err_full_combined_vals,
        liquid_cut_error = err1_cut_vals,
        solid_cut_error = err2_cut_vals,
        combined_cut_error = err_cut_combined_vals,
        position_error = pos_error_vals
    )
    
    metadata = DataFrame(
        parameter = ["p1_global", "p2_global", "p_combined", 
                     "p1_full", "p2_full", "p_full_combined",
                     "p1_cut", "p2_cut", "p_cut_combined",
                     "pos_rate"],
        value = [p1_global, p2_global, p_combined,
                 p1_full, p2_full, p_full_combined,
                 p1_cut, p2_cut, p_cut_combined,
                 pos_rate]
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
        (err1_vals, err2_vals, err_combined_vals),
        (err1_full_vals, err2_full_vals, err_full_combined_vals),
        (err1_cut_vals, err2_cut_vals, err_cut_combined_vals),
        pos_error_vals,
        (p1_global, p2_global, p_combined),
        (p1_full, p2_full, p_full_combined),
        (p1_cut, p2_cut, p_cut_combined),
        pos_rate,
        run_dir
    )
end

"""
    plot_stefan_convergence_results(h_vals, global_errors, full_errors, cut_errors, 
                                    pos_error_vals, global_rates, full_rates, cut_rates, 
                                    pos_rate; style=:temperature, save_dir=".")

Create publication-quality convergence plots for Stefan problem.
Style can be :temperature (phase errors) or :position (interface position error).
"""
function plot_stefan_convergence_results(
    h_vals::Vector{Float64},
    global_errors::Tuple,
    full_errors::Tuple, 
    cut_errors::Tuple,
    pos_error_vals::Vector{Float64},
    global_rates::Tuple,
    full_rates::Tuple,
    cut_rates::Tuple,
    pos_rate::Float64;
    style::Symbol=:temperature,
    save_dir::String="."
)
    # Create directory if it doesn't exist
    mkpath(save_dir)
    
    # Scientific color palette (colorblind-friendly)
    colors = [:darkblue, :darkred, :black]
    symbols = [:circle, :rect, :diamond]
    
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

    # Function to compute rates using only last 3 points
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
    
    # Unpack error values
    (err1_vals, err2_vals, err_combined_vals) = global_errors
    (err1_full_vals, err2_full_vals, err_full_combined_vals) = full_errors
    (err1_cut_vals, err2_cut_vals, err_cut_combined_vals) = cut_errors
    
    # Unpack rates
    (p1_global, p2_global, p_combined) = global_rates
    (p1_full, p2_full, p_full_combined) = full_rates
    (p1_cut, p2_cut, p_cut_combined) = cut_rates
    
    # Calculate last 3 point rates
    last3_p1_global = compute_last3_rate(h_vals, err1_vals)
    last3_p2_global = compute_last3_rate(h_vals, err2_vals)
    last3_p_combined = compute_last3_rate(h_vals, err_combined_vals)
    last3_pos_rate = compute_last3_rate(h_vals, pos_error_vals)
    
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

        # Plot global error data and fitted curve
        scatter!(ax, h_vals, err1_vals, color=colors[1], marker=symbols[1],
                markersize=10, label="Liquid (p = $(last3_p1_global))")
        lines!(ax, h_fine, power_fit(h_fine, last3_p1_global, h_vals, err1_vals), 
              color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax, h_vals, err2_vals, color=colors[2], marker=symbols[2],
                markersize=10, label="Solid (p = $(last3_p2_global))")
        lines!(ax, h_fine, power_fit(h_fine, last3_p2_global, h_vals, err2_vals), 
              color=colors[2], linestyle=:dash, linewidth=2)
        
        scatter!(ax, h_vals, err_combined_vals, color=colors[3], marker=symbols[3],
                markersize=10, label="Combined (p = $(last3_p_combined))")
        lines!(ax, h_fine, power_fit(h_fine, last3_p_combined, h_vals, err_combined_vals), 
              color=colors[3], linestyle=:dash, linewidth=2)
        
        # Add reference slopes
        h_ref = h_vals[end]
        err_ref = err_combined_vals[end] * 0.5
        
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
        scatter!(ax, h_vals, pos_error_vals, color=:purple, marker=:star5,
                markersize=12, label="Interface Position (p = $(last3_pos_rate))")
        lines!(ax, h_fine, power_fit(h_fine, last3_pos_rate, h_vals, pos_error_vals), 
              color=:purple, linestyle=:dash, linewidth=2)
        
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
    plot_stefan_temperature_profiles(nx, uL, uS, alphaL, alphaS, lambda, t_final, run_dir; save_dir=".")

Plot temperature profiles for the finest mesh solution compared with analytical solution.
"""
function plot_stefan_temperature_profiles(
    nx::Int,
    uL::Float64,
    uS::Float64,
    alphaL::Float64,
    alphaS::Float64,
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
    config_file = joinpath(run_dir, "..", "config.csv")
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
            u_analytical[i] = analytical_u_L(x, t_final, uL, lambda, alphaL)
        else
            u_analytical[i] = analytical_u_S(x - xint_ana, t_final, uS, lambda, alphaL, alphaS)
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
         position = (0.5*lx, 0.1*uL),
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
    plot_stefan_interface_position(nx, alphaL, lambda, run_dir; save_dir=".")

Plot interface position evolution over time compared with analytical solution.
"""
function plot_stefan_interface_position(
    nx::Int,
    alphaL::Float64,
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
           label="Analytical: 2λ√(αt)")
    
    # Add legend
    axislegend(ax, position=:best, framevisible=true, backgroundcolor=(:white, 0.7))
    
    # Save the figure
    mkpath(save_dir)
    save(joinpath(save_dir, "stefan_interface_evolution.pdf"), fig, pt_per_unit=1)
    save(joinpath(save_dir, "stefan_interface_evolution.png"), fig, px_per_unit=4)
    
    return fig
end

"""
    plot_stefan_results(nx_list, uL, uS, alphaL, alphaS, SteL, SteS, t_final, results_dir; save_dir=".")

Create a comprehensive set of plots for the Stefan problem results.
"""
function plot_stefan_results(
    nx_list::Vector{Int},
    uL::Float64,
    uS::Float64,
    alphaL::Float64,
    alphaS::Float64,
    SteL::Float64,
    SteS::Float64,
    t_final::Float64,
    results_dir::String;
    save_dir::String="."
)
    # Create output directory
    mkpath(save_dir)
    
    # Find the most recent run if results_dir is a directory
    if isdir(results_dir)
        # Get all subdirectories
        runs = filter(isdir, map(d -> joinpath(results_dir, d), readdir(results_dir)))
        
        # Sort by modification time (most recent first)
        sort!(runs, by=mtime, rev=true)
        
        if isempty(runs)
            error("No run directories found in $results_dir")
        end
        
        results_dir = runs[1]
    end
    
    # Load results
    summary_file = joinpath(results_dir, "summary.csv")
    rates_file = joinpath(results_dir, "convergence_rates.csv")
    
    if !isfile(summary_file) || !isfile(rates_file)
        error("Missing summary or rates file in $results_dir")
    end
    
    df = CSV.read(summary_file, DataFrame)
    rates = CSV.read(rates_file, DataFrame)
    
    # Extract data
    h_vals = df.mesh_size
    
    # Extract errors
    err1_vals = df.liquid_error
    err2_vals = df.solid_error
    err_combined_vals = df.combined_error
    
    err1_full_vals = df.liquid_full_error
    err2_full_vals = df.solid_full_error
    err_full_combined_vals = df.combined_full_error
    
    err1_cut_vals = df.liquid_cut_error
    err2_cut_vals = df.solid_cut_error
    err_cut_combined_vals = df.combined_cut_error
    
    pos_error_vals = df.position_error
    
    # Get convergence rates
    p1_global = rates[rates.parameter .== "p1_global", :value][1]
    p2_global = rates[rates.parameter .== "p2_global", :value][1]
    p_combined = rates[rates.parameter .== "p_combined", :value][1]
    
    p1_full = rates[rates.parameter .== "p1_full", :value][1]
    p2_full = rates[rates.parameter .== "p2_full", :value][1]
    p_full_combined = rates[rates.parameter .== "p_full_combined", :value][1]
    
    p1_cut = rates[rates.parameter .== "p1_cut", :value][1]
    p2_cut = rates[rates.parameter .== "p2_cut", :value][1]
    p_cut_combined = rates[rates.parameter .== "p_cut_combined", :value][1]
    
    pos_rate = rates[rates.parameter .== "pos_rate", :value][1]
    
    # Calculate lambda
    lambda = find_lambda(SteL, SteS, alphaL, alphaS)
    
    # Create plots
    println("Creating temperature field convergence plot...")
    fig_temp = plot_stefan_convergence_results(
        h_vals, 
        (err1_vals, err2_vals, err_combined_vals),
        (err1_full_vals, err2_full_vals, err_full_combined_vals),
        (err1_cut_vals, err2_cut_vals, err_cut_combined_vals),
        pos_error_vals,
        (p1_global, p2_global, p_combined),
        (p1_full, p2_full, p_full_combined),
        (p1_cut, p2_cut, p_cut_combined),
        pos_rate,
        style=:temperature,
        save_dir=save_dir
    )
    
    println("Creating interface position convergence plot...")
    fig_pos = plot_stefan_convergence_results(
        h_vals, 
        (err1_vals, err2_vals, err_combined_vals),
        (err1_full_vals, err2_full_vals, err_full_combined_vals),
        (err1_cut_vals, err2_cut_vals, err_cut_combined_vals),
        pos_error_vals,
        (p1_global, p2_global, p_combined),
        (p1_full, p2_full, p_full_combined),
        (p1_cut, p2_cut, p_cut_combined),
        pos_rate,
        style=:position,
        save_dir=save_dir
    )
    
    # Use the finest mesh for temperature and interface plots
    nx_finest = maximum(nx_list)
    
    println("Creating temperature profile plot...")
    fig_profile = plot_stefan_temperature_profiles(
        nx_finest,
        uL,
        uS,
        alphaL,
        alphaS,
        lambda,
        t_final,
        results_dir,
        save_dir=save_dir
    )
    
    println("Creating interface position plot...")
    fig_interface = plot_stefan_interface_position(
        nx_finest,
        alphaL,
        lambda,
        results_dir,
        save_dir=save_dir
    )
    
    return (fig_temp, fig_pos, fig_profile, fig_interface)
end

# Update the benchmark function

function run_stefan_benchmark()
    # Define test parameters
    nx_list = [32, 64, 128, 256, 512]
    uL = 1.0      # Temperature in liquid region
    uS = 0.0      # Temperature in solid region
    alphaL = 1.0  # Thermal diffusivity of liquid
    alphaS = 1.0  # Thermal diffusivity of solid
    SteL = 1.0    # Stefan number for liquid
    SteS = 0.0    # Stefan number for solid
    lx = 2.0      # Domain length
    
    # Calculate lambda and expected interface position
    lambda = find_lambda(SteL, SteS, alphaL, alphaS)
    
    # Set initial interface at 5% of domain length
    xint_init = 0.05 * lx
    
    # Calculate the initial time corresponding to this interface position
    # From xint_init = 2*lambda*sqrt(alphaL*t_init) -> solve for t_init
    t_start = (xint_init / (2*lambda*sqrt(alphaL)))^2
    
    # Set final time to run for 0.1 time units after initial state
    t_final = t_start + 0.1
    
    xint_final = interface_position(t_final, lambda, alphaL)
    
    # Run convergence tests
    println("Running Stefan problem mesh convergence...")
    println("Analytical solution: λ = $lambda")
    println("Initial interface position: x = $xint_init at t = $t_start")
    println("Expected final interface position at t=$t_final: x = $xint_final")
    
    results = run_stefan_1d_mesh_convergence(
        nx_list,
        uL,
        uS,
        alphaL,
        alphaS,
        SteL,
        SteS,
        lx=lx,  
        x0=0.0,
        xint_init=xint_init,
        Tstart=t_start,
        Tend=t_final,
        norm=2,
        relative=true
    )
    
    # Create publication-quality plots
    h_vals, global_errors, full_errors, cut_errors, pos_error_vals,
    global_rates, full_rates, cut_rates, pos_rate, run_dir = results
    
    plots = plot_stefan_results(
        nx_list,
        uL,
        uS,
        alphaL,
        alphaS,
        SteL,
        SteS,
        t_final,
        run_dir
    )
    
    return results, plots
end


results, plots = run_stefan_benchmark()
