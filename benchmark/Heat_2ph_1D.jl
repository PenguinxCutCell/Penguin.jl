using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using SpecialFunctions
using Roots
using CSV, DataFrames  # Replace DelimitedFiles with CSV and DataFrames
using Dates
using Printf
using Statistics

"""
Run mesh convergence study for diphasic heat transfer problem
"""
function run_diphasic_mesh_convergence(
    nx_list::Vector{Int},
    u1_analytical::Function,
    u2_analytical::Function;
    lx::Float64=8.0,
    x0::Float64=0.0,
    xint::Float64=4.0,
    Tend::Float64=0.1,
    He::Float64=0.5,
    D1::Float64=1.0,
    D2::Float64=1.0,
    norm::Real=2,
    relative::Bool=false,
    npts::Int=3,
    output_dir::String="diphasic_heat_convergence_results"
)
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Create a timestamp for this run
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    run_dir = joinpath(output_dir, timestamp)
    mkpath(run_dir)
    
    # Initialize storage arrays
    h_vals = Float64[]
    dt_vals = Float64[] # To track dt values
    
    # Global errors (all cells)
    err1_vals = Float64[]  # Phase 1
    err2_vals = Float64[]  # Phase 2
    err_combined_vals = Float64[]  # Combined phases
    
    # Full cell errors
    err1_full_vals = Float64[]  # Phase 1
    err2_full_vals = Float64[]  # Phase 2
    err_full_combined_vals = Float64[]  # Combined phases
    
    # Cut cell errors
    err1_cut_vals = Float64[]  # Phase 1
    err2_cut_vals = Float64[]  # Phase 2
    err_cut_combined_vals = Float64[]  # Combined phases

    # Save test configuration
    config_df = DataFrame(
        parameter = ["lx", "x0", "xint", "Tend", "He", "D1", "D2", "norm", "relative"],
        value = [lx, x0, xint, Tend, He, D1, D2, norm, relative]
    )
    CSV.write(joinpath(run_dir, "config.csv"), config_df)

    # For each mesh resolution
    for nx in nx_list
        println("\n===== Testing mesh size nx = $nx =====")
        
        # Build mesh
        mesh = Penguin.Mesh((nx,), (lx,), (x0,))
        
        # Define the body
        body = (x, _=0) -> (x - xint)
        body_c = (x, _=0) -> -(x - xint)
        
        # Define the capacity
        capacity = Capacity(body, mesh)
        capacity_c = Capacity(body_c, mesh)
        
        # Define the operators
        operator = DiffusionOps(capacity)
        operator_c = DiffusionOps(capacity_c)
        
        # Define the boundary conditions
        bc1 = Dirichlet(0.0)
        bc0 = Dirichlet(1.0)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc0, :bottom => bc1))
        
        # Interface conditions
        ic = InterfaceConditions(ScalarJump(1.0, He, 0.0), FluxJump(1.0, 1.0, 0.0))
        
        # Define the source term
        f1 = (x,y,z,t)->0.0
        f2 = (x,y,z,t)->0.0
        
        # Define diffusion coefficients
        D1_func = (x,y,z)->D1
        D2_func = (x,y,z)->D2
        
        # Define the phases
        Fluide_1 = Phase(capacity, operator, f1, D1_func)
        Fluide_2 = Phase(capacity_c, operator_c, f2, D2_func)
        
        # Initial condition
        u0ₒ1 = zeros(nx+1)
        u0ᵧ1 = zeros(nx+1)
        u0ₒ2 = ones(nx+1)
        u0ᵧ2 = ones(nx+1)
        
        u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)
        
        # Time step based on mesh size
        Δt = 0.5 * (lx/nx)^2
        push!(dt_vals, Δt)
        
        # Define the solver
        solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "CN")
        
        # Solve the problem
        solve_DiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, Δt, Tend, bc_b, ic, "CN"; method=Base.:\)
        
        # Compute errors
        (ana_sols, num_sols, global_errs, full_errs, cut_errs, empty_errs) = 
            check_convergence_diph(u1_analytical, u2_analytical, solver, capacity, capacity_c, norm, relative)
        
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
            combined_cut_error = err_cut_combined
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
    p1_global, _ = do_fit(log.(err1_vals))
    p2_global, _ = do_fit(log.(err2_vals))
    p_combined, _ = do_fit(log.(err_combined_vals))
    
    p1_full, _ = do_fit(log.(err1_full_vals))
    p2_full, _ = do_fit(log.(err2_full_vals))
    p_full_combined, _ = do_fit(log.(err_full_combined_vals))
    
    p1_cut, _ = do_fit(log.(err1_cut_vals))
    p2_cut, _ = do_fit(log.(err2_cut_vals))
    p_cut_combined, _ = do_fit(log.(err_cut_combined_vals))
    
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
    
    # Print convergence rates
    println("\n===== Convergence Rates =====")
    println("\n--- Global Errors (All Cells) ---")
    println("Phase 1: p = $p1_global (last $npts)")
    println("Phase 2: p = $p2_global (last $npts)")
    println("Combined: p = $p_combined (last $npts)")
    
    println("\n--- Full Cell Errors ---")
    println("Phase 1: p = $p1_full (last $npts)")
    println("Phase 2: p = $p2_full (last $npts)")
    println("Combined: p = $p_full_combined (last $npts)")
    
    println("\n--- Cut Cell Errors ---")
    println("Phase 1: p = $p1_cut (last $npts)")
    println("Phase 2: p = $p2_cut (last $npts)")
    println("Combined: p = $p_cut_combined (last $npts)")

    # Save final summary results to CSV
    df = DataFrame(
        mesh_size = h_vals,
        nx = nx_list,
        dt = dt_vals,
        phase1_error = err1_vals,
        phase2_error = err2_vals,
        combined_error = err_combined_vals,
        phase1_full_error = err1_full_vals,
        phase2_full_error = err2_full_vals,
        combined_full_error = err_full_combined_vals,
        phase1_cut_error = err1_cut_vals,
        phase2_cut_error = err2_cut_vals,
        combined_cut_error = err_cut_combined_vals
    )
    
    metadata = DataFrame(
        parameter = ["p1_global", "p2_global", "p_combined", 
                     "p1_full", "p2_full", "p_full_combined",
                     "p1_cut", "p2_cut", "p_cut_combined"],
        value = [p1_global, p2_global, p_combined,
                 p1_full, p2_full, p_full_combined,
                 p1_cut, p2_cut, p_cut_combined]
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
        (p1_global, p2_global, p_combined),
        (p1_full, p2_full, p_full_combined),
        (p1_cut, p2_cut, p_cut_combined),
        run_dir
    )
end

"""
Load most recent convergence results for diphasic heat transfer problem
"""
function load_latest_diphasic_convergence_results(output_dir="diphasic_heat_convergence_results")
    if !isdir(output_dir)
        error("Output directory not found: $output_dir")
    end
    
    # Get all subdirectories (runs)
    runs = filter(isdir, map(d -> joinpath(output_dir, d), readdir(output_dir)))
    
    # Sort by modification time (most recent first)
    sort!(runs, by=mtime, rev=true)
    
    if isempty(runs)
        error("No run directories found in $output_dir")
    end
    
    latest_run = runs[1]
    
    # Load summary and rates
    summary_file = joinpath(latest_run, "summary.csv")
    rates_file = joinpath(latest_run, "convergence_rates.csv")
    
    if !isfile(summary_file) || !isfile(rates_file)
        error("Missing summary or rates file in $latest_run")
    end
    
    df = CSV.read(summary_file, DataFrame)
    rates = CSV.read(rates_file, DataFrame)
    
    # Extract data
    h_vals = df.mesh_size
    
    # Extract global errors
    err1_vals = df.phase1_error
    err2_vals = df.phase2_error
    err_combined_vals = df.combined_error
    
    # Extract full cell errors
    err1_full_vals = df.phase1_full_error
    err2_full_vals = df.phase2_full_error
    err_full_combined_vals = df.combined_full_error
    
    # Extract cut cell errors
    err1_cut_vals = df.phase1_cut_error
    err2_cut_vals = df.phase2_cut_error
    err_cut_combined_vals = df.combined_cut_error
    
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
    
    println("Loaded results from $latest_run")
    
    return (
        h_vals,
        (err1_vals, err2_vals, err_combined_vals),
        (err1_full_vals, err2_full_vals, err_full_combined_vals),
        (err1_cut_vals, err2_cut_vals, err_cut_combined_vals),
        (p1_global, p2_global, p_combined),
        (p1_full, p2_full, p_full_combined),
        (p1_cut, p2_cut, p_cut_combined),
        latest_run
    )
end
function plot_diphasic_convergence_results(
    h_vals::Vector{Float64},
    global_errors::Tuple,
    full_errors::Tuple, 
    cut_errors::Tuple,
    global_rates::Tuple,
    full_rates::Tuple,
    cut_rates::Tuple;
    style::Symbol=:by_phase,  # :by_phase or :by_error_type
    norm::Int=2,
    save_dir::String="."
)
    # Unpack error values
    (err1_vals, err2_vals, err_combined_vals) = global_errors
    (err1_full_vals, err2_full_vals, err_full_combined_vals) = full_errors
    (err1_cut_vals, err2_cut_vals, err_cut_combined_vals) = cut_errors
    
    # Unpack rates
    (p1_global, p2_global, p_combined) = global_rates
    (p1_full, p2_full, p_full_combined) = full_rates
    (p1_cut, p2_cut, p_cut_combined) = cut_rates
    
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
    
    # Calculate last 3 point rates
    last3_p1_global = compute_last3_rate(h_vals, err1_vals)
    last3_p2_global = compute_last3_rate(h_vals, err2_vals)
    last3_p_combined = compute_last3_rate(h_vals, err_combined_vals)
    
    last3_p1_full = compute_last3_rate(h_vals, err1_full_vals)
    last3_p2_full = compute_last3_rate(h_vals, err2_full_vals)
    
    last3_p1_cut = compute_last3_rate(h_vals, err1_cut_vals)
    last3_p2_cut = compute_last3_rate(h_vals, err2_cut_vals)
    
    h_fine = 10 .^ range(log10(minimum(h_vals)), log10(maximum(h_vals)), length=100)
    
    if style == :by_error_type
        # Create figure with two panels side by side for Phase 1 and Phase 2
        fig = Figure(resolution=(1200, 600), fontsize=14)
        
        # Create axes for Phase 1 and Phase 2
        ax_phase1 = Axis(
            fig[1, 1],
            xlabel = "Mesh size (h)",
            ylabel = "Relative Error (L$norm norm)",
            title = "Phase 1",
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
        
        ax_phase2 = Axis(
            fig[1, 2],
            xlabel = "Mesh size (h)",
            ylabel = "Relative Error (L$norm norm)",
            title = "Phase 2",
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
        
        # Plot Phase 1 data - different error types
        scatter!(ax_phase1, h_vals, err1_vals, color=colors[1], marker=symbols[1],
                markersize=10, label="Global (p = $(last3_p1_global))")
        lines!(ax_phase1, h_fine, power_fit(h_fine, last3_p1_global, h_vals, err1_vals), 
              color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax_phase1, h_vals, err1_full_vals, color=colors[2], marker=symbols[2],
                markersize=10, label="Full (p = $(last3_p1_full))")
        lines!(ax_phase1, h_fine, power_fit(h_fine, last3_p1_full, h_vals, err1_full_vals), 
              color=colors[2], linestyle=:dash, linewidth=2)
        
        scatter!(ax_phase1, h_vals, err1_cut_vals, color=colors[3], marker=symbols[3],
                markersize=10, label="Cut (p = $(last3_p1_cut))")
        lines!(ax_phase1, h_fine, power_fit(h_fine, last3_p1_cut, h_vals, err1_cut_vals), 
              color=colors[3], linestyle=:dash, linewidth=2)
        
        # Plot Phase 2 data - different error types
        scatter!(ax_phase2, h_vals, err2_vals, color=colors[1], marker=symbols[1],
                markersize=10, label="Global (p = $(last3_p2_global))")
        lines!(ax_phase2, h_fine, power_fit(h_fine, last3_p2_global, h_vals, err2_vals), 
              color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax_phase2, h_vals, err2_full_vals, color=colors[2], marker=symbols[2],
                markersize=10, label="Full (p = $(last3_p2_full))")
        lines!(ax_phase2, h_fine, power_fit(h_fine, last3_p2_full, h_vals, err2_full_vals), 
              color=colors[2], linestyle=:dash, linewidth=2)
        
        scatter!(ax_phase2, h_vals, err2_cut_vals, color=colors[3], marker=symbols[3],
                markersize=10, label="Cut (p = $(last3_p2_cut))")
        lines!(ax_phase2, h_fine, power_fit(h_fine, last3_p2_cut, h_vals, err2_cut_vals), 
              color=colors[3], linestyle=:dash, linewidth=2)
        
        # Add reference slopes to both panels
        for ax in [ax_phase1, ax_phase2]
            # Reference slopes
            h_ref = h_vals[end]
            err_ref = minimum([minimum(err1_vals), minimum(err2_vals)]) * 5.0
            
            lines!(ax, h_fine, err_ref * (h_fine/h_ref).^2, color=:black, linestyle=:dot, 
                  linewidth=1.5, label="O(h²)")
            
            lines!(ax, h_fine, err_ref * (h_fine/h_ref), color=:black, linestyle=:dashdot, 
                  linewidth=1.5, label="O(h)")
        end
        
        # Add legends
        axislegend(ax_phase1, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        axislegend(ax_phase2, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        
        # Save the figure
        save(joinpath(save_dir, "diphasic_by_error_type.pdf"), fig, pt_per_unit=1)
        save(joinpath(save_dir, "diphasic_by_error_type.png"), fig, px_per_unit=4)
        
        return fig
        
    else # style == :by_phase
        # Create comprehensive figure with all results
        fig = Figure(resolution=(1200, 800), fontsize=14)
        
        # Global errors panel
        ax_comp_global = Axis(
            fig[1, 1],
            xlabel = "Mesh size (h)",
            ylabel = "Relative Error (L$norm norm)",
            title  = "Global Errors",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true, 
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
        )
        
        # Full cell errors panel
        ax_comp_full = Axis(
            fig[1, 2],
            xlabel = "Mesh size (h)",
            ylabel = "Relative Error (L$norm norm)",
            title  = "Full Cell Errors",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true, 
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
        )
        
        # Cut cell errors panel
        ax_comp_cut = Axis(
            fig[2, 1],
            xlabel = "Mesh size (h)",
            ylabel = "Relative Error (L$norm norm)",
            title  = "Cut Cell Errors",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true, 
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
        )
        
        # Combined errors panel - compare between cell types for each phase
        ax_comp_combined = Axis(
            fig[2, 2],
            xlabel = "Mesh size (h)",
            ylabel = "Relative Error (L$norm norm)",
            title  = "Combined Errors",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true, 
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
        )
        
        # Plot in global panel - different phases
        scatter!(ax_comp_global, h_vals, err1_vals, color=colors[1], marker=symbols[1], 
                markersize=10, label="Phase 1 (p = $(last3_p1_global))")
        lines!(ax_comp_global, h_fine, power_fit(h_fine, last3_p1_global, h_vals, err1_vals), 
              color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax_comp_global, h_vals, err2_vals, color=colors[2], marker=symbols[2], 
                markersize=10, label="Phase 2 (p = $(last3_p2_global))")
        lines!(ax_comp_global, h_fine, power_fit(h_fine, last3_p2_global, h_vals, err2_vals), 
              color=colors[2], linestyle=:dash, linewidth=2)
        
        scatter!(ax_comp_global, h_vals, err_combined_vals, color=colors[3], marker=symbols[3], 
                markersize=10, label="Combined (p = $(last3_p_combined))")
        lines!(ax_comp_global, h_fine, power_fit(h_fine, last3_p_combined, h_vals, err_combined_vals), 
              color=colors[3], linestyle=:dash, linewidth=2)
        
        # Plot in full cell panel - different phases
        scatter!(ax_comp_full, h_vals, err1_full_vals, color=colors[1], marker=symbols[1], 
                markersize=10, label="Phase 1 (p = $(last3_p1_full))")
        lines!(ax_comp_full, h_fine, power_fit(h_fine, last3_p1_full, h_vals, err1_full_vals), 
              color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax_comp_full, h_vals, err2_full_vals, color=colors[2], marker=symbols[2], 
                markersize=10, label="Phase 2 (p = $(last3_p2_full))")
        lines!(ax_comp_full, h_fine, power_fit(h_fine, last3_p2_full, h_vals, err2_full_vals), 
              color=colors[2], linestyle=:dash, linewidth=2)
        
        scatter!(ax_comp_full, h_vals, err_full_combined_vals, color=colors[3], marker=symbols[3], 
                markersize=10, label="Combined (p = $(p_full_combined))")
        lines!(ax_comp_full, h_fine, power_fit(h_fine, p_full_combined, h_vals, err_full_combined_vals), 
              color=colors[3], linestyle=:dash, linewidth=2)
        
        # Plot in cut cell panel - different phases
        scatter!(ax_comp_cut, h_vals, err1_cut_vals, color=colors[1], marker=symbols[1], 
                markersize=10, label="Phase 1 (p = $(last3_p1_cut))")
        lines!(ax_comp_cut, h_fine, power_fit(h_fine, last3_p1_cut, h_vals, err1_cut_vals), 
              color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax_comp_cut, h_vals, err2_cut_vals, color=colors[2], marker=symbols[2], 
                markersize=10, label="Phase 2 (p = $(last3_p2_cut))")
        lines!(ax_comp_cut, h_fine, power_fit(h_fine, last3_p2_cut, h_vals, err2_cut_vals), 
              color=colors[2], linestyle=:dash, linewidth=2)
        
        scatter!(ax_comp_cut, h_vals, err_cut_combined_vals, color=colors[3], marker=symbols[3], 
                markersize=10, label="Combined (p = $(p_cut_combined))")
        lines!(ax_comp_cut, h_fine, power_fit(h_fine, p_cut_combined, h_vals, err_cut_combined_vals), 
              color=colors[3], linestyle=:dash, linewidth=2)
        
        # Plot in combined panel - compare between cell types
        scatter!(ax_comp_combined, h_vals, err_combined_vals, color=:black, marker=:circle, 
                markersize=10, label="Global")
        lines!(ax_comp_combined, h_vals, err_combined_vals, color=:black)
        
        scatter!(ax_comp_combined, h_vals, err_full_combined_vals, color=:darkgreen, marker=:rect, 
                markersize=10, label="Full")
        lines!(ax_comp_combined, h_vals, err_full_combined_vals, color=:darkgreen)
        
        scatter!(ax_comp_combined, h_vals, err_cut_combined_vals, color=:purple, marker=:diamond, 
                markersize=10, label="Cut")
        lines!(ax_comp_combined, h_vals, err_cut_combined_vals, color=:purple)
        
        # Add reference slopes to all panels
        # Calculate reference values
        h_ref = h_vals[end]
        err_ref = err_combined_vals[end] * 0.5
        
        for ax in [ax_comp_global, ax_comp_full, ax_comp_cut, ax_comp_combined]
            lines!(ax, h_fine, err_ref * (h_fine/h_ref).^2, color=:black, linestyle=:dot, 
                  linewidth=1.5, label="O(h²)")
            lines!(ax, h_fine, err_ref * (h_fine/h_ref), color=:black, linestyle=:dashdot, 
                  linewidth=1.5, label="O(h)")
        end
        
        # Add legends
        axislegend(ax_comp_global, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        axislegend(ax_comp_full, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        axislegend(ax_comp_cut, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        axislegend(ax_comp_combined, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        
        # Save comprehensive plot
        save(joinpath(save_dir, "diphasic_comprehensive_convergence.pdf"), fig, pt_per_unit=1)
        save(joinpath(save_dir, "diphasic_comprehensive_convergence.png"), fig, px_per_unit=4)
        
        return fig
    end
end

# Create both plots (by_error_type and by_phase)
function plot_all_diphasic_convergence_plots(
    h_vals, 
    global_errors, 
    full_errors, 
    cut_errors, 
    global_rates, 
    full_rates, 
    cut_rates;
    norm::Int=2,
    save_dir::String="."
)
    fig_by_error = plot_diphasic_convergence_results(
        h_vals, 
        global_errors, 
        full_errors, 
        cut_errors, 
        global_rates, 
        full_rates, 
        cut_rates,
        style=:by_error_type,
        norm=norm,
        save_dir=save_dir
    )
    
    fig_by_phase = plot_diphasic_convergence_results(
        h_vals, 
        global_errors, 
        full_errors, 
        cut_errors, 
        global_rates, 
        full_rates, 
        cut_rates,
        style=:by_phase,
        norm=norm,
        save_dir=save_dir
    )
    
    return fig_by_error, fig_by_phase
end

# Example usage
function run_diphasic_convergence_analysis()
    # Define test parameters
    nx_list = [40, 80, 160, 320, 640, 1280]
    xint = 4.0
    Tend = 0.1
    He = 1.0
    D1 = 1.0
    D2 = 1.0
    
    # Define analytical solutions
    function T1(x)
        t = Tend
        x = x - xint
        return - He/(1+He*sqrt(D1/D2))*(erfc(x/(2*sqrt(D1*t))) - 2)
    end
    
    function T2(x)
        t = Tend
        x = x - xint
        return - He/(1+He*sqrt(D1/D2))*erfc(x/(2*sqrt(D2*t))) + 1
    end
    
    # Run convergence study
    println("Running diphasic heat transfer convergence study...")
    results = run_diphasic_mesh_convergence(
        nx_list,
        T1,
        T2,
        lx=8.0,
        x0=0.0,
        xint=xint,
        Tend=Tend,
        He=He,
        D1=D1,
        D2=D2,
        norm=2,
        relative=false,
        npts=3
    )
    
    h_vals, global_errors, full_errors, cut_errors, global_rates, full_rates, cut_rates, run_dir = results
    
    # Create publication-quality plots (both styles)
    println("Creating publication-quality plots...")
    fig_by_error, fig_by_phase = plot_all_diphasic_convergence_plots(
        h_vals,
        global_errors, 
        full_errors, 
        cut_errors,
        global_rates,
        full_rates,
        cut_rates,
        norm=2,
        save_dir=run_dir
    )
    
    println("\nDiphasic heat transfer convergence study completed!")
    println("Results saved to: $(run_dir)")
    
    return results, (fig_by_error, fig_by_phase)
end

# Run the analysis
results, plots = run_diphasic_convergence_analysis()