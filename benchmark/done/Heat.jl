using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using SpecialFunctions
using Roots
using CSV, DataFrames
using Dates
using Printf
using Statistics

function run_mesh_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    norm,
    output_dir::String="heat_convergence_results"
)
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Create a timestamp for this run
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    run_dir = joinpath(output_dir, timestamp)
    mkpath(run_dir)
    
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]

    for (i, (nx, ny)) in enumerate(zip(nx_list, ny_list))
        println("Running mesh test $i of $(length(nx_list)): nx=$nx, ny=$ny")
        
        # Build mesh
        x0, y0 = 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

        # Define the body
        circle = (x,y,_=0) -> (sqrt((x-center[1])^2 + (y-center[2])^2) - radius)
            
        # Define capacity/operator
        capacity = Capacity(circle, mesh)
        operator = DiffusionOps(capacity)

        # BC + solver
        w0 = 0.0
        wr = 1.0
        bc_boundary = Robin(1.0,1.0,1.0*wr)
        bc0 = Dirichlet(w0)
        bc_b = BorderConditions(Dict(
            :left   => bc0,
            :right  => bc0,
            :top    => bc0,
            :bottom => bc0
        ))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        u0ₒ = ones((nx+1)*(ny+1)) * w0
        u0ᵧ = zeros((nx+1)*(ny+1)) * w0
        u0 = vcat(u0ₒ, u0ᵧ)

        Δt = 0.25*(lx/nx)^2
        Tend = 0.1

        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "BE") # Start by a backward Euler scheme

        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, "CN"; method=Base.:\)

        # Compute errors
        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm)

        # Representative mesh size ~ 1 / min(nx, ny)
        h = 1.0 / min(nx, ny)
        push!(h_vals, h)
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        
        # Save individual test result to CSV
        test_df = DataFrame(
            mesh_size = h,
            nx = nx,
            ny = ny,
            global_error = global_err,
            full_error = full_err,
            cut_error = cut_err,
            empty_error = empty_err
        )
        
        # Use formatted nx/ny in filename for easier reading
        test_filename = @sprintf("mesh_%04dx%04d.csv", nx, ny)
        test_filepath = joinpath(run_dir, test_filename)
        CSV.write(test_filepath, test_df)
        println("  Saved result to $(test_filepath)")
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

    # Round
    p_global = round(p_global, digits=2)
    p_full   = round(p_full, digits=2)
    p_cut    = round(p_cut, digits=2)

    println("Estimated order of convergence (global) = ", p_global)
    println("Estimated order of convergence (full)   = ", p_full)
    println("Estimated order of convergence (cut)    = ", p_cut)
    
    # Save final summary results
    df = DataFrame(
        mesh_size = h_vals,
        nx = nx_list,
        ny = ny_list,
        global_error = err_vals,
        full_error = err_full_vals,
        cut_error = err_cut_vals,
        empty_error = err_empty_vals
    )
    
    metadata = DataFrame(
        parameter = ["p_global", "p_full", "p_cut"],
        value = [p_global, p_full, p_cut]
    )
    
    # Write final results
    summary_file = joinpath(run_dir, "summary.csv")
    rates_file = joinpath(run_dir, "convergence_rates.csv")
    CSV.write(summary_file, df)
    CSV.write(rates_file, metadata)
    
    println("Results saved to $(run_dir)")
    println("  Summary: $(summary_file)")
    println("  Convergence rates: $(rates_file)")

    # Plot in log-log scale
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "h",
        ylabel = "L$norm error",
        title  = "Convergence plot",
        xscale = log10,
        yscale = log10
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
    
    # Save the figure
    plot_file = joinpath(run_dir, "convergence_plot.png")
    save(plot_file, fig)
    
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
        run_dir
    )
end

# Example usage:
nx_list = [16, 32, 64, 128, 256]
ny_list = nx_list
radius, center = 1.0, (2.01, 2.01)
function radial_heat_(x, y)
    t=0.1
    R=1.0
    k=1.0
    a=1.0
    wr = 1.0
    w0 = 0.0

    function j0_zeros_robin(N, k, R; guess_shift = 0.25)
        # Define the function for alpha J1(alpha) - k R J0(alpha) = 0
        eq(alpha) = alpha * besselj1(alpha) - k * R * besselj0(alpha)
    
        zs = zeros(Float64, N)
        for m in 1:N
            # Approximate location around (m - guess_shift)*π
            x_left  = (m - guess_shift - 0.5) * π
            x_right = (m - guess_shift + 0.5) * π
            x_left  = max(x_left, 1e-6)  # Ensure bracket is positive
            zs[m]   = find_zero(eq, (x_left, x_right))
        end
        return zs
    end

    alphas = j0_zeros_robin(1000, k, R)
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
        An = 2.0 * k * R / ((k^2 * R^2 + αm^2) * besselj0(αm))
        s += An * exp(- a * αm^2 * t/R^2) * besselj0(αm * (r / R))
    end
    return (1.0 - s) * (wr - w0) + w0
end

# Run with organized output directory
output_dir = "heat_convergence_results"
#run_mesh_convergence(nx_list, ny_list, radius, center, radial_heat_, norm=2, output_dir=output_dir)

function plot_convergence_results(
    h_vals::Vector{Float64},
    err_vals::Vector{Float64},
    err_full_vals::Vector{Float64},
    err_cut_vals::Vector{Float64},
    p_global::Float64,
    p_full::Float64,
    p_cut::Float64;
    save_path::String="convergence_plot_publication.pdf"
)
    # Scientific color palette (colorblind-friendly)
    colors = [:darkblue, :darkred, :darkgreen]
    symbol = [:circle, :rect, :diamond]
    
    # Create figure with larger size and higher resolution for publication
    fig = Figure(resolution=(800, 600), fontsize=14)
    
    # Create axis with LaTeX formatting
    ax = Axis(
        fig[1, 1],
        xlabel = "Mesh size (h)",
        ylabel = "Relative Error (L₂ norm)",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true,
        yminorticksvisible = true,
        xminorgridvisible = true,
        yminorgridvisible = true,
        xminorticks = IntervalsBetween(10),
        yminorticks = IntervalsBetween(10),
        xticks = LogTicks(WilkinsonTicks(5)),
        yticks = LogTicks(WilkinsonTicks(5)),
    )

    # Calculate fit lines
    h_fine = 10 .^ range(log10(minimum(h_vals)), log10(maximum(h_vals)), length=100)
    
    # Compute rates using only last 3 points
    function compute_last3_rate(h_data, err_data)
        # Get last 3 points (or fewer if not enough data)
        n = length(h_data)
        idx_start = max(1, n-1)
        last_h = h_data[idx_start:n]
        last_err = err_data[idx_start:n]
        
        # Compute rate using linear regression on log-log data
        log_h = log.(last_h)
        log_err = log.(last_err)
        
        # Simple linear regression slope formula: slope = cov(x,y)/var(x)
        if length(log_h) > 1
            h_mean = mean(log_h)
            err_mean = mean(log_err)
            numerator = sum((log_h .- h_mean) .* (log_err .- err_mean))
            denominator = sum((log_h .- h_mean).^2)
            rate = numerator / denominator
            return round(rate, digits=2)
        else
            return p_global  # Fall back to original if not enough points
        end
    end
    
    # Calculate last 3 point rates
    last3_p_global = compute_last3_rate(h_vals, err_vals)
    last3_p_full = compute_last3_rate(h_vals, err_full_vals)
    last3_p_cut = compute_last3_rate(h_vals, err_cut_vals)
    
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

    # Plot data points with last 3 point rates in the legend
    scatter!(ax, h_vals, err_vals, color=colors[1], marker =symbol[1],
             markersize=10, label="Global error (p = $(last3_p_global))")
    
    scatter!(ax, h_vals, err_full_vals, color=colors[2], marker =symbol[2],
             markersize=10, label="Full error (p = $(last3_p_cut))")
    
    scatter!(ax, h_vals, err_cut_vals, color=colors[3], marker =symbol[3],
             markersize=10, label="Cut error (p = $(last3_p_full))")
    
    # Plot fitted curves using the last 3 point rates instead of original rates
    lines!(ax, h_fine, power_fit(h_fine, last3_p_global, h_vals, err_vals), 
           color=colors[1], linestyle=:dash, linewidth=2)
    
    lines!(ax, h_fine, power_fit(h_fine, last3_p_full, h_vals, err_full_vals), 
           color=colors[2], linestyle=:dash, linewidth=2)
    
    lines!(ax, h_fine, power_fit(h_fine, last3_p_cut, h_vals, err_cut_vals), 
           color=colors[3], linestyle=:dash, linewidth=2)
    
    # Reference slopes
    h_ref = h_vals[end]
    err_ref = err_vals[end] * 0.5
    
    lines!(ax, h_fine, err_ref * (h_fine/h_ref).^2, color=:black, linestyle=:dot, 
           linewidth=1.5, label="O(h²)")
    
    lines!(ax, h_fine, err_ref * (h_fine/h_ref), color=:black, linestyle=:dashdot, 
           linewidth=1.5, label="O(h)")
    
    # Add legend
    axislegend(ax, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
    
    # Set limits
    limits!(ax, 
        minimum(h_vals)*0.8, maximum(h_vals)*1.2,
        minimum([minimum(err_vals), minimum(err_full_vals), minimum(err_cut_vals)])*0.8,
        maximum([maximum(err_vals), maximum(err_full_vals), maximum(err_cut_vals)])*1.2
    )
    
    # Add grid for readability
    ax.xgridvisible = true
    ax.ygridvisible = true
    
    # Save to file
    save(save_path, fig, pt_per_unit=1)
    
    return fig
end
function load_convergence_results(dir_path::String="heat_convergence_results")
    # Check if directory exists
    if !isdir(dir_path)
        error("Directory $dir_path does not exist")
    end
    
    # Find all run directories (they're timestamps)
    run_dirs = filter(d -> isdir(joinpath(dir_path, d)), readdir(dir_path))
    
    if isempty(run_dirs)
        error("No run directories found in $dir_path")
    end
    
    # Sort by timestamp to get the most recent one
    sort!(run_dirs)
    latest_run = run_dirs[end]
    
    # Construct path to the latest run
    run_path = joinpath(dir_path, latest_run)
    
    # Check for summary file
    summary_file = joinpath(run_path, "summary.csv")
    rates_file = joinpath(run_path, "convergence_rates.csv")
    
    if !isfile(summary_file) || !isfile(rates_file)
        error("Missing summary.csv or convergence_rates.csv in $run_path")
    end
    
    # Load the data
    df = CSV.read(summary_file, DataFrame)
    rates_df = CSV.read(rates_file, DataFrame)
    
    # Extract values
    h_vals = df.mesh_size
    
    # Global, full, cut and empty errors
    err_vals = df.global_error
    err_full_vals = df.full_error
    err_cut_vals = df.cut_error
    err_empty_vals = df.empty_error
    
    # Extract rates
    p_global = rates_df[rates_df.parameter .== "p_global", :value][1]
    p_full = rates_df[rates_df.parameter .== "p_full", :value][1]
    p_cut = rates_df[rates_df.parameter .== "p_cut", :value][1]
    
    return (
        h_vals,
        err_vals,
        err_full_vals,
        err_cut_vals,
        err_empty_vals,
        p_global,
        p_full,
        p_cut,
        run_path
    )
end
# Load from the most recent run
results = load_convergence_results()

# Or load from a specific run directory
# results = load_convergence_results("/path/to/specific/run/directory")

# Extract the data
h_vals, err_vals, err_full_vals, err_cut_vals, _, p_global, p_full, p_cut, _ = results

# Create publication plot
fig = plot_convergence_results(
    h_vals, err_vals, err_full_vals, err_cut_vals, 
    p_global, p_full, p_cut,
    save_path="heat_convergence_publication.pdf"
)

# Display it
display(fig)