using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using SpecialFunctions
using Roots
using CSV, DataFrames
using Dates
using Printf
using Statistics

function run_mesh_convergence_moving(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius_mean::Float64, 
    radius_amp::Float64,
    period::Float64,
    center::Tuple{Float64,Float64},
    D::Float64,
    Φ_ana::Function,
    source_term::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    Tend::Float64=0.1, 
    norm::Real=2,
    output_dir::String="oscillating_disk_convergence_results"
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
    dt_vals = Float64[]  # Also track dt values
    
    # Save test configuration
    config_df = DataFrame(
        parameter = ["radius_mean", "radius_amp", "period", "center_x", "center_y", "D", "Tend"],
        value = [radius_mean, radius_amp, period, center[1], center[2], D, Tend]
    )
    CSV.write(joinpath(run_dir, "config.csv"), config_df)

    for (i, (nx, ny)) in enumerate(zip(nx_list, ny_list))
        println("Running mesh test $i of $(length(nx_list)): nx=$nx, ny=$ny")
        
        # Build mesh
        x0, y0 = 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
        
        # Calculate time step based on mesh size (CFL condition)
        Δt = 0.5*(lx/nx)^2
        push!(dt_vals, Δt)
        
        Tstart = Δt  # Small positive time to avoid t=0 singularity
        
        # Define the oscillating body as a level set function
        function oscillating_body(x, y, t)
            R_t = radius_mean + radius_amp * sin(2π * t / period)
            return (sqrt((x-center[1])^2 + (y-center[2])^2) - R_t)
        end
        
        # Define space-time mesh
        STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
        
        # Define capacity/operator
        capacity = Capacity(oscillating_body, STmesh)
        operator = DiffusionOps(capacity)

        # Define boundary conditions
        bc = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(
            :left   => bc,
            :right  => bc,
            :top    => bc,
            :bottom => bc
        ))
        
        # Dirichlet boundary condition for the disk interface
        robin_bc = Dirichlet((x, y, t) -> Φ_ana(x, y, t))
        
        # Define the phase with source term
        phase = Phase(capacity, operator, source_term, (x,y,z) -> D)
        
        # Initialize with analytical solution at t=Tstart
        u0ₒ = zeros((nx+1)*(ny+1))
        u0ᵧ = zeros((nx+1)*(ny+1))
        
        # Fill with initial analytical solution
        for i in 1:nx+1
            for j in 1:ny+1
                idx = (j-1)*(nx+1) + i
                x = mesh.nodes[1][i]
                y = mesh.nodes[2][j]
                u0ₒ[idx] = Φ_ana(x, y, Tstart)
            end
        end
        
        u0 = vcat(u0ₒ, u0ᵧ)
        
        # Define the solver
        solver = MovingDiffusionUnsteadyMono(phase, bc_b, robin_bc, Δt, u0, mesh, "BE")
        
        # Solve the problem
        solve_MovingDiffusionUnsteadyMono!(solver, phase, oscillating_body, Δt, Tstart, Tend, bc_b, robin_bc, mesh, "BE"; method=Base.:\)
        
        # Check errors based on last body position
        body_tend = (x, y,_=0) ->  begin
            # Calculate oscillating radius at Tend
            R_t = radius_mean + radius_amp * sin(2π * Tend / period)
            return sqrt((x - center[1])^2 + (y - center[2])^2) - R_t
        end
        
        capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)
        Φ_ana_tend(x, y) = Φ_ana(x, y, Tend)
        
        # Calculate errors
        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(Φ_ana_tend, solver, capacity_tend, norm)

        # Representative mesh size ~ 1 / min(nx, ny)
        h = lx / nx
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
            dt = Δt,
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
        dt = dt_vals,
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
        dt_vals,
        p_global,
        p_full,
        p_cut,
        run_dir
    )
end

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
        yticks = LogTicks(WilkinsonTicks(2)),
    )

    # Calculate fit lines
    h_fine = 10 .^ range(log10(minimum(h_vals)), log10(maximum(h_vals)), length=100)
    
    # Compute rates using only last 3 points
    function compute_last3_rate(h_data, err_data)
        # Get last 3 points (or fewer if not enough data)
        n = length(h_data)
        idx_start = max(1, n-2)
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

    function compute_last3_rate_cut(h_data, err_data)
        # Get last 3 points (or fewer if not enough data)
        n = length(h_data)
        idx_start = max(1, n-2)
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
    last3_p_cut = compute_last3_rate_cut(h_vals, err_cut_vals)
    
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
    scatter!(ax, h_vals, err_vals, color=colors[1], marker=symbol[1],
             markersize=10, label="Global error (p = $(last3_p_global))")
    
    scatter!(ax, h_vals, err_full_vals, color=colors[2], marker=symbol[2],
             markersize=10, label="Full error (p = $(last3_p_full))")
    
    scatter!(ax, h_vals, err_cut_vals, color=colors[3], marker=symbol[3],
             markersize=10, label="Cut error (p = $(last3_p_cut))")
    
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
        maximum([maximum(err_vals), maximum(err_full_vals), maximum(err_cut_vals)])*1.5
    )
    
    # Add grid for readability
    ax.xgridvisible = true
    ax.ygridvisible = true
    
    # Save to file
    save(save_path, fig, pt_per_unit=1)
    
    return fig
end

function load_latest_convergence_results(output_dir="oscillating_disk_convergence_results")
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
    err_vals = df.global_error
    err_full_vals = df.full_error
    err_cut_vals = df.cut_error
    err_empty_vals = df.empty_error
    dt_vals = df.dt
    
    # Get convergence rates
    p_global = rates[rates.parameter .== "p_global", :value][1]
    p_full = rates[rates.parameter .== "p_full", :value][1]
    p_cut = rates[rates.parameter .== "p_cut", :value][1]
    
    println("Loaded results from $latest_run")
    
    return (
        h_vals,
        err_vals,
        err_full_vals,
        err_cut_vals,
        err_empty_vals,
        dt_vals,
        p_global,
        p_full,
        p_cut,
        latest_run
    )
end

# Example usage - define parameters for oscillating disk test
nx_list = [8, 16, 32, 64, 128]  # Test mesh resolutions
ny_list = nx_list

# Disk parameters
radius_mean = 1.0
radius_amp = 0.5
period = 1.0
center = (2.0, 2.0)
D = 1.0

# Define analytical solution and source term for manufactured solution
function Φ_ana_disk(x, y, t)
    # Calculate radius from center
    r = sqrt((x - center[1])^2 + (y - center[2])^2)
    
    # Calculate oscillating radius at this time
    R_t = radius_mean + radius_amp * sin(2π * t / period)
    
    # Handle points outside the domain
    if r > R_t
        return 0.0
    end
    
    # Calculate analytical solution (1 + 0.5*sin(2πt/T))*cos(πx)*cos(πy)
    return (1 + 0.5 * sin(2π * t / period)) * cos(π * x) * cos(π * y)
end

function source_term_disk(x, y, z, t)
    # Calculate radius from center
    r = sqrt((x - center[1])^2 + (y - center[2])^2)
    
    # Calculate oscillating radius
    R_t = radius_mean + radius_amp * sin(2π * t / period)
    
    # Handle points outside domain
    if r > R_t
        return 0.0
    end
    
    # Calculate source term using the manufactured solution formula
    # f(x,y,t) = (π/T)*cos(πx)*cos(πy)*cos(2πt/T) + 2π²*D*(1 + 0.5*sin(2πt/T))*cos(πx)*cos(πy)
    term1 = (π / period) * cos(π * x) * cos(π * y) * cos(2π * t / period)
    term2 = 2 * π^2 * D * (1 + 0.5 * sin(2π * t / period)) * cos(π * x) * cos(π * y)
    
    return term1 + term2
end

"""
# Run convergence tests - uncomment to execute
results = run_mesh_convergence_moving(
    nx_list, ny_list,
    radius_mean, radius_amp, period,
    center, D,
    Φ_ana_disk, source_term_disk,
    norm=2
 )
"""

# To load and plot the most recent results
results = load_latest_convergence_results()
h_vals, err_vals, err_full_vals, err_cut_vals, _, _, p_global, p_full, p_cut, _ = results
 
fig = plot_convergence_results(
    h_vals, err_vals, err_full_vals, err_cut_vals, 
    p_global, p_full, p_cut,
    save_path="oscillating_disk_convergence_publication.pdf"
 )
 
display(fig)