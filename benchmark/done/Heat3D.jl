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
    nz_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64,Float64},
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    lz::Float64=4.0,
    norm,
    output_dir::String="heat3D_convergence_results"
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

    for (i, (nx, ny, nz)) in enumerate(zip(nx_list, ny_list, nz_list))
        println("Running mesh test $i of $(length(nx_list)): nx=$nx, ny=$ny, nz=$nz")
        
        # Build mesh
        x0, y0, z0 = 0.0, 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))

        # Define the body
        circle = (x,y,z) -> (
            sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2) - radius
        )
            
        # Define capacity/operator
        capacity = Capacity(circle, mesh)
        operator = DiffusionOps(capacity)

        # BC + solver
        bc_boundary = Dirichlet(1.0)  # Homogeneous Dirichlet on domain boundaries
        bc0 = Dirichlet(1.0)
        bc_b = BorderConditions(Dict(
            :left   => bc0,
            :right  => bc0,
            :top    => bc0,
            :bottom => bc0
        ))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        u0ₒ = zeros((nx+1)*(ny+1)*(nz+1)) * 1.0
        u0ᵧ = zeros((nx+1)*(ny+1)*(nz+1)) * 1.0
        u0 = vcat(u0ₒ, u0ᵧ)

        Δt = 0.75*(lx/nx)^2
        Tend = 0.1

        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "BE") # Start by a backward Euler scheme

        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, "CN"; method=Base.:\)
        write_vtk("heat_3d", mesh, solver)

        # Compute errors
        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm)

        # Representative mesh size ~ 1 / min(nx, ny, nz)
        h = 1.0 / min(nx, ny, nz)
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
            nz = nz,
            global_error = global_err,
            full_error = full_err,
            cut_error = cut_err,
            empty_error = empty_err
        )
        
        # Use formatted nx/ny in filename for easier reading
        test_filename = @sprintf("mesh_%04dx%04dx%04d.csv", nx, ny, nz)
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
        nz = nz_list,
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
nx_list = [10, 20, 30, 40, 50, 60, 70]
ny_list = [10, 20, 30, 40, 50, 60, 70]
nz_list = [10, 20, 30, 40, 50, 60, 70]
radius, center = 1.0, (2.01, 2.01, 2.01)
function heat3d_analytical(x, y, z, t, center, radius, D, T0, Tb)
    # Calculate radial distance from center
    r = sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2)
    
    # Return boundary temperature if outside or on the sphere
    if r >= radius
        return Tb
    end
    
    # For numerical stability, avoid r=0 (center of sphere)
    if r < 1e-10
        # At center, all sin(πnr/R) terms become 0 except as r→0, 
        # use L'Hôpital's rule: sin(πnr/R)/(πr/R) → 1 as r→0
        # This gives 2∑(-1)^(n+1)exp(-Dπ²n²t/R²) at r=0
        sum_val = 0.0
        max_terms = 20
        for n in 1:max_terms
            term = ((-1)^(n+1)) * exp(-D * π^2 * n^2 * t / radius^2)
            sum_val += term
            if abs(term) < 1e-10
                break
            end
        end
        return Tb + (T0 - Tb) * 2 * sum_val
    end
    
    # Normal calculation for 0 < r < R
    max_terms = 20
    sum_val = 0.0
    
    for n in 1:max_terms
        # Calculate term in the series
        term = ((-1)^(n+1)) / n * sin(π * n * r / radius) * exp(-D * π^2 * n^2 * t / radius^2)
        sum_val += term
        
        # Early stopping if terms become very small
        if abs(term) < 1e-10
            break
        end
    end
    
    # Complete analytical solution
    T = Tb + (T0 - Tb) * (2 * radius / (π * r)) * sum_val
    
    return T
end


# Example parameters
center = (2.0, 2.0, 2.0)  # center of sphere
radius = 1.0              # radius of sphere
D = 1.0                   # thermal diffusivity
T0 = 0.0                # initial temperature
Tb = 1.0                # boundary temperature
t = 0.1                   # time

# Function to pass to the convergence analysis
u_analytical(x, y, z) = heat3d_analytical(x, y, z, t, center, radius, D, T0, Tb)

# Run the benchmark
output_dir = "heat3D_convergence_results"
#run_mesh_convergence(nx_list, ny_list, nz_list, radius, center, u_analytical, norm=Inf; output_dir=output_dir)

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
        ylabel = "Relative Error (L∞ norm)",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true,
        yminorticksvisible = true,
        xminorgridvisible = true,
        yminorgridvisible = true,
        xminorticks = IntervalsBetween(10),
        yminorticks = IntervalsBetween(10),
        xticks = LogTicks(WilkinsonTicks(5)),
        yticks = LogTicks(WilkinsonTicks(3)),
    )

    # Calculate fit lines
    h_fine = 10 .^ range(log10(minimum(h_vals)), log10(maximum(h_vals)), length=100)
    
    # Compute rates using only last 3 points
    function compute_last3_rate(h_data, err_data)
        # Get last 3 points (or fewer if not enough data)
        n = length(h_data)
        idx_start =  max(1, n-5)
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

    function compute_lastcut_rate(h_data, err_data)
        # Get last 3 points (or fewer if not enough data)
        n = length(h_data)
        idx_start = max(1, n-0)
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
    last3_p_cut = compute_lastcut_rate(h_vals, err_cut_vals)
    
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
             markersize=10, label="Full error (p = $(last3_p_full))")
    
    scatter!(ax, h_vals, err_cut_vals, color=colors[3], marker =symbol[3],
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
        maximum([maximum(err_vals), maximum(err_full_vals), maximum(err_cut_vals)])*1.2
    )
    
    # Add grid for readability
    ax.xgridvisible = true
    ax.ygridvisible = true
    
    # Save to file
    save(save_path, fig, pt_per_unit=1)
    
    return fig
end

function load_convergence_results(dir_path::String="heat3D_convergence_results")
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