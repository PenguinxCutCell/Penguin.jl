using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using SpecialFunctions
using QuadGK
using DelimitedFiles
using Printf
using DataFrames  # Add this
using CSV         # Add this
using Dates       # Add this
# Define parameters
lx, ly = 8.0, 8.0
center = (lx/2, ly/2)
radius = lx/4
Tend = 0.1
Dg, Dl = 1.0, 1.0
cg0, cl0 = 1.0, 0.0
He = 0.5    
D = sqrt(Dg/Dl)

"""
Run mesh convergence study for 2D diphasic heat transfer problem with circular interface
"""
function run_diphasic_mesh_convergence_2D(
    nx_list::Vector{Int},
    u1_analytical::Function,
    u2_analytical::Function;
    lx::Float64=8.0,
    ly::Float64=8.0,
    x0::Float64=0.0,
    y0::Float64=0.0,
    radius::Float64=1.0,
    center::Tuple{Float64,Float64}=(4.0,4.0),
    Tend::Float64=0.5,
    He::Float64=1.0,
    D1::Float64=1.0,
    D2::Float64=1.0,
    norm::Real=2,
    relative::Bool=false,
    npts::Int=3,
    output_dir::String="diphasic_heat_results",
)
    # Initialize storage arrays
    h_vals = Float64[]
    
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

    # For each mesh resolution
    for nx in nx_list
        ny = nx  # Use square meshes
        println("\n===== Testing mesh size nx = ny = $nx =====")
        
        # Build mesh
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
        
        # Define the body functions
        circle = (x,y,_=0) -> sqrt((x-center[1])^2 + (y-center[2])^2) - radius
        circle_c = (x,y,_=0) -> -(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)
        
        # Define capacities
        capacity = Capacity(circle, mesh)
        capacity_c = Capacity(circle_c, mesh)
        
        # Define operators
        operator = DiffusionOps(capacity)
        operator_c = DiffusionOps(capacity_c)
        
        # Define boundary conditions
        bc = Dirichlet(0.0)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}())
        
        # Interface conditions
        ic = InterfaceConditions(ScalarJump(He, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))
        
        # Define source terms
        f1 = (x,y,z,t)->0.0
        f2 = (x,y,z,t)->0.0
        
        # Define diffusion coefficients
        D1_func = (x,y,z)->D1
        D2_func = (x,y,z)->D2
        
        # Define the phases
        Fluide_1 = Phase(capacity, operator, f1, D1_func)
        Fluide_2 = Phase(capacity_c, operator_c, f2, D2_func)
        
        # Initial condition
        u0ₒ1 = ones((nx+1)*(ny+1))
        u0ᵧ1 = ones((nx+1)*(ny+1))
        u0ₒ2 = zeros((nx+1)*(ny+1))
        u0ᵧ2 = zeros((nx+1)*(ny+1))
        u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)
        
        # Time step based on mesh size
        Δt = 0.5 * (lx/nx)^2  # Stability factor for diffusion
        
        # Define the solver
        solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "BE")
        
        # Solve the problem
        solve_DiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, Δt, Tend, bc_b, ic, "BE"; method=Base.:\)
        
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
                # Add this code after computing errors (around line 95)
        # and before the visualization code (if nx == maximum(nx_list))
        
        # Create output directory for this run
        output_dir = "diphasic_heat_results"
        mkpath(output_dir)
        
        # Create summary dataframe for current mesh size
        mesh_df = DataFrame(
            mesh_size = lx / nx,
            nx = nx,
            ny = ny,
            global_error_phase1 = err1,
            global_error_phase2 = err2,
            global_error_combined = err_combined,
            full_error_phase1 = err1_full,
            full_error_phase2 = err2_full,
            full_error_combined = err_full_combined,
            cut_error_phase1 = err1_cut,
            cut_error_phase2 = err2_cut,
            cut_error_combined = err_cut_combined
        )
        
        # Save individual mesh results
        mesh_filename = @sprintf("diphasic_mesh_%04dx%04d.csv", nx, ny)
        mesh_filepath = joinpath(output_dir, mesh_filename)
        CSV.write(mesh_filepath, mesh_df)
        println("  Saved mesh results to $(mesh_filepath)")
        
        # For the largest mesh, save a visualization of the solution
        if nx == maximum(nx_list)
            # Create a grid for visualization
            x = range(x0, stop=x0+lx, length=nx+1)
            y = range(y0, stop=y0+ly, length=ny+1)
            
            # Extract solutions
            (u1_ana, u2_ana) = ana_sols
            (u1_num, u2_num) = num_sols
            
            # Reshape solutions for 2D visualization
            u1_ana_2d = reshape(u1_ana, (ny+1, nx+1))
            u2_ana_2d = reshape(u2_ana, (ny+1, nx+1))
            u1_num_2d = reshape(u1_num, (ny+1, nx+1))
            u2_num_2d = reshape(u2_num, (ny+1, nx+1))
            
            # Create masks for inside/outside the circle
            mask_inside = zeros(Bool, ny+1, nx+1)
            mask_outside = zeros(Bool, ny+1, nx+1)
            
            for j in 1:ny+1
                for i in 1:nx+1
                    # Check if point is inside or outside circle
                    if circle(x[i], y[j]) <= 0
                        mask_inside[j, i] = true
                    else
                        mask_outside[j, i] = true
                    end
                end
            end
            
            # Apply masks to solutions
            u1_ana_masked = copy(u1_ana_2d)
            u2_ana_masked = copy(u2_ana_2d)
            u1_num_masked = copy(u1_num_2d)
            u2_num_masked = copy(u2_num_2d)
            
            u1_ana_masked[.!mask_inside] .= NaN
            u2_ana_masked[.!mask_outside] .= NaN
            u1_num_masked[.!mask_inside] .= NaN
            u2_num_masked[.!mask_outside] .= NaN
            
            # Compute errors
            err1_2d = abs.(u1_ana_2d .- u1_num_2d)
            err2_2d = abs.(u2_ana_2d .- u2_num_2d)
            err1_masked = copy(err1_2d)
            err2_masked = copy(err2_2d)
            err1_masked[.!mask_inside] .= NaN
            err2_masked[.!mask_outside] .= NaN
            
            # Plot analytical solution
            fig_ana = Figure(size=(1000, 800))
            ax_ana = Axis(fig_ana[1, 1], 
                title="Analytical Solution (t = $Tend)",
                xlabel="x",
                ylabel="y")
            
            hm1 = heatmap!(ax_ana, x, y, u1_ana_masked, colormap=:viridis, colorrange=(0, 1))
            hm2 = heatmap!(ax_ana, x, y, u2_ana_masked, colormap=:plasma, colorrange=(0, 1))
            
            # Add circle boundary
            theta = range(0, 2π, length=100)
            circle_x = center[1] .+ radius .* cos.(theta)
            circle_y = center[2] .+ radius .* sin.(theta)
            lines!(ax_ana, circle_x, circle_y, color=:white, linewidth=2)
            
            Colorbar(fig_ana[1, 2], hm1, label="Phase 1 concentration")
            Colorbar(fig_ana[1, 3], hm2, label="Phase 2 concentration")
            
            # Plot numerical solution
            fig_num = Figure(size=(1000, 800))
            ax_num = Axis(fig_num[1, 1], 
                title="Numerical Solution (t = $Tend, nx = $nx)",
                xlabel="x",
                ylabel="y")
            
            hm1_num = heatmap!(ax_num, x, y, u1_num_masked, colormap=:viridis, colorrange=(0, 1))
            hm2_num = heatmap!(ax_num, x, y, u2_num_masked, colormap=:plasma, colorrange=(0, 1))
            
            # Add circle boundary
            lines!(ax_num, circle_x, circle_y, color=:white, linewidth=2)
            
            Colorbar(fig_num[1, 2], hm1_num, label="Phase 1 concentration")
            Colorbar(fig_num[1, 3], hm2_num, label="Phase 2 concentration")
            
            # Plot error
            fig_err = Figure(size=(1000, 800))
            ax_err = Axis(fig_err[1, 1], 
                title="Absolute Error (t = $Tend, nx = $nx)",
                xlabel="x",
                ylabel="y")
            
            hm1_err = heatmap!(ax_err, x, y, err1_masked, colormap=:viridis)
            hm2_err = heatmap!(ax_err, x, y, err2_masked, colormap=:plasma)
            
            # Add circle boundary
            lines!(ax_err, circle_x, circle_y, color=:white, linewidth=2)
            
            Colorbar(fig_err[1, 2], hm1_err, label="Phase 1 error")
            Colorbar(fig_err[1, 3], hm2_err, label="Phase 2 error")
            
            display(fig_ana)
            display(fig_num)
            display(fig_err)
        end
    end
    
    # Fit convergence rates
    # Model for curve_fit
    function fit_model(x, p)
        p[1] .* x .+ p[2]
    end
    
    # Fit each on log scale: log(err) = p*log(h) + c
    log_h = log.(h_vals)
    
    function do_fit(log_err, use_last_n=3)
        # Use only the last n points (default 3)
        n = min(use_last_n, length(log_h))
        idx = length(log_h) - n + 1 : length(log_h)
        
        # Fit using only those points
        fit_result = curve_fit(fit_model, log_h[idx], log_err[idx], [-1.0, 0.0])
        return fit_result.param[1], fit_result.param[2]  # (p_est, c_est)
    end
    
    # Fit convergence rates for each phase and cell type (all points)
    p1_global_all, _ = do_fit(log.(err1_vals), length(err1_vals))
    p2_global_all, _ = do_fit(log.(err2_vals), length(err2_vals))
    p_combined_all, _ = do_fit(log.(err_combined_vals), length(err_combined_vals))
    
    p1_full_all, _ = do_fit(log.(err1_full_vals), length(err1_full_vals))
    p2_full_all, _ = do_fit(log.(err2_full_vals), length(err2_full_vals))
    p_full_combined_all, _ = do_fit(log.(err_full_combined_vals), length(err_full_combined_vals))
    
    p1_cut_all, _ = do_fit(log.(err1_cut_vals), length(err1_cut_vals))
    p2_cut_all, _ = do_fit(log.(err2_cut_vals), length(err2_cut_vals))
    p_cut_combined_all, _ = do_fit(log.(err_cut_combined_vals), length(err_cut_combined_vals))
    
    # Fit convergence rates using only last n points
    p1_global, _ = do_fit(log.(err1_vals), npts)
    p2_global, _ = do_fit(log.(err2_vals), npts)
    p_combined, _ = do_fit(log.(err_combined_vals), npts)
    
    p1_full, _ = do_fit(log.(err1_full_vals), npts)
    p2_full, _ = do_fit(log.(err2_full_vals), npts)
    p_full_combined, _ = do_fit(log.(err_full_combined_vals), npts)
    
    p1_cut, _ = do_fit(log.(err1_cut_vals), npts)
    p2_cut, _ = do_fit(log.(err2_cut_vals), npts)
    p_cut_combined, _ = do_fit(log.(err_cut_combined_vals), npts)
    
    # Round for display
    p1_global_all = round(p1_global_all, digits=2)
    p2_global_all = round(p2_global_all, digits=2)
    p_combined_all = round(p_combined_all, digits=2)
    
    p1_full_all = round(p1_full_all, digits=2)
    p2_full_all = round(p2_full_all, digits=2)
    p_full_combined_all = round(p_full_combined_all, digits=2)
    
    p1_cut_all = round(p1_cut_all, digits=2)
    p2_cut_all = round(p2_cut_all, digits=2)
    p_cut_combined_all = round(p_cut_combined_all, digits=2)
    
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
    println("Phase 1: p = $p1_global (last $npts), p = $p1_global_all (all)")
    println("Phase 2: p = $p2_global (last $npts), p = $p2_global_all (all)")
    println("Combined: p = $p_combined (last $npts), p = $p_combined_all (all)")
    
    println("\n--- Full Cell Errors ---")
    println("Phase 1: p = $p1_full (last $npts), p = $p1_full_all (all)")
    println("Phase 2: p = $p2_full (last $npts), p = $p2_full_all (all)")
    println("Combined: p = $p_full_combined (last $npts), p = $p_full_combined_all (all)")
    
    println("\n--- Cut Cell Errors ---")
    println("Phase 1: p = $p1_cut (last $npts), p = $p1_cut_all (all)")
    println("Phase 2: p = $p2_cut (last $npts), p = $p2_cut_all (all)")
    println("Combined: p = $p_cut_combined (last $npts), p = $p_cut_combined_all (all)")
    
    # Plot global errors
    fig_global = Figure()
    ax_global = Axis(
        fig_global[1, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Global Errors (All Cells)",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )
    
    scatter!(ax_global, h_vals, err1_vals, 
             label="Phase 1 (p = $p1_global)", 
             markersize=12, color=:blue)
    lines!(ax_global, h_vals, err1_vals, color=:blue)
    
    scatter!(ax_global, h_vals, err2_vals, 
             label="Phase 2 (p = $p2_global)", 
             markersize=12, color=:red)
    lines!(ax_global, h_vals, err2_vals, color=:red)
    
    scatter!(ax_global, h_vals, err_combined_vals, 
             label="Combined (p = $p_combined)", 
             markersize=12, color=:black)
    lines!(ax_global, h_vals, err_combined_vals, color=:black)
    
    # Add reference lines
    lines!(ax_global, h_vals, 0.1*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax_global, h_vals, 0.1*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    
    # Fix the error by getting the c_est value properly from the do_fit function
    # The issue is that you're trying to use "_" which is a write-only placeholder
    
    # Add fitted line for last 3 points (for Phase 1)
    last_3_idx = length(h_vals)-2:length(h_vals)
    h_range = exp.(range(log(h_vals[last_3_idx[1]]), log(h_vals[last_3_idx[end]]), length=100))
    
    # Get the coefficient for the fitted curve
    _, c_est_p1 = do_fit(log.(err1_vals), 3)
    c_est_p1 = exp(c_est_p1)  # Convert from log space
    
    # Use the coefficient to generate the fitted line
    err_fit1 = c_est_p1 * h_range.^p1_global
    lines!(ax_global, h_range, err_fit1, 
           color=:blue, linestyle=:dot, linewidth=2, 
           label="Last $npts fit (p = $p1_global)")
    
    axislegend(ax_global, position=:rb)
    display(fig_global)
    
    # Plot full cell errors
    fig_full = Figure()
    ax_full = Axis(
        fig_full[1, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Full Cell Errors",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )
    
    scatter!(ax_full, h_vals, err1_full_vals, label="Phase 1 ($p1_full)", markersize=12, color=:blue)
    lines!(ax_full, h_vals, err1_full_vals, color=:blue)
    
    scatter!(ax_full, h_vals, err2_full_vals, label="Phase 2 ($p2_full)", markersize=12, color=:red)
    lines!(ax_full, h_vals, err2_full_vals, color=:red)
    
    scatter!(ax_full, h_vals, err_full_combined_vals, label="Combined ($p_full_combined)", markersize=12, color=:black)
    lines!(ax_full, h_vals, err_full_combined_vals, color=:black)
    
    # Add reference lines
    lines!(ax_full, h_vals, 0.1*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax_full, h_vals, 0.1*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    
    axislegend(ax_full, position=:rb)
    display(fig_full)
    
    # Plot cut cell errors
    fig_cut = Figure()
    ax_cut = Axis(
        fig_cut[1, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Cut Cell Errors",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )
    
    scatter!(ax_cut, h_vals, err1_cut_vals, label="Phase 1 ($p1_cut)", markersize=12, color=:blue)
    lines!(ax_cut, h_vals, err1_cut_vals, color=:blue)
    
    scatter!(ax_cut, h_vals, err2_cut_vals, label="Phase 2 ($p2_cut)", markersize=12, color=:red)
    lines!(ax_cut, h_vals, err2_cut_vals, color=:red)
    
    scatter!(ax_cut, h_vals, err_cut_combined_vals, label="Combined ($p_cut_combined)", markersize=12, color=:black)
    lines!(ax_cut, h_vals, err_cut_combined_vals, color=:black)
    
    # Add reference lines
    lines!(ax_cut, h_vals, 0.1*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax_cut, h_vals, 0.1*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    
    axislegend(ax_cut, position=:rb)
    display(fig_cut)
    
    # Create a comprehensive convergence plot for both phases
    fig_comp = Figure(resolution=(1200, 800))
    
    # Global errors panel
    ax_comp_global = Axis(
        fig_comp[1, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Global Errors",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )
    
    # Full cell errors panel
    ax_comp_full = Axis(
        fig_comp[1, 2],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Full Cell Errors",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )
    
    # Cut cell errors panel
    ax_comp_cut = Axis(
        fig_comp[2, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Cut Cell Errors",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )
    
    # Combined errors panel
    ax_comp_combined = Axis(
        fig_comp[2, 2],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Combined Errors",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )
    
    # Plot in global panel
    scatter!(ax_comp_global, h_vals, err1_vals, label="Phase 1 ($p1_global)", markersize=10, color=:blue)
    lines!(ax_comp_global, h_vals, err1_vals, color=:blue)
    scatter!(ax_comp_global, h_vals, err2_vals, label="Phase 2 ($p2_global)", markersize=10, color=:red)
    lines!(ax_comp_global, h_vals, err2_vals, color=:red)
    
    # Plot in full cell panel
    scatter!(ax_comp_full, h_vals, err1_full_vals, label="Phase 1 ($p1_full)", markersize=10, color=:blue)
    lines!(ax_comp_full, h_vals, err1_full_vals, color=:blue)
    scatter!(ax_comp_full, h_vals, err2_full_vals, label="Phase 2 ($p2_full)", markersize=10, color=:red)
    lines!(ax_comp_full, h_vals, err2_full_vals, color=:red)
    
    # Plot in cut cell panel
    scatter!(ax_comp_cut, h_vals, err1_cut_vals, label="Phase 1 ($p1_cut)", markersize=10, color=:blue)
    lines!(ax_comp_cut, h_vals, err1_cut_vals, color=:blue)
    scatter!(ax_comp_cut, h_vals, err2_cut_vals, label="Phase 2 ($p2_cut)", markersize=10, color=:red)
    lines!(ax_comp_cut, h_vals, err2_cut_vals, color=:red)
    
    # Plot in combined panel
    scatter!(ax_comp_combined, h_vals, err_combined_vals, label="Global ($p_combined)", markersize=10, color=:black)
    lines!(ax_comp_combined, h_vals, err_combined_vals, color=:black)
    scatter!(ax_comp_combined, h_vals, err_full_combined_vals, label="Full ($p_full_combined)", markersize=10, color=:green)
    lines!(ax_comp_combined, h_vals, err_full_combined_vals, color=:green)
    scatter!(ax_comp_combined, h_vals, err_cut_combined_vals, label="Cut ($p_cut_combined)", markersize=10, color=:purple)
    lines!(ax_comp_combined, h_vals, err_cut_combined_vals, color=:purple)
    
    # Add reference slopes to all panels
    for ax in [ax_comp_global, ax_comp_full, ax_comp_cut, ax_comp_combined]
        lines!(ax, h_vals, 0.1*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
        lines!(ax, h_vals, 0.1*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    end
    
    # Add legends
    axislegend(ax_comp_global, position=:rb)
    axislegend(ax_comp_full, position=:rb)
    axislegend(ax_comp_cut, position=:rb)
    axislegend(ax_comp_combined, position=:rb)
    
    display(fig_comp)
    
        # Save errors and rates to a file
    open("diphasic_convergence_results.txt", "w") do io
        write(io, "# Diphasic Heat Transfer Convergence Study\n\n")
        
        # Write parameters
        write(io, "Parameters:\n")
        write(io, "  Mesh sizes: $nx_list\n")
        write(io, "  Norm: L$norm\n")
        write(io, "  Relative error: $relative\n")
        write(io, "  Final time: $Tend\n")
        write(io, "  Henry coefficient: $He\n")
        write(io, "  Diffusion coefficients: D1=$D1, D2=$D2\n\n")
        
        # Write convergence rates
        write(io, "Convergence Rates:\n")
        write(io, "  Global Errors:\n")
        write(io, "    Phase 1: $p1_global (last 3), $p1_global_all (all)\n")
        write(io, "    Phase 2: $p2_global (last 3), $p2_global_all (all)\n")
        write(io, "    Combined: $p_combined (last 3), $p_combined_all (all)\n\n")
        
        write(io, "  Full Cell Errors:\n")
        write(io, "    Phase 1: $p1_full (last 3), $p1_full_all (all)\n")
        write(io, "    Phase 2: $p2_full (last 3), $p2_full_all (all)\n")
        write(io, "    Combined: $p_full_combined (last 3), $p_full_combined_all (all)\n\n")
        
        write(io, "  Cut Cell Errors:\n")
        write(io, "    Phase 1: $p1_cut (last 3), $p1_cut_all (all)\n")
        write(io, "    Phase 2: $p2_cut (last 3), $p2_cut_all (all)\n")
        write(io, "    Combined: $p_cut_combined (last 3), $p_cut_combined_all (all)\n\n")
        
        # Write error data
        write(io, "Raw Data:\n")
        write(io, "h,err1,err2,err_combined,err1_full,err2_full,err_full_combined,err1_cut,err2_cut,err_cut_combined\n")
        
        for i in 1:length(h_vals)
            write(io, @sprintf("%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                h_vals[i],
                err1_vals[i], err2_vals[i], err_combined_vals[i],
                err1_full_vals[i], err2_full_vals[i], err_full_combined_vals[i],
                err1_cut_vals[i], err2_cut_vals[i], err_cut_combined_vals[i]
            ))
        end
    end

        # Add this before the return statement at the end of run_diphasic_mesh_convergence_2D
    
    # Save final summary results
    df = DataFrame(
        mesh_size = h_vals,
        nx = nx_list,
        ny = nx_list,  # Using nx_list since they're equal in your case
        global_error_phase1 = err1_vals,
        global_error_phase2 = err2_vals,
        global_error_combined = err_combined_vals,
        full_error_phase1 = err1_full_vals,
        full_error_phase2 = err2_full_vals,
        full_error_combined = err_full_combined_vals,
        cut_error_phase1 = err1_cut_vals,
        cut_error_phase2 = err2_cut_vals,
        cut_error_combined = err_cut_combined_vals
    )
    
    metadata = DataFrame(
        parameter = [
            "p_global_phase1", "p_global_phase2", "p_global_combined",
            "p_full_phase1", "p_full_phase2", "p_full_combined",
            "p_cut_phase1", "p_cut_phase2", "p_cut_combined",
            "Henry_coefficient", "D1", "D2", "final_time"
        ],
        value = [
            p1_global, p2_global, p_combined,
            p1_full, p2_full, p_full_combined,
            p1_cut, p2_cut, p_cut_combined,
            He, D1, D2, Tend
                    ]
                )
                
                # Write final results
                timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
                summary_file = joinpath(output_dir, "summary_$(timestamp).csv")
                rates_file = joinpath(output_dir, "convergence_rates_$(timestamp).csv")
                CSV.write(summary_file, df)
                CSV.write(rates_file, metadata)
                
                println("\nFinal results saved to $(output_dir)")
                println("  Summary: $(summary_file)")
                println("  Convergence rates: $(rates_file)")
    
    return (
        h_vals,
        (err1_vals, err2_vals, err_combined_vals),
        (err1_full_vals, err2_full_vals, err_full_combined_vals),
        (err1_cut_vals, err2_cut_vals, err_cut_combined_vals),
        (p1_global, p2_global, p_combined),     # Last 3 points
        (p1_full, p2_full, p_full_combined),    # Last 3 points
        (p1_cut, p2_cut, p_cut_combined),       # Last 3 points
        (p1_global_all, p2_global_all, p_combined_all),  # All points
        (p1_full_all, p2_full_all, p_full_combined_all), # All points
        (p1_cut_all, p2_cut_all, p_cut_combined_all)     # All points
    )
end

# Analytical solution functions
function compute_cg_analytical(x, y, t)
    r = sqrt((x-center[1])^2 + (y-center[2])^2)
    if r >= radius
        return 0.0
    end
    
    prefactor = (4*cg0*Dg*Dl*Dl*He)/(π^2*radius)
    Umax = 5.0/sqrt(Dg*t)
    val, _ = quadgk(u->cg_integrand(u, x, y, t), 0, Umax; atol=1e-6, rtol=1e-6)
    return prefactor*val
end

function compute_cl_analytical(x, y, t)
    r = sqrt((x-center[1])^2 + (y-center[2])^2)
    if r < radius
        return 0.0
    end
    
    prefactor = (2*cg0*Dg*sqrt(Dl)*He)/π
    Umax = 5.0/sqrt(Dg*t)
    val, _ = quadgk(u->cl_integrand(u, x, y, t), 0, Umax; atol=1e-6, rtol=1e-6)
    return prefactor*val
end

function Phi(u)
    term1 = Dg*sqrt(Dl)*besselj1(u*radius)*bessely0(D*u*radius)
    term2 = He*Dl*sqrt(Dg)*besselj0(u*radius)*bessely1(D*u*radius)
    return term1 - term2
end

function Psi(u)
    term1 = Dg*sqrt(Dl)*besselj1(u*radius)*besselj0(D*u*radius)
    term2 = He*Dl*sqrt(Dg)*besselj0(u*radius)*besselj1(D*u*radius)
    return term1 - term2
end

function cg_integrand(u, x, y, t)
    r = sqrt((x-center[1])^2 + (y-center[2])^2)
    Φu = Phi(u)
    Ψu = Psi(u)
    denom = u^2*(Φu^2 + Ψu^2)
    num   = exp(-Dg*u^2*t)*besselj0(u*r)*besselj1(u*radius)
    return iszero(denom) ? 0.0 : num/denom
end

function cl_integrand(u, x, y, t)
    r = sqrt((x-center[1])^2 + (y-center[2])^2)
    Φu = Phi(u)
    Ψu = Psi(u)
    denom = u*(Φu^2 + Ψu^2)
    term1 = besselj0(D*u*r)*Φu
    term2 = bessely0(D*u*r)*Ψu
    num   = exp(-Dg*u^2*t)*besselj1(u*radius)*(term1 - term2)
    return iszero(denom) ? 0.0 : num/denom
end

# Create analytical solution functions that capture the final time
final_time = Tend
cg_analytical(x, y) = compute_cg_analytical(x, y, final_time)
cl_analytical(x, y) = compute_cl_analytical(x, y, final_time)

# Define mesh sizes to test
nx_list = [16, 32, 64, 128, 256]


# Run the convergence study
results = run_diphasic_mesh_convergence_2D(
    nx_list,
    cg_analytical,
    cl_analytical;
    lx=lx,
    ly=ly,
    x0=0.0,
    y0=0.0,
    radius=radius,
    center=center,
    Tend=final_time,
    He=He,
    D1=Dg,
    D2=Dl,
    norm=2
)


using CairoMakie, CSV, DataFrames, Dates, Statistics

function load_diphasic_convergence_results(dir_path::String="diphasic_heat_results")
    # Find the most recent summary file
    summary_files = filter(file -> startswith(file, "summary_"), readdir(dir_path))
    
    if isempty(summary_files)
        error("No summary files found in $dir_path")
    end
    
    # Sort by timestamp in filename to get the most recent
    sort!(summary_files, by=file -> match(r"summary_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", file).captures[1])
    latest_summary = summary_files[end]
    
    # Load the data
    df = CSV.read(joinpath(dir_path, latest_summary), DataFrame)
    
    # Find corresponding rates file
    rates_file = replace(latest_summary, "summary" => "convergence_rates")
    rates_df = CSV.read(joinpath(dir_path, rates_file), DataFrame)
    
    # Extract values
    h_vals = df.mesh_size
    
    # Global errors
    err1_vals = df.global_error_phase1
    err2_vals = df.global_error_phase2
    err_combined_vals = df.global_error_combined
    
    # Full cell errors
    err1_full_vals = df.full_error_phase1
    err2_full_vals = df.full_error_phase2
    err_full_combined_vals = df.full_error_combined
    
    # Cut cell errors
    err1_cut_vals = df.cut_error_phase1
    err2_cut_vals = df.cut_error_phase2
    err_cut_combined_vals = df.cut_error_combined
    
    # Extract rates
    p1_global = rates_df[rates_df.parameter .== "p_global_phase1", :value][1]
    p2_global = rates_df[rates_df.parameter .== "p_global_phase2", :value][1]
    p_combined = rates_df[rates_df.parameter .== "p_global_combined", :value][1]
    
    p1_full = rates_df[rates_df.parameter .== "p_full_phase1", :value][1]
    p2_full = rates_df[rates_df.parameter .== "p_full_phase2", :value][1]
    p_full_combined = rates_df[rates_df.parameter .== "p_full_combined", :value][1]
    
    p1_cut = rates_df[rates_df.parameter .== "p_cut_phase1", :value][1]
    p2_cut = rates_df[rates_df.parameter .== "p_cut_phase2", :value][1]
    p_cut_combined = rates_df[rates_df.parameter .== "p_cut_combined", :value][1]
    
    return (
        h_vals,
        (err1_vals, err2_vals, err_combined_vals),
        (err1_full_vals, err2_full_vals, err_full_combined_vals),
        (err1_cut_vals, err2_cut_vals, err_cut_combined_vals),
        (p1_global, p2_global, p_combined),
        (p1_full, p2_full, p_full_combined),
        (p1_cut, p2_cut, p_cut_combined)
    )
end

function plot_diphasic_convergence(
    h_vals::Vector{Float64},
    global_errs::Tuple,
    full_errs::Tuple,
    cut_errs::Tuple,
    global_rates::Tuple,
    full_rates::Tuple,
    cut_rates::Tuple;
    style::Symbol=:by_error_type,  # :by_error_type or :by_phase
    save_path::String="diphasic_convergence_plot.pdf",
    output_dir::String="diphasic_heat_results"
)
    # Unpack error values
    (err1_vals, err2_vals, err_combined_vals) = global_errs
    (err1_full_vals, err2_full_vals, err_full_combined_vals) = full_errs
    (err1_cut_vals, err2_cut_vals, err_cut_combined_vals) = cut_errs
    
    # Function to compute power law curve using ALL points
    function power_fit_all(h, h_data, err_data)
        # Convert to log space
        log_h_data = log.(h_data)
        log_err_data = log.(err_data)
        
        # Simple linear regression in log space to find p (slope)
        h_mean = mean(log_h_data)
        err_mean = mean(log_err_data)
        numerator = sum((log_h_data .- h_mean) .* (log_err_data .- err_mean))
        denominator = sum((log_h_data .- h_mean).^2)
        p = numerator / denominator
        
        # Find intercept (log(C))
        log_C = err_mean - p * h_mean
        C = exp(log_C)
        
        # Return the fitted curve and the computed rate
        return C .* h.^p, round(p, digits=2)
    end
    
    # Function to compute power law curve using only the last 3 points
    function power_fit_last3(h, h_data, err_data)
        # Get the last 3 points (or fewer if not enough data)
        n = length(h_data)
        idx_start = max(1, n-2)
        last_three_h = h_data[idx_start:n]
        last_three_err = err_data[idx_start:n]
        
        # Convert to log space
        log_h_data = log.(last_three_h)
        log_err_data = log.(last_three_err)
        
        # Simple linear regression in log space to find p (slope)
        h_mean = mean(log_h_data)
        err_mean = mean(log_err_data)
        numerator = sum((log_h_data .- h_mean) .* (log_err_data .- err_mean))
        denominator = sum((log_h_data .- h_mean).^2)
        p = numerator / denominator
        
        # Find intercept (log(C))
        log_C = err_mean - p * h_mean
        C = exp(log_C)
        
        # Return the fitted curve and the computed rate
        return C .* h.^p, round(p, digits=2)
    end
    
    # Calculate all rates
    h_fine = 10 .^ range(log10(minimum(h_vals)), log10(maximum(h_vals)), length=100)
    
    # Calculate rates and curves - all points for global and full, last 3 for cut
    err1_curve, p1_global_all = power_fit_all(h_fine, h_vals, err1_vals)
    err2_curve, p2_global_all = power_fit_all(h_fine, h_vals, err2_vals)
    err_combined_curve, p_combined_all = power_fit_all(h_fine, h_vals, err_combined_vals)
    
    err1_full_curve, p1_full_all = power_fit_all(h_fine, h_vals, err1_full_vals)
    err2_full_curve, p2_full_all = power_fit_all(h_fine, h_vals, err2_full_vals)
    err_full_combined_curve, p_full_combined_all = power_fit_all(h_fine, h_vals, err_full_combined_vals)
    
    # Using last 3 points for cut cells as requested
    err1_cut_curve, p1_cut_last3 = power_fit_last3(h_fine, h_vals, err1_cut_vals)
    err2_cut_curve, p2_cut_last3 = power_fit_last3(h_fine, h_vals, err2_cut_vals)
    err_cut_combined_curve, p_cut_combined_last3 = power_fit_last3(h_fine, h_vals, err_cut_combined_vals)
    
    # Scientific color palette (colorblind-friendly)
    if style == :by_error_type
        # For style by error type, use different colors for different error types
        # and different markers for phases
        colors = [:darkblue, :darkred, :darkgreen]
        symbols = [:circle, :rect, :diamond]
        linestyles = [:solid, :dash, :dot]
        
        # Create figure with larger size and higher resolution for publication
        fig = Figure(resolution=(1200, 800), fontsize=14)
        
        # Create separate axes for Phase 1, Phase 2
        ax_phase1 = Axis(
            fig[1, 1],
            xlabel = "Mesh size (h)",
            ylabel = "Error (L₂ norm)",
            title = "Phase 1",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticks = IntervalsBetween(10),
            xticks = LogTicks(WilkinsonTicks(5)),
            yticks = (10.0 .^ [-4, -3, -2, -1], ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹"])
        )
        
        ax_phase2 = Axis(
            fig[1, 2],
            xlabel = "Mesh size (h)",
            ylabel = "Error (L₂ norm)",
            title = "Phase 2",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticks = IntervalsBetween(10),
            xticks = LogTicks(WilkinsonTicks(5)),
            yticks = (10.0 .^ [-4, -3, -2, -1], ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹"])
        )
        
        # Phase 1 - Plot data points with ALL-point rates for global and full, last 3 for cut
        scatter!(ax_phase1, h_vals, err1_vals, color=colors[1], marker=symbols[1],
                markersize=10, label="Global (p=$(p1_global_all))")
        lines!(ax_phase1, h_fine, err1_curve, 
            color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax_phase1, h_vals, err1_full_vals, color=colors[2], marker=symbols[2],
                markersize=10, label="Full (p=$(p1_full_all))")
        lines!(ax_phase1, h_fine, err1_full_curve, 
            color=colors[2], linestyle=:dash, linewidth=2)
        
        # Note the change to p1_cut_last3 here
        scatter!(ax_phase1, h_vals, err1_cut_vals, color=colors[3], marker=symbols[3],
                markersize=10, label="Cut (p=$(p1_cut_last3))")
        lines!(ax_phase1, h_fine, err1_cut_curve, 
            color=colors[3], linestyle=:dash, linewidth=2)
                
        # Phase 2 - Same approach
        scatter!(ax_phase2, h_vals, err2_vals, color=colors[1], marker=symbols[1],
                markersize=10, label="Global (p=$(p2_global_all))")
        lines!(ax_phase2, h_fine, err2_curve, 
            color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax_phase2, h_vals, err2_full_vals, color=colors[2], marker=symbols[2],
                markersize=10, label="Full (p=$(p2_full_all))")
        lines!(ax_phase2, h_fine, err2_full_curve, 
            color=colors[2], linestyle=:dash, linewidth=2)
        
        # Note the change to p2_cut_last3 here
        scatter!(ax_phase2, h_vals, err2_cut_vals, color=colors[3], marker=symbols[3],
                markersize=10, label="Cut (p=$(p2_cut_last3))")
        lines!(ax_phase2, h_fine, err2_cut_curve, 
            color=colors[3], linestyle=:dash, linewidth=2)

        # Reference slopes for each panel
        for ax in [ax_phase1, ax_phase2]
            # Reference slopes
            h_ref = h_vals[end]
            err_ref = 0.000005
            
            lines!(ax, h_fine, err_ref * (h_fine/h_ref).^2, color=:black, linestyle=:dot, 
                linewidth=1.5, label="O(h²)")
            
            lines!(ax, h_fine, err_ref * (h_fine/h_ref), color=:black, linestyle=:dashdot, 
                linewidth=1.5, label="O(h)")
        end
        
        # Add legends
        axislegend(ax_phase1, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        axislegend(ax_phase2, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        
    else # style == :by_phase
        # For style by phase, use different colors for different phases
        # and different markers for error types
        colors = [:darkblue, :darkred, :darkgreen]
        symbols = [:circle, :rect, :diamond]
        
        # Create figure with larger size and higher resolution for publication
        fig = Figure(resolution=(1200, 800), fontsize=14)
        
        # Create separate axes for Global, Full, and Cut errors
        ax_global = Axis(
            fig[1, 1],
            xlabel = "Mesh size (h)",
            ylabel = "Error (L₂ norm)",
            title = "Global Errors",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticks = IntervalsBetween(10)
        )
        
        ax_full = Axis(
            fig[1, 2],
            xlabel = "Mesh size (h)",
            ylabel = "Error (L₂ norm)",
            title = "Full Cell Errors",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticks = IntervalsBetween(10)
        )
        
        ax_cut = Axis(
            fig[2, 1:2],
            xlabel = "Mesh size (h)",
            ylabel = "Error (L₂ norm)",
            title = "Cut Cell Errors",
            xscale = log10,
            yscale = log10,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticks = IntervalsBetween(10)
        )
        
        # Global - Plot data points with ALL-point rates in the legend
        scatter!(ax_global, h_vals, err1_vals, color=colors[1], marker=symbols[1],
                markersize=10, label="Phase 1 (p=$(p1_global_all))")
        lines!(ax_global, h_fine, err1_curve, 
            color=colors[1], linestyle=:solid, linewidth=2)
        
        scatter!(ax_global, h_vals, err2_vals, color=colors[2], marker=symbols[2],
                markersize=10, label="Phase 2 (p=$(p2_global_all))")
        lines!(ax_global, h_fine, err2_curve, 
            color=colors[2], linestyle=:solid, linewidth=2)
        
        scatter!(ax_global, h_vals, err_combined_vals, color=colors[3], marker=symbols[3],
                markersize=10, label="Combined (p=$(p_combined_all))")
        lines!(ax_global, h_fine, err_combined_curve, 
            color=colors[3], linestyle=:solid, linewidth=2)
        
        # Full - Plot data points with ALL-point rates in the legend
        scatter!(ax_full, h_vals, err1_full_vals, color=colors[1], marker=symbols[1],
                markersize=10, label="Phase 1 (p=$(p1_full_all))")
        lines!(ax_full, h_fine, err1_full_curve, 
            color=colors[1], linestyle=:solid, linewidth=2)
        
        scatter!(ax_full, h_vals, err2_full_vals, color=colors[2], marker=symbols[2],
                markersize=10, label="Phase 2 (p=$(p2_full_all))")
        lines!(ax_full, h_fine, err2_full_curve, 
            color=colors[2], linestyle=:solid, linewidth=2)
        
        scatter!(ax_full, h_vals, err_full_combined_vals, color=colors[3], marker=symbols[3],
                markersize=10, label="Combined (p=$(p_full_combined_all))")
        lines!(ax_full, h_fine, err_full_combined_curve, 
            color=colors[3], linestyle=:solid, linewidth=2)
        
        # Cut - Plot data points with LAST 3 rates in the legend
        scatter!(ax_cut, h_vals, err1_cut_vals, color=colors[1], marker=symbols[1],
                markersize=10, label="Phase 1 (p=$(p1_cut_last3))")
        lines!(ax_cut, h_fine, err1_cut_curve, 
            color=colors[1], linestyle=:solid, linewidth=2)
        
        scatter!(ax_cut, h_vals, err2_cut_vals, color=colors[2], marker=symbols[2],
                markersize=10, label="Phase 2 (p=$(p2_cut_last3))")
        lines!(ax_cut, h_fine, err2_cut_curve, 
            color=colors[2], linestyle=:solid, linewidth=2)
        
        scatter!(ax_cut, h_vals, err_cut_combined_vals, color=colors[3], marker=symbols[3],
                markersize=10, label="Combined (p=$(p_cut_combined_last3))")
        lines!(ax_cut, h_fine, err_cut_combined_curve, 
            color=colors[3], linestyle=:solid, linewidth=2)
        
        # Reference slopes for each panel
        for ax in [ax_global, ax_full, ax_cut]
            # Reference slopes
            h_ref = h_vals[end]
            err_ref = 0.005
            
            lines!(ax, h_fine, err_ref * (h_fine/h_ref).^2, color=:black, linestyle=:dot, 
                linewidth=1.5, label="O(h²)")
            
            lines!(ax, h_fine, err_ref * (h_fine/h_ref), color=:black, linestyle=:dashdot, 
                linewidth=1.5, label="O(h)")
        end
        
        # Add legends
        axislegend(ax_global, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        axislegend(ax_full, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        axislegend(ax_cut, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
    end
    
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Save to file
    save(joinpath(output_dir, save_path), fig)
    
    return fig
end

# Example usage:
# Load the results from the CSV files
results = load_diphasic_convergence_results("diphasic_heat_results")


# Make plots in both styles
fig_by_error = plot_diphasic_convergence(
    results[1],  # h_vals
    results[2],  # global_errs
    results[3],  # full_errs
    results[4],  # cut_errs
    results[5],  # global_rates
    results[6],  # full_rates
    results[7],  # cut_rates
    style=:by_error_type,
    save_path="diphasic_by_error_type.pdf"
)

fig_by_phase = plot_diphasic_convergence(
    results[1],  # h_vals
    results[2],  # global_errs
    results[3],  # full_errs
    results[4],  # cut_errs
    results[5],  # global_rates
    results[6],  # full_rates
    results[7],  # cut_rates
    style=:by_phase,
    save_path="diphasic_by_phase.pdf"
)
# Display the plots
display(fig_by_error)
display(fig_by_phase)



using CairoMakie, DelimitedFiles

function plot_sherwood_numbers()
    # Directory containing the Sherwood number files
    sherwood_dir = "sherwood"
    
    # Grid sizes to process
    grid_sizes = [16, 32, 64, 128, 256]
    
    # Scientific color palette (colorblind-friendly)
    colors = [:steelblue, :indianred, :seagreen, :darkorchid, :goldenrod]
    markers = [:circle, :square, :diamond, :cross, :star5]
    
    # Create figure with two side-by-side panels
    fig = Figure(resolution=(1200, 600), fontsize=14)
    
    # Create two axes - left for full view, right for zoomed view
    ax_full = Axis(
        fig[1, 1],
        xlabel = "Time",
        ylabel = "Normalized Interface Flux",
        xgridvisible = true,
        ygridvisible = true,
        title = "Full Time Range"
    )
    
    ax_zoom = Axis(
        fig[1, 2],
        xlabel = "Time",
        ylabel = "Normalized Interface Flux",
        xgridvisible = true,
        ygridvisible = true,
        title = "Zoom at End (t=0.8 to 1.0)"
    )
    
    # Define the zoom range (last 20% of time)
    zoom_start = 0.8
    zoom_end = 1.02
    
    # Load and plot each file
    for (i, nx) in enumerate(grid_sizes)
        filename = joinpath(sherwood_dir, "Sherwood_number_$(nx).txt")
        
        # Skip if file doesn't exist
        if !isfile(filename)
            @warn "File not found: $filename"
            continue
        end
        
        # Read the Sherwood number data
        lines = readlines(filename)
        sherwood_data = Float64[]
        
        for line in lines
            line = strip(line)
            # Skip empty lines or comment lines
            if isempty(line) || startswith(line, "#") || startswith(line, "//")
                continue
            end
            
            try
                push!(sherwood_data, parse(Float64, line))
            catch
                # Skip lines that can't be parsed as float
            end
        end
        
        # Create time steps (0 to 1)
        n_points = length(sherwood_data)
        time_steps = range(0, 1, length=n_points)
        
        # Plot full data on left panel
        scatter!(ax_full, collect(time_steps), sherwood_data, 
                label="nx=$nx", 
                color=colors[i],
                marker=markers[i],
                markersize=8)
        
        # Find indices for zoom region
        zoom_indices = findall(t -> zoom_start <= t <= zoom_end, time_steps)
        
        # Plot zoomed data on right panel
        scatter!(ax_zoom, 
                collect(time_steps[zoom_indices]), 
                sherwood_data[zoom_indices], 
                color=colors[i],
                marker=markers[i],
                markersize=8)
    end
    
    # Add analytical reference line using the nx=256 data
    analytical_file = joinpath(sherwood_dir, "Sherwood_number_256.txt")
    if isfile(analytical_file)
        lines = readlines(analytical_file)
        analytical_data = Float64[]
        
        for line in lines
            line = strip(line)
            if isempty(line) || startswith(line, "#") || startswith(line, "//")
                continue
            end
            
            try
                push!(analytical_data, parse(Float64, line))
            catch
                # Skip lines that can't be parsed as float
            end
        end
        
        # Use time range from 0 to 1
        n_points = length(analytical_data)
        time_steps = range(0, 1, length=n_points)
        
        # Plot analytical data as a dashed line on both panels
        lines!(ax_full, collect(time_steps), analytical_data, 
               label="Analytical", 
               color=:black, 
               linestyle=:dash, 
               linewidth=3)
               
        # Find indices for zoom region
        zoom_indices = findall(t -> zoom_start <= t <= zoom_end, time_steps)
        
        # Plot zoomed analytical data
        lines!(ax_zoom, 
              collect(time_steps[zoom_indices]), 
              analytical_data[zoom_indices], 
              color=:black, 
              linestyle=:dash, 
              linewidth=3)
    end
    
    # Set specific limits for the zoomed plot
    xlims!(ax_zoom, zoom_start, zoom_end)
    
    # Add visual indicator of zoom area on the full plot
    vspan!(ax_full, zoom_start, zoom_end, color=(:gray, 0.2))
    
    # Add a legend with better positioning (only on the full plot)
    axislegend(ax_full, position=:rt, framevisible=true, backgroundcolor=(:white, 0.8))
    
    
    # Save the plot
    save(joinpath(sherwood_dir, "sherwood_number_comparison_with_zoom.pdf"), fig)
    save(joinpath(sherwood_dir, "sherwood_number_comparison_with_zoom.png"), fig)
    
    return fig
end
# Generate the plot
fig = plot_sherwood_numbers()
display(fig)