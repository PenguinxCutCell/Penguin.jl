using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using SpecialFunctions
using QuadGK
using DelimitedFiles
using Printf
using DataFrames
using CSV
using Dates
using Statistics

### Moving Vertical Interface Heat Equation Benchmark
# Parameters for the problem
lx, ly = 4.0, 4.0
D_plus = 1.0   # Diffusion coefficient in phase 1 (right)
D_minus = 1.0  # Diffusion coefficient in phase 2 (left)
cp_plus = 1.0  # Heat capacity in phase 1 (right)
cp_minus = 1.0 # Heat capacity in phase 2 (left)
Tend = 0.1

# Interface motion: s(t) = s0 + A*sin(ω t)
s0, A, ω = 2.0, 0.1, 2π
s(t) = s0 + A*sin(ω*t)     # Interface position
sdot(t) = A*ω*cos(ω*t)     # Interface velocity

"""
Run mesh convergence study for 2D diphasic heat transfer problem with moving vertical interface
"""
function run_moving_vertical_interface_convergence_2D(
    nx_list::Vector{Int},
    u1_analytical::Function,
    u2_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    x0::Float64=0.0,
    y0::Float64=0.0,
    Tend::Float64=1.0,
    D1::Float64=D_plus,
    D2::Float64=D_minus,
    norm::Real=2,
    relative::Bool=false,
    npts::Int=3,
    output_dir::String="moving_vertical_interface_results",
)
    # Initialize storage arrays
    h_vals = Float64[]
    
    # Error storage 
    err1_vals = Float64[]
    err2_vals = Float64[]
    err_combined_vals = Float64[]
    err1_full_vals = Float64[]
    err2_full_vals = Float64[]
    err_full_combined_vals = Float64[]
    err1_cut_vals = Float64[]
    err2_cut_vals = Float64[]
    err_cut_combined_vals = Float64[]

    # Create output directory
    mkpath(output_dir)

    # For each mesh resolution
    for nx in nx_list
        ny = nx  # Use square meshes
        println("\n===== Testing mesh size nx = ny = $nx =====")
        
        # Build mesh
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
        
        # Define the space-time mesh
        Δt = 0.5*(lx/nx)^2  # Time step based on mesh size
        STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
        
        # Level set function for the vertical plane:
        # body(x,y,t) > 0 for x > s(t)  (right side, phase 1)
        # body(x,y,t) < 0 for x < s(t)  (left side, phase 2)
        body = (x,y,t) -> (x - s(t))
        body_c = (x,y,t) -> (s(t) - x)
        
        # Define capacities with appropriate cp values
        capacity = Capacity(body, STmesh)
        capacity_c = Capacity(body_c, STmesh)
        
        # Define operators with appropriate D values
        operator = DiffusionOps(capacity)
        operator_c = DiffusionOps(capacity_c)
        
        # Define boundary conditions (Dirichlet = 0 on all boundaries)
        bc = Dirichlet(0.0)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
            :left => bc, :right => bc, :top => bc, :bottom => bc))
        
        # Interface conditions
        ic = InterfaceConditions(
            ScalarJump(1.0, 1.0, 0.0),    # [φ] = 0
            FluxJump(D1, D2, 0.0)         # [D∂nφ] = 0
        )
        
        # Define the source terms based on the manufactured solution
        function f1(x, y, z, t)
            if x >= s(t)  # right side, phase 1
                # Time derivative term
                time_term = -cp_plus * D2 * exp(-t) * g(x,y) * (cos(ω*t) + (x - s(t)))
                
                # Spatial derivative term
                laplacian_g = (8.0 - 6.0*x + 2.0*s(t)) * g2(y) - 2.0 * (x - s(t)) * g1(x)
                space_term = -D1 * exp(-t) * laplacian_g
                
                return time_term + space_term
            else
                return 0.0
            end
        end
        
        function f2(x, y, z, t)
            if x < s(t)  # left side, phase 2
                # Time derivative term
                time_term = -cp_minus * D1 * exp(-t) * g(x,y) * (cos(ω*t) + (x - s(t)))
                
                # Spatial derivative term
                laplacian_g = (8.0 - 6.0*x + 2.0*s(t)) * g2(y) - 2.0 * (x - s(t)) * g1(x)
                space_term = -D2 * exp(-t) * laplacian_g
                
                return time_term + space_term
            else
                return 0.0
            end
        end
        
        # Define diffusion coefficients
        K1_func = (x, y, z) -> D1
        K2_func = (x, y, z) -> D2
        
        # Define the phases
        Phase_1 = Phase(capacity, operator, f1, K1_func)
        Phase_2 = Phase(capacity_c, operator_c, f2, K2_func)
        
        # Define the initial condition (at t=0)
        u0ₒ1 = zeros((nx+1)*(ny+1))
        u0ᵧ1 = zeros((nx+1)*(ny+1))
        u0ₒ2 = zeros((nx+1)*(ny+1))
        u0ᵧ2 = zeros((nx+1)*(ny+1))
        
        # Fill initial conditions from the manufactured solution at t=0
        for i in 1:(nx+1)
            for j in 1:(ny+1)
                x = x0 + (i-1) * lx/nx
                y = y0 + (j-1) * ly/ny
                idx = (j-1)*(nx+1) + i
                
                if x >= s(0)  # Phase 1 (right)
                    u0ₒ1[idx] = u1_analytical(x, y, 0)
                    u0ᵧ1[idx] = u1_analytical(x, y, 0)
                else         # Phase 2 (left)
                    u0ₒ2[idx] = u2_analytical(x, y, 0)
                    u0ᵧ2[idx] = u2_analytical(x, y, 0)
                end
            end
        end
        
        u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)
        
        # Define the solver
        solver = MovingDiffusionUnsteadyDiph(Phase_1, Phase_2, bc_b, ic, Δt, u0, mesh, "BE")
        
        # Solve the problem
        solve_MovingDiffusionUnsteadyDiph!(solver, Phase_1, Phase_2, body, body_c, Δt, Tend, bc_b, ic, mesh, "BE"; method=Base.:\)
        
        # Check errors based on last body position
        body_tend = (x, y, _=0) -> (x - s(Tend))
        body_tend_c = (x, y, _=0) -> (s(Tend) - x)
        
        capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)
        capacity_tend_c = Capacity(body_tend_c, mesh; compute_centroids=false)
        
        # Compute errors
        (ana_sols, num_sols, global_errs, full_errs, cut_errs, empty_errs) = 
            check_convergence_diph(u1_analytical, u2_analytical, solver, capacity_tend, capacity_tend_c, norm, relative)
        
        # Store results and continue with error processing
        push!(h_vals, lx / nx)
        (err1, err2, err_combined) = global_errs
        push!(err1_vals, err1)
        push!(err2_vals, err2)
        push!(err_combined_vals, err_combined)
        
        (err1_full, err2_full, err_full_combined) = full_errs
        push!(err1_full_vals, err1_full)
        push!(err2_full_vals, err2_full)
        push!(err_full_combined_vals, err_full_combined)
        
        (err1_cut, err2_cut, err_cut_combined) = cut_errs
        push!(err1_cut_vals, err1_cut)
        push!(err2_cut_vals, err2_cut)
        push!(err_cut_combined_vals, err_cut_combined)
        
        # Save individual mesh results to CSV
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
        
        # Save individual mesh results to CSV
        mesh_filename = @sprintf("vertical_mesh_%04dx%04d.csv", nx, ny)
        mesh_filepath = joinpath(output_dir, mesh_filename)
        CSV.write(mesh_filepath, mesh_df)
        println("  Saved mesh results to $(mesh_filepath)")
        
        
    end
    
    # Fit convergence rates using curve_fit
    function fit_model(x, p)
        p[1] .* x .+ p[2]
    end
    
    # Fit on log scale: log(err) = p*log(h) + c
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

    # Create comprehensive convergence plot for both phases
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
    
    save(joinpath(output_dir, "convergence_plots.png"), fig_comp)
    display(fig_comp)
    
    # Save errors and rates to a text file
    open(joinpath(output_dir, "vertical_interface_convergence_results.txt"), "w") do io
        write(io, "# Moving Vertical Interface Heat Transfer Convergence Study\n\n")
        
        # Write parameters
        write(io, "Parameters:\n")
        write(io, "  Mesh sizes: $nx_list\n")
        write(io, "  Norm: L$norm\n")
        write(io, "  Relative error: $relative\n")
        write(io, "  Final time: $Tend\n")
        write(io, "  Diffusion coefficients: D1=$D1, D2=$D2\n")
        write(io, "  Interface motion: s(t) = $s0 + $A*sin($ω*t)\n\n")
        
        # Write convergence rates
        write(io, "Convergence Rates:\n")
        write(io, "  Global Errors:\n")
        write(io, "    Phase 1: $p1_global (last $npts), $p1_global_all (all)\n")
        write(io, "    Phase 2: $p2_global (last $npts), $p2_global_all (all)\n")
        write(io, "    Combined: $p_combined (last $npts), $p_combined_all (all)\n\n")
        
        write(io, "  Full Cell Errors:\n")
        write(io, "    Phase 1: $p1_full (last $npts), $p1_full_all (all)\n")
        write(io, "    Phase 2: $p2_full (last $npts), $p2_full_all (all)\n")
        write(io, "    Combined: $p_full_combined (last $npts), $p_full_combined_all (all)\n\n")
        
        write(io, "  Cut Cell Errors:\n")
        write(io, "    Phase 1: $p1_cut (last $npts), $p1_cut_all (all)\n")
        write(io, "    Phase 2: $p2_cut (last $npts), $p2_cut_all (all)\n")
        write(io, "    Combined: $p_cut_combined (last $npts), $p_cut_combined_all (all)\n\n")
        
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
    
    # Save final summary results to CSV
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
    df = DataFrame(
        mesh_size = h_vals,
        nx = nx_list,
        ny = nx_list,
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
            "D_plus", "D_minus", "cp_plus", "cp_minus", "final_time",
            "interface_s0", "interface_A", "interface_omega"
        ],
        value = [
            p1_global, p2_global, p_combined,
            p1_full, p2_full, p_full_combined,
            p1_cut, p2_cut, p_cut_combined,
            D1, D2, cp_plus, cp_minus, Tend,
            s0, A, ω
        ]
    )
    
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

# Helper functions for the manufactured solution
g1(x) = x * (4.0 - x)
g2(y) = y * (4.0 - y)
g(x, y) = g1(x) * g2(y)
theta(t) = exp(-t)

# Manufactured solution for each phase
function compute_u1_analytical(x, y, t)
    if x >= s(t)  # Phase 1 (right)
        return D_minus * (x - s(t)) * g(x, y) * theta(t)
    else
        return 0.0
    end
end

function compute_u2_analytical(x, y, t)
    if x < s(t)  # Phase 2 (left)
        return D_plus * (x - s(t)) * g(x, y) * theta(t)
    else
        return 0.0
    end
end

# Create analytical solution functions that capture time t
u1_analytical(x, y, t=Tend) = compute_u1_analytical(x, y, t)
u2_analytical(x, y, t=Tend) = compute_u2_analytical(x, y, t)

# Define mesh sizes to test
nx_list = [16, 32, 64, 128, 256]

"""
# Run the convergence study
results = run_moving_vertical_interface_convergence_2D(
    nx_list,
    u1_analytical,
    u2_analytical;
    lx=lx,
    ly=ly,
    x0=0.0,
    y0=0.0,
    Tend=Tend,
    D1=D_plus,
    D2=D_minus,
    norm=2,
    output_dir="vertical_interface_results"
)
"""



function read_diphasic_convergence_data(csv_path::String; compute_rates::Bool=true, npts::Int=3)
    # Read data from CSV
    df = CSV.read(csv_path, DataFrame)
    
    # Determine column names - allow for different naming conventions
    h_col = findfirst(col -> occursin("mesh_size", lowercase(string(col))) || 
                             occursin("h", lowercase(string(col))), names(df))
    
    # Extract mesh sizes
    h_vals = df[!, h_col]
    
    # Sort by mesh size (coarse to fine)
    sort_idx = sortperm(h_vals, rev=true)
    h_vals = h_vals[sort_idx]
    
    # Extract error values
    function find_column(pattern)
        col = findfirst(col -> occursin(pattern, lowercase(string(col))), names(df))
        if isnothing(col)
            @warn "Column matching '$pattern' not found in CSV. Using zeros."
            return zeros(size(df, 1))
        end
        return df[sort_idx, col]
    end
    
    # Global errors
    err1_vals = find_column("global_error_phase1")
    err2_vals = find_column("global_error_phase2")
    err_combined_vals = find_column("global_error_combined")
    
    # Full cell errors
    err1_full_vals = find_column("full_error_phase1")
    err2_full_vals = find_column("full_error_phase2")
    err_full_combined_vals = find_column("full_error_combined")
    
    # Cut cell errors
    err1_cut_vals = find_column("cut_error_phase1")
    err2_cut_vals = find_column("cut_error_phase2")
    err_cut_combined_vals = find_column("cut_error_combined")
    
    # Compute convergence rates if requested
    if compute_rates
        function do_fit(h_data, err_data, use_last_n=npts)
            # Convert to log space
            log_h_data = log.(h_data)
            log_err_data = log.(err_data)
            
            # Use only the last n points
            n = min(use_last_n, length(log_h_data))
            idx = length(log_h_data) - n + 1 : length(log_h_data)
            
            # Simple linear regression in log space
            h_mean = mean(log_h_data[idx])
            err_mean = mean(log_err_data[idx])
            numerator = sum((log_h_data[idx] .- h_mean) .* (log_err_data[idx] .- err_mean))
            denominator = sum((log_h_data[idx] .- h_mean).^2)
            
            # If denominator is too close to zero, return a default rate
            if abs(denominator) < 1e-10
                @warn "Nearly constant data detected, convergence rate calculation may be inaccurate."
                return 0.0
            end
            
            p = numerator / denominator
            return round(p, digits=2)
        end
        
        # Calculate rates using all points
        p1_global_all = do_fit(h_vals, err1_vals, length(h_vals))
        p2_global_all = do_fit(h_vals, err2_vals, length(h_vals))
        p_combined_all = do_fit(h_vals, err_combined_vals, length(h_vals))
        
        p1_full_all = do_fit(h_vals, err1_full_vals, length(h_vals))
        p2_full_all = do_fit(h_vals, err2_full_vals, length(h_vals))
        p_full_combined_all = do_fit(h_vals, err_full_combined_vals, length(h_vals))
        
        p1_cut_all = do_fit(h_vals, err1_cut_vals, length(h_vals))
        p2_cut_all = do_fit(h_vals, err2_cut_vals, length(h_vals))
        p_cut_combined_all = do_fit(h_vals, err_cut_combined_vals, length(h_vals))
        
        # Calculate rates using last n points
        p1_global = do_fit(h_vals, err1_vals, npts)
        p2_global = do_fit(h_vals, err2_vals, npts)
        p_combined = do_fit(h_vals, err_combined_vals, npts)
        
        p1_full = do_fit(h_vals, err1_full_vals, npts)
        p2_full = do_fit(h_vals, err2_full_vals, npts)
        p_full_combined = do_fit(h_vals, err_full_combined_vals, npts)
        
        p1_cut = do_fit(h_vals, err1_cut_vals, npts)
        p2_cut = do_fit(h_vals, err2_cut_vals, npts)
        p_cut_combined = do_fit(h_vals, err_cut_combined_vals, npts)
        
        global_rates = (p1_global, p2_global, p_combined)
        full_rates = (p1_full, p2_full, p_full_combined)
        cut_rates = (p1_cut, p2_cut, p_cut_combined)
    else
        # If not computing rates, just use placeholders
        global_rates = (0.0, 0.0, 0.0)
        full_rates = (0.0, 0.0, 0.0)
        cut_rates = (0.0, 0.0, 0.0)
    end
    
    # Package the data into the format expected by plot_diphasic_convergence
    global_errs = (err1_vals, err2_vals, err_combined_vals)
    full_errs = (err1_full_vals, err2_full_vals, err_full_combined_vals)
    cut_errs = (err1_cut_vals, err2_cut_vals, err_cut_combined_vals)
    
    return h_vals, global_errs, full_errs, cut_errs, global_rates, full_rates, cut_rates
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
        idx_start = max(1, n-3)
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
    
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
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
        
        
    elseif style == :by_phase
        # For style by phase, use different colors for different phases
        # and different markers for error types
        colors = [:blue, :red, :black]  # Phase 1, Phase 2, Combined
        symbols = [:circle, :rect, :diamond]  # Global, Full, Cut
        
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
            yminorticks = IntervalsBetween(10),
            xticks = LogTicks(WilkinsonTicks(5)),
            yticks = (10.0 .^ [-4, -3, -2, -1], ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹"])
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
            yminorticks = IntervalsBetween(10),
            xticks = LogTicks(WilkinsonTicks(5)),
            yticks = (10.0 .^ [-4, -3, -2, -1], ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹"])
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
            yminorticks = IntervalsBetween(10),
            xticks = LogTicks(WilkinsonTicks(5)),
            yticks = (10.0 .^ [-4, -3, -2, -1], ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹"])
        )
        
        # Global errors panel
        scatter!(ax_global, h_vals, err1_vals, color=colors[1], marker=symbols[1],
                markersize=10, label="Phase 1 (p=$(p1_global_all))")
        lines!(ax_global, h_fine, err1_curve, 
            color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax_global, h_vals, err2_vals, color=colors[2], marker=symbols[1],
                markersize=10, label="Phase 2 (p=$(p2_global_all))")
        lines!(ax_global, h_fine, err2_curve, 
            color=colors[2], linestyle=:dash, linewidth=2)
        
        scatter!(ax_global, h_vals, err_combined_vals, color=colors[3], marker=symbols[1],
                markersize=10, label="Combined (p=$(p_combined_all))")
        lines!(ax_global, h_fine, err_combined_curve, 
            color=colors[3], linestyle=:dash, linewidth=2)
        
        # Full cell errors panel
        scatter!(ax_full, h_vals, err1_full_vals, color=colors[1], marker=symbols[2],
                markersize=10, label="Phase 1 (p=$(p1_full_all))")
        lines!(ax_full, h_fine, err1_full_curve, 
            color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax_full, h_vals, err2_full_vals, color=colors[2], marker=symbols[2],
                markersize=10, label="Phase 2 (p=$(p2_full_all))")
        lines!(ax_full, h_fine, err2_full_curve, 
            color=colors[2], linestyle=:dash, linewidth=2)
        
        scatter!(ax_full, h_vals, err_full_combined_vals, color=colors[3], marker=symbols[2],
                markersize=10, label="Combined (p=$(p_full_combined_all))")
        lines!(ax_full, h_fine, err_full_combined_curve, 
            color=colors[3], linestyle=:dash, linewidth=2)
        
        # Cut cell errors panel (using last 3 points for the fit)
        scatter!(ax_cut, h_vals, err1_cut_vals, color=colors[1], marker=symbols[3],
                markersize=10, label="Phase 1 (p=$(p1_cut_last3))")
        lines!(ax_cut, h_fine, err1_cut_curve, 
            color=colors[1], linestyle=:dash, linewidth=2)
        
        scatter!(ax_cut, h_vals, err2_cut_vals, color=colors[2], marker=symbols[3],
                markersize=10, label="Phase 2 (p=$(p2_cut_last3))")
        lines!(ax_cut, h_fine, err2_cut_curve, 
            color=colors[2], linestyle=:dash, linewidth=2)
        
        scatter!(ax_cut, h_vals, err_cut_combined_vals, color=colors[3], marker=symbols[3],
                markersize=10, label="Combined (p=$(p_cut_combined_last3))")
        lines!(ax_cut, h_fine, err_cut_combined_curve, 
            color=colors[3], linestyle=:dash, linewidth=2)
        
        # Reference slopes for each panel
        for ax in [ax_global, ax_full, ax_cut]
            # Reference slopes
            h_ref = h_vals[end]
            err_ref = 0.000005
            
            lines!(ax, h_fine, err_ref * (h_fine/h_ref).^2, color=:black, linestyle=:dot, 
                linewidth=1.5, label="O(h²)")
            
            lines!(ax, h_fine, err_ref * (h_fine/h_ref), color=:black, linestyle=:dashdot, 
                linewidth=1.5, label="O(h)")
        end
        
        # Add legends
        axislegend(ax_global, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        axislegend(ax_full, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        axislegend(ax_cut, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))
        
        # Add a main title
        Label(fig[0, :], "Diphasic Heat Transfer Convergence: By Phase", fontsize=18)
    else
        error("Unknown plot style: $style. Use :by_error_type or :by_phase")
    end
    
    # Save the figure
    full_save_path = joinpath(output_dir, save_path)
    save(full_save_path, fig)
    println("Saved convergence plot to: $full_save_path")
    
    # Display if requested
    display(fig)
    
    # Return the figure and computed rates for possible further use
    return fig, (
        (p1_global_all, p2_global_all, p_combined_all),
        (p1_full_all, p2_full_all, p_full_combined_all),
        (p1_cut_last3, p2_cut_last3, p_cut_combined_last3)
    )
end

# Read data from CSV and generate convergence plot
h_vals, global_errs, full_errs, cut_errs, global_rates, full_rates, cut_rates = 
    read_diphasic_convergence_data("vertical_interface_results/summary.csv")

# Create the plot
fig, rates = plot_diphasic_convergence(
    h_vals, 
    global_errs,
    full_errs,
    cut_errs,
    global_rates,
    full_rates,
    cut_rates,
    output_dir="diphasic_results",
    save_path="convergence_by_error_type.pdf"
)