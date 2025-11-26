using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie

function run_mesh_convergence_ft(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    norm,
    relative::Bool=false
)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]

    for (nx, ny) in zip(nx_list, ny_list)
        println("Running Front Tracking with mesh $(nx)×$(ny)...")
        # Build mesh
        x0, y0 = 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

        # Create a front tracker with resolution proportional to mesh size
        front = FrontTracker()
        # Higher resolution for finer meshes to ensure accurate geometry
        resolution = max(500, min(nx, ny) * 10)
        create_circle!(front, center[1], center[2], radius, resolution)

        # Define capacity using front tracking
        capacity = Capacity(front, mesh)
        operator = DiffusionOps(capacity)

        # BC + solver
        bc_boundary = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(
            :left   => bc_boundary,
            :right  => bc_boundary,
            :top    => bc_boundary,
            :bottom => bc_boundary
        ))
        phase = Phase(capacity, operator, (x,y,_)->4.0, (x,y,_)->1.0)
        solver = DiffusionSteadyMono(phase, bc_b, Dirichlet(0.0))

        solve_DiffusionSteadyMono!(solver; method=Base.:\)

        # Compute errors
        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm, relative)

        # Representative mesh size ~ 1 / min(nx, ny)
        push!(h_vals, 1.0 / min(nx, ny))

        push!(err_vals,       global_err)
        push!(err_full_vals,  full_err)
        push!(err_cut_vals,   cut_err)
        push!(err_empty_vals, empty_err)
        
        println("  Global error: $global_err")
        println("  Full cells error: $full_err")
        println("  Cut cells error: $cut_err")
    end

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

    # Get convergence rates
    p_global_all, _ = do_fit(log.(err_vals), length(h_vals))
    p_full_all,   _ = do_fit(log.(err_full_vals), length(h_vals))
    p_cut_all,    _ = do_fit(log.(err_cut_vals), length(h_vals))

    # Get convergence rates for just the last 3 points
    p_global, _ = do_fit(log.(err_vals), 3)
    p_full,   _ = do_fit(log.(err_full_vals), 3)
    p_cut,    _ = do_fit(log.(err_cut_vals), 3)

    # Round
    p_global = round(p_global, digits=1)
    p_full   = round(p_full, digits=1)
    p_cut    = round(p_cut, digits=1)

    p_global_all = round(p_global_all, digits=1)
    p_full_all   = round(p_full_all, digits=1)
    p_cut_all    = round(p_cut_all, digits=1)

    println("\nFRONT TRACKING METHOD RESULTS:")
    println("Estimated order of convergence (all points):")
    println("  - Global = ", p_global_all)
    println("  - Full   = ", p_full_all)
    println("  - Cut    = ", p_cut_all)

    println("\nEstimated order of convergence (last 3 points):")
    println("  - Global = ", p_global)
    println("  - Full   = ", p_full)
    println("  - Cut    = ", p_cut)

    # Plot in log-log scale
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "h",
        ylabel = "L$norm error",
        title  = "Convergence plot (Front Tracking)",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )

    scatter!(ax, h_vals, err_vals,       label="All cells ($p_global)", markersize=12)
    lines!(ax, h_vals, err_vals,         color=:black)
    scatter!(ax, h_vals, err_full_vals,  label="Full cells ($p_full)",   markersize=12)
    lines!(ax, h_vals, err_full_vals,    color=:black)
    scatter!(ax, h_vals, err_cut_vals,   label="Cut cells ($p_cut)",     markersize=12)
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
        fig
    )
end

function run_mesh_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    norm,
    relative::Bool=false
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
        circle = (x,y) -> (sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

        # Define capacity/operator
        capacity = Capacity(circle, mesh; method="ImplicitIntegration")
        operator = DiffusionOps(capacity)

        # BC + solver
        bc_boundary = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(
            :left   => bc_boundary,
            :right  => bc_boundary,
            :top    => bc_boundary,
            :bottom => bc_boundary
        ))
        phase = Phase(capacity, operator, (x,y,_)->4.0, (x,y,_)->1.0)
        solver = DiffusionSteadyMono(phase, bc_b, Dirichlet(0.0))

        solve_DiffusionSteadyMono!(solver; method=Base.:\)

        # Compute errors
        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm, relative)

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

    function do_fit(log_err, use_last_n=3)
        # Use only the last n points (default 3)
        n = min(use_last_n, length(log_h))
        idx = length(log_h) - n + 1 : length(log_h)
        
        # Fit using only those points
        fit_result = curve_fit(fit_model, log_h[idx], log_err[idx], [-1.0, 0.0])
        return fit_result.param[1], fit_result.param[2]  # (p_est, c_est)
    end

    # Get convergence rates for all points
    p_global_all, _ = do_fit(log.(err_vals), length(h_vals))
    p_full_all,   _ = do_fit(log.(err_full_vals), length(h_vals))
    p_cut_all,    _ = do_fit(log.(err_cut_vals), length(h_vals))

    # Get convergence rates for just the last 3 points
    p_global, _ = do_fit(log.(err_vals), 3)
    p_full,   _ = do_fit(log.(err_full_vals), 3)
    p_cut,    _ = do_fit(log.(err_cut_vals), 3)

    # Round
    p_global = round(p_global, digits=1)
    p_full   = round(p_full, digits=1)
    p_cut    = round(p_cut, digits=1)

    p_global_all = round(p_global_all, digits=1)
    p_full_all   = round(p_full_all, digits=1)
    p_cut_all    = round(p_cut_all, digits=1)

    println("Estimated order of convergence (all points):")
    println("  - Global = ", p_global_all)
    println("  - Full   = ", p_full_all)
    println("  - Cut    = ", p_cut_all)

    println("\nEstimated order of convergence (last 3 points):")
    println("  - Global = ", p_global)
    println("  - Full   = ", p_full)
    println("  - Cut    = ", p_cut)


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
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )

    scatter!(ax, h_vals, err_vals,       label="All cells ($p_global)", markersize=12)
    lines!(ax, h_vals, err_vals,         color=:black)
    scatter!(ax, h_vals, err_full_vals,  label="Full cells ($p_full)",   markersize=12)
    lines!(ax, h_vals, err_full_vals,    color=:black)
    scatter!(ax, h_vals, err_cut_vals,   label="Cut cells ($p_cut)",     markersize=12)
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

# Run convergence test for Front Tracking and regular VOFI method
function compare_methods()
    nx_list = [20, 40, 80, 160]
    ny_list = [20, 40, 80, 160]
    radius, center = 1.0, (2.0, 2.0)
    u_analytical(x,y) = 1.0 - (x-center[1])^2 - (y-center[2])^2
    
    # Run Front Tracking convergence
    println("Running Front Tracking convergence tests...")
    h_vals, err_vals, err_full_vals, err_cut_vals, err_empty_vals, 
    p_global, p_full, p_cut, fig_ft = run_mesh_convergence_ft(
        nx_list, ny_list, radius, center, u_analytical, norm=2, relative=false
    )
    
    # Save results
    CairoMakie.save("ft_convergence.png", fig_ft)
    
    # Run the original VOFI method too for comparison
    println("\nRunning VOFI convergence tests for comparison...")
    h_vals_vofi, err_vals_vofi, err_full_vals_vofi, err_cut_vals_vofi, 
    err_empty_vals_vofi, p_global_vofi, p_full_vofi, p_cut_vofi = run_mesh_convergence(
        nx_list, ny_list, radius, center, u_analytical, norm=2, relative=false
    )
    
    # Create comparison plot
    fig_comp = Figure(size=(900, 600))
    ax_comp = Axis(
        fig_comp[1, 1],
        xlabel = "h",
        ylabel = "L2 error",
        title  = "Front Tracking vs VOFI Convergence",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )

    # Plot global errors
    scatter!(ax_comp, h_vals, err_vals, label="FT Global ($p_global)",
             markersize=12, color=:blue)
    lines!(ax_comp, h_vals, err_vals, color=:blue)
    scatter!(ax_comp, h_vals_vofi, err_vals_vofi, label="VOFI Global ($p_global_vofi)",
             markersize=12, color=:red)
    lines!(ax_comp, h_vals_vofi, err_vals_vofi, color=:red)

    # Plot cut cell errors
    scatter!(ax_comp, h_vals, err_cut_vals, label="FT Cut ($p_cut)",
             markersize=8, color=:blue, marker=:diamond)
    lines!(ax_comp, h_vals, err_cut_vals, color=:blue, linestyle=:dash)
    scatter!(ax_comp, h_vals_vofi, err_cut_vals_vofi, label="VOFI Cut ($p_cut_vofi)",
             markersize=8, color=:red, marker=:diamond)
    lines!(ax_comp, h_vals_vofi, err_cut_vals_vofi, color=:red, linestyle=:dash)

    # Reference lines
    lines!(ax_comp, h_vals, 10.0*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax_comp, h_vals, 1.0*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    axislegend(ax_comp, position=:rb)
    
    display(fig_comp)
    CairoMakie.save("ft_vs_vofi_comparison.png", fig_comp)
    
    # Print summary comparison
    println("\n====== CONVERGENCE RATE COMPARISON ======")
    println("                   FRONT TRACKING    VOFI")
    println("Global convergence:      $(p_global)           $(p_global_vofi)")
    println("Full cells:              $(p_full)           $(p_full_vofi)")
    println("Cut cells:               $(p_cut)           $(p_cut_vofi)")
    
    # Save data for further analysis
    open("convergence_comparison.txt", "w") do io
        println(io, "method, h, global_err, full_err, cut_err")
        for i in 1:length(h_vals)
            println(io, "FT, $(h_vals[i]), $(err_vals[i]), $(err_full_vals[i]), $(err_cut_vals[i])")
            println(io, "VOFI, $(h_vals_vofi[i]), $(err_vals_vofi[i]), $(err_full_vals_vofi[i]), $(err_cut_vals_vofi[i])")
        end
    end
    
    return (h_vals, err_vals, err_cut_vals, p_global, p_cut,
            h_vals_vofi, err_vals_vofi, err_cut_vals_vofi, p_global_vofi, p_cut_vofi)
end

# Run the comparison
compare_methods()