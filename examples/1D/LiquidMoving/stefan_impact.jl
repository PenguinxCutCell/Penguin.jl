using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Statistics
using Printf

"""
Analyze the impact of Newton iteration count on convergence properties
"""
function analyze_newton_iterations_impact(
    max_iterations_list=[1, 2, 3, 5, 10, 20]; 
    base_nx=320,
    reference_nx=1280, 
    Δt=0.001, 
    Tend=0.01,
    tol=1e-10
)
    # Results containers
    final_position_errors = Dict()
    newton_iterations_counts = Dict()
    final_positions = Dict()
    all_residuals = Dict()
    xf_histories = Dict()
    
    # First compute reference solution on finest mesh
    println("Computing reference solution on mesh size $(reference_nx)...")
    ref_final_position, _solver_ref, ref_residuals, _history_ref = 
        compute_solution(reference_nx, Δt, Tend, max_iter=50, tol=tol)
    
    # For each maximum iteration count
    for max_iter in max_iterations_list
        println("\nRunning with maximum $(max_iter) Newton iteration(s)...")
        
        # Compute solution with current max iterations
        final_position, _solver, residuals, _history = 
            compute_solution(base_nx, Δt, Tend, max_iter=max_iter, tol=tol)
        
        # Store final position
        final_positions[max_iter] = final_position
        
        # Store all residuals for this iteration count
        all_residuals[max_iter] = residuals
        
        # Compute error against reference solution
        final_position_errors[max_iter] = abs(final_position - ref_final_position)
        
        # Store max iterations actually used
        if length(residuals) > 0
            newton_iterations_counts[max_iter] = [length(x) for x in residuals if !isempty(x)]
        else
            newton_iterations_counts[max_iter] = []
        end
        
        println("  - Final position: $(final_position)")
        println("  - Final position error: $(final_position_errors[max_iter])")
        if !isempty(newton_iterations_counts[max_iter])
            println("  - Average iterations used: $(mean(newton_iterations_counts[max_iter]))")
            println("  - Maximum iterations used: $(maximum(newton_iterations_counts[max_iter]))")
            println("  - Total iterations: $(sum(newton_iterations_counts[max_iter]))")
        else
            println("  - No iteration data available")
        end
    end
    
    # === CONVERGENCE ORDER ANALYSIS ===
    fig_order = Figure(resolution=(1000, 800))
    ax_order = Axis(fig_order[1, 1], 
                   xlabel="Number of Max Newton Iterations", 
                   ylabel="Final Position Error (log scale)", 
                   title="Convergence Order Analysis",
                   xscale=log10, yscale=log10)
    
    final_errors = [final_position_errors[max_iter] for max_iter in max_iterations_list]
    scatter!(ax_order, max_iterations_list, final_errors, markersize=15)
    lines!(ax_order, max_iterations_list, final_errors)
    
    # Fit power law to determine convergence order: error ∝ N^(-p)
    if length(max_iterations_list) >= 3
        model(x, p) = p[1] .* x .^ (-p[2])
        try
            powerfit = curve_fit(model, Float64.(max_iterations_list), final_errors, [1.0, 1.0])
            order = powerfit.param[2]
            
            # Add fitted curve
            x_fit = range(minimum(max_iterations_list), maximum(max_iterations_list), length=100)
            y_fit = model(x_fit, powerfit.param)
            lines!(ax_order, x_fit, y_fit, linestyle=:dash, linewidth=2, color=:red,
                  label="Order ≈ $(round(order, digits=2))")
            
            text!(ax_order, "Error ∝ N^(-$(round(order, digits=2)))", 
                 position=(max_iterations_list[end÷2], final_errors[end÷2] * 0.5), 
                 fontsize=14)
        catch e
            println("Couldn't fit power law: $e")
        end
    end
    
    axislegend(ax_order, position=:lt)
    
    # === NEWTON ITERATIONS ANALYSIS ===
    fig_iterations = Figure(resolution=(1000, 800))
    ax_iterations = Axis(fig_iterations[1, 1],
                        xlabel="Time Step",
                        ylabel="Newton Iterations Used",
                        title="Newton Iterations Required")
    
    for max_iter in max_iterations_list
        if !isempty(newton_iterations_counts[max_iter])
            timesteps = 1:length(newton_iterations_counts[max_iter])
            scatter!(ax_iterations, timesteps, newton_iterations_counts[max_iter],
                    label="Max iter = $(max_iter)")
            lines!(ax_iterations, timesteps, newton_iterations_counts[max_iter])
        end
    end
    
    axislegend(ax_iterations, position=:lt)
    
    # === NEWTON CONVERGENCE BEHAVIOR ===
    # Choose a representative time step to analyze convergence behavior
    fig_newton = Figure(resolution=(1200, 800))
    
    # Choose 3 representative iterations to display convergence behavior
    if !isempty(all_residuals)
        time_steps = [5, 8]  # Example time steps to analyze
        
        for (t_idx, step) in enumerate(time_steps)
            ax_newton = Axis(fig_newton[1, t_idx],
                            xlabel="Newton Iteration",
                            ylabel="log₁₀(Residual)",
                            title="Newton convergence at time step $(step)")
                            
            for max_iter in max_iterations_list
                if haskey(all_residuals, max_iter) && 
                   step <= length(all_residuals[max_iter]) && 
                   !isempty(all_residuals[max_iter][step])
                    
                    res = all_residuals[max_iter][step]
                    iterations = 1:length(res)
                    
                    # Plot residual evolution
                    lines!(ax_newton, iterations, log10.(res), 
                          linewidth=2,
                          label="Max iter = $(max_iter)")
                    
                    # Mark final residual
                    scatter!(ax_newton, [length(res)], [log10(res[end])],
                            markersize=10)
                end
            end
            
            # Add tolerance line
            hlines!(ax_newton, [log10(tol)], 
                    color=:red, 
                    linestyle=:dash, 
                    label="Tolerance")
            
            axislegend(ax_newton, position=:rt)
        end
    end
    
    # === COMPUTATIONAL EFFICIENCY ===
    fig_efficiency = Figure(resolution=(1000, 800))
    ax_efficiency = Axis(fig_efficiency[1, 1],
                        xlabel="Total Newton Iterations",
                        ylabel="Final Position Error",
                        title="Computational Efficiency",
                        yscale=log10)
    
    total_iterations = [isempty(newton_iterations_counts[max_iter]) ? 
                        0 : sum(newton_iterations_counts[max_iter]) 
                        for max_iter in max_iterations_list]
    
    scatter!(ax_efficiency, total_iterations, final_errors, markersize=15)
    
    # Add labels for each point showing max_iter
    for (i, max_iter) in enumerate(max_iterations_list)
        text!(ax_efficiency, "$(max_iter)", 
             position=(total_iterations[i], final_errors[i]), 
             fontsize=12)
    end
    
    lines!(ax_efficiency, total_iterations, final_errors)
    
    # === SUMMARY FIGURE ===
    fig_summary = Figure(resolution=(1200, 900))
    
    ax_sum1 = Axis(fig_summary[1, 1], 
                  xlabel="Max Newton Iterations", 
                  ylabel="Final Position Error",
                  title="(a) Convergence Order Analysis",
                  xscale=log10, yscale=log10)
                  
    ax_sum2 = Axis(fig_summary[1, 2], 
                  xlabel="Max iterations allowed", 
                  ylabel="Final Interface Position",
                  title="(b) Final Interface Position")
                  
    ax_sum3 = Axis(fig_summary[2, 1], 
                  xlabel="Time Step", 
                  ylabel="Newton Iterations Used",
                  title="(c) Iterations Required per Time Step")
                  
    ax_sum4 = Axis(fig_summary[2, 2], 
                  xlabel="Total Newton Iterations", 
                  ylabel="Final Position Error",
                  title="(d) Computational Efficiency",
                  yscale=log10)
    
    # Add data to summary plots
    scatter!(ax_sum1, max_iterations_list, final_errors, markersize=15)
    lines!(ax_sum1, max_iterations_list, final_errors)
    
    position_values = [final_positions[max_iter] for max_iter in max_iterations_list]
    scatter!(ax_sum2, max_iterations_list, position_values, markersize=15) 
    lines!(ax_sum2, max_iterations_list, position_values)
    
    # Add reference line
    hlines!(ax_sum2, [ref_final_position], color=:red, linestyle=:dash, 
            label="Reference")
    
    # Add iterations per time step
    for max_iter in max_iterations_list
        if !isempty(newton_iterations_counts[max_iter])
            timesteps = 1:length(newton_iterations_counts[max_iter])
            lines!(ax_sum3, timesteps, newton_iterations_counts[max_iter],
                  label="Max iter = $(max_iter)")
        end
    end
    
    # Add efficiency plot
    scatter!(ax_sum4, total_iterations, final_errors, markersize=15)
    lines!(ax_sum4, total_iterations, final_errors)
    
    # Add labels for each point showing max_iter
    for (i, max_iter) in enumerate(max_iterations_list)
        text!(ax_sum4, "$(max_iter)", 
             position=(total_iterations[i], final_errors[i]), 
             fontsize=12)
    end
    
    # Add convergence order to summary
    if length(max_iterations_list) >= 3
        try
            powerfit = curve_fit(model, Float64.(max_iterations_list), final_errors, [1.0, 1.0])
            order = powerfit.param[2]
            
            x_fit = range(minimum(max_iterations_list), maximum(max_iterations_list), length=100)
            y_fit = model(x_fit, powerfit.param)
            lines!(ax_sum1, x_fit, y_fit, linestyle=:dash, linewidth=2, color=:red,
                  label="Order ≈ $(round(order, digits=2))")
        catch e
            println("Couldn't fit power law for summary: $e")
        end
    end
    
    axislegend(ax_sum1, position=:lt)
    axislegend(ax_sum2, position=:lt)
    axislegend(ax_sum3, position=:lt)
    
    # Create tables with numerical results
    println("\n=== NUMERICAL RESULTS ===")
    println("| Max Iterations | Final Position | Error vs Reference | Avg Iterations Used | Total Iterations |")
    println("|----------------|---------------|-------------------|-------------------|----------------|")
    
    for max_iter in max_iterations_list
        pos = final_positions[max_iter]
        err = final_position_errors[max_iter]
        avg_iter = isempty(newton_iterations_counts[max_iter]) ? 
                  0 : mean(newton_iterations_counts[max_iter])
        tot_iter = isempty(newton_iterations_counts[max_iter]) ? 
                  0 : sum(newton_iterations_counts[max_iter])
        
        @printf("| %14d | %13.10f | %18.10e | %19.2f | %16d |\n", 
               max_iter, pos, err, avg_iter, tot_iter)
    end
    
    # Save figures
    save("newton_impact_summary.png", fig_summary)
    save("newton_convergence_order.png", fig_order)
    save("newton_iterations_required.png", fig_iterations)
    save("newton_convergence_behavior.png", fig_newton)
    save("newton_computational_efficiency.png", fig_efficiency)
    
    return final_position_errors, newton_iterations_counts, final_positions, all_residuals
end

"""
Run a single Stefan problem with specified parameters
"""
function compute_solution(nx, Δt, Tend; max_iter=1000, tol=1e-10, reltol=1e-10, α=1.0)
    # Define the spatial mesh
    lx = 1.0
    x0 = 0.0
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    
    # Define the body and initial interface position
    xf = 0.05*lx
    
    # Define the Space-Time mesh
    STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
    
    # Define capacity, operator, etc.
    capacity = Capacity((x,t,_=0)->(x - xf), STmesh)
    operator = DiffusionOps(capacity)
    
    # Create Phase object
    Fluide = Phase(capacity, operator, 
                  (x,y,z,t)-> 0.0,  # f_func
                  (x,y,z)-> 1.0)    # K_func
    
    # Define boundary conditions
    bc = Dirichlet(0.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
    ρ, L = 1.0, 1.0
    stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))
    
    # Initial condition
    u0ₒ = zeros((nx+1))
    u0ᵧ = zeros((nx+1))
    u0 = vcat(u0ₒ, u0ᵧ)
    
    # Newton parameters - with max_iter control
    Newton_params = (max_iter, tol, reltol, α)
    
    # Define the solver
    solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")
    
    # Solve the problem
    solver, residuals, xf_log, timestep_history = solve_MovingLiquidDiffusionUnsteadyMono!(
        solver, Fluide, xf, Δt, Tend, bc_b, bc, stef_cond, mesh, "BE"; 
        Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\
    )
    
    # Return the final position - handle empty xf_log case
    final_position = isempty(xf_log) ? xf : xf_log[end]
    
    return final_position, solver, residuals, timestep_history
end

# Run the analysis with different max iterations settings
position_errors, newton_counts, final_positions, all_residuals = analyze_newton_iterations_impact(
    [1, 2, 3, 5, 10, 20, 50], 
    base_nx=320, 
    reference_nx=1280,
    Δt=0.001,
    Tend=0.01,
    tol=1e-10
)