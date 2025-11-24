using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations
using Statistics

# Import the analyze functions from fixedpoint_analysis.jl
# Assuming the file is in the same directory
include("fixedpoint_analysis.jl")

"""
Perform a mesh convergence study for the fixed-point analysis.
"""
function mesh_convergence_study(mesh_sizes=[20, 40, 80, 160]; 
                               Δt=0.001, xf_initial=0.05, alpha=1.0,
                               num_time_steps=3, num_samples=100)
    # Store results for each mesh size
    results = Dict{Int, Any}()
    
    # Create figure for comparison
    fig = Figure(resolution=(1200, 900))
    
    # Plot for Lipschitz constants
    ax_lipschitz = Axis(fig[1, 1], 
                        xlabel="Time Step", 
                        ylabel="Lipschitz Constant",
                        title="Lipschitz Constant vs. Mesh Size")
    
    # Plot for interface positions
    ax_interface = Axis(fig[2, 1], 
                       xlabel="Time Step", 
                       ylabel="Interface Position",
                       title="Interface Position vs. Mesh Size")
    
    # Plot for convergence of L
    ax_conv = Axis(fig[1, 2], 
                  xlabel="1/h (Mesh Points)", 
                  ylabel="Lipschitz Constant",
                  title="Mesh Convergence of Lipschitz Constant")
                  
    # Plot for convergence of xf
    ax_xf_conv = Axis(fig[2, 2], 
                     xlabel="1/h (Mesh Points)", 
                     ylabel="Interface Position",
                     title="Mesh Convergence of Interface Position")
    
    # Colors for different mesh sizes
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :cyan]
    
    # For storing mesh convergence data
    h_values = Float64[]
    L_at_final = Float64[]
    xf_at_final = Float64[]
    
    # Run analysis for each mesh size
    for (i, nx) in enumerate(mesh_sizes)
        println("\n=== Analyzing Mesh Size: $nx ===")
        
        # Set color for this mesh size
        color = colors[mod1(i, length(colors))]
        
        # Run the analysis
        L_values, fixed_points, xf_values, _ = analyze_stefan_fixed_point_over_time(
            xf_initial, num_time_steps, 
            mesh_size=nx, Δt=Δt, num_samples=num_samples, alpha=alpha
        )
        
        # Store results
        results[nx] = Dict(
            "L_values" => L_values,
            "fixed_points" => fixed_points,
            "xf_values" => xf_values
        )
        
        # Plot Lipschitz constants
        time_steps = 1:num_time_steps
        lines!(ax_lipschitz, time_steps, L_values, 
              linewidth=3, color=color, label="nx = $nx")
        scatter!(ax_lipschitz, time_steps, L_values, 
                markersize=10, color=color)
        
        # Plot interface positions
        lines!(ax_interface, 0:num_time_steps, xf_values, 
              linewidth=3, color=color, label="nx = $nx")
        scatter!(ax_interface, 0:num_time_steps, xf_values, 
                markersize=10, color=color)
        
        # Store data for convergence plot
        push!(h_values, nx)
        push!(L_at_final, L_values[end])
        push!(xf_at_final, xf_values[end])
    end
    
    # Add reference line at L=1
    hlines!(ax_lipschitz, [1.0], color=:black, linestyle=:dash, linewidth=2, 
           label="Critical value L=1")
    
    # Add legends
    axislegend(ax_lipschitz, position=:lt)
    axislegend(ax_interface, position=:lt)
    
    # Plot mesh convergence for Lipschitz constant
    scatter!(ax_conv, h_values, L_at_final, 
            markersize=12, color=:blue, label="Last time step")
    lines!(ax_conv, h_values, L_at_final, 
          linewidth=2, color=:blue)
    
    # Plot mesh convergence for interface position
    scatter!(ax_xf_conv, h_values, xf_at_final, 
            markersize=12, color=:blue, label="Last time step")
    lines!(ax_xf_conv, h_values, xf_at_final, 
          linewidth=2, color=:blue)
    
    # Save figure
    save("mesh_convergence_study.png", fig)
    
    return fig, results
end

"""
Perform a time step study for the fixed-point analysis.
"""
function time_step_study(time_steps=[0.0001, 0.0005, 0.001, 0.005]; 
                        mesh_size=80, xf_initial=0.05, alpha=1.0,
                        num_time_steps=3, num_samples=100)
    # Store results for each time step
    results = Dict{Float64, Any}()
    
    # Create figure for comparison
    fig = Figure(resolution=(1200, 900))
    
    # Plot for Lipschitz constants
    ax_lipschitz = Axis(fig[1, 1], 
                        xlabel="Time Step Index", 
                        ylabel="Lipschitz Constant",
                        title="Lipschitz Constant vs. Δt")
    
    # Plot for interface positions
    ax_interface = Axis(fig[2, 1], 
                       xlabel="Time Step Index", 
                       ylabel="Interface Position",
                       title="Interface Position vs. Δt")
    
    # Plot for relationship between Δt and L
    ax_dt_L = Axis(fig[1, 2], 
                  xlabel="Δt (log scale)", 
                  ylabel="Lipschitz Constant",
                  xscale=log10,
                  title="Effect of Δt on Lipschitz Constant")
                  
    # Plot for relationship between Δt and xf
    ax_dt_xf = Axis(fig[2, 2], 
                   xlabel="Δt (log scale)", 
                   ylabel="Final Interface Position",
                   xscale=log10,
                   title="Effect of Δt on Interface Position")
    
    # Colors for different time steps
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :cyan]
    
    # For storing time step study data
    dt_values = Float64[]
    L_at_final = Float64[]
    xf_at_final = Float64[]
    
    # Run analysis for each time step
    for (i, dt) in enumerate(time_steps)
        println("\n=== Analyzing Time Step: $dt ===")
        
        # Set color for this time step
        color = colors[mod1(i, length(colors))]
        
        # Run the analysis
        L_values, fixed_points, xf_values, _ = analyze_stefan_fixed_point_over_time(
            xf_initial, num_time_steps, 
            mesh_size=mesh_size, Δt=dt, num_samples=num_samples, alpha=alpha
        )
        
        # Store results
        results[dt] = Dict(
            "L_values" => L_values,
            "fixed_points" => fixed_points,
            "xf_values" => xf_values
        )
        
        # Plot Lipschitz constants
        time_indices = 1:num_time_steps
        lines!(ax_lipschitz, time_indices, L_values, 
              linewidth=3, color=color, label="Δt = $dt")
        scatter!(ax_lipschitz, time_indices, L_values, 
                markersize=10, color=color)
        
        # Plot interface positions
        lines!(ax_interface, 0:num_time_steps, xf_values, 
              linewidth=3, color=color, label="Δt = $dt")
        scatter!(ax_interface, 0:num_time_steps, xf_values, 
                markersize=10, color=color)
        
        # Store data for relationship plots
        push!(dt_values, dt)
        push!(L_at_final, L_values[end])
        push!(xf_at_final, xf_values[end])
    end
    
    # Add reference line at L=1
    hlines!(ax_lipschitz, [1.0], color=:black, linestyle=:dash, linewidth=2, 
           label="Critical value L=1")
    
    # Add legends
    axislegend(ax_lipschitz, position=:lt)
    axislegend(ax_interface, position=:lt)
    
    # Plot relationship between Δt and Lipschitz constant
    scatter!(ax_dt_L, dt_values, L_at_final, 
            markersize=12, color=:blue, label="Last time step")
    lines!(ax_dt_L, dt_values, L_at_final, 
          linewidth=2, color=:blue)
    
    # Reference line at L=1
    hlines!(ax_dt_L, [1.0], color=:black, linestyle=:dash, linewidth=2,
           label="Critical value L=1")
    
    # Plot relationship between Δt and interface position
    scatter!(ax_dt_xf, dt_values, xf_at_final, 
            markersize=12, color=:blue, label="Last time step")
    lines!(ax_dt_xf, dt_values, xf_at_final, 
          linewidth=2, color=:blue)
    
    # Save figure
    save("time_step_study.png", fig)
    
    return fig, results
end

"""
Perform an alpha parameter study for the fixed-point analysis.
"""
function alpha_study(alpha_values=[0.1, 0.5, 0.75, 1.0, 1.25, 1.5]; 
                    mesh_size=80, Δt=0.001, xf_initial=0.05,
                    num_time_steps=3, num_samples=100)
    # Store results for each alpha
    results = Dict{Float64, Any}()
    
    # Create figure for comparison
    fig = Figure(resolution=(1200, 900))
    
    # Plot for Lipschitz constants
    ax_lipschitz = Axis(fig[1, 1], 
                        xlabel="Time Step", 
                        ylabel="Lipschitz Constant",
                        title="Lipschitz Constant vs. α")
    
    # Plot for interface positions
    ax_interface = Axis(fig[2, 1], 
                       xlabel="Time Step", 
                       ylabel="Interface Position",
                       title="Interface Position vs. α")
    
    # Plot for relationship between α and L
    ax_alpha_L = Axis(fig[1, 2], 
                     xlabel="α", 
                     ylabel="Lipschitz Constant",
                     title="Effect of α on Lipschitz Constant")
    
    # Plot for iterations required vs α
    ax_iter = Axis(fig[2, 2],
                  xlabel="α",
                  ylabel="Est. Iterations to Converge",
                  title="Effect of α on Newton Convergence Rate")
    
    # Colors for different alpha values
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :cyan]
    
    # For storing alpha study data
    alpha_for_plot = Float64[]
    L_at_final = Float64[]
    iter_est = Float64[]
    
    # Run analysis for each alpha
    for (i, a) in enumerate(alpha_values)
        println("\n=== Analyzing α = $a ===")
        
        # Set color for this alpha
        color = colors[mod1(i, length(colors))]
        
        # Run the analysis
        L_values, fixed_points, xf_values, _ = analyze_stefan_fixed_point_over_time(
            xf_initial, num_time_steps, 
            mesh_size=mesh_size, Δt=Δt, num_samples=num_samples, alpha=a
        )
        
        # Store results
        results[a] = Dict(
            "L_values" => L_values,
            "fixed_points" => fixed_points,
            "xf_values" => xf_values
        )
        
        # Plot Lipschitz constants
        time_steps = 1:num_time_steps
        lines!(ax_lipschitz, time_steps, L_values, 
              linewidth=3, color=color, label="α = $a")
        scatter!(ax_lipschitz, time_steps, L_values, 
                markersize=10, color=color)
        
        # Plot interface positions
        lines!(ax_interface, 0:num_time_steps, xf_values, 
              linewidth=3, color=color, label="α = $a")
        scatter!(ax_interface, 0:num_time_steps, xf_values, 
                markersize=10, color=color)
        
        # Store data for relationship plots
        push!(alpha_for_plot, a)
        push!(L_at_final, L_values[end])
        
        # Estimate iterations required based on L
        if L_values[end] >= 1.0
            iter = Inf  # Divergent or very slow
        else
            # For L < 1, iterations ≈ log(tol)/log(L)
            tol = 1e-8
            iter = log(tol)/log(L_values[end])
        end
        push!(iter_est, min(iter, 100))  # Cap at 100 for visualization
    end
    
    # Add reference line at L=1
    hlines!(ax_lipschitz, [1.0], color=:black, linestyle=:dash, linewidth=2, 
           label="Critical value L=1")
    
    # Add legends
    axislegend(ax_lipschitz, position=:lt)
    axislegend(ax_interface, position=:lt)
    
    # Plot relationship between α and Lipschitz constant
    scatter!(ax_alpha_L, alpha_for_plot, L_at_final, 
            markersize=12, color=:blue)
    lines!(ax_alpha_L, alpha_for_plot, L_at_final, 
          linewidth=2, color=:blue)
    
    # Reference line at L=1
    hlines!(ax_alpha_L, [1.0], color=:black, linestyle=:dash, linewidth=2,
           label="Critical value L=1")
    
    # Plot estimated iterations vs alpha
    scatter!(ax_iter, alpha_for_plot, iter_est, 
            markersize=12, color=:blue)
    lines!(ax_iter, alpha_for_plot, iter_est, 
          linewidth=2, color=:blue)
    
    # Add "diverges" label for L >= 1
    for (i, L) in enumerate(L_at_final)
        if L >= 1.0
            scatter!(ax_iter, [alpha_for_plot[i]], [90], 
                    marker=:xcross, markersize=15, color=:red)
            text!(ax_iter, "Likely diverges", 
                 position=(alpha_for_plot[i], 95), 
                 fontsize=12, color=:red, align=(:center, :bottom))
        end
    end
    
    # Save figure
    save("alpha_parameter_study.png", fig)
    
    return fig, results
end

"""
Perform a study of initial interface position.
"""
function initial_position_study(xf_initials=[0.01, 0.02, 0.05, 0.1, 0.2]; 
                               mesh_size=80, Δt=0.001, alpha=1.0,
                               num_time_steps=3, num_samples=100)
    # Store results for each initial position
    results = Dict{Float64, Any}()
    
    # Create figure for comparison
    fig = Figure(resolution=(1200, 900))
    
    # Plot for Lipschitz constants
    ax_lipschitz = Axis(fig[1, 1], 
                        xlabel="Time Step", 
                        ylabel="Lipschitz Constant",
                        title="Lipschitz Constant vs. Initial Position")
    
    # Plot for interface positions
    ax_interface = Axis(fig[2, 1], 
                       xlabel="Time Step", 
                       ylabel="Interface Position",
                       title="Interface Position vs. Initial Position")
    
    # Plot for growth rate
    ax_growth = Axis(fig[1, 2], 
                    xlabel="Initial Position", 
                    ylabel="Position Growth Rate",
                    title="Effect of Initial Position on Growth Rate")
    
    # Plot for speed vs position
    ax_speed = Axis(fig[2, 2], 
                   xlabel="Time Step", 
                   ylabel="Interface Speed",
                   title="Interface Speed vs. Initial Position")
    
    # Colors for different initial positions
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :cyan]
    
    # For storing growth rate data
    xf0_values = Float64[]
    growth_rates = Float64[]
    
    # Run analysis for each initial position
    for (i, xf0) in enumerate(xf_initials)
        println("\n=== Analyzing Initial Position: $xf0 ===")
        
        # Set color for this initial position
        color = colors[mod1(i, length(colors))]
        
        # Run the analysis
        L_values, fixed_points, xf_values, _ = analyze_stefan_fixed_point_over_time(
            xf0, num_time_steps, 
            mesh_size=mesh_size, Δt=Δt, num_samples=num_samples, alpha=alpha
        )
        
        # Store results
        results[xf0] = Dict(
            "L_values" => L_values,
            "fixed_points" => fixed_points,
            "xf_values" => xf_values
        )
        
        # Plot Lipschitz constants
        time_steps = 1:num_time_steps
        lines!(ax_lipschitz, time_steps, L_values, 
              linewidth=3, color=color, label="xf₀ = $xf0")
        scatter!(ax_lipschitz, time_steps, L_values, 
                markersize=10, color=color)
        
        # Plot interface positions
        lines!(ax_interface, 0:num_time_steps, xf_values, 
              linewidth=3, color=color, label="xf₀ = $xf0")
        scatter!(ax_interface, 0:num_time_steps, xf_values, 
                markersize=10, color=color)
        
        # Calculate interface speeds
        speeds = diff(xf_values) ./ Δt
        
        # Plot interface speeds
        lines!(ax_speed, 1:num_time_steps, speeds, 
              linewidth=3, color=color, label="xf₀ = $xf0")
        scatter!(ax_speed, 1:num_time_steps, speeds, 
                markersize=10, color=color)
        
        # Calculate overall growth rate (position[end]/position[1])
        growth_rate = xf_values[end] / xf0
        push!(xf0_values, xf0)
        push!(growth_rates, growth_rate)
    end
    
    # Add reference line at L=1
    hlines!(ax_lipschitz, [1.0], color=:black, linestyle=:dash, linewidth=2, 
           label="Critical value L=1")
    
    # Add legends
    axislegend(ax_lipschitz, position=:lt)
    axislegend(ax_interface, position=:lt)
    axislegend(ax_speed, position=:lt)
    
    # Plot growth rate vs initial position
    scatter!(ax_growth, xf0_values, growth_rates, 
            markersize=12, color=:blue)
    lines!(ax_growth, xf0_values, growth_rates, 
          linewidth=2, color=:blue)
    
    # Save figure
    save("initial_position_study.png", fig)
    
    return fig, results
end

"""
Create a comprehensive parameter study dashboard.
"""
function create_parameter_dashboard()
    # Create a large figure for the dashboard
    fig = Figure(resolution=(1600, 1200))
    
    # Define title
    Label(fig[1, 1:2], "Stefan Problem Fixed Point Analysis\nParameter Study Dashboard", 
          fontsize=24, font=:bold)
    
    # Define base parameters
    mesh_size_base = 80
    dt_base = 0.001
    xf_initial_base = 0.05
    alpha_base = 1.0
    
    # ========== Mesh Size Study ==========
    mesh_sizes = [20, 40, 80, 160]
    
    # Create data
    L_mesh_results = Dict{Int, Float64}()
    xf_mesh_results = Dict{Int, Float64}()
    
    for nx in mesh_sizes
        println("\nAnalyzing mesh size: $nx")
        
        # Run a quick single step analysis
        L_values, _, xf_values, _ = analyze_stefan_fixed_point_over_time(
            xf_initial_base, 1, mesh_size=nx, Δt=dt_base, 
            num_samples=50, alpha=alpha_base
        )
        
        L_mesh_results[nx] = L_values[1]
        xf_mesh_results[nx] = xf_values[2]  # Second value is after first time step
    end
    
    # Plot mesh size vs L
    ax_mesh_L = Axis(fig[2, 1], 
                     xlabel="Mesh Size", ylabel="Lipschitz Constant",
                     title="Effect of Mesh Size on Lipschitz Constant")
    
    scatter!(ax_mesh_L, collect(keys(L_mesh_results)), collect(values(L_mesh_results)), 
            markersize=10, color=:blue)
    scatter!(ax_mesh_L, collect(keys(L_mesh_results)), collect(values(L_mesh_results)), 
          color=:blue)
    
    # Add reference line at L=1
    hlines!(ax_mesh_L, [1.0], color=:red, linestyle=:dash, linewidth=2)
    
    # ========== Time Step Study ==========
    time_steps = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    
    # Create data
    L_dt_results = Dict{Float64, Float64}()
    xf_dt_results = Dict{Float64, Float64}()
    
    for dt in time_steps
        println("\nAnalyzing time step: $dt")
        
        # Run a quick single step analysis
        L_values, _, xf_values, _ = analyze_stefan_fixed_point_over_time(
            xf_initial_base, 1, mesh_size=mesh_size_base, Δt=dt, 
            num_samples=50, alpha=alpha_base
        )
        
        L_dt_results[dt] = L_values[1]
        xf_dt_results[dt] = xf_values[2]  # Second value is after first time step
    end
    
    # Plot time step vs L
    ax_dt_L = Axis(fig[2, 2], 
                   xlabel="Δt", ylabel="Lipschitz Constant",
                   title="Effect of Time Step on Lipschitz Constant")
    
    scatter!(ax_dt_L, collect(keys(L_dt_results)), collect(values(L_dt_results)), 
            markersize=10, color=:blue)
    scatter!(ax_dt_L, collect(keys(L_dt_results)), collect(values(L_dt_results)), 
          color=:blue)
    
    # Add reference line at L=1
    hlines!(ax_dt_L, [1.0], color=:red, linestyle=:dash, linewidth=2)
    
    # ========== Alpha Study ==========
    alphas = [0.75, 1.0, 1.25]
    
    # Create data
    L_alpha_results = Dict{Float64, Float64}()
    xf_alpha_results = Dict{Float64, Float64}()
    
    for a in alphas
        println("\nAnalyzing alpha: $a")
        
        # Run a quick single step analysis
        L_values, _, xf_values, _ = analyze_stefan_fixed_point_over_time(
            xf_initial_base, 1, mesh_size=mesh_size_base, Δt=dt_base, 
            num_samples=50, alpha=a
        )
        
        L_alpha_results[a] = L_values[1]
        xf_alpha_results[a] = xf_values[2]  # Second value is after first time step
    end
    
    # Plot alpha vs L
    ax_alpha_L = Axis(fig[3, 1], 
                      xlabel="α", ylabel="Lipschitz Constant",
                      title="Effect of Newton Parameter α on Lipschitz Constant")
    
    scatter!(ax_alpha_L, collect(keys(L_alpha_results)), collect(values(L_alpha_results)), 
            markersize=10, color=:blue)
    scatter!(ax_alpha_L, collect(keys(L_alpha_results)), collect(values(L_alpha_results)), 
          color=:blue)
    
    # Add reference line at L=1
    hlines!(ax_alpha_L, [1.0], color=:red, linestyle=:dash, linewidth=2)
    
    # Find optimal alpha (where L is minimized)
    optimal_alpha = argmin(L_alpha_results)
    vlines!(ax_alpha_L, [optimal_alpha], color=:green, linestyle=:dash, linewidth=2)
    text!(ax_alpha_L, "Optimal α ≈ $(round(optimal_alpha, digits=2))",
         position=(optimal_alpha, minimum(values(L_alpha_results))-0.05),
         fontsize=12, align=(:center, :top))
    
    # ========== Initial Position Study ==========
    xf_initials = [0.05, 0.1, 0.2, 0.3, 0.4]
    
    # Create data
    L_xf0_results = Dict{Float64, Float64}()
    xf_xf0_results = Dict{Float64, Float64}()
    
    for xf0 in xf_initials
        println("\nAnalyzing initial position: $xf0")
        
        # Run a quick single step analysis
        L_values, _, xf_values, _ = analyze_stefan_fixed_point_over_time(
            xf0, 1, mesh_size=mesh_size_base, Δt=dt_base, 
            num_samples=50, alpha=alpha_base
        )
        
        L_xf0_results[xf0] = L_values[1]
        xf_xf0_results[xf0] = xf_values[2]  # Second value is after first time step
    end
    
    # Plot initial position vs L
    ax_xf0_L = Axis(fig[3, 2], 
                    xlabel="Initial Position", ylabel="Lipschitz Constant",
                    title="Effect of Initial Position on Lipschitz Constant")
    
    scatter!(ax_xf0_L, collect(keys(L_xf0_results)), collect(values(L_xf0_results)), 
            markersize=10, color=:blue)
    scatter!(ax_xf0_L, collect(keys(L_xf0_results)), collect(values(L_xf0_results)), 
          color=:blue)
    
    # Add reference line at L=1
    hlines!(ax_xf0_L, [1.0], color=:red, linestyle=:dash, linewidth=2)
    
    # ========== Recommendations and Analysis ==========
    parameter_text = "Parameter Study Summary:\n" *
                    "- Mesh size effect: $(L_mesh_results[minimum(keys(L_mesh_results))]) → $(L_mesh_results[maximum(keys(L_mesh_results))])\n" *
                    "- Time step effect: $(round(L_dt_results[minimum(keys(L_dt_results))], digits=4)) → $(round(L_dt_results[maximum(keys(L_dt_results))], digits=4))\n" *
                    "- Alpha effect: $(round(minimum(values(L_alpha_results)), digits=4)) (best) vs $(round(maximum(values(L_alpha_results)), digits=4)) (worst)\n" *
                    "- Initial position effect: $(round(minimum(values(L_xf0_results)), digits=4)) → $(round(maximum(values(L_xf0_results)), digits=4))"
    
    # Find optimal parameters
    best_mesh = argmin(L_mesh_results)
    best_dt = argmin(L_dt_results)
    best_alpha = argmin(L_alpha_results)
    best_xf0 = argmin(L_xf0_results)
    
    recommendations = "Recommended Parameter Values:\n" *
                     "- Mesh size: nx = $best_mesh\n" *
                     "- Time step: Δt = $best_dt\n" *
                     "- Alpha: α = $(round(best_alpha, digits=4))\n" *
                     "- Initial position has less impact, but xf₀ = $best_xf0 gives best results"
    
    # Add text boxes
    text_box1 = Axis(fig[4, 1], aspect=DataAspect())
    hidedecorations!(text_box1)
    hidespines!(text_box1)
    text!(text_box1, parameter_text, position=(0.05, 0.7), 
         fontsize=14, align=(:left, :center))
    
    text_box2 = Axis(fig[4, 2], aspect=DataAspect())
    hidedecorations!(text_box2)
    hidespines!(text_box2)
    text!(text_box2, recommendations, position=(0.05, 0.7), 
         fontsize=14, align=(:left, :center))
    
    # Save the dashboard
    save("stefan_parameter_dashboard.png", fig)
    
    return fig
end

# Execute the parameter studies
println("Running Stefan Problem Parameter Study")
println("=====================================")

# Create the comprehensive dashboard
dashboard_fig = create_parameter_dashboard()
display(dashboard_fig)

# Comment/uncomment these for detailed individual studies
#mesh_fig, mesh_results = mesh_convergence_study(Δt=0.001, xf_initial=0.1, alpha=1.0, num_time_steps=3, num_samples=100)
#time_fig, time_results = time_step_study()
#alpha_fig, alpha_results = alpha_study([0.5, 0.75, 1.0, 1.25])
#pos_fig, pos_results = initial_position_study([0.05, 0.1, 0.2, 0.3])

println("\nParameter study complete!")
println("Dashboard saved as: stefan_parameter_dashboard.png")
println("\nRecommendations:")
println("For optimal convergence of the fixed-point iteration:")
println("1. Use the finest mesh computationally feasible")
println("2. Use smaller time steps when possible")
println("3. Tune the relaxation parameter α (typically 0.5-1.0 works well)")
println("4. Initial position has less impact on convergence properties")

"""

"""