using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations
using Statistics

"""
Analyze a discrete function by computing its Lipschitz constant and visualizing
its behavior, slopes, and potential fixed points.
"""
function analyze_discrete_function(x_points, y_points; plot_title="Discrete Function Analysis", 
                                  fit_curve=false, interpolation_method=:linear,
                                  mesh_size=0, domain_range=(0.0, 1.0))
    # Check inputs
    if length(x_points) != length(y_points)
        error("Length of x_points and y_points must match")
    end
    if length(x_points) < 2
        error("Need at least two points to compute a Lipschitz constant")
    end
    
    # Make sure points are sorted by x
    sorted_indices = sortperm(x_points)
    x_sorted = x_points[sorted_indices]
    y_sorted = y_points[sorted_indices]
    
    # Compute differences between consecutive points
    dx = diff(x_sorted)
    dy = diff(y_sorted)
    
    # Calculate slopes
    slopes = dy ./ dx
    
    # Compute Lipschitz constant (maximum absolute slope)
    L = maximum(abs.(slopes))
    
    # Create plot
    fig = Figure(resolution=(1000, 800))
    ax = Axis(fig[1, 1], 
              xlabel="x", ylabel="f(x)", 
              title="$plot_title\nLipschitz constant: L = $(round(L, digits=4))")
    
    # Plot the original points
    lines!(ax, x_sorted, y_sorted, color=:blue, linewidth=2, label="f(x)")
    #scatter!(ax, x_sorted, y_sorted, color=:blue, markersize=8, label="Sampled points")
    
    # Connect the points with straight lines
    lines!(ax, x_sorted, y_sorted, color=:blue, linewidth=2, label="f(x)")
    
    # Add identity line y=x
    identity_line = range(minimum(x_sorted), maximum(x_sorted), length=100)
    lines!(ax, identity_line, identity_line, 
           color=:black, linestyle=:dash, linewidth=2, label="y = x")
    
    # Add mesh face lines if mesh_size is provided
    if mesh_size > 0
        x0, lx = domain_range
        dx_mesh = (lx - x0) / mesh_size
        
        # Generate mesh face positions
        mesh_faces = [dx_mesh/2 + i*dx_mesh for i in 0:mesh_size]
        
        # Plot vertical lines for mesh faces
        for face_pos in mesh_faces
            if face_pos >= minimum(x_sorted) && face_pos <= maximum(x_sorted)
                vlines!(ax, [face_pos], color=:gray, linestyle=:dot, linewidth=2, 
                       label=(face_pos == mesh_faces[1] ? "Mesh faces" : nothing))
            end
        end
    end
    
    # Plot derivatives/slopes
    x_mid = (x_sorted[1:end-1] .+ x_sorted[2:end]) ./ 2
    
    # Add derivative plot
    ax2 = Axis(fig[2, 1], 
               xlabel="x", ylabel="f'(x)", 
               title="Derivative / Slope Analysis")
    
    #scatter!(ax2, x_mid, slopes, color=:red, markersize=8, label="Point-to-point slopes")
    lines!(ax2, x_mid, slopes, color=:red, linewidth=2)
    
    # Add mesh face lines to derivative plot too
    if mesh_size > 0
        x0, lx = domain_range
        dx_mesh = (lx - x0) / mesh_size
        
        # Generate mesh face positions
        mesh_faces = [dx_mesh/2 + i*dx_mesh for i in 0:mesh_size]
        
        # Plot vertical lines for mesh faces
        for face_pos in mesh_faces
            if face_pos >= minimum(x_mid) && face_pos <= maximum(x_mid)
                vlines!(ax2, [face_pos], color=:gray, linestyle=:dot, linewidth=2, 
                        label=(face_pos == mesh_faces[1] ? "Mesh faces" : nothing))
            end
        end
    end
    
    # Find the max slope point
    max_slope_idx = findmax(abs.(slopes))[2]
    max_slope_x = x_mid[max_slope_idx]
    max_slope_y = slopes[max_slope_idx]
    
    # Highlight the max slope
    scatter!(ax2, [max_slope_x], [max_slope_y], color=:green, markersize=15, 
             marker=:star5, label="Max slope: $(round(max_slope_y, digits=4))")
    
    # Add horizontal lines for Lipschitz constant
    hlines!(ax2, [L], color=:darkgreen, linestyle=:dash, linewidth=2, 
            label="Lipschitz constant: $(round(L, digits=4))")
    hlines!(ax2, [-L], color=:darkgreen, linestyle=:dash, linewidth=2)
    
    # Find fixed points (where f(x) ≈ x)
    fixed_points = Float64[]
    for i in 1:length(x_sorted)
        if abs(y_sorted[i] - x_sorted[i]) < 1e-6
            push!(fixed_points, x_sorted[i])
        end
    end
    
    # Use linear interpolation to find more precise fixed points
    if isempty(fixed_points) && length(x_sorted) > 2
        # Create a function g(x) = f(x) - x
        g_vals = y_sorted - x_sorted
        
        # Look for sign changes in g(x)
        for i in 1:length(g_vals)-1
            if g_vals[i] * g_vals[i+1] <= 0
                # Sign change detected, approximate fixed point by linear interpolation
                x1, x2 = x_sorted[i], x_sorted[i+1]
                g1, g2 = g_vals[i], g_vals[i+1]
                
                # Linear interpolation to find where g(x) = 0
                x_fixed = x1 - g1 * (x2 - x1) / (g2 - g1)
                push!(fixed_points, x_fixed)
            end
        end
    end
    
    # If fixed points found, highlight them
    if !isempty(fixed_points)
        # Interpolate to get f(fixed_point)
        itp = cubic_spline_interpolation(x_sorted, y_sorted, extrapolation_bc=Flat())
        fixed_y = itp.(fixed_points)
        
        scatter!(ax, fixed_points, fixed_y, 
                color=:magenta, markersize=15, marker=:star8, 
                label="Fixed points: x = f(x)")
                
        # Add text labels for fixed point values
        for (i, (fx, fy)) in enumerate(zip(fixed_points, fixed_y))
            text!(ax, "x* = $(round(fx, digits=5))", 
                  position=(fx, fy + 0.02*maximum(y_sorted)), 
                  fontsize=12, align=(:center, :bottom), color=:magenta)
        end
    end
    
    # Add legends
    axislegend(ax, position=:rt)
    axislegend(ax2, position=:rt)
    
    # Return the figure and Lipschitz constant
    return fig, L, fixed_points
end

"""
Sample and analyze the fixed-point function of the Stefan problem solver
for a specific time step.
"""
function analyze_stefan_fixed_point_function(xf_current, xf_range, num_samples; 
                                           mesh_size=40, Δt=0.001, max_iter=20, 
                                           alpha=1.0, time_step_index=1)
    # Define domain parameters
    nx = mesh_size
    lx = 1.0
    x0 = 0.0
    domain = ((x0, lx),)
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    
    # Define the Space-Time mesh
    STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
    
    # Sample points across the range
    xf_samples = range(xf_range[1], xf_range[2], length=num_samples)
    xf_next = zeros(length(xf_samples))
    
    # Prepare progress display
    println("Sampling fixed point function at $(length(xf_samples)) points...")
    
    # Initialize temp vectors to store intermediate results
    residuals = Float64[]
    
    # For each sample point, compute one step of the fixed-point iteration
    for (i, xf) in enumerate(xf_samples)
        # Define the body function for current xf
        body = (x, t, _=0) -> (x - xf)
        
        # Define capacity, operator, source, etc.
        capacity = Capacity(body, STmesh)
        operator = DiffusionOps(capacity)
        f = (x, y, z, t) -> 0.0  # Source term
        K = (x, y, z) -> 1.0     # Diffusion coefficient
        
        # Define the phase
        phase = Phase(capacity, operator, f, K)
        
        # Define boundary conditions
        bc = Dirichlet(0.0)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
        ρL = 1.0  # Latent heat parameter
        stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρL))
        
        # Initial condition
        u0ₒ = zeros(nx+1)
        u0ᵧ = zeros(nx+1)
        u0 = vcat(u0ₒ, u0ᵧ)
        
        # Define the solver
        solver = MovingLiquidDiffusionUnsteadyMono(phase, bc_b, bc, Δt, u0, mesh, "BE")
        
        # Solve one iteration
        solve_system!(solver, method=Base.:\)
        
        # Get temperature field
        Tᵢ = solver.x
        
        # Extract operator dimensions and size
        dims = phase.operator.size
        len_dims = length(dims)
        cap_index = len_dims
        
        # Create the 1D or 2D indices
        if len_dims == 2
            # 1D case
            nx, _ = dims
            n = nx
        elseif len_dims == 3
            # 2D case
            nx, ny, _ = dims
            n = nx*ny
        end
        
        # Update volumes / compute new interface position
        Vn_1 = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
        Vn   = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
        Hₙ   = sum(diag(Vn))
        Hₙ₊₁ = sum(diag(Vn_1))
        
        # Compute interface flux
        W! = phase.operator.Wꜝ[1:n, 1:n]
        G = phase.operator.G[1:n, 1:n]
        H = phase.operator.H[1:n, 1:n]
        V = phase.operator.V[1:n, 1:n]
        Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id = Id[1:n, 1:n]
        Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
        Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        Interface_term = 1/ρL * sum(Interface_term)
        
        # Compute residual
        res = Hₙ₊₁ - Hₙ - Interface_term
        push!(residuals, res)
        
        # Apply Newton update rule to get next position
        #xf_next[i] = xf + alpha * res
        xf_next[i] = Hₙ + alpha * res

        # Show progress for every 20% of points
        if i % max(1, num_samples ÷ 5) == 0
            println("  Processed $(i)/$(num_samples) points ($(round(i/num_samples*100))%)")
        end
    end
    
    # Analyze the discrete function
    title = "Stefan Problem Fixed-Point Analysis (Time step $(time_step_index), α=$(alpha))"
    fig, L, fixed_points = analyze_discrete_function(
        xf_samples, xf_next, 
        plot_title=title,
        mesh_size=mesh_size,
        domain_range=(0.0, 1.0)
    )
    
    # Add residual plot
    ax3 = Axis(fig[3, 1], 
              xlabel="xf", ylabel="Residual", 
              title="Residual Function")
    
    lines!(ax3, xf_samples, residuals, linewidth=2, color=:red)
    scatter!(ax3, xf_samples, residuals, markersize=5, color=:red)
    
    # Add zero line
    hlines!(ax3, [0.0], color=:black, linestyle=:dash, linewidth=1)
    
    # Highlight zero crossings (fixed points of the original function)
    if !isempty(fixed_points)
        # Interpolate to get residual at fixed points
        itp_res = linear_interpolation(xf_samples, residuals, extrapolation_bc=Flat())
        res_at_fixed = itp_res.(fixed_points)
        
        scatter!(ax3, fixed_points, res_at_fixed, 
                color=:magenta, markersize=15, marker=:star8, 
                label="Fixed points (zero residual)")
    end
    
    # Add convergence prediction
    if L < 1.0
        label = "L < 1: Contraction, converges to fixed point"
        color = :green
    else
        label = "L > 1: Not contractive, may not converge"
        color = :red
    end
    
    text!(fig[1, 1], label, position=(mean(xf_range), minimum(xf_next)), 
          fontsize=14, color=color, align=(:center, :bottom))
    
    # Add iteration simulation if there is a fixed point
    if !isempty(fixed_points)
        ax4 = Axis(fig[4, 1], 
                  xlabel="Iteration", ylabel="Interface Position", 
                  title="Iteration Simulation")
        
        # Create interpolation function for xf_next
        interp_func = linear_interpolation(xf_samples, xf_next, extrapolation_bc=Flat())
        
        # Starting from xf_current, simulate iterations
        iterations = [xf_current]
        iter_x = 0:max_iter
        
        x_curr = xf_current
        for i in 1:max_iter
            x_next = interp_func(x_curr)
            push!(iterations, x_next)
            
            # Check for convergence
            if abs(x_next - x_curr) < 1e-6
                iter_x = 0:i
                break
            end
            
            x_curr = x_next
        end
        
        # Plot iterations
        lines!(ax4, iter_x, iterations, linewidth=2, color=:blue)
        scatter!(ax4, iter_x, iterations, markersize=8, color=:blue, 
                label="Iterations starting from x₀=$(round(xf_current, digits=5))")
        
        # Add horizontal line for fixed point if any
        if !isempty(fixed_points)
            hlines!(ax4, [fixed_points[1]], color=:magenta, linestyle=:dash, linewidth=2, 
                   label="Fixed point x*=$(round(fixed_points[1], digits=5))")
        end
        
        axislegend(ax4, position=:rt)
    end
        
    # Save figure to file
    save("stefan_fixed_point_analysis_step$(time_step_index).png", fig)
    
    return fig, L, fixed_points, residuals
end

"""
Analyze the fixed-point function at multiple time steps to observe its evolution.
"""
function analyze_stefan_fixed_point_over_time(xf_initial, num_time_steps=5; 
                                            mesh_size=40, Δt=0.001, num_samples=50,
                                            alpha=1.0)
    # Store Lipschitz constants over time
    time_steps = collect(1:num_time_steps)
    L_values = zeros(num_time_steps)
    fixed_points_history = Vector{Vector{Float64}}(undef, num_time_steps)
    xf_values = [xf_initial]
    
    # Run initial model to get the true evolution of xf
    nx = mesh_size
    lx = 1.0
    x0 = 0.0
    domain = ((x0, lx),)
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    
    # Define the body
    body = (x, t, _=0) -> (x - xf_initial)
    
    # Define the Space-Time mesh
    STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
    
    # Define capacity and operators
    capacity = Capacity(body, STmesh)
    operator = DiffusionOps(capacity)
    
    # Define boundary conditions
    bc = Dirichlet(0.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
    ρ, L = 1.0, 1.0
    stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))
    
    # Define the source term
    f = (x, y, z, t) -> 0.0
    K = (x, y, z) -> 1.0
    
    # Define the phase
    phase = Phase(capacity, operator, f, K)
    
    # Initial condition
    u0ₒ = zeros((nx+1))
    u0ᵧ = zeros((nx+1))
    u0 = vcat(u0ₒ, u0ᵧ)
    
    # Newton parameters
    Newton_params = (1000, 1e-15, 1e-15, alpha)
    
    # Define the solver
    solver = MovingLiquidDiffusionUnsteadyMono(phase, bc_b, bc, Δt, u0, mesh, "BE")
    
    # Solve for few steps to get xf evolution
    solver, _, xf_log, _ = solve_MovingLiquidDiffusionUnsteadyMono!(
        solver, phase, xf_initial, Δt, Δt*(num_time_steps+1), 
        bc_b, bc, stef_cond, mesh, "BE", 
        Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\
    )
    
    # For each time step, analyze the fixed-point function
    for i in 1:num_time_steps
        println("\nAnalyzing Time Step $(i)...")
        current_xf = xf_log[i]
        push!(xf_values, current_xf)
        
        # Define range around the current xf for sampling
        sample_width = 0.5*current_xf
        xf_range = (current_xf - sample_width, current_xf + sample_width)
        
        # Analyze fixed point function
        _, L, fixed_points, _ = analyze_stefan_fixed_point_function(
            current_xf, xf_range, num_samples, 
            mesh_size=mesh_size, Δt=Δt, alpha=alpha, time_step_index=i
        )
        
        # Store results
        L_values[i] = L
        fixed_points_history[i] = fixed_points
    end
    
    # Create summary figure
    fig_summary = Figure(resolution=(900, 600))
    
    # Plot the Lipschitz constant evolution
    ax1 = Axis(fig_summary[1, 1], 
              xlabel="Time Step", ylabel="Lipschitz Constant",
              title="Evolution of Lipschitz Constant")
    
    lines!(ax1, time_steps, L_values, linewidth=3, color=:blue)
    scatter!(ax1, time_steps, L_values, markersize=10, color=:blue)
    
    # Add reference line at L=1
    hlines!(ax1, [1.0], color=:red, linestyle=:dash, linewidth=2, 
           label="Critical value L=1")
    
    # Color regions for convergence
    for i in 1:length(L_values)-1
        if L_values[i] < 1.0
            color = (:green, 0.1)
        else
            color = (:red, 0.1)
        end
        poly!(ax1, [i, i+1, i+1, i], [0, 0, L_values[i+1], L_values[i]], color=color)
    end
    
    # Add convergence region labels
    text!(ax1, "Convergent (L<1)", position=(mean(time_steps), 0.5), 
          fontsize=14, color=:darkgreen, align=(:center, :bottom))
    text!(ax1, "Divergent (L>1)", position=(mean(time_steps), 1.5), 
          fontsize=14, color=:darkred, align=(:center, :bottom))
    
    axislegend(ax1)
    
    # Plot the interface position evolution
    ax2 = Axis(fig_summary[2, 1],
              xlabel="Time Step", ylabel="Interface Position",
              title="Evolution of Interface Position")
    
    lines!(ax2, 0:num_time_steps, xf_values, linewidth=3, color=:blue)
    scatter!(ax2, 0:num_time_steps, xf_values, markersize=10, color=:blue, 
            label="xf evolution")
    
    # Add convergence info based on L values
    for i in 1:num_time_steps
        if L_values[i] < 1.0
            marker = :circle
            color = :green
            size = 12
        else
            marker = :xcross
            color = :red
            size = 12
        end
        
        scatter!(ax2, [i], [xf_values[i+1]], marker=marker, 
                color=color, markersize=size)
    end
    
    axislegend(ax2)
    
    # Save the summary figure
    save("stefan_fixed_point_summary.png", fig_summary)
    
    return L_values, fixed_points_history, xf_values, fig_summary
end


# Run the analysis
println("Starting Stefan Problem Fixed-Point Analysis")
println("============================================")

# Set parameters
mesh_size = 80
Δt = 0.001
xf_initial = 0.05
alpha = 1.0  # Newton relaxation parameter
num_time_steps = 5
num_samples = 1000

# Run the analysis
L_values, fixed_points, xf_values, fig_summary = analyze_stefan_fixed_point_over_time(
    xf_initial, num_time_steps, 
    mesh_size=mesh_size, Δt=Δt, num_samples=num_samples, alpha=alpha
)

# Display results summary
println("\nAnalysis Complete!")
println("==================")
println("Fixed-point analysis across $(num_time_steps) time steps:")
for i in 1:num_time_steps
    println("Time Step $(i):")
    println("  - Interface Position: $(round(xf_values[i+1], digits=6))")
    println("  - Lipschitz Constant: $(round(L_values[i], digits=4))")
    
    if !isempty(fixed_points[i])
        println("  - Fixed Points: $(round.(fixed_points[i], digits=6))")
    else
        println("  - No fixed points found in sampled range")
    end
    
    # Convergence prediction
    if L_values[i] < 1.0
        println("  - Prediction: Convergent (L < 1)")
    else
        println("  - Prediction: May diverge (L > 1)")
    end
    println()
end

display(fig_summary)
println("Individual time step analysis plots saved as: stefan_fixed_point_analysis_step*.png")
println("Summary plot saved as: stefan_fixed_point_summary.png")

using Penguin, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using Printf
using Statistics
"""
Plot the fixed point functions from multiple time steps on a single figure
with improved visibility and clarity.
"""
function plot_fixed_point_functions_improved(xf_samples_list, xf_next_list, time_steps;
                                           fixed_points_list=nothing, L_values=nothing,
                                           title="Évolution des Fonctions de Point Fixe",
                                           domain_range=(0.0, 1.0))
    # Create figure with larger size for better visibility
    fig = Figure(resolution=(1200, 900))
    
    # Main plot for fixed point functions - make it bigger by using more space
    ax = Axis(fig[1:2, 1:2], 
              xlabel="xf", ylabel="f(xf)",
              title=title,
              aspect=1.5)  # Better aspect ratio for visibility
    
    # Add identity line
    x_min = minimum([minimum(xs) for xs in xf_samples_list])
    x_max = maximum([maximum(xs) for xs in xf_samples_list])
    identity_line = range(x_min, x_max, length=100)
    lines!(ax, identity_line, identity_line, 
           color=:black, linestyle=:dash, linewidth=3, label="y = x")
    
    # Use a colormap for better differentiation between time steps
    n_steps = length(time_steps)
    cmap = cgrad(:viridis, n_steps, categorical=true)
    
    # Store legend entries for better organization
    legend_entries = []
    
    # Plot each fixed point function with better visibility
    for (i, (xf_samples, xf_next)) in enumerate(zip(xf_samples_list, xf_next_list))
        # Use color from colormap
        color = cmap[i]
        
        # Plot with thicker lines for better visibility
        lines!(ax, xf_samples, xf_next, 
              linewidth=3, color=color,
              label="Pas $(time_steps[i])" * 
                   (L_values !== nothing ? ", L=$(round(L_values[i], digits=2))" : ""))
        
        # Add to legend entries
        push!(legend_entries, (
            LineElement(linewidth=3, color=color),
            "Pas $(time_steps[i]), L=$(round(L_values[i], digits=2))"
        ))
        
        # Highlight fixed points if provided
        if fixed_points_list !== nothing && !isempty(fixed_points_list[i])
            # Create interpolation to get y values at fixed points
            itp = linear_interpolation(xf_samples, xf_next, extrapolation_bc=Flat())
            
            # Plot fixed points
            for fp in fixed_points_list[i]
                try
                    fp_y = itp(fp)
                    scatter!(ax, [fp], [fp_y],
                            markersize=15, color=color, marker=:star8)
                    
                    # Add label for fixed point
                    text!(ax, "x* = $(round(fp, digits=4))",
                         position=(fp, fp_y + 0.01),
                         fontsize=12, color=color, 
                         align=(:center, :bottom))
                catch e
                    println("Warning: Could not plot fixed point $(fp) for time step $(i)")
                end
            end
        end
    end
    
    # Add a well-formatted legend with custom entries
    #Legend(fig[1:2, 3], legend_entries, ["Fonctions de point fixe"])
    
    return fig
end

"""
Enhanced version of the analyze_stefan_fixed_point_over_time function
with more time steps and improved visualization.
"""
function analyze_stefan_fixed_point_over_time_improved(xf_initial, num_time_steps=20; 
                                                     mesh_size=40, Δt=0.001, num_samples=50,
                                                     alpha=1.0)
    # Store Lipschitz constants over time
    time_steps = collect(1:num_time_steps)
    L_values = zeros(num_time_steps)
    fixed_points_history = Vector{Vector{Float64}}(undef, num_time_steps)
    xf_values = [xf_initial]
    
    # Store data for combined plot
    xf_samples_list = []
    xf_next_list = []
    
    # Run initial model to get the true evolution of xf
    nx = mesh_size
    lx = 1.0
    x0 = 0.0
    domain = ((x0, lx),)
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    
    # Define the body
    body = (x, t, _=0) -> (x - xf_initial)
    
    # Define the Space-Time mesh
    STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
    
    # Define capacity and operators
    capacity = Capacity(body, STmesh)
    operator = DiffusionOps(capacity)
    
    # Define boundary conditions
    bc = Dirichlet(0.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
    ρ, L = 1.0, 1.0
    stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))
    
    # Define the source term
    f = (x, y, z, t) -> 0.0
    K = (x, y, z) -> 1.0
    
    # Define the phase
    phase = Phase(capacity, operator, f, K)
    
    # Initial condition
    u0ₒ = zeros((nx+1))
    u0ᵧ = zeros((nx+1))
    u0 = vcat(u0ₒ, u0ᵧ)
    
    # Newton parameters
    Newton_params = (1000, 1e-15, 1e-15, alpha)
    
    # Define the solver
    solver = MovingLiquidDiffusionUnsteadyMono(phase, bc_b, bc, Δt, u0, mesh, "BE")
    
    # Solve for all time steps to get xf evolution
    solver, _, xf_log, _ = solve_MovingLiquidDiffusionUnsteadyMono!(
        solver, phase, xf_initial, Δt, Δt*(num_time_steps+1), 
        bc_b, bc, stef_cond, mesh, "BE", 
        Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\
    )
    
    # For each time step, analyze the fixed-point function
    for i in 1:num_time_steps
        println("\nAnalyzing Time Step $(i)...")
        current_xf = xf_log[i]
        push!(xf_values, current_xf)
        
        # Define range around the current xf for sampling
        sample_width = 0.5*current_xf
        xf_range = (max(0.001, current_xf - sample_width), 
                   min(1.0, current_xf + sample_width))
        
        # Generate samples for this time step
        xf_samples = range(xf_range[1], xf_range[2], length=num_samples)
        xf_next = zeros(length(xf_samples))
        
        # For each sample point, compute one step of the fixed-point iteration
        for (j, xf) in enumerate(xf_samples)
            # Define the body function for current xf
            body = (x, t, _=0) -> (x - xf)
            
            # Define capacity, operator, source, etc.
            capacity = Capacity(body, STmesh)
            operator = DiffusionOps(capacity)
            f = (x, y, z, t) -> 0.0
            K = (x, y, z) -> 1.0
            phase = Phase(capacity, operator, f, K)
            
            # Define boundary conditions
            bc = Dirichlet(0.0)
            bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
                :top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
            ρL = 1.0
            
            # Initial condition
            u0ₒ = zeros(nx+1)
            u0ᵧ = zeros(nx+1)
            u0 = vcat(u0ₒ, u0ᵧ)
            
            # Define the solver
            solver = MovingLiquidDiffusionUnsteadyMono(phase, bc_b, bc, Δt, u0, mesh, "BE")
            
            # Solve one iteration
            solve_system!(solver, method=Base.:\)
            
            # Extract dimensions
            dims = phase.operator.size
            len_dims = length(dims)
            cap_index = len_dims
            
            if len_dims == 2
                nx, _ = dims
                n = nx
            elseif len_dims == 3
                nx, ny, _ = dims
                n = nx*ny
            end
            
            # Update volumes / compute new interface position
            Vn_1 = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
            Vn = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
            Hₙ = sum(diag(Vn))
            Hₙ₊₁ = sum(diag(Vn_1))
            
            # Compute interface flux
            Tᵢ = solver.x
            W! = phase.operator.Wꜝ[1:n, 1:n]
            G = phase.operator.G[1:n, 1:n]
            H = phase.operator.H[1:n, 1:n]
            V = phase.operator.V[1:n, 1:n]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:n, 1:n]
            Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
            Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            Interface_term = 1/ρL * sum(Interface_term)
            
            # Apply update rule
            xf_next[j] = Hₙ + alpha * Interface_term
        end
        
        # Store data for combined plot
        push!(xf_samples_list, xf_samples)
        push!(xf_next_list, xf_next)
        
        # Compute Lipschitz constant and find fixed points
        # Make sure points are sorted by x
        sorted_indices = sortperm(xf_samples)
        x_sorted = xf_samples[sorted_indices]
        y_sorted = xf_next[sorted_indices]
        
        # Compute differences between consecutive points
        dx = diff(x_sorted)
        dy = diff(y_sorted)
        
        # Calculate slopes
        slopes = dy ./ dx
        
        # Compute Lipschitz constant (maximum absolute slope)
        L = maximum(abs.(slopes))
        
        # Find fixed points (where f(x) ≈ x)
        fixed_points = Float64[]
        g_vals = y_sorted - x_sorted
        
        # Look for sign changes in g(x)
        for k in 1:length(g_vals)-1
            if g_vals[k] * g_vals[k+1] <= 0
                # Sign change detected, approximate fixed point by linear interpolation
                x1, x2 = x_sorted[k], x_sorted[k+1]
                g1, g2 = g_vals[k], g_vals[k+1]
                
                # Linear interpolation to find where g(x) = 0
                x_fixed = x1 - g1 * (x2 - x1) / (g2 - g1)
                push!(fixed_points, x_fixed)
            end
        end
        
        # Store results
        L_values[i] = L
        fixed_points_history[i] = fixed_points
    end
    
    # Create the combined fixed point function plot
    fig_fixed_points = plot_fixed_point_functions_improved(
        xf_samples_list, xf_next_list, time_steps,
        fixed_points_list=fixed_points_history, L_values=L_values,
        title="Évolution des Fonctions de Point Fixe (α=$alpha, Δt=$Δt, $num_time_steps pas)",
        domain_range=(0.0, 1.0)
    )
    
    # Save and display figure
    save("stefan_fixed_point_functions_combined.png", fig_fixed_points)
    display(fig_fixed_points)
    
    # Create summary figure
    fig_summary = Figure(resolution=(1000, 800))
    
    # Plot the Lipschitz constant evolution
    ax1 = Axis(fig_summary[1, 1], 
              xlabel="Pas de temps", ylabel="Constante de Lipschitz",
              title="Évolution de la Constante de Lipschitz")
    
    lines!(ax1, time_steps, L_values, linewidth=3, color=:blue)
    scatter!(ax1, time_steps, L_values, markersize=10, color=:blue)
    
    # Add reference line at L=1
    hlines!(ax1, [1.0], color=:red, linestyle=:dash, linewidth=2, 
           label="Valeur critique L=1")
    
    # Color regions for convergence
    for i in 1:length(L_values)-1
        if L_values[i] < 1.0
            color = (:green, 0.1)
        else
            color = (:red, 0.1)
        end
        poly!(ax1, [i, i+1, i+1, i], [0, 0, L_values[i+1], L_values[i]], color=color)
    end
    
    axislegend(ax1)
    
    # Plot the interface position evolution
    ax2 = Axis(fig_summary[2, 1],
              xlabel="Pas de temps", ylabel="Position de l'interface",
              title="Évolution de la Position de l'Interface")
    
    lines!(ax2, 0:num_time_steps, xf_values, linewidth=3, color=:blue)
    scatter!(ax2, 0:num_time_steps, xf_values, markersize=10, color=:blue, 
            label="Évolution xf")
    
    # Add convergence info based on L values
    for i in 1:num_time_steps
        if L_values[i] < 1.0
            marker = :circle
            color = :green
            size = 12
        else
            marker = :xcross
            color = :red
            size = 12
        end
        
        scatter!(ax2, [i], [xf_values[i+1]], marker=marker, 
                color=color, markersize=size)
    end
    
    axislegend(ax2)
    
    # Save the summary figure
    save("stefan_fixed_point_summary_improved.png", fig_summary)
    display(fig_summary)
    
    return L_values, fixed_points_history, xf_values, fig_fixed_points, fig_summary
end

# Run the improved analysis
println("Starting Enhanced Stefan Problem Fixed-Point Analysis")
println("==================================================")

# Set parameters
mesh_size = 80
Δt = 0.001
xf_initial = 0.05
alpha = 1.0  # Newton relaxation parameter
num_time_steps = 20  # Increased from 5 to 20
num_samples = 200    # Reduced for faster computation with more time steps

# Run the analysis
L_values, fixed_points, xf_values, fig_fixed_points, fig_summary = analyze_stefan_fixed_point_over_time_improved(
    xf_initial, num_time_steps, 
    mesh_size=mesh_size, Δt=Δt, num_samples=num_samples, alpha=alpha
)