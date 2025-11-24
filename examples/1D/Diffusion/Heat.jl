using Penguin
using IterativeSolvers
using CairoMakie  # For plotting
using Statistics  # For mean calculation
### 1D Test Case : Monophasic Unsteady Diffusion Equation 
# Define the mesh
nx = 160
lx = 4.0
x0 = 0.0
domain=((x0,lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
xint = 2.0 + 0.1
body = (x, _=0) -> (x - xint)

# Define the capacity
capacity = Capacity(body, mesh)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc0 = Dirichlet(0.0)
bc1 = Dirichlet(1.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc0, :bottom => bc1))

# Define the source term
f = (x,y,z,t)->0.0
D = (x,y,z)->10.0

# Define the phase
Fluide = Phase(capacity, operator, f, D)

# Initial condition
u0ₒ = zeros(nx+1)
u0ᵧ = zeros(nx+1)
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
Δt = 4.0*(lx/nx)^2
Tend = 1.0
solver = DiffusionUnsteadyMono(Fluide, bc_b, bc0, Δt, u0, "CN")

# Solver parameters
tol = 1e-8
max_iter = 500

# Collection to store convergence histories - will be populated after solving
all_convergence_histories = []
time_values = []

# Solve the problem directly with bicgstabl
solve_DiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, bc0, "CN"; 
                            method=IterativeSolvers.bicgstabl,
                            log=true,
                            reltol=eps())

# After solving, access the convergence histories stored in solver.ch
all_convergence_histories = solver.ch
time_points = collect(0:Δt:Tend+2Δt)

# Updated function to plot convergence histories
function plot_convergence_history(histories, time_points=nothing; 
                                 max_lines=10, plot_final=true)
    fig = Figure(resolution=(1000, 800))
    
    # Create main axis for selected histories
    ax_main = Axis(fig[1, 1], 
                  xlabel="Iteration", 
                  ylabel="Residual Norm", 
                  title="Convergence History",
                  yscale=log10)
    
    # Create axis for final timestep (if requested)
    if plot_final
        ax_final = Axis(fig[1, 2],
                       xlabel="Iteration",
                       ylabel="Residual Norm",
                       title="Final Timestep Convergence",
                       yscale=log10)
    end
    
    # Create axis for iteration count over time
    ax_iters = Axis(fig[2, 1:2],
                    xlabel="Time",
                    ylabel="Iteration Count",
                    title="Iterations Per Timestep")
    
    # Store iteration counts for all timesteps
    iters_per_step = Int[]
    
    # Calculate how many histories to skip to stay under max_lines
    n_histories = length(histories)
    
    if n_histories > max_lines
        # Skip some histories to avoid overcrowding the plot
        step = max(1, floor(Int, n_histories / max_lines))
        indices = 1:step:n_histories
        # Always include the final history
        if indices[end] != n_histories
            indices = [indices..., n_histories]
        end
    else
        indices = 1:n_histories
    end
    
    # Color map for the lines
    colors = cgrad(:viridis, length(indices), categorical=true)
    
    # Plot selected histories on main axis
    for (idx, i) in enumerate(indices)
        ch = histories[i]
        
        # Extract residual norms
        if haskey(ch.data, :resnorm)
            residuals = ch.data[:resnorm]
            iterations = 1:length(residuals)
            
            # Create label
            t_val = isnothing(time_points) ? i : time_points[i]
            label = "t = $(round(t_val, digits=3))"
            
            # Plot on main axis
            lines!(ax_main, iterations, residuals, 
                  label=label, color=colors[idx])
        end
        
        # Store iteration count
        push!(iters_per_step, ch.iters)
    end
    
    # Plot iterations per timestep
    # Plot iterations per timestep
    if !isnothing(time_points) && length(time_points) == length(histories)
        # Ensure time_points and iters_per_step have the same length
        valid_times = time_points[1:length(iters_per_step)]
        
        scatter!(ax_iters, valid_times, iters_per_step, 
                color=:blue, markersize=5)
        lines!(ax_iters, valid_times, iters_per_step, 
              color=:blue, alpha=0.5)
    else
        # If no time_points or length mismatch, use indices instead
        indices = 1:length(iters_per_step)
        scatter!(ax_iters, indices, iters_per_step, 
                color=:blue, markersize=5)
        lines!(ax_iters, indices, iters_per_step, 
              color=:blue, alpha=0.5)
    end
    
    # Plot final timestep in detail
    if plot_final && !isempty(histories)
        final_ch = histories[end]
        
        if haskey(final_ch.data, :resnorm)
            residuals = final_ch.data[:resnorm]
            iterations = 1:length(residuals)
            
            lines!(ax_final, iterations, residuals, 
                  color=:blue, linewidth=2)
            
            # Add convergence stats
            stats_text = """
            Iterations: $(final_ch.iters)
            Converged: $(final_ch.isconverged)
            Mat-Vec Products: $(nprods(final_ch))
            """
            
            # Position text in lower left
            text!(ax_final, 1, minimum(residuals)*10, 
                 text=stats_text, align=(:left, :bottom))
        end
    end
    
    # Add legend to main plot
    axislegend(ax_main, position=:rt)
    
    # Add tolerance line if available
    if !isempty(histories) && haskey(histories[1].data, :tol)
        tol = histories[1].data[:tol]
        hlines!(ax_main, [tol], color=:red, linestyle=:dash, 
               label="Tolerance")
        
        if plot_final
            hlines!(ax_final, [tol], color=:red, linestyle=:dash, 
                   label="Tolerance")
        end
    end
    
    # Main title
    Label(fig[0, :], "IterativeSolvers Convergence History Analysis", 
          fontsize=20)
    
    return fig
end

# Plot the convergence histories with the updated function
convergence_fig = plot_convergence_history(all_convergence_histories, time_points)
display(convergence_fig)
save("convergence_analysis.png", convergence_fig)

# If you want to see just a few specific timesteps
selected_indices = [1, 5, 10, length(all_convergence_histories)]
selected_histories = all_convergence_histories[selected_indices]
selected_times = time_points[selected_indices]

specific_fig = plot_convergence_history(selected_histories, selected_times)
display(specific_fig)
save("selected_convergence_history.png", specific_fig)

# To examine a single convergence history in detail
if !isempty(all_convergence_histories)
    ch = all_convergence_histories[end]  # Last timestep
    
    # Create a focused figure for just this history
    fig_single = Figure(resolution=(800, 600))
    ax = Axis(fig_single[1, 1],
             xlabel="Iteration",
             ylabel="Residual Norm",
             title="Final Timestep Convergence Detail",
             yscale=log10)
    
    if haskey(ch.data, :resnorm)
        residuals = ch.data[:resnorm]
        iterations = 1:length(residuals)
        
        # Plot residuals
        lines!(ax, iterations, residuals, linewidth=2)
        scatter!(ax, iterations, residuals, markersize=6)
        
        # Add tolerance line
        if haskey(ch.data, :tol)
            tol = ch.data[:tol]
            hlines!(ax, [tol], color=:red, linestyle=:dash, 
                   label="Tolerance $(tol)")
        end
        
        # Show convergence rate
        if length(residuals) >= 2
            conv_rates = [log10(residuals[i]/residuals[i-1]) 
                         for i in 2:length(residuals)]
            avg_rate = mean(conv_rates)
            
            text!(ax, 1, minimum(residuals)*10, 
                 text="Avg. convergence rate: $(round(avg_rate, digits=3))",
                 align=(:left, :bottom))
        end
    end
    
    display(fig_single)
    save("final_convergence_detail.png", fig_single)
end

# Extract the last residual norm value from each convergence history
final_residuals = Float64[]
for ch in all_convergence_histories
    if haskey(ch.data, :resnorm)
        residuals = ch.data[:resnorm]
        push!(final_residuals, residuals[end])
    else
        # If no residual data is available, use NaN
        push!(final_residuals, NaN)
    end
end

# Create a figure for plotting final residuals over time
fig_final_res = Figure(resolution=(800, 600))
ax = Axis(fig_final_res[1, 1],
          xlabel="Time",
          ylabel="Conservation Residual Norm",
          yscale=log10)

# Plot the final residuals
# Make sure time_points has the right length
valid_times = time_points[1:length(final_residuals)]
scatter!(ax, valid_times, final_residuals, color=:blue, markersize=6)
lines!(ax, valid_times, final_residuals, color=:blue, alpha=0.7)

# Add some annotations
# Find max and min values
max_res = maximum(filter(!isnan, final_residuals))
max_idx = findfirst(x -> x == max_res, final_residuals)
min_res = minimum(filter(!isnan, final_residuals))
min_idx = findfirst(x -> x == min_res, final_residuals)


# Add a horizontal line showing the average final residual
avg_res = mean(filter(!isnan, final_residuals))
hlines!(ax, [avg_res], color=:red, linestyle=:dash, 
        label="Avg: $(round(avg_res, digits=8))")

# Add legend
axislegend(ax, position=:rb)

# Display and save
display(fig_final_res)
save("final_residuals_vs_time.png", fig_final_res)