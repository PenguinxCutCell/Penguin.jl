using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie


"""
Calculate the dimensionless parameter λ for the one-phase Stefan problem.
"""
function find_lambda(Stefan_number)
    f = (λ) -> λ*exp(λ^2)*erf(λ) - Stefan_number/sqrt(π)
    # Use initial guess of 0.1 to avoid issues at λ=0
    lambda = find_zero(f, 10.0)
    return lambda
end

"""
Analytical temperature distribution for the one-phase Stefan problem.
"""
function analytical_temperature(x, t, T₀, k, lambda)
    if t <= 0
        return T₀
    end
    return T₀ - T₀/erf(lambda) * (erf(x/(2*sqrt(k*t))))
end

"""
Analytical interface position for the one-phase Stefan problem.
"""
function analytical_position(t, k, lambda)
    return 2*lambda*sqrt(k*t)
end


### 1D Test Case : One-phase Stefan Problem
# Define the spatial mesh
nx = 5
lx = 5.
x0 = 0.
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
xf = 0.51*lx   # Interface position
body = (x,t, _=0)->(x - xf)

# Define the Space-Time mesh
Δt = 0.5*(lx/nx)^2  # Time step based on stability condition
Tstart = 0.0
Tend = 0.01
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
ρ, L = 1.0, 1.0
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))

# Define the source term
f = (x,y,z,t)-> 0.0 #sin(x)*cos(10*y)
K = (x,y,z)-> 1.0

# Define the phase
Fluide = Phase(capacity, operator, f, K)

# Initial condition
u0ₒ = zeros((nx+1))
u0ᵧ = zeros((nx+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Newton parameters
max_iter = 1000
tol = eps()
reltol = eps()
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log, timestep_history = solve_MovingLiquidDiffusionUnsteadyMono!(solver, Fluide, xf, Δt, Tstart, Tend, bc_b, bc, stef_cond, mesh, "CN"; Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\)


# Save residuals in csv
using CSV
using DataFrames

# Initialize DataFrame to store residuals
df = DataFrame(timestep=Int[], iteration=Int[], residual=Float64[])

# Sort the keys to ensure time steps are processed in order
timesteps = sort(collect(keys(residuals)))

for timestep in timesteps
    # Skip empty residual vectors
    if isempty(residuals[timestep])
        continue
    end
    
    # Process each residual in the current time step
    for (iteration, r) in enumerate(residuals[timestep])
        # Check if the residual is a vector and extract a scalar value if needed
        if isa(r, Vector) || isa(r, AbstractArray)
            # Use the norm as a scalar representation of the vector
            scalar_residual = norm(r)
        else
            scalar_residual = r
        end
        
        # Add to DataFrame
        push!(df, (timestep=timestep, iteration=iteration, residual=scalar_residual))
    end
end

# Write to CSV file
CSV.write("residuals.csv", df)


using CSV
using DataFrames

function read_residuals_from_csv(file_path="residuals.csv")
    # Read the CSV file
    println("Reading residuals from $(file_path)...")
    df = CSV.read(file_path, DataFrame)
    
    # Check if the file contains the expected columns
    required_columns = [:timestep, :iteration, :residual]
   
    
    # Initialize the residuals dictionary
    residuals = Dict{Int, Vector{Float64}}()
    
    # Group by timestep
    timesteps = sort(unique(df.timestep))
    
    # Process each timestep
    for ts in timesteps
        # Get rows for this timestep
        ts_data = filter(row -> row.timestep == ts, df)
        
        # Sort by iteration to ensure correct order
        sort!(ts_data, :iteration)
        
        # Extract residuals for this timestep
        res_values = ts_data.residual
        
        # Add to dictionary
        residuals[ts] = res_values
    end
    
    println("Successfully loaded $(length(residuals)) timesteps.")
    return residuals
end

# Example usage
residuals = read_residuals_from_csv("residuals.csv")

# Then use with your plotting function
function plot_residuals_publication(residuals::Dict{Int, Vector{Float64}}; 
                                    selected_timesteps=nothing,
                                    Δt=0.5*(lx/nx)^2,
                                    save_path="newton_residuals_publication")
    # Set up figure with appropriate size and resolution for publication
    fig = Figure(resolution=(1800, 800), fontsize=14)
    
    # Create main plot for convergence
    ax1 = Axis(fig[1, 1], 
               xlabel="Newton Iteration", 
               ylabel="Residual Norm", 
               yscale=log10,
               title="Convergence of Newton Iterations",
               titlesize=18,
               xlabelsize=16,
               ylabelsize=16,
               xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticks = IntervalsBetween(10),
            yticks = LogTicks(WilkinsonTicks(5)))
    
    # Create zoomed plot for early iterations
    ax2 = Axis(fig[1, 2], 
               xlabel="Newton Iteration", 
               ylabel="", 
               yscale=log10,
               title="Zoom (Iterations 0-20)",
               titlesize=18,
               xlabelsize=16,
               ylabelsize=16,
               xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            yminorticks = IntervalsBetween(10),
            yticks = LogTicks(WilkinsonTicks(5)))
    
    # Set up colors using scientific color palette
    colors = cgrad(:viridis, 12, categorical=true)
    
    # Sort the keys to ensure time steps are processed in order
    timesteps = sort(collect(keys(residuals)))
    
    # Select a subset of timesteps if specified, otherwise use all
    if isnothing(selected_timesteps)
        # Select evenly distributed timesteps (maximum 8-10 for readability)
        n_to_show = min(8, length(timesteps))
        step_size = max(1, div(length(timesteps), n_to_show))
        selected_timesteps = timesteps[1:step_size:end]
    end
    
    # Track minimum and maximum for axis limits
    min_res, max_res = Inf, -Inf
    
    # Plot each selected timestep on both axes
    for (i, timestep) in enumerate(selected_timesteps)
        # Skip empty residual vectors
        if isempty(residuals[timestep]) || length(residuals[timestep]) < 2
            continue
        end
        
        # Get residuals and iterations for this timestep
        res = residuals[timestep]
        iterations = collect(1:length(res))
        
        # Update min/max values
        min_res = min(min_res, minimum(res))
        max_res = max(max_res, maximum(res))
        
        # Plot on main axis
        lines!(ax1, iterations, res, 
               color=colors[mod1(i, 12)], 
               linewidth=2.5,
               label="t = $(round(timestep*Δt, digits=5))")
        
        scatter!(ax1, iterations, res, 
                color=colors[mod1(i, 12)], 
                markersize=8)
                
        # Plot on zoomed axis (same data)
        lines!(ax2, iterations, res, 
               color=colors[mod1(i, 12)], 
               linewidth=2.5)
        
        scatter!(ax2, iterations, res, 
                color=colors[mod1(i, 12)], 
                markersize=8)
    end
    
    # Add reference lines for convergence orders
    ref_x = [1, 10]
    ref_y = [max_res/10, max_res/10^10]
    
    # Add reference line to both plots
    lines!(ax1, ref_x, ref_y, 
           color=:black, 
           linestyle=:dash, 
           linewidth=1.5,
           label="First order (slope -1)")
           
    lines!(ax2, ref_x, ref_y, 
           color=:black, 
           linestyle=:dash, 
           linewidth=1.5,
           label="First order (slope -1)")
    
    # Set axis limits with appropriate padding
    xlims!(ax1, 0.0, maximum([length(residuals[ts]) for ts in selected_timesteps]) + 5.0)
    ylims!(ax1, min_res/10, max_res*10)
    
    # Set axis limits for zoom plot
    xlims!(ax2, 0.0, 20.5)  # Zoom to iterations 0-20
    ylims!(ax2, min_res/10, max_res*10)  # Same y-limits as main plot
    
    # Add grid for better readability
    ax1.xgridvisible = true
    ax1.ygridvisible = true
    ax2.xgridvisible = true
    ax2.ygridvisible = true
    
    # Add legend to main plot only
    axislegend(ax1, position=:rt, framevisible=true, labelsize=12, 
               backgroundcolor=(:white, 0.85), patchsize=(30, 15))
    
    # Link y-axes for synchronized zooming/panning
    linkyaxes!(ax1, ax2)
    
    # Add column gaps for better spacing
    colgap!(fig.layout, 20)
    
    # Save in multiple formats for publication
    save(save_path * ".pdf", fig, pt_per_unit=1)
    save(save_path * ".png", fig, px_per_unit=4)
    
    display(fig)
    return fig
end

# Call the function with default parameters
plot_residuals_publication(residuals)

# Call the function with default parameters

# For specific timesteps (e.g., to highlight interesting behavior)
# plot_residuals_publication(residuals, selected_timesteps=[1, 5, 10, 20])

# Calculate Stefan number and lambda
Stefan_number = 1.0  # Assuming Stefan number = 1
lambda = find_lambda(Stefan_number)

println("  Computing errors...")
body_tend = (x, _=0) -> (x - xf_log[end])  # Use the last interface position
capacity_tend = Capacity(body_tend, mesh)

T_analytical = (x) -> analytical_temperature(x, Tend, 1.0, 1.0, lambda)

# Check convergence using the analytical solution
(u_ana, u_num, global_err, full_err, cut_err, empty_err) =
    check_convergence(T_analytical, solver, capacity_tend, 2, false)


function plot_temperature_comparison(
    solver,
    mesh::Penguin.Mesh,
    T₀::Float64,
    k::Float64,
    lambda::Float64,
    t_final::Float64,
    xf_final::Float64;
    save_path::String="temperature_comparison.png"
)
    # Extract mesh info
    x_nodes = mesh.nodes[1]
    nx = length(x_nodes) - 1
    
    # Get numerical solution (bulk field)
    u_num = solver.x[1:(nx+1)]
    
    # Calculate analytical solution
    u_analytical = zeros(nx+1)
    for i in 1:nx+1
        if x_nodes[i] < xf_final
            u_analytical[i] = analytical_temperature(x_nodes[i], t_final, T₀, k, lambda)
        else
            u_analytical[i] = 0.0  # Outside liquid domain
        end
    end
    
    # Create figure
    fig = Figure(resolution=(900, 600), fontsize=14)
    
    ax = Axis(
        fig[1, 1],
        xlabel = "Position (x)",
        ylabel = "Temperature",
        title = "Stefan Problem: Analytical vs Numerical Solution at t = $t_final"
    )
    
    # Plot solutions
    lines!(ax, x_nodes, u_analytical, color=:red, linewidth=3, 
           label="Analytical Solution")
    scatter!(ax, x_nodes, u_num, color=:blue, markersize=6, 
           label="Numerical Solution")
    
    # Mark interface position
    vlines!(ax, xf_final, color=:black, linewidth=2, linestyle=:dash,
            label="Interface Position")
    
    # Calculate error metrics
    l2_error = sqrt(sum((u_analytical - u_num).^2)) / sqrt(sum(u_analytical.^2))
    max_error = maximum(abs.(u_analytical - u_num))
    
    # Add error information
    text!(ax, "L₂ Error: $(round(l2_error, digits=6))\nMax Error: $(round(max_error, digits=6))",
         position = (0.6*maximum(x_nodes), 0.8*T₀),
         fontsize = 14,
         color = :black)
    
    # Add legend
    axislegend(ax, position=:rt, framevisible=true, backgroundcolor=(:white, 0.7))
    
    # Save the figure if requested
    if !isempty(save_path)
        save(save_path, fig, px_per_unit=4)
    end
    
    display(fig)
    return fig
end

# Plot comparison
plot_temperature_comparison(
    solver,
    mesh,
    1.0,   # T₀
    1.0,   # k
    lambda,
    Tend,  # final time
    xf_log[end]  # final interface position
)
readline()
# Animation
animate_solution(solver, mesh, body)

# Plot residuals   
#residuals[i] might be empty, remove them
residuals = filter(x -> !isempty(x), residuals)

figure = Figure()
ax = Axis(figure[1,1], xlabel = "Newton Iterations", ylabel = "Residuals", title = "Residuals")
for i in 1:length(residuals)
    lines!(ax, log10.(residuals[i]), label = "Time = $(i*Δt)")
end
#axislegend(ax)
display(figure)

# Plot the position
figure = Figure()
ax = Axis(figure[1,1], xlabel = "Time", ylabel = "Interface position", title = "Interface position")
lines!(ax, xf_log, label = "Interface position")
display(figure)

# Add convergence rate analysis for residuals
# Add convergence rate analysis for residuals
function analyze_convergence_rates(residuals::Dict{Int, Vector{Float64}})
    figure = Figure(resolution=(1000, 800))
    ax1 = Axis(figure[1, 1], 
               xlabel="Newton Iteration (k)", 
               ylabel="log₁₀(Residual)",
               title="Residual Convergence Analysis")
    
    # For rate estimates
    ax2 = Axis(figure[2, 1], 
               xlabel="Time Step", 
               ylabel="Convergence Rate (-slope)",
               title="Convergence Rate Evolution")
    
    rates = Float64[]
    times = Float64[]
    
    # Process each time step in order
    time_steps = sort(collect(keys(residuals)))
    
    for i in time_steps
        res = residuals[i]
        if length(res) < 3
            println("Skipping time step $i (not enough iterations)")
            continue
        end
        
        # Take log of residuals
        log_res = log10.(res)
        
        # Create iteration indices
        iterations = collect(1:length(log_res))
        
        # Only fit the linear part of the convergence (after initial iterations)
        # Try to automatically detect where linear convergence begins
        start_idx = 1
        for j in 2:length(log_res)-1
            # Check if we have steady decrease for 3 consecutive points
            if log_res[j] < log_res[j-1] && log_res[j+1] < log_res[j]
                start_idx = j
                break
            end
        end
        
        # Ensure we have enough points for a meaningful fit
        if length(log_res) - start_idx + 1 < 3
            println("Skipping time step $i (not enough data for linear fit)")
            continue
        end
        
        # Fit linear model log(residual) = A*iteration + B
        model(x, p) = p[1] .* x .+ p[2]
        linear_fit = curve_fit(model, iterations[start_idx:end], log_res[start_idx:end], [-1.0, 0.0])
        
        # Extract convergence rate
        rate = -linear_fit.param[1]  # Negative slope is the convergence rate
        push!(rates, rate)
        push!(times, i*Δt)
        
        # Plot data points and fit on first plot
        scatter!(ax1, iterations, log_res, label="Time = $(i*Δt)")
        
        # Plot fitted line
        fit_line = model(collect(start_idx:length(log_res)), linear_fit.param)
        lines!(ax1, start_idx:length(log_res), fit_line, 
               linestyle=:dash, linewidth=2, 
               label="Rate = $(round(rate, digits=2))")
        
        if mod(i, 15) == 0
            text!(ax1, 
                  iterations[end], log_res[end]-0.5, 
                  text="$(round(rate, digits=2))", 
                  fontsize=12, 
                  align=(:left, :bottom))
        end
    end
    
    # Safety check if we have any rates
    if isempty(rates)
        println("Warning: No convergence rates could be calculated.")
        return Float64[], Float64[]
    end
    
    # Plot the evolution of convergence rates
    scatter!(ax2, times, rates, markersize=10)
    lines!(ax2, times, rates)
    
    # Add average rate line
    avg_rate = sum(rates) / length(rates)
    hlines!(ax2, [avg_rate], color=:red, linestyle=:dash, 
            label="Average: $(round(avg_rate, digits=2))")
    
    # Format plots
    axislegend(ax2, position=:lb)
    
    # Calculate statistics
    println("Average convergence rate: $(avg_rate)")
    println("Minimum convergence rate: $(minimum(rates))")
    println("Maximum convergence rate: $(maximum(rates))")
    
    display(figure)
    return rates, times
end

# Call the analysis function
convergence_rates, time_steps = analyze_convergence_rates(residuals)

# Plot log-log of convergence rates if desired
figure = Figure()
ax = Axis(figure[1, 1],
         xlabel="Residual at iteration k",
         ylabel="Residual at iteration k+1",
         title="Convergence Order Analysis")

# For each time step, plot residual[k+1] vs residual[k] in log-log scale
for (i, res) in enumerate(residuals)
    if length(res) < 4  # Need at least 4 points for this analysis
        continue
    end
    
    # Skip the first iteration which might be far off
    x_vals = res[2:end-1]
    y_vals = res[3:end]
    
    scatter!(ax, x_vals, y_vals, label="Time = $(i*Δt)")
    
    # For reference, add lines showing quadratic and linear convergence
    if i == 1
        x_range = 10.0 .^ range(log10(minimum(x_vals)), log10(maximum(x_vals)), length=100)
        # Quadratic convergence: y ∝ x²
        quad_factor = y_vals[1] / x_vals[1]^2
        lines!(ax, x_range, quad_factor .* x_range.^2, 
              linestyle=:dash, color=:red, linewidth=2,
              label="Quadratic: r[k+1] ∝ r[k]²")
        
        # Linear convergence: y ∝ x
        lin_factor = y_vals[1] / x_vals[1]
        lines!(ax, x_range, lin_factor .* x_range, 
              linestyle=:dot, color=:blue, linewidth=2,
              label="Linear: r[k+1] ∝ r[k]")
    end
end

ax.xscale = log10
ax.yscale = log10
#axislegend(ax, position=:lt)
display(figure)

# save xf_log
open("xf_log_$nx.txt", "w") do io
    for i in 1:length(xf_log)
        println(io, xf_log[i])
    end
end

# Plot the solution
plot_solution(solver, mesh, body, capacity; state_i=10)

# create a directory to save solver.states[i]
if !isdir("solver_states_$nx")
    mkdir("solver_states_$nx")
end
# save solver.states[i]
for i in 1:length(solver.states)
    open("solver_states_$nx/solver_states_$i.txt", "w") do io
        for j in 1:length(solver.states[i])
            println(io, solver.states[i][j])
        end
    end
end