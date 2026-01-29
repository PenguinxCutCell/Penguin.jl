using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations
using Colors
using Statistics
using FFTW
using DSP
using Roots
using FrontCutTracking

### 2D Test Case: Melting Circle (Stefan Problem with Circular Interface)
### Ice sphere melting in warm liquid with self-similar solution

# Define physical parameters
L = 1.0      # Latent heat
c = 1.0      # Specific heat capacity
TM = 0.0     # Melting temperature (at interface)
T∞ = 0.5     # Far field temperature (warm liquid) - POSITIVE for melting
α = 1.0      # Thermal diffusivity

# Calculate the Stefan number (now positive for melting)
Ste = (c * (T∞ - TM)) / L
println("Stefan number: $Ste (positive for melting)")

# Define F(s) function using the exponential integral E₁
function F(s)
    return expint(s^2/4)  # E₁(s²/4)
end

# Determine the similarity parameter λ from the transcendental Stefan condition
stefan_eq(k) = k^2 * exp(-k^2) / (-expint(k^2)) - Ste
λ = 0.6893
println("Similarity parameter λ = $λ")

# Set initial conditions
R0 = 3.0      # Initial radius (larger to see melting progress)
t_init = 0.1
t_final = 0.2 # Final time

# Analytical temperature function for melting
function analytical_temperature(r, t)
    s_interface = interface_position(t) / sqrt(α * t)
    s = r / sqrt(α * t)

    if s < s_interface
        return TM  # At and inside the solid (ice)
    else
        # In liquid region, use the similarity solution
        return T∞ * (1.0 - F(s)/F(s_interface))
    end
end

# Function to calculate the interface position at time t (decreasing radius)
function interface_position(t)
    return max(R0 - 2 * λ * sqrt(α * t), 0.0)
end

function analytical_flux(t)
    s_interface = interface_position(t) / sqrt(α * t)
    κ = 1.0  # Thermal conductivity
    return (κ * T∞ / F(s_interface)) * exp(-s_interface^2) / sqrt(α * t)  # Note sign change for melting
end

# Print information about the simulation
println("Initial radius at t=$t_init: R=$(interface_position(t_init))")

# Define the spatial mesh
nx, ny = 32, 32
lx, ly = 16.0, 16.0
x0, y0 = -8.0, -8.0
Δx, Δy = lx/(nx), ly/(ny)
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

println("Mesh created with dimensions: $(nx) x $(ny), Δx=$(Δx), Δy=$(Δy)")

# Create the front-tracking body
nmarkers = 100
front = FrontTracker() 
create_circle!(front, 0.01, 0.01, interface_position(t_init), nmarkers)

# Define the initial position of the front
body = (x, y, t, _=0) -> -sdf(front, x, y)

# Define the Space-Time mesh
Δt = 0.25*(lx / nx)^2  # Time step size
t_final = t_init + 11Δt
println("Final radius at t=$(t_init + Δt): R=$(interface_position(t_init + Δt))")

STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc_b = Dirichlet(T∞)  # Far field temperature (now warm)
bc = Dirichlet(TM)    # Temperature at the interface (melting point)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc_b, :right => bc_b, :top => bc_b, :bottom => bc_b))

# Stefan condition at the interface - Note: direction of heat flow is reversed for melting
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, L))

# Define the source term (no source)
f = (x,y,z,t) -> 0.0
K = (x,y,z) -> 1.0  # Thermal conductivity

Fluide = Phase(capacity, operator, f, K)

# Set up initial condition
u0ₒ = zeros((nx+1)*(ny+1))
body_init = (x,y,_=0) -> -sdf(front, x, y)
cap_init = Capacity(body_init, mesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
centroids = cap_init.C_ω

# Initialize the temperature
for idx in 1:length(centroids)
    centroid = centroids[idx]
    x, y = centroid[1], centroid[2]
    r = sqrt(x^2 + y^2)
    u0ₒ[idx] = analytical_temperature(r, t_init)
end
u0ᵧ = ones((nx+1)*(ny+1))*TM
u0 = vcat(u0ₒ, u0ᵧ)

# Visualize initial temperature field
fig_init = Figure(size=(800, 600))
ax_init = Axis(fig_init[1, 1], 
               title="Initial Temperature Field (Melting Circle)", 
               xlabel="x", ylabel="y",
               aspect=DataAspect())
hm = heatmap!(ax_init, mesh.nodes[1], mesh.nodes[2], 
              reshape(u0ₒ, (nx+1, ny+1)),
              colormap=:ice)
Colorbar(fig_init[1, 2], hm, label="Temperature")

# Add interface contour
markers = get_markers(front)
marker_x = [m[1] for m in markers]
marker_y = [m[2] for m in markers]
lines!(ax_init, marker_x, marker_y, color=:black, linewidth=2)

display(fig_init)

# Newton parameters
Newton_params = (1, 1e-6, 1e-6, 1.0) # max_iter, tol, reltol, α

# Run the simulation
solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log, timestep_history, phase, position_increments = solve_StefanMono2D!(
    solver, Fluide, front, Δt, t_init, t_final, bc_b, bc, stef_cond, mesh, "BE";
    Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\
)

# Create the animation
function create_temperature_interface_animation(solver, mesh, xf_log, timestep_history)
    # Mesh dimensions
    xi = mesh.nodes[1]
    yi = mesh.nodes[2]
    nx1, ny1 = length(xi), length(yi)
    npts = nx1 * ny1
    
    # Extract temperature limits for consistent color scale
    all_temps = Float64[]
    for Tstate in solver.states
        Tw = Tstate[1:npts]
        push!(all_temps, extrema(Tw)...)
    end
    temp_limits = (minimum(all_temps), maximum(all_temps))
    
    # Get all interface timesteps
    all_timesteps = sort(collect(keys(xf_log)))
    
    # Create animation directly with record
    fig = Figure(size=(800, 700))
    ax = Axis(fig[1, 1], 
            title="Temperature & Melting Interface Evolution", 
            xlabel="x", ylabel="y", aspect=DataAspect())
    
    # Create results directory if needed
    results_dir = joinpath(pwd(), "simulation_results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Record animation
    record(fig, joinpath(results_dir, "melting_circle_animation.mp4"), 1:length(solver.states)) do i
        empty!(ax)  # Clear axis for new frame
        
        # Update title with current time
        current_time = timestep_history[min(i, length(timestep_history))][1]
        ax.title = "Melting Circle, t=$(round(current_time, digits=3))"
        
        # Extract and reshape temperature for this timestep
        Tstate = solver.states[i]
        Tw = Tstate[1:npts]
        Tmat = reshape(Tw, (nx1, ny1))
        
        # Display temperature heatmap
        hm = heatmap!(ax, xi, yi, Tmat, colormap=:ice, colorrange=temp_limits)
        Colorbar(fig[1, 2], hm, label="Temperature")
        
        # Overlay interface if available for this timestep
        if i <= length(all_timesteps)
            timestep_index = all_timesteps[min(i, length(all_timesteps))]
            markers = xf_log[timestep_index]
            
            # Extract coordinates for plotting
            marker_x = [m[1] for m in markers]
            marker_y = [m[2] for m in markers]
            
            # Plot interface as a closed line
            lines!(ax, marker_x, marker_y, color=:white, linewidth=3)
            scatter!(ax, marker_x, marker_y, color=:white, markersize=3, alpha=0.7)
        end
        
        # Display progress
        println("Frame $i of $(length(solver.states))")
    end
    
    println("\nAnimation saved to: $(joinpath(results_dir, "melting_circle_animation.mp4"))")
    return joinpath(results_dir, "melting_circle_animation.mp4")
end

# Call animation function after simulation
animation_file = create_temperature_interface_animation(solver, mesh, xf_log, timestep_history)

# Compare results with analytical solution
function plot_simulation_results(residuals, xf_log, timestep_history, Ste)
    # Create results directory
    results_dir = joinpath(pwd(), "simulation_results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Plot interface evolution
    fig_interface = Figure(size=(800, 800))
    ax_interface = Axis(fig_interface[1, 1], 
                       title="Melting Interface Evolution", 
                       xlabel="x", 
                       ylabel="y",
                       aspect=DataAspect())
    
    # Get all timesteps and sort them
    all_timesteps = sort(collect(keys(xf_log)))
    num_timesteps = length(all_timesteps)
    
    # Generate color gradient based on timestep
    colors = cgrad(:viridis, num_timesteps)
    
    # Plot each interface position
    for (i, timestep) in enumerate(all_timesteps)
        markers = xf_log[timestep]
        
        # For closed interfaces, ensure proper visualization
        marker_x = [m[1] for m in markers]
        marker_y = [m[2] for m in markers]
        
        # Add the first marker again to close the loop if needed
        if !isapprox(marker_x[1], marker_x[end]) || !isapprox(marker_y[1], marker_y[end])
            push!(marker_x, marker_x[1])
            push!(marker_y, marker_y[1])
        end
        
        # Plot interface as a closed curve
        lines!(ax_interface, marker_x, marker_y, 
              color=colors[i], 
              linewidth=2)
        
        # Add timestamp label near the interface
        if i == 1 || i == num_timesteps || i % max(1, div(num_timesteps, 5)) == 0
            time_value = timestep_history[min(timestep, length(timestep_history))][1]
            text!(ax_interface, mean(marker_x), mean(marker_y), 
                 text="t=$(round(time_value, digits=2))",
                 align=(:center, :center),
                 fontsize=10)
        end
    end
    
    # Add colorbar to show timestep progression
    Colorbar(fig_interface[1, 2], limits=(1, num_timesteps),
            colormap=:viridis, label="Timestep")
    
    save(joinpath(results_dir, "melting_interface_evolution.png"), fig_interface)
    
    # Plot radius evolution over time
    times = [hist[1] for hist in timestep_history]
    radii = Float64[]
    
    for timestep in all_timesteps
        markers = xf_log[timestep]
        
        # Calculate geometric center
        center_x = sum(m[1] for m in markers) / length(markers)
        center_y = sum(m[2] for m in markers) / length(markers)
        
        # Calculate mean radius
        mean_radius = mean([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in markers])
        push!(radii, mean_radius)
    end
    
    fig_radius = Figure(size=(800, 600))
    ax_radius = Axis(fig_radius[1, 1], 
                    title="Melting Interface Radius Evolution", 
                    xlabel="Time", 
                    ylabel="Mean Radius")
    
    # Create correct time values for plotting
    times_for_plot = Float64[]
    for ts in all_timesteps
        if ts <= length(timestep_history)
            push!(times_for_plot, timestep_history[ts][1])
        else
            push!(times_for_plot, times_for_plot[end] + timestep_history[end][2])
        end
    end

    # Add analytical solution for comparison
    analytical_times = range(t_init, stop=t_final, length=100)
    analytical_radii = [interface_position(t) for t in analytical_times]

    # Plot simulation results
    scatter!(ax_radius, times_for_plot, radii, 
            label="Simulation", markersize=6)

    # Plot analytical solution
    lines!(ax_radius, analytical_times, analytical_radii,
        label="Analytical", linewidth=2, color=:red, linestyle=:dash)
        
    axislegend(ax_radius)
    
    save(joinpath(results_dir, "radius_evolution.png"), fig_radius)
    display(fig_radius)
    
    return results_dir
end

# Call plotting function after simulation
results_dir = plot_simulation_results(residuals, xf_log, timestep_history, Ste)
println("\nSimulation results visualization saved to: $results_dir")