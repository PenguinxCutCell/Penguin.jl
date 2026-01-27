using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CairoMakie
using Interpolations
using Colors
using Statistics
using FrontCutTracking

### 2D Test Case: Growing Crystal with Perturbed Interface
### Crystal growing in an undercooled liquid

# Define physical parameters
L = 1.0                 # Latent heat
c = 1.0                 # Specific heat capacity
TM = 0.0                # Melting temperature (solid phase) as specified
T∞ = -1.0               # Far field temperature (undercooled liquid)

# Calculate the Stefan number
Ste = (c * (TM - T∞)) / L
println("Stefan number: $Ste")

# Define the spatial mesh
nx, ny = 64, 64 # Number of grid points in x and y directions
lx, ly = 16.0, 16.0
x0, y0 = -8.0, -8.0
Δx, Δy = lx/nx, ly/ny
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

println("Mesh created with dimensions: $(nx) x $(ny), Δx=$(Δx), Δy=$(Δy)")

# Initial time settings
t_init = 1.0
R0 = 1.0  # Base radius for the crystal

# Create front tracker with perturbed crystal shape
front = FrontTracker()
nmarkers = 200  # Number of markers

# Create the perturbed crystal
create_crystal!(front, 0.0, 0.0, R0, 6, 0.1, nmarkers)

# Define the body using the SDF from the front tracker
body = (x, y, t, _=0) -> -sdf(front, x, y)

# Define the Space-Time mesh
Δt = 0.25 * (Δx)^2  # Time step size based on mesh spacing
t_final = t_init + 10*Δt  # Run for 10 time steps

STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_final], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh; compute_centroids=false)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc_b = Dirichlet(T∞)  # Far field temperature
bc = Dirichlet(TM)    # Temperature at the interface (melting temperature)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :left => bc_b, :right => bc_b, :top => bc_b, :bottom => bc_b))

# Stefan condition at the interface
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, L))

# Define the source term (no source)
f = (x,y,z,t) -> 0.0
K = (x,y,z) -> 1.0  # Thermal conductivity

Fluide = Phase(capacity, operator, f, K)

# Initialize temperature fields
u0ₒ = zeros((nx+1)*(ny+1))
body_init = (x,y,_=0) -> sdf(front, x, y)
cap_init = Capacity(body_init, mesh; compute_centroids=false)
centroids = cap_init.C_ω
factor = 4.0  # Controls width of transition (adjust as needed)
L0 = R0       # Characteristic length (use initial radius)
# Initialize the temperature with properly scaled tanh profile
for idx in 1:length(centroids)
    centroid = centroids[idx]
    x, y = centroid[1], centroid[2]
    
    # Distance signée (négative à l'intérieur, positive à l'extérieur)
    val = body_init(x, y)
    
    if val <= 0
        # À l'intérieur ou sur l'interface: température = TM
        u0ₒ[idx] = TM
    else
        # À l'extérieur: transition de TM à T∞ avec tanh
        # Utilisation de tanh qui va de 0 à l'interface à 1 loin
        transition = tanh(val * L0 / factor)
        
        # Interpoler de TM (à l'interface) à T∞ (loin)
        u0ₒ[idx] = TM + transition * (T∞ - TM)
    end
end
#u0ₒ = ones((nx+1)*(ny+1)) * T∞ 
u0ᵧ = ones((nx+1)*(ny+1)) * TM  # Interface temperature
u0 = vcat(u0ₒ, u0ᵧ)

# Visualize initial temperature field
fig_init = Figure(size=(900, 700))
ax_init = Axis(fig_init[1, 1], 
               title="Initial Temperature Field", 
               xlabel="x", ylabel="y",
               aspect=DataAspect())
hm = heatmap!(ax_init, mesh.nodes[1], mesh.nodes[2], 
              reshape(u0ₒ, (nx+1, ny+1)),
              colormap=:thermal)
Colorbar(fig_init[1, 2], hm, label="Temperature")

# Add interface contour
markers = get_markers(front)
marker_x = [m[1] for m in markers]
marker_y = [m[2] for m in markers]
lines!(ax_init, marker_x, marker_y, color=:black, linewidth=2)

display(fig_init)

# Newton parameters
Newton_params = (1, 1e-6, 1e-6, 1.0)  # max_iter, tol, reltol, α

# Run the simulation
solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log, timestep_history = solve_StefanMono2D!(
    solver, Fluide, front, Δt, t_init, t_final, bc_b, bc, stef_cond, mesh, "BE";
    Newton_params=Newton_params, 
    smooth_factor=1.0, 
    window_size=10, 
    method=Base.:\
)

# Function to visualize results
function visualize_results(solver, mesh, xf_log, timestep_history)
    results_dir = joinpath(pwd(), "growing_crystal_results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Plot temperature field and interface at final time
    xi = mesh.nodes[1]
    yi = mesh.nodes[2]
    nx1, ny1 = length(xi), length(yi)
    npts = nx1 * ny1
    
    # Get final state
    final_state = solver.states[end]
    Tw_final = final_state[1:npts]
    T_final = reshape(Tw_final, (nx1, ny1))
    
    # Plot final temperature and interface
    fig_final = Figure(size=(900, 700))
    ax_final = Axis(fig_final[1, 1], 
                   title="Final Temperature Field", 
                   xlabel="x", ylabel="y",
                   aspect=DataAspect())
    
    hm = heatmap!(ax_final, xi, yi, T_final, colormap=:thermal)
    Colorbar(fig_final[1, 2], hm, label="Temperature")
    
    # Get final interface
    final_timestep = maximum(keys(xf_log))
    final_markers = xf_log[final_timestep]
    
    # Extract coordinates
    marker_x = [m[1] for m in final_markers]
    marker_y = [m[2] for m in final_markers]
    
    # Plot final interface
    lines!(ax_final, marker_x, marker_y, color=:black, linewidth=2)
    scatter!(ax_final, marker_x, marker_y, color=:white, markersize=4)
    
    save(joinpath(results_dir, "final_temperature.png"), fig_final)
    
    # Plot interface evolution
    fig_evolution = Figure(size=(800, 800))
    ax_evolution = Axis(fig_evolution[1, 1], 
                       title="Interface Evolution", 
                       xlabel="x", ylabel="y",
                       aspect=DataAspect())
    
    # Plot interface at different times
    all_timesteps = sort(collect(keys(xf_log)))
    num_timesteps = length(all_timesteps)
    colors = cgrad(:viridis, num_timesteps)
    
    for (i, timestep) in enumerate(all_timesteps)
        markers = xf_log[timestep]
        marker_x = [m[1] for m in markers]
        marker_y = [m[2] for m in markers]
        
        lines!(ax_evolution, marker_x, marker_y, 
              color=colors[i], 
              linewidth=2)
        
        # Add time label
        if i == 1 || i == num_timesteps || i % max(1, div(num_timesteps, 4)) == 0
            time_value = timestep_history[min(timestep, length(timestep_history))][1]
            text!(ax_evolution, mean(marker_x), mean(marker_y), 
                 text="t=$(round(time_value, digits=2))",
                 align=(:center, :center),
                 fontsize=10)
        end
    end
    
    Colorbar(fig_evolution[1, 2], limits=(1, num_timesteps),
            colormap=:viridis, label="Timestep")
    
    save(joinpath(results_dir, "interface_evolution.png"), fig_evolution)
    
    # Create animation of temperature and interface
    create_animation(solver, mesh, xf_log, results_dir)
    
    return results_dir
end

# Function to create animation
function create_animation(solver, mesh, xf_log, results_dir)
    # Mesh dimensions
    xi = mesh.nodes[1]
    yi = mesh.nodes[2]
    nx1, ny1 = length(xi), length(yi)
    npts = nx1 * ny1
    
    # Get temperature range for consistent colormap
    all_temps = Float64[]
    for state in solver.states
        Tw = state[1:npts]
        push!(all_temps, extrema(Tw)...)
    end
    temp_limits = (minimum(all_temps), maximum(all_temps))
    
    # Get timesteps
    all_timesteps = sort(collect(keys(xf_log)))
    
    # Create animation
    fig = Figure(size=(900, 700))
    ax = Axis(fig[1, 1], 
            title="Growing Crystal Simulation", 
            xlabel="x", ylabel="y", 
            aspect=DataAspect())
    
    # Record animation
    record(fig, joinpath(results_dir, "growing_crystal_animation.gif"), 
           1:length(solver.states); framerate=5) do i
        empty!(ax)
        
        # Update title
        ax.title = "Growing Crystal, t=$(round(i * 0.1, digits=2))"
        
        # Plot temperature
        Tstate = solver.states[i]
        Tw = Tstate[1:npts]
        Tmat = reshape(Tw, (nx1, ny1))
        
        hm = heatmap!(ax, xi, yi, Tmat, colormap=:thermal, colorrange=temp_limits)
        Colorbar(fig[1, 2], hm, label="Temperature")
        
        # Plot interface
        if i <= length(all_timesteps)
            markers = xf_log[all_timesteps[i]]
            marker_x = [m[1] for m in markers]
            marker_y = [m[2] for m in markers]
            
            lines!(ax, marker_x, marker_y, color=:black, linewidth=3)
            scatter!(ax, marker_x, marker_y, color=:white, markersize=4)
        end
    end
    
    println("Animation saved to: $(joinpath(results_dir, "growing_crystal_animation.gif"))")
end

# Visualize results
results_dir = visualize_results(solver, mesh, xf_log, timestep_history)
println("Results saved to: $results_dir")