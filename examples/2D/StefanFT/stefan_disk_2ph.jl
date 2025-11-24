using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations
using Colors
using Statistics

### 2D Test Case: Two-phase Stefan Problem with Circular Interface
# Phase 1: Inside the circle (solid)
# Phase 2: Outside the circle (liquid)

# Define physical parameters
ρ = 1.0      # Density
L = 1.0      # Latent heat
c = 1.0      # Specific heat capacity
Tm = 0.0     # Melting temperature at interface
T1 = -0.5    # Cold temperature (inside circle)
T2 = 0.5     # Hot temperature (outside circle)

# Diffusion coefficients
D1 = 1.0     # Solid phase
D2 = 1.0     # Liquid phase

# Calculate Stefan numbers
Ste1 = c * (Tm - T1) / L
Ste2 = c * (T2 - Tm) / L
println("Stefan numbers: Ste1 = $Ste1, Ste2 = $Ste2")

# Set initial conditions
R0 = 2.0       # Initial radius
t_init = 0.0   # Initial time
t_final = 0.1  # Final time

# Define the spatial mesh
nx, ny = 64, 64
lx, ly = 10.0, 10.0
x0, y0 = -5.0, -5.0
Δx, Δy = lx/nx, ly/ny
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

println("Mesh created with dimensions: $(nx) x $(ny), Δx=$(Δx), Δy=$(Δy), domain=[$x0, $(x0+lx)] x [$y0, $(y0+ly)]")

# Create the front-tracking object for circular interface
nmarkers = 100
front = FrontTracker() 
create_circle!(front, 0.0, 0.0, R0, nmarkers)

# Define the initial signed distance function for each phase
# Phase 1: Inside circle (negative distance inside)
# Phase 2: Outside circle (negative distance outside)
body1 = (x, y, t, _=0) -> sdf(front, x, y)        # Phase 1 (inside circle)
body2 = (x, y, t, _=0) -> -sdf(front, x, y)       # Phase 2 (outside circle)

# Define the time step
Δt = 0.1*(lx/nx)^2  # Time step based on mesh size
time_interval = [t_init, t_init + Δt]

# Create the space-time mesh
STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)

# Define the capacities for both phases
capacity1 = Capacity(body1, STmesh; compute_centroids=false)
capacity2 = Capacity(body2, STmesh; compute_centroids=false)

# Define the diffusion operators
operator1 = DiffusionOps(capacity1)
operator2 = DiffusionOps(capacity2)

# Define source terms (no sources)
f1 = (x,y,z,t) -> 0.0
f2 = (x,y,z,t) -> 0.0

# Define diffusion coefficients
K1 = (x,y,z) -> D1
K2 = (x,y,z) -> D2

# Define the phases
Phase1 = Phase(capacity1, operator1, f1, K1)
Phase2 = Phase(capacity2, operator2, f2, K2)

# Define boundary conditions
# Far-field boundary conditions
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :left   => Dirichlet(T2),
    :right  => Dirichlet(T2),
    :top    => Dirichlet(T2),
    :bottom => Dirichlet(T2)
))

# Interface conditions
interface_cond = InterfaceConditions(
    ScalarJump(1.0, 1.0, Tm),       # Temperature jump (T₁ = T₂ = Tm at interface)
    FluxJump(D1, D2, ρ*L)           # Flux jump (latent heat release)
)

# Set up initial condition - Phase 1 (inside circle)
u1ₒ = ones((nx+1)*(ny+1)) * T1     # Bulk initial temperature
u1ᵧ = ones((nx+1)*(ny+1)) * Tm     # Interface temperature

# Set up initial condition - Phase 2 (outside circle)
u2ₒ = ones((nx+1)*(ny+1)) * T2     # Bulk initial temperature
u2ᵧ = ones((nx+1)*(ny+1)) * Tm     # Interface temperature

# Combine all initial values
u0 = vcat(u1ₒ, u1ᵧ, u2ₒ, u2ᵧ)

# Newton parameters
Newton_params = (10, 1e-6, 1e-6, 1.0) # max_iter, tol, reltol, α

# Run the simulation
solver = StefanDiph2D(Phase1, Phase2, bc_b, interface_cond, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log, timestep_history, phase1_final, phase2_final, position_increments = 
    solve_StefanDiph2D!(solver, Phase1, Phase2, front, Δt, t_init, t_final, bc_b, interface_cond, mesh, "BE";
                       Newton_params=Newton_params, method=Base.:\)


# Create results directory
results_dir = joinpath(pwd(), "simulation_results")
if !isdir(results_dir)
    mkdir(results_dir)
end
println("\nVisualisations des résultats enregistrées dans: $results_dir")

# Plot interface evolution
fig_interface = Figure(size=(800, 800))
ax_interface = Axis(fig_interface[1, 1], 
                   title="Interface Evolution", 
                   xlabel="x", ylabel="y",
                   aspect=DataAspect())

# Get all timesteps and sort them
all_timesteps = sort(collect(keys(xf_log)))
num_timesteps = length(all_timesteps)

# Generate color gradient based on timestep
colors = cgrad(:viridis, num_timesteps)

for (i, timestep) in enumerate(all_timesteps)
    markers = xf_log[timestep]
    marker_x = [m[1] for m in markers]
    marker_y = [m[2] for m in markers]
    
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

save(joinpath(results_dir, "interface_evolution.png"), fig_interface)
display(fig_interface)

# Plot residuals for each timestep
fig_residuals = Figure(size=(900, 600))
ax_residuals = Axis(fig_residuals[1, 1], 
                    title="Convergence History", 
                    xlabel="Iteration", 
                    ylabel="Residual Norm (log scale)",
                    yscale=log10)

for (timestep, residual_vec) in sort(collect(residuals))
    lines!(ax_residuals, 1:length(residual_vec), residual_vec, 
          label="Timestep $timestep", 
          linewidth=2)
end

Legend(fig_residuals[1, 2], ax_residuals)
save(joinpath(results_dir, "residuals.png"), fig_residuals)
display(fig_residuals)

# Plot radius evolution
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
                title="Interface Radius Evolution", 
                xlabel="Time", 
                ylabel="Mean Radius")

# Create time values for plotting
times_for_plot = Float64[]
for ts in all_timesteps
    if ts == 1
        push!(times_for_plot, timestep_history[1][1])
    else
        time_index = min(ts, length(timestep_history))
        push!(times_for_plot, timestep_history[time_index][1])
    end
end

# Plot simulation results
scatter!(ax_radius, times_for_plot, radii, 
        label="Simulation", markersize=6)

save(joinpath(results_dir, "radius_evolution.png"), fig_radius)
display(fig_radius)

println("Simulation completed successfully!")