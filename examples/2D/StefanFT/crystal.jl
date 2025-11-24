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

### 2D Test Case: Frank Sphere (Stefan Problem with Circular Interface)
### Ice crysyal decaying

# Define physical parameters
L = 1.0      # Latent heat
c = 1.0      # Specific heat capacity
TM = 0.0     # Melting temperature (inside sphere)
T∞ = 1.0    # Far field temperature (undercooled liquid)

# Calculate the Stefan number
Ste = (c * (TM - T∞)) / L
println("Stefan number: $Ste")

# Define F(s) function using the exponential integral E₁
function F(s)
    return expint(s^2/4)  # E₁(s²/4)
end

# Calculate the similarity parameter S
S = 1.56
println("Similarity parameter S = $S")

# Set initial conditions as specified
R0 = 1.0      # Initial radius
t_init = 1.0  # Initial time
t_final = 1.1   # Final time

# Analytical temperature function
function analytical_temperature(r, t)
    # Calculate similarity variable
    s = r / sqrt(t)
    
    if s < S
        return TM  # Inside the solid (ice)
    else
        # In liquid region, use the similarity solution
        return T∞ * (1.0 - F(s)/F(S))
    end
end

# Function to calculate the interface position at time t
function interface_position(t)
    return S * sqrt(t)
end

function analytical_flux(t)
    return - (k * Tinf / F(S)) * exp(-S^2) / sqrt(t)
end

# Print information about the simulation
println("Initial radius at t=$t_init: R=$(interface_position(t_init))")

# Plot the analytical solution
radii = [interface_position(t) for t in range(t_init, stop=t_final, length=100)]
temperatures = [analytical_temperature(r, t_final) for r in radii]

# Define the spatial mesh
nx, ny = 32, 32  # Number of grid points in x and y directions
lx, ly = 4.0, 4.0
x0, y0 = -2.0, -2.0
Δx, Δy = lx/(nx), ly/(ny)
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

println("Mesh created with dimensions: $(nx) x $(ny), Δx=$(Δx), Δy=$(Δy), domain=[$x0, $(x0+lx)], [$y0, $(y0+ly)]")

# Création du cristal - remplace votre body et place_markers_from_body!
front = FrontTracker()
nmarkers = 150  # Nombre de marqueurs pour le cercle

# Pour un cercle parfait (amplitude=0)
#create_circle!(front, 0.0, 0.0, interface_position(t_init), nmarkers)
#create_crystal!(front, 0.0, 0.0, R0, 6, 0.0, nmarkers)
# Pour un cristal avec perturbation
create_crystal!(front, 0.0, 0.0, R0, 4, 0.1, nmarkers)

# Définir le corps (body) en utilisant la SDF du front
body = (x, y, t, _=0) -> -sdf(front, x, y)

# Define the Space-Time mesh
#Δt = 0.5*(lx / nx)^2  # Time step size
Δt = 0.01
t_final = t_init + 10Δt
println("Final radius at t=$(t_init + Δt): R=$(interface_position(t_init + Δt))")

STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh; compute_centroids=false)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc_b = Dirichlet(T∞)  # Far field temperature
bc = Dirichlet(TM)  # Temperature at the interface
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc_b, :right => bc_b, :top => bc_b, :bottom => bc_b))

# Stefan condition at the interface
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, L))

# Define the source term (no source)
f = (x,y,z,t) -> 0.0
K = (x,y,z) -> 1.0  # Thermal conductivity

Fluide = Phase(capacity, operator, f, K)

# Set up initial condition with tanh profile
# Parameters for tanh profile
factor = 0.5  # Controls width of transition (adjust as needed)
L0 = R0       # Characteristic length (use initial radius)

# Initialize temperature fields
u0ₒ = zeros((nx+1)*(ny+1))

# Create a smooth initial temperature field using normalized tanh
body_init = (x,y,_=0) -> sdf(front, x, y)
cap_init = Capacity(body_init, mesh; compute_centroids=false)
centroids = cap_init.C_ω

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

# Visualize initial temperature field to verify
fig_init = Figure(size=(800, 600))
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
Newton_params = (5, 1e-6, 1e-6, 1.0) # max_iter, tol, reltol, α

# Run the simulation
solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log, timestep_history = solve_StefanMono2D!(solver, Fluide, front, Δt, t_init, t_final,bc_b, bc, stef_cond, mesh, "BE";
   Newton_params=Newton_params, smooth_factor=0.75, window_size=3, enable_stencil_fusion=false,
   method=Base.:\)

# Plot the results
function plot_simulation_results(residuals, xf_log, timestep_history, Ste=nothing)
    # Create results directory
    results_dir = joinpath(pwd(), "simulation_results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # 1. Plot residuals for each timestep
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
    save(joinpath(results_dir, "residuals_$(timestep_history[1][1]).png"), fig_residuals)
    
    # 2. Plot interface positions at different timesteps
    fig_interface = Figure(size=(800, 800))
    ax_interface = Axis(fig_interface[1, 1], 
                       title="Interface Evolution", 
                       xlabel="x", 
                       ylabel="y",
                       aspect=DataAspect())
    
    # Get all timesteps and sort them
    all_timesteps = sort(collect(keys(xf_log)))
    num_timesteps = length(all_timesteps)
    
    # Generate color gradient based on timestep
    colors = cgrad(:viridis, num_timesteps)
    
    # Modify the plotting loop in the interface evolution section:
    for (i, timestep) in enumerate(all_timesteps)
        markers = xf_log[timestep]
        
        # For closed interfaces, make sure first and last markers match
        # This ensures proper visualization of closed curves
        if length(markers) > 1 && isapprox(markers[1][1], markers[end][1], atol=1e-10) && 
           isapprox(markers[1][2], markers[end][2], atol=1e-10)
            # It's a closed interface
            is_closed = true
        else
            is_closed = false
        end
        
        # Extract marker coordinates for plotting
        if is_closed
            # For closed interfaces, exclude the last marker (which should be a duplicate)
            marker_x = [m[1] for m in markers[1:end-1]]
            marker_y = [m[2] for m in markers[1:end-1]]
            
            # Add the first marker again to close the loop properly
            push!(marker_x, markers[1][1])
            push!(marker_y, markers[1][2])
        else
            # For open interfaces, use all markers
            marker_x = [m[1] for m in markers]
            marker_y = [m[2] for m in markers]
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
    
    save(joinpath(results_dir, "interface_evolution.png"), fig_interface)
    
    # 3. Plot radius evolution over time
    times = [hist[1] for hist in timestep_history]
    radii = Float64[]
    radius_stds = Float64[]
    
    for timestep in all_timesteps
        markers = xf_log[timestep]
        
        # Calculate geometric center
        center_x = sum(m[1] for m in markers) / length(markers)
        center_y = sum(m[2] for m in markers) / length(markers)
        
        # Calculate radii and statistics
        marker_radii = [sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in markers]
        mean_radius = sum(marker_radii) / length(marker_radii)
        radius_std = sqrt(sum((r - mean_radius)^2 for r in marker_radii) / length(marker_radii))
        
        push!(radii, mean_radius)
        push!(radius_stds, radius_std)
    end
    
    fig_radius = Figure(size=(800, 600))
    ax_radius = Axis(fig_radius[1, 1], 
                    title="Interface Radius Evolution", 
                    xlabel="Time", 
                    ylabel="Mean Radius")
    
    # Plot radius vs time - CORRECTION
    timestep_to_time = Dict{Int, Float64}()
    for (timestep, hist) in enumerate(timestep_history)
        timestep_to_time[timestep] = hist[1]  # Store actual time for each timestep 
    end

    # Create correct time values for each radius
    times_for_plot = Float64[]
    for ts in all_timesteps
        # Map each timestep to its actual simulation time
        # The timestep number is ts, which starts at 1 (first timestep)
        # xf_log keys correspond to original timestep numbers
        if ts == 1
            # Initial time
            push!(times_for_plot, timestep_history[1][1])
        else
            # For subsequent timesteps
            time_index = min(ts, length(timestep_history))
            push!(times_for_plot, timestep_history[time_index][1])
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
    
    # 4. Plot adaptive timestep history
    fig_dt = Figure(size=(800, 400))
    ax_dt = Axis(fig_dt[1, 1], 
                title="Adaptive Timestep History", 
                xlabel="Time", 
                ylabel="Δt")
    
    dt_times = [hist[1] for hist in timestep_history]
    dt_values = [hist[2] for hist in timestep_history]
    
    lines!(ax_dt, dt_times, dt_values, 
          linewidth=2)
    
    save(joinpath(results_dir, "timestep_history.png"), fig_dt)
    
    return results_dir
end

results_dir = plot_simulation_results(residuals, xf_log, timestep_history, Ste)
println("\nSimulation results visualization saved to: $results_dir")


function plot_temperature_heatmaps(solver, mesh, timestep_history)
    # On suppose que solver.states[i] == vcat(Tw, Tg)
    xi = mesh.nodes[1]           # vecteur x de taille nx+1
    yi = mesh.nodes[2]           # vecteur y de taille ny+1
    nx1, ny1 = length(xi), length(yi)
    npts = nx1 * ny1

    for (i, Tstate) in enumerate(solver.states)
        # extraire et reshaper Tw
        Tw = Tstate[1:npts]
        Tmat = reshape(Tw, (nx1, ny1))

        # créer une figure
        fig = Figure(size=(600,500))
        ax = Axis(fig[1,1],
                  title="Temperature bulk, pas de temps $(i)",
                  xlabel="x", ylabel="y", aspect=DataAspect())
        hm = heatmap!(ax, xi, yi, Tmat; colormap=:thermal)
        Colorbar(fig[1,2], hm, label="T")

        display(fig)
        save(joinpath("simulation_results","temp_step_$(i).png"), fig)
    end
end

# appel après la simulation
plot_temperature_heatmaps(solver, mesh, timestep_history)

function create_temperature_interface_animation(solver, mesh, xf_log, timestep_history)
    # Dimensions du maillage
    xi = mesh.nodes[1]
    yi = mesh.nodes[2]
    nx1, ny1 = length(xi), length(yi)
    npts = nx1 * ny1
    
    # Extraire les limites de température pour une échelle de couleur cohérente
    all_temps = Float64[]
    for Tstate in solver.states
        Tw = Tstate[1:npts]
        push!(all_temps, extrema(Tw)...)
    end
    temp_limits = (minimum(all_temps), maximum(all_temps))
    
    # Récupérer tous les pas de temps d'interface
    all_timesteps = sort(collect(keys(xf_log)))
    
    # Créer l'animation directement avec record
    fig = Figure(size=(800, 700))
    ax = Axis(fig[1, 1], 
            title="Temperature & Interface Evolution", 
            xlabel="x", ylabel="y", aspect=DataAspect())
    
    # Créer le répertoire des résultats si nécessaire
    results_dir = joinpath(pwd(), "simulation_results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Enregistrer l'animation
    record(fig, joinpath(results_dir, "temperature_interface_animation.mp4"), 1:length(solver.states)) do i
        empty!(ax)  # Effacer l'axe pour la nouvelle frame
        
        # Mettre à jour le titre avec le temps actuel
        current_time = i
        ax.title = "Temperature & Interface Evolution, t=$(round(current_time, digits=3))"
        
        # Extraire et reformater la température pour ce pas de temps
        Tstate = solver.states[i]
        Tw = Tstate[1:npts]
        Tmat = reshape(Tw, (nx1, ny1))
        
        # Afficher le heatmap de température
        hm = heatmap!(ax, xi, yi, Tmat, colormap=:thermal, colorrange=temp_limits)
        Colorbar(fig[1, 2], hm, label="Temperature")
        
        # Superposer l'interface si disponible pour ce pas de temps
        ts = i
        if ts <= length(all_timesteps)
            markers = xf_log[all_timesteps[ts]]
            
            # Extraire les coordonnées pour le tracé
            marker_x = [m[1] for m in markers]
            marker_y = [m[2] for m in markers]
            
            # Tracer l'interface comme une ligne fermée
            lines!(ax, marker_x, marker_y, color=:white, linewidth=3)
            scatter!(ax, marker_x, marker_y, color=:white, markersize=3, alpha=0.7)
        end
        
        # Afficher la progression
        println("Frame $i of $(length(solver.states))")
    end
    
    println("\nAnimation saved to: $(joinpath(results_dir, "temperature_interface_animation.mp4"))")
    
    # Créer une version GIF également (plus compatible)
    record(fig, joinpath(results_dir, "temperature_interface_animation.gif"), 1:length(solver.states)) do i
        empty!(ax)
        
        current_time = i 
        ax.title = "Temperature & Interface Evolution, t=$(round(current_time, digits=3))"
        
        Tstate = solver.states[i]
        Tw = Tstate[1:npts]
        Tmat = reshape(Tw, (nx1, ny1))
        
        hm = heatmap!(ax, xi, yi, Tmat, colormap=:thermal, colorrange=temp_limits)
        Colorbar(fig[1, 2], hm, label="Temperature")
        
        ts = i
        if ts <= length(all_timesteps)
            markers = xf_log[all_timesteps[ts]]
            marker_x = [m[1] for m in markers]
            marker_y = [m[2] for m in markers]
            lines!(ax, marker_x, marker_y, color=:white, linewidth=3)
            scatter!(ax, marker_x, marker_y, color=:white, markersize=3, alpha=0.7)
        end
    end
    
    println("GIF animation saved to: $(joinpath(results_dir, "temperature_interface_animation.gif"))")
    
    return joinpath(results_dir, "temperature_interface_animation.mp4")
end

# Appeler la fonction après la simulation
animation_file = create_temperature_interface_animation(solver, mesh, xf_log, timestep_history)