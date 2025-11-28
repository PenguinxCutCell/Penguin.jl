using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using Interpolations
using Statistics
using FrontCutTracking

### 3D Test Case: Frank Sphere (Stefan Problem with Spherical Interface)
### Ice sphere growing in undercooled liquid with self-similar solution

# Argument to enable/disable plotting with GLMakie
# Set to true to generate 3D visualizations
const ENABLE_PLOTTING = get(ENV, "STEFAN3D_PLOT", "false") == "true"

if ENABLE_PLOTTING
    using GLMakie
    println("3D plotting enabled with GLMakie")
else
    using CairoMakie
    println("Plotting disabled for GLMakie, using CairoMakie for 2D fallback plots only")
end

# Define physical parameters
L = 1.0      # Latent heat
c = 1.0      # Specific heat capacity
TM = 0.0     # Melting temperature (inside sphere)
T∞ = -0.5    # Far field temperature (undercooled liquid)

# Calculate the Stefan number
Ste = (c * (TM - T∞)) / L
println("Stefan number: $Ste")

# Define F(s) function using the exponential integral E₁
function F(s)
    return expint(s^2/4)  # E₁(s²/4)
end

# Calculate the similarity parameter S (for 3D spherical growth)
# S is determined from the Stefan condition: ρL*S² = 2*k*(T∞-TM)/F(S)
# where F(s) = E₁(s²/4) is the exponential integral
# For Ste=0.5 in 3D spherical coordinates, S ≈ 1.56
S = 1.56
println("Similarity parameter S = $S")

# Set initial conditions
# Using R(t) = S*sqrt(t) as the self-similar solution
t_init = 1.0   # Initial time
t_final = 1.1  # Final time

# Analytical temperature function (radial symmetry)
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

# Print information about the simulation
println("Initial radius at t=$t_init: R=$(interface_position(t_init))")

# Define the spatial mesh (3D)
nx, ny, nz = 4, 4, 4  # Lower resolution for 3D to manage computational cost
lx, ly, lz = 12.0, 12.0, 12.0
x0, y0, z0 = -6.0, -6.0, -6.0
Δx, Δy, Δz = lx/(nx), ly/(ny), lz/(nz)
mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))

println("3D Mesh created with dimensions: $(nx) x $(ny) x $(nz)")
println("Δx=$(Δx), Δy=$(Δy), Δz=$(Δz)")
println("Domain: [$x0, $(x0+lx)] x [$y0, $(y0+ly)] x [$z0, $(z0+lz)]")

# Create the 3D front-tracking body (sphere)
nmarkers = 20  # Number of markers on the sphere surface
front = FrontTracker3D()
create_sphere!(front, 0.0, 0.0, 0.0, interface_position(t_init), nmarkers)

# Define the initial position of the front using the SDF
body = (x, y, z, t, _=0) -> -sdf(front, x, y, z)

# Define the Space-Time mesh
Δt = 0.1*(lx / nx)^2  # Time step size (smaller for 3D stability)
t_final = t_init + 5Δt  # Run for 5 time steps
println("Time step Δt = $Δt")
println("Final radius at t=$(t_init + Δt): R=$(interface_position(t_init + Δt))")

STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc_b = Dirichlet(T∞)  # Far field temperature
bc = Dirichlet(TM)    # Temperature at the interface

# 3D border conditions (6 faces)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :left => bc_b, 
    :right => bc_b, 
    :top => bc_b, 
    :bottom => bc_b,
    :front => bc_b,
    :back => bc_b
))

# Stefan condition at the interface
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, L))

# Define the source term (no source) and thermal conductivity
f = (x, y, z, t) -> 0.0
K = (x, y, z, t) -> 1.0  # Thermal conductivity

source(x, y, z, t,teval)= f(x, y, z, t)
Fluide = Phase(capacity, operator, source, K)

# Set up initial condition for 3D
n_total = (nx+1)*(ny+1)*(nz+1)
u0ₒ = zeros(n_total)
body_init = (x, y, z, _=0) -> -sdf(front, x, y, z)
cap_init = Capacity(body_init, mesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
centroids = cap_init.C_ω

# Initialize the temperature field
for idx in 1:length(centroids)
    centroid = centroids[idx]
    x, y, z = centroid[1], centroid[2], centroid[3]
    r = sqrt(x^2 + y^2 + z^2)
    u0ₒ[idx] = analytical_temperature(r, t_init)
end
u0ᵧ = ones(n_total)*TM  # Initial temperature at the interface
u0 = vcat(u0ₒ, u0ᵧ)

println("\nInitial temperature field set up")
println("Number of DOFs: $(length(u0))")

# Newton parameters
Newton_params = (20, 1e-5, 1e-5, 1.0)  # max_iter, tol, reltol, α

# Run the simulation
println("\nStarting 3D Stefan problem solver...")
solver = StefanMono3D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log, timestep_history, phase, position_increments = solve_StefanMono3D!(
    solver, Fluide, front, Δt, t_init, t_final, bc_b, bc, stef_cond, mesh, "BE";
    Newton_params=Newton_params,
    method=Base.:\,
    plot_results=ENABLE_PLOTTING
)

println("\n=== Simulation completed ===")

# Function to plot position increments (2D plot, works with CairoMakie)
function plot_position_increments(residuals, position_increments, timestep_history)
    results_dir = joinpath(pwd(), "simulation_results_3D")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    fig_increments = Figure(size=(900, 600))
    ax_increments = Axis(fig_increments[1, 1], 
                      title="Position Changes Between Iterations (3D Stefan)", 
                      xlabel="Iteration", 
                      ylabel="Position change (log scale)",
                      yscale=log10)
    
    for (timestep, inc_vec) in sort(collect(position_increments))
        if !isempty(inc_vec)
            lines!(ax_increments, 1:length(inc_vec), inc_vec, 
                  label="Timestep $timestep", 
                  linewidth=2)
        end
    end
    
    Legend(fig_increments[1, 2], ax_increments)
    save(joinpath(results_dir, "position_increments_3D.png"), fig_increments)
    
    return fig_increments
end

# Function to plot convergence history (2D plot)
function plot_simulation_results_3D(residuals, xf_log, timestep_history)
    results_dir = joinpath(pwd(), "simulation_results_3D")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Plot residuals for each timestep
    fig_residuals = Figure(size=(900, 600))
    ax_residuals = Axis(fig_residuals[1, 1], 
                        title="Convergence History (3D Stefan)", 
                        xlabel="Iteration", 
                        ylabel="Residual Norm (log scale)",
                        yscale=log10)
    
    for (timestep, residual_vec) in sort(collect(residuals))
        if !isempty(residual_vec)
            lines!(ax_residuals, 1:length(residual_vec), residual_vec, 
                  label="Timestep $timestep", 
                  linewidth=2)
        end
    end
    
    Legend(fig_residuals[1, 2], ax_residuals)
    save(joinpath(results_dir, "residuals_3D.png"), fig_residuals)
    
    # Plot radius evolution over time
    all_timesteps = sort(collect(keys(xf_log)))
    times = Float64[]
    radii = Float64[]
    
    for timestep in all_timesteps
        markers = xf_log[timestep]
        if !isempty(markers)
            # Calculate geometric center
            center_x = sum(m[1] for m in markers) / length(markers)
            center_y = sum(m[2] for m in markers) / length(markers)
            center_z = sum(m[3] for m in markers) / length(markers)
            
            # Calculate mean radius
            marker_radii = [sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2 + (m[3] - center_z)^2) for m in markers]
            mean_radius = sum(marker_radii) / length(marker_radii)
            
            push!(radii, mean_radius)
            
            # Get time for this timestep
            time_idx = min(timestep, length(timestep_history))
            push!(times, timestep_history[time_idx][1])
        end
    end
    
    fig_radius = Figure(size=(800, 600))
    ax_radius = Axis(fig_radius[1, 1], 
                    title="Interface Radius Evolution (3D Stefan)", 
                    xlabel="Time", 
                    ylabel="Mean Radius")
    
    # Plot simulation results
    scatter!(ax_radius, times, radii, label="Simulation", markersize=8)
    
    # Add analytical solution for comparison
    if !isempty(times)
        analytical_times = range(times[1], stop=times[end], length=100)
        analytical_radii = [interface_position(t) for t in analytical_times]
        lines!(ax_radius, analytical_times, analytical_radii,
            label="Analytical", linewidth=2, color=:red, linestyle=:dash)
    end
    
    axislegend(ax_radius)
    save(joinpath(results_dir, "radius_evolution_3D.png"), fig_radius)
    
    return results_dir
end

# Plot 3D interface using GLMakie (only if enabled)
function plot_3D_interface(xf_log, timestep_history)
    if !ENABLE_PLOTTING
        println("3D plotting is disabled. Set STEFAN3D_PLOT=true to enable.")
        return nothing
    end
    
    results_dir = joinpath(pwd(), "simulation_results_3D")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Plot interface evolution in 3D
    all_timesteps = sort(collect(keys(xf_log)))
    
    fig = Figure(size=(1000, 800))
    ax = Axis3(fig[1, 1], 
               title="3D Interface Evolution (Sphere)", 
               xlabel="x", ylabel="y", zlabel="z",
               aspect=:data)
    
    # Generate color gradient based on timestep
    colors = cgrad(:viridis, length(all_timesteps))
    
    for (idx, timestep) in enumerate(all_timesteps)
        markers = xf_log[timestep]
        if !isempty(markers)
            xs = [m[1] for m in markers]
            ys = [m[2] for m in markers]
            zs = [m[3] for m in markers]
            
            scatter!(ax, xs, ys, zs, color=colors[idx], markersize=3, 
                    label="t=$(round(timestep_history[min(timestep, length(timestep_history))][1], digits=3))")
        end
    end
    
    save(joinpath(results_dir, "interface_3D.png"), fig)
    display(fig)
    
    return fig
end

# Run 2D diagnostics (always)
println("\nGenerating 2D diagnostic plots...")
results_dir = plot_simulation_results_3D(residuals, xf_log, timestep_history)
plot_position_increments(residuals, position_increments, timestep_history)
println("Results saved to: $results_dir")

# Run 3D visualization (only if enabled)
if ENABLE_PLOTTING
    println("\nGenerating 3D visualization...")
    plot_3D_interface(xf_log, timestep_history)
end

# Print final statistics
println("\n=== Final Statistics ===")
if !isempty(xf_log)
    final_timestep = maximum(keys(xf_log))
    final_markers = xf_log[final_timestep]
    if !isempty(final_markers)
        center_x = sum(m[1] for m in final_markers) / length(final_markers)
        center_y = sum(m[2] for m in final_markers) / length(final_markers)
        center_z = sum(m[3] for m in final_markers) / length(final_markers)
        final_radii = [sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2 + (m[3] - center_z)^2) for m in final_markers]
        final_mean_radius = sum(final_radii) / length(final_radii)
        final_std_radius = sqrt(sum((r - final_mean_radius)^2 for r in final_radii) / length(final_radii))
        
        expected_radius = interface_position(timestep_history[end][1])
        
        println("Final mean radius: $(round(final_mean_radius, digits=6))")
        println("Expected radius: $(round(expected_radius, digits=6))")
        println("Radius std dev: $(round(final_std_radius, digits=6)) ($(round(100*final_std_radius/final_mean_radius, digits=2))%)")
        println("Relative error: $(round(100*abs(final_mean_radius - expected_radius)/expected_radius, digits=2))%")
    end
end

println("\n3D Stefan problem example complete.")
