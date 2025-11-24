using Penguin
using IterativeSolvers, SpecialFunctions
using Roots
using CairoMakie

### 2D Test Case: Monophasic Unsteady Diffusion Equation inside an oscillating Disk with Manufactured Solution
# Define the mesh - larger domain to accommodate oscillation
nx, ny = 64, 64
lx, ly = 4.0, 4.0  
x0, y0 = 0.0, 0.0
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define translation parameters
radius = 1.0       # Constant radius of disk
velocity_x = 1.0    # Translation velocity in x-direction
velocity_y = 0.0    # Translation velocity in y-direction
x_0_initial = 2.0   # Initial x-position
y_0_initial = ly/2  # Initial y-position
D = 1.0             # diffusion coefficient
println("Peclet number: ", (velocity_x * (lx/nx)) / D)

# Define the translating body as a level set function
function translating_body(x, y, t)
    # Calculate current position of center
    x_t = x_0_initial + velocity_x * t
    y_t = y_0_initial + velocity_y * t
    
    # Return signed distance function to disk
    return sqrt((x - x_t)^2 + (y - y_t)^2) - radius
end

# Define the Space-Time mesh
Δt = 0.5*(lx/nx)^2  # Time step based on mesh size
Tstart = 0.01  # Start at small positive time to avoid t=0 singularity
Tend = 0.2
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(translating_body, STmesh)

# Initialize the velocity field to match disk translation
# Number of nodes
N = (nx + 1) * (ny + 1)*2

# Initialize velocity components - set to constant field equal to disk translation
uₒx = ones(N) * velocity_x  # Constant velocity in x-direction
uₒy = ones(N) * velocity_y  # Constant velocity in y-direction
uₒt = zeros(N)  # No time component needed for this problem
uₒ = (uₒx, uₒy, uₒt)

# For boundary velocities
uᵧ = zeros(3*N) * 1.0
uᵧx = ones(N) * velocity_x  # Constant velocity in x-direction
uᵧy = ones(N) * velocity_y  # Constant velocity in y-direction
uᵧt = zeros(N)  # No time component needed for this problem
uᵧ = vcat(uᵧx, uᵧy, uᵧt)

# Define the operators
operator = ConvectionOps(capacity, uₒ, uᵧ)

# Define the boundary conditions
# Homogeneous Dirichlet on domain boundaries
bc = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc))

# Dirichlet condition on the disk boundary
ic = Dirichlet(1.0)

f = (x,y,z,t) -> 0.0

# Define the phase with source term from manufactured solution
Fluide = Phase(capacity, operator, f, (x,y,z) -> D)

# Create initial condition arrays with Gaussian bump
# Calculate mesh nodes coordinates
mesh_nodes_x = repeat(mesh.nodes[1], outer=length(mesh.nodes[2]))
mesh_nodes_y = repeat(mesh.nodes[2], inner=length(mesh.nodes[1]))
npts = (nx+1)*(ny+1)

# Parameters for the Gaussian bump
amplitude = 1.0       # Maximum temperature value
sigma = radius / 10.0  # Controls the width of the Gaussian (related to disk radius)

# Initialize temperature arrays
T0ₒ = zeros((nx+1)*(ny+1))
T0ᵧ = zeros((nx+1)*(ny+1))

# Fill arrays with Gaussian values
for i in 1:length(mesh_nodes_x)
    x = mesh_nodes_x[i]
    y = mesh_nodes_y[i]
    
    # Distance from initial disk center
    dist = sqrt((x - x_0_initial)^2 + (y - y_0_initial)^2)
    
    # Gaussian function
    if dist <= 2*radius  # Only apply within a reasonable distance
        T0ₒ[i] = amplitude * exp(-(dist^2) / (2*sigma^2))
    end
    # Boundary values remain at zero
end

# Set the same values for boundary temperatures
T0ᵧ = copy(T0ₒ)

T0 = vcat(T0ₒ, T0ᵧ)

# Define the solver
solver = MovingAdvDiffusionUnsteadyMono(Fluide, bc_b, ic, Δt, T0, mesh, "BE")

# Solve the problem
solve_MovingAdvDiffusionUnsteadyMono!(solver, Fluide, translating_body, Δt, Tstart, Tend, bc_b, ic, mesh, "BE", uₒ, uᵧ; method=Base.:\)

# Check errors based on last body 
body_tend = (x, y,_=0) ->  begin
    # Calculate center position at Tend
    x_t = x_0_initial + velocity_x * Tend
    y_t = y_0_initial + velocity_y * Tend
    # Return signed distance function to disk at Tend
    return sqrt((x - x_t)^2 + (y - y_t)^2) - radius
end
capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)
#Φ_ana_tend(x, y) = Φ_ana(x, y, Tend)
#u_ana1, u_num1, global_err1, full_err1, cut_err1, empty_err1 = check_convergence(Φ_ana_tend, solver, capacity_tend, 2, false)

using CSV, DataFrames
# Function to save simulation data to CSV files - adapted for translating disk
function save_simulation_data(solver, mesh, body_func)
    # Create directory for data
    data_dir = joinpath(pwd(), "simulation_data")
    mkpath(data_dir)
    
    # Extract mesh dimensions
    xi = mesh.centers[1]
    yi = mesh.centers[2]
    nx = length(xi)
    ny = length(yi)
    npts = (nx+1) * (ny+1)
    
    # Save mesh information
    mesh_df = DataFrame(
        x = repeat(mesh.nodes[1], outer=length(mesh.nodes[2])),
        y = repeat(mesh.nodes[2], inner=length(mesh.nodes[1]))
    )
    CSV.write(joinpath(data_dir, "mesh.csv"), mesh_df)
    
    # Save simulation parameters for translating disk
    params_df = DataFrame(
        parameter = ["radius", "velocity_x", "velocity_y", "x_0_initial", "y_0_initial", "D", "dt", "Tend"],
        value = [radius, velocity_x, velocity_y, x_0_initial, y_0_initial, D, Δt, Tend]
    )
    CSV.write(joinpath(data_dir, "parameters.csv"), params_df)
    
    # Save temperature data for each timestep
    for (i, state) in enumerate(solver.states)
        t = Tstart + (i-1) * Δt
        
        # Extract temperature data
        Tw = state[1:npts]
        
        # Create dataframe with position and temperature
        temp_df = DataFrame(
            x = mesh_df.x,
            y = mesh_df.y,
            temperature = Tw
        )
        
        # Save to CSV
        CSV.write(joinpath(data_dir, "temperature_t$(i-1).csv"), temp_df)
        
        # Save disk parameters at this time
        x_t = x_0_initial + velocity_x * t
        y_t = y_0_initial + velocity_y * t
        
        disk_df = DataFrame(
            time = t,
            x_center = x_t,
            y_center = y_t,
            radius = radius
        )
        
        # Append to disk positions file
        if i == 1
            CSV.write(joinpath(data_dir, "disk_parameters.csv"), disk_df)
        else
            CSV.write(joinpath(data_dir, "disk_parameters.csv"), disk_df, append=true)
        end
    end
    
    # Create a timestep index file
    timesteps_df = DataFrame(
        timestep = 0:(length(solver.states)-1),
        time = [Tstart + i * Δt for i in 0:(length(solver.states)-1)]
    )
    CSV.write(joinpath(data_dir, "timesteps.csv"), timesteps_df)
    
    println("Simulation data saved to $(data_dir)")
    return data_dir
end

# Function to generate snapshots from saved data - adapted for translating disk
function plot_snapshots_from_csv(data_dir=joinpath(pwd(), "simulation_data"))
    # Check if data directory exists
    if !isdir(data_dir)
        error("Data directory not found: $data_dir")
    end
    
    # Create directory for publication figures
    pub_dir = joinpath(pwd(), "publication_figures")
    mkpath(pub_dir)
    
    # Load parameters
    params_df = CSV.read(joinpath(data_dir, "parameters.csv"), DataFrame)
    params = Dict(Symbol(row.parameter) => row.value for row in eachrow(params_df))
    
    # Load mesh data
    mesh_df = CSV.read(joinpath(data_dir, "mesh.csv"), DataFrame)
    
    # Load timesteps
    timesteps_df = CSV.read(joinpath(data_dir, "timesteps.csv"), DataFrame)
    
    # Load disk parameters
    disk_params_df = CSV.read(joinpath(data_dir, "disk_parameters.csv"), DataFrame)
    
    # Determine all available temperature files
    temp_files = filter(f -> startswith(f, "temperature_t"), readdir(data_dir))
    
    # Extract unique grid points (x,y) to reshape temperature data
    x_unique = unique(mesh_df.x)
    y_unique = unique(mesh_df.y)
    nx = length(x_unique) - 1
    ny = length(y_unique) - 1
    Δx = x_unique[2] - x_unique[1]
    Δy = y_unique[2] - y_unique[1]
    
    # Find temperature limits across all timesteps
    temp_min = Inf
    temp_max = -Inf
    
    for file in temp_files
        temp_df = CSV.read(joinpath(data_dir, file), DataFrame)
        temp_min = 0.0
        temp_max = 1.0
    end
    
    temp_limits = (temp_min, temp_max)
    
    # Calculate total travel distance
    total_time = timesteps_df.time[end]
    total_x_travel = params[:velocity_x] * total_time
    total_y_travel = params[:velocity_y] * total_time
    
    # Choose 4 evenly spaced snapshots
    num_snapshots = 4
    snapshot_indices = round.(Int, range(1, length(timesteps_df.timestep), length=num_snapshots))
    snapshot_times = timesteps_df.time[snapshot_indices]
    
    # Create figure with 4 panels
    fig = Figure(resolution=(1500, 400), fontsize=12)
    
    # Create common colorbar
    Colorbar(fig[1:1, 5], colormap=:viridis, 
             label="Temperature", labelsize=14, size=25, ticklabelsize=12)
    
    # Determine global plot limits to keep consistent view
    x_min = params[:x_0_initial] - 2
    x_max = params[:x_0_initial] + total_x_travel + 2
    y_min = params[:y_0_initial] - 2
    y_max = params[:y_0_initial] + total_y_travel + 2
    
    for (i, t) in enumerate(snapshot_times)
        # Find closest timestep
        closest_idx = argmin(abs.(timesteps_df.time .- t))
        timestep = timesteps_df.timestep[closest_idx]
        t_exact = timesteps_df.time[closest_idx]
        
        # Load temperature data for this timestep
        temp_df = CSV.read(joinpath(data_dir, "temperature_t$(timestep).csv"), DataFrame)
        
        # Get disk parameters at this time
        disk_row = disk_params_df[abs.(disk_params_df.time .- t_exact) .< 1e-10, :]
        x_center = disk_row.x_center[1]
        y_center = disk_row.y_center[1]
        disk_radius = disk_row.radius[1]
        
        # Create subplot
        ax = Axis(fig[1, i], 
                  title="t=$(round(t_exact, digits=3)), x=$(round(x_center, digits=2)), y=$(round(y_center, digits=2))",
                  xlabel="x", ylabel="y",
                  aspect=DataAspect(),
                  limits=(x_min, x_max, y_min, y_max))
        
        # Reshape temperature data for heatmap
        T_mat = reshape(temp_df.temperature, (length(x_unique), length(y_unique)))'

        # Replace zeros with NaN for better visualization
        T_mat[T_mat .== 0.0] .= NaN
        
        # Plot temperature field
        hm = heatmap!(ax, x_unique, y_unique, T_mat', colormap=:viridis)
        
        # Draw the disk interface
        n_interface_points = 200
        theta = range(0, 2π, length=n_interface_points)
        
        interface_x = x_center .+ disk_radius .* cos.(theta)
        interface_y = y_center .+ disk_radius .* sin.(theta)
        
        # Plot interface with higher visibility
        lines!(ax, interface_x, interface_y, color=:black, linewidth=2.5)
        
        # Draw velocity vector
        arrow_length = 0.5
        arrows!(ax, [x_center], [y_center], 
                [arrow_length * params[:velocity_x]], [arrow_length * params[:velocity_y]], 
                color=:red, linewidth=1.5, arrowsize=15)
        
        # Draw the disk's path
        path_x = [params[:x_0_initial] + params[:velocity_x] * t for t in 0:0.01:total_time]
        path_y = [params[:y_0_initial] + params[:velocity_y] * t for t in 0:0.01:total_time]
        lines!(ax, path_x, path_y, color=:black, linestyle=:dash, linewidth=1.0)
        
        # Mark current position on the path
        scatter!(ax, [x_center], [y_center], color=:red, markersize=8)
    end
    
    # Adjust layout
    fig[1, 1:4] = [GridLayout() for _ in 1:4]
    colgap!(fig.layout, 10)
    
    # Save the figure
    save(joinpath(pub_dir, "translating_disk_snapshots.pdf"), fig, pt_per_unit=2)
    save(joinpath(pub_dir, "translating_disk_snapshots.png"), fig, px_per_unit=4)
    
    # Create position vs time figure
    time_fig = Figure(resolution=(800, 600), fontsize=12)
    
    # Plot x position
    ax_x = Axis(time_fig[1, 1], 
               title="Disk Position Over Time",
               xlabel="Time (t)",
               ylabel="Position")
                  
    # Plot position evolution
    times = disk_params_df.time
    x_positions = disk_params_df.x_center
    y_positions = disk_params_df.y_center
    
    lines!(ax_x, times, x_positions, color=:blue, linewidth=2.0, label="x position")
    lines!(ax_x, times, y_positions, color=:red, linewidth=2.0, label="y position")
    
    # Mark snapshot times
    for t in snapshot_times
        closest_idx = argmin(abs.(times .- t))
        t_exact = times[closest_idx]
        x_pos = x_positions[closest_idx]
        y_pos = y_positions[closest_idx]
        
        scatter!(ax_x, [t_exact], [x_pos], color=:blue, markersize=8)
        scatter!(ax_x, [t_exact], [y_pos], color=:red, markersize=8)
    end
    
    axislegend(ax_x)
    
    save(joinpath(pub_dir, "position_vs_time.pdf"), time_fig)
    save(joinpath(pub_dir, "position_vs_time.png"), time_fig, px_per_unit=4)
    
    println("Publication figures from CSV data saved to $(pub_dir)")
    return fig
end

# After running your simulation
data_dir = save_simulation_data(solver, mesh, translating_body)

# Either right away or in a separate session
plot_snapshots_from_csv()  # Uses default data directory