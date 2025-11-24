using Penguin
using IterativeSolvers, SpecialFunctions
using Roots
using CairoMakie

### 2D Test Case: Diphasic Unsteady Diffusion Equation inside an oscillating Disk 
# Define the mesh - larger domain to accommodate oscillation
nx, ny = 64, 64
lx, ly = 4.0, 4.0  
x0, y0 = 0.0, 0.0
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define translation parameters
radius = 1.0       # Constant radius of disk
velocity_x = 0.0    # Translation velocity in x-direction
velocity_y = 0.0    # Translation velocity in y-direction
x_0_initial = 2.01   # Initial x-position
y_0_initial = 2.01  # Initial y-position
D = 1.0             # diffusion coefficient

# Define the translating body as a level set function
function translating_body(x, y, t)
    # Calculate current position of center
    x_t = x_0_initial + velocity_x * t
    y_t = y_0_initial + velocity_y * t
    
    # Return signed distance function to disk
    return sqrt((x - x_t)^2 + (y - y_t)^2) - radius
end

# Define the translating body as a level set function
function translating_body_c(x, y, t)
    # Calculate current position of center
    x_t = x_0_initial + velocity_x * t
    y_t = y_0_initial + velocity_y * t
    
    # Return signed distance function to disk
    return -(sqrt((x - x_t)^2 + (y - y_t)^2) - radius)
end

# Define the Space-Time mesh
Δt = 0.5*(lx/nx)^2  # Time step based on mesh size
Tstart = 0.0  # Start at small positive time to avoid t=0 singularity
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(translating_body, STmesh)
capacity_c = Capacity(translating_body_c, STmesh)

# Initialize the velocity field to match disk translation
# Number of nodes
N = (nx + 1) * (ny + 1)*2

# Initialize velocity components - set to constant field equal to disk translation
uₒx = fill(velocity_x, N)
uₒy = fill(velocity_y, N)
uₒt = zeros(N)  # No time component needed for this problem
uₒ = (uₒx, uₒy, uₒt)

# For boundary velocities
uᵧ = zeros(3*N)
uᵧx = fill(velocity_x, N)
uᵧy = fill(velocity_y, N)
uᵧt = zeros(N)  # No time component needed for this problem
uᵧ = vcat(uᵧx, uᵧy, uᵧt)
uᵧ = zeros(3*N)


# Define the operators
operator = ConvectionOps(capacity, uₒ, uᵧ)
operator_c = ConvectionOps(capacity_c, uₒ, uᵧ)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc))

ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Define the source term
f1 = (x,y,z,t)->0.0
f2 = (x,y,z,t)->0.0

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, (x,y,_)->1.0)
Fluide_2 = Phase(capacity_c, operator_c, f2, (x,y,_)->1.0)

# Initial condition
T0ₒ1 = ones((nx+1)*(ny+1))
T0ᵧ1 = ones((nx+1)*(ny+1))
T0ₒ2 = zeros((nx+1)*(ny+1))
T0ᵧ2 = zeros((nx+1)*(ny+1))
T0 = vcat(T0ₒ1, T0ᵧ1, T0ₒ2, T0ᵧ2)

# Define the solver
solver = MovingAdvDiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, T0, mesh, "BE")

# Solve the problem
solve_MovingAdvDiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, translating_body, translating_body_c, Δt, Tstart, Tend, bc_b, ic, mesh, "BE", uₒ, uᵧ; method=Base.:\)

using CSV, DataFrames

# Function to save simulation data for diphasic translating disk
function save_simulation_data_diphasic(solver, mesh, body_func)
    # Create directory for data
    data_dir = joinpath(pwd(), "simulation_data_diphasic")
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
        
        # Extract temperature data for both phases
        # Phase 1 - Inside disk (first quarter of state vector)
        T1 = state[1:npts]
        
        # Phase 2 - Outside disk (third quarter of state vector)
        T2 = state[2*npts+1:3*npts]
        
        # Combine data for phase 1 and 2
        temp_df = DataFrame(
            x = mesh_df.x,
            y = mesh_df.y,
            temperature_phase1 = T1,
            temperature_phase2 = T2
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
    
    println("Diphasic simulation data saved to $(data_dir)")
    return data_dir
end

# Function to plot diphasic simulation snapshots
function plot_diphasic_snapshots_from_csv(data_dir=joinpath(pwd(), "simulation_data_diphasic"))
    # Check if data directory exists
    if !isdir(data_dir)
        error("Data directory not found: $data_dir")
    end
    
    # Create directory for publication figures
    pub_dir = joinpath(pwd(), "publication_figures_diphasic")
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
    nx = length(x_unique)
    ny = length(y_unique)
    Δx = x_unique[2] - x_unique[1]
    Δy = y_unique[2] - y_unique[1]
    
    # Find temperature limits for each phase
    temp1_min, temp1_max = Inf, -Inf
    temp2_min, temp2_max = Inf, -Inf
    
    for file in temp_files
        temp_df = CSV.read(joinpath(data_dir, file), DataFrame)
        temp1_min = min(temp1_min, minimum(skipmissing(temp_df.temperature_phase1)))
        temp1_max = max(temp1_max, maximum(skipmissing(temp_df.temperature_phase1)))
        temp2_min = min(temp2_min, minimum(skipmissing(temp_df.temperature_phase2)))
        temp2_max = max(temp2_max, maximum(skipmissing(temp_df.temperature_phase2)))
    end
    
    # Using automatic limits if values are extremes
    if !isfinite(temp1_min) || !isfinite(temp1_max)
        temp1_min, temp1_max = 0.0, 1.0
    end
    if !isfinite(temp2_min) || !isfinite(temp2_max)
        temp2_min, temp2_max = 0.0, 1.0
    end

    temp1_min, temp1_max = 0.0, 1.0
    temp2_min, temp2_max = 0.0, 1.0
    
    # Use a global temperature range for visualization
    temp_min = min(temp1_min, temp2_min)
    temp_max = max(temp1_max, temp2_max)
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
    Colorbar(fig[1:1, 5], colormap=:turbo, 
             label="Temperature", labelsize=14, size=25, ticklabelsize=12)
    
    # Determine global plot limits to keep consistent view
    x_min = params[:x_0_initial] - 2
    x_max = params[:x_0_initial] + total_x_travel + 2
    y_min = params[:y_0_initial] - 2
    y_max = params[:y_0_initial] + total_y_travel + 2

    interface_x = []
    interface_y = []
    
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
        
        # Create a proper temperature grid for visualization
        T_grid = zeros(ny, nx)
        
        # Populate the grid with combined temperature data from both phases
        for (idx, row) in enumerate(eachrow(temp_df))
            # Find the indices in the grid that match the coordinates
            x_idx = findfirst(x -> isapprox(x, row.x, atol=1e-10), x_unique)
            y_idx = findfirst(y -> isapprox(y, row.y, atol=1e-10), y_unique)
            
            if !isnothing(x_idx) && !isnothing(y_idx)
                # Calculate distance from disk center
                dist = sqrt((row.x - x_center)^2 + (row.y - y_center)^2)
                
                # Use phase1 temp inside disk, phase2 temp outside disk
                if dist <= disk_radius
                    T_grid[y_idx, x_idx] = row.temperature_phase1
                else
                    T_grid[y_idx, x_idx] = row.temperature_phase2
                end
            end
        end
        
        # Plot combined temperature field
        hm = heatmap!(ax, x_unique, y_unique, T_grid', colormap=:turbo)
        
        # Draw the disk interface
        n_interface_points = 200
        theta = range(0, 2π, length=n_interface_points)
        
        interface_x = x_center .+ disk_radius .* cos.(theta)
        interface_y = y_center .+ disk_radius .* sin.(theta)
        
        # Plot interface with higher visibility
        lines!(ax, interface_x, interface_y, color=:white, linewidth=2.5)
        
        # Draw velocity vector
        arrow_length = 0.5
        arrows!(ax, [x_center], [y_center], 
                [arrow_length * params[:velocity_x]], [arrow_length * params[:velocity_y]], 
                color=:white, linewidth=1.5, arrowsize=15)
        
        # Draw the disk's path
        path_x = [params[:x_0_initial] + params[:velocity_x] * t for t in 0:0.01:total_time]
        path_y = [params[:y_0_initial] + params[:velocity_y] * t for t in 0:0.01:total_time]
        lines!(ax, path_x, path_y, color=:white, linestyle=:dash, linewidth=1.0)
        
        # Mark current position on the path
        scatter!(ax, [x_center], [y_center], color=:white, markersize=8)
    end
    
    # Adjust layout
    fig[1, 1:4] = [GridLayout() for _ in 1:4]
    colgap!(fig.layout, 10)
    
    # Save the figure
    save(joinpath(pub_dir, "translating_disk_diphasic_snapshots.pdf"), fig, pt_per_unit=2)
    save(joinpath(pub_dir, "translating_disk_diphasic_snapshots.png"), fig, px_per_unit=4)
    
    # Also create a figure showing phases separately
    separate_fig = Figure(resolution=(1500, 800), fontsize=12)
    
    # Add two colorbars for separate phases
    Colorbar(separate_fig[1, 5], limits=(temp1_min, temp1_max), colormap=:viridis, 
             label="Temperature Phase 1 (Inside)", labelsize=14, size=25, ticklabelsize=12)
    Colorbar(separate_fig[2, 5], limits=(temp2_min, temp2_max), colormap=:plasma, 
             label="Temperature Phase 2 (Outside)", labelsize=14, size=25, ticklabelsize=12)
    
    for (i, t) in enumerate(snapshot_times)
        closest_idx = argmin(abs.(timesteps_df.time .- t))
        timestep = timesteps_df.timestep[closest_idx]
        t_exact = timesteps_df.time[closest_idx]
        
        # Load temperature data
        temp_df = CSV.read(joinpath(data_dir, "temperature_t$(timestep).csv"), DataFrame)
        
        # Get disk parameters
        disk_row = disk_params_df[abs.(disk_params_df.time .- t_exact) .< 1e-10, :]
        x_center = disk_row.x_center[1]
        y_center = disk_row.y_center[1]
        disk_radius = disk_row.radius[1]
        
        # Create phase 1 temperature grid (inside disk)
        T1_grid = zeros(ny, nx)
        for (idx, row) in enumerate(eachrow(temp_df))
            x_idx = findfirst(x -> isapprox(x, row.x, atol=1e-10), x_unique)
            y_idx = findfirst(y -> isapprox(y, row.y, atol=1e-10), y_unique)
            
            if !isnothing(x_idx) && !isnothing(y_idx)
                # Only show phase 1 inside the disk
                dist = sqrt((row.x - x_center)^2 + (row.y - y_center)^2)
                if dist <= disk_radius
                    T1_grid[y_idx, x_idx] = row.temperature_phase1
                else
                    T1_grid[y_idx, x_idx] = NaN  # Outside disk is transparent
                end
            end
        end
        
        # Create phase 2 temperature grid (outside disk)
        T2_grid = zeros(ny, nx)
        for (idx, row) in enumerate(eachrow(temp_df))
            x_idx = findfirst(x -> isapprox(x, row.x, atol=1e-10), x_unique)
            y_idx = findfirst(y -> isapprox(y, row.y, atol=1e-10), y_unique)
            
            if !isnothing(x_idx) && !isnothing(y_idx)
                # Only show phase 2 outside the disk
                dist = sqrt((row.x - x_center)^2 + (row.y - y_center)^2)
                if dist > disk_radius
                    T2_grid[y_idx, x_idx] = row.temperature_phase2
                else
                    T2_grid[y_idx, x_idx] = NaN  # Inside disk is transparent
                end
            end
        end
        
        # Plot phase 1 (top row)
        ax1 = Axis(separate_fig[1, i], 
                  title="Phase 1 (Inside) - t=$(round(t_exact, digits=3))",
                  xlabel="x", ylabel="y",
                  aspect=DataAspect(),
                  limits=(x_min, x_max, y_min, y_max))
        
        heatmap!(ax1, x_unique, y_unique, T1_grid, colormap=:viridis, colorrange=(temp1_min, temp1_max))
        lines!(ax1, interface_x, interface_y, color=:white, linewidth=2.0)
        
        # Plot phase 2 (bottom row)
        ax2 = Axis(separate_fig[2, i], 
                  title="Phase 2 (Outside) - t=$(round(t_exact, digits=3))",
                  xlabel="x", ylabel="y",
                  aspect=DataAspect(),
                  limits=(x_min, x_max, y_min, y_max))
        
        heatmap!(ax2, x_unique, y_unique, T2_grid, colormap=:plasma, colorrange=(temp2_min, temp2_max))
        lines!(ax2, interface_x, interface_y, color=:white, linewidth=2.0)
    end
    
    # Adjust layout
    separate_fig[1, 1:4] = [GridLayout() for _ in 1:4]
    separate_fig[2, 1:4] = [GridLayout() for _ in 1:4]
    colgap!(separate_fig.layout, 10)
    
    # Save the separate phases figure
    save(joinpath(pub_dir, "translating_disk_separate_phases.pdf"), separate_fig, pt_per_unit=2)
    save(joinpath(pub_dir, "translating_disk_separate_phases.png"), separate_fig, px_per_unit=4)
    
    println("Publication figures from CSV data saved to $(pub_dir)")
    return fig, separate_fig
end

# After running your simulation
data_dir = save_simulation_data_diphasic(solver, mesh, translating_body)

# Either right away or in a separate session
plot_diphasic_snapshots_from_csv()  # Uses default data directory