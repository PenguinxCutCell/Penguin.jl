using Penguin
using IterativeSolvers, SpecialFunctions
using Roots
using CairoMakie

### 2D Test Case: Monophasic Unsteady Diffusion Equation inside an oscillating Disk with Manufactured Solution
# Define the mesh - larger domain to accommodate oscillation
nx, ny = 32, 32
lx, ly = 4.0, 4.0  
x0, y0 = 0.0, 0.0
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define oscillation parameters
radius_mean = 1.0    # mean radius of disk
radius_amp = 0.5     # amplitude of oscillation 
period = 1.0         # oscillation period T
x_0 = lx/2           # center x-position (fixed)
y_0 = ly/2           # center y-position (fixed)
ν = 1.0              # damping parameter for analytical solution
D = 1.0              # diffusion coefficient

# Define the oscillating body as a level set function
function oscillating_body(x, y, t)
    # Calculate oscillating radius
    R_t = radius_mean + radius_amp * sin(2π * t / period)
    
    # Return signed distance function to disk
    return (sqrt((x - x_0)^2 + (y - y_0)^2) - R_t)
end

# Analytical solution Φ_ana(r,t) = exp(-νt - r²/(4D*t*R(t)²))
function Φ_ana(x, y, t)
    # Calculate radius from center
    r = sqrt((x - x_0)^2 + (y - y_0)^2)
    
    # Calculate oscillating radius at this time
    R_t = radius_mean + radius_amp * sin(2π * t / period)
    
    # Handle t=0 case separately to avoid division by zero
    if t ≈ 0.0
        # For t→0, exp(-r²/(4D*t*R(t)²)) → 0 for r>0
        return 0.0  # Initial condition at t=0
    end
    
    # Handle points outside the domain
    if r > R_t
        return 0.0
    end
    
    # Calculate analytical solution
    return R_t*cos(π*x)*cos(π*y)
end

# Implement source term f(x,y,t)
function source_term(x, y, z, t)
    # Calculate radius from center
    r = sqrt((x - x_0)^2 + (y - y_0)^2)
    
    # Calculate oscillating radius
    R_t = radius_mean + radius_amp * sin(2π * t / period)
    
    # Handle points outside domain
    if r > R_t
        return 0.0
    end
    
    # Calculate source term using the manufactured solution formula
    # f(x,y,t) = (π/T)*cos(πx)*cos(πy)*cos(2πt/T) + 2π²*D*(1 + 0.5*sin(2πt/T))*cos(πx)*cos(πy)
    term1 = (π / period) * cos(π * x) * cos(π * y) * cos(2π * t / period)
    term2 = 2 * π^2 * D * (1 + 0.5 * sin(2π * t / period)) * cos(π * x) * cos(π * y)
    
    return term1 + term2
end

# Define the Space-Time mesh
Δt = 0.5*(lx/nx)^2   # Time step based on stability criterion
Tstart = Δt  # Start at small positive time to avoid t=0 singularity
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Tstart+Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(oscillating_body, STmesh; compute_centroids=true, method="VOFI", integration_method=:vofijul)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
# Homogeneous Dirichlet on domain boundaries
bc = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc))



# Dirichlet boundary condition for the disk interface - explicitly showing r=R(t)
robin_bc = Dirichlet((x, y, t) -> begin
    # Calculate radius from center at boundary point
    r = sqrt((x - x_0)^2 + (y - y_0)^2)
    
    # Calculate oscillating radius at this time
    R_t = radius_mean + radius_amp * sin(2π * t / period)
    
    # Return analytical solution evaluated at the boundary
    return Φ_ana(x, y, t)
end)

# Define the phase with source term from manufactured solution
Fluide = Phase(capacity, operator, source_term, (x,y,z) -> D)

# Initialize with analytical solution at t=Tstart
function init_condition(x, y)
    return Φ_ana(x, y, Tstart)
end

# Create initial condition arrays
u0ₒ = zeros((nx+1)*(ny+1))
u0ᵧ = zeros((nx+1)*(ny+1))

# Fill with initial analytical solution
for i in 1:nx+1
    for j in 1:ny+1
        idx = (j-1)*(nx+1) + i
        x = mesh.nodes[1][i]
        y = mesh.nodes[2][j]
        u0ₒ[idx] = init_condition(x, y)
    end
end

u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
solver = MovingDiffusionUnsteadyMono(Fluide, bc_b, robin_bc, Δt, Tstart, u0, mesh, "BE")

# Solve the problem
solve_MovingDiffusionUnsteadyMono!(solver, Fluide, oscillating_body, Δt, Tstart, Tend, bc_b, robin_bc, mesh, "BE"; method=Base.:\, geometry_method="VOFI", integration_method=:vofijul)

# Check errors based on last body 
body_tend = (x, y,_=0) ->  begin
    # Calculate oscillating radius at Tend
    R_t = radius_mean + radius_amp * sin(2π * (Tend) / period)
    return sqrt((x - x_0)^2 + (y - y_0)^2) - R_t
end
capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)
Φ_ana_tend(x, y) = Φ_ana(x, y, Tend)
u_ana1, u_num1, global_err1, full_err1, cut_err1, empty_err1 = check_convergence(Φ_ana_tend, solver, capacity_tend, 2, false)

using CSV, DataFrames

# Function to save simulation data to CSV files - adapted for oscillating disk
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
    
    # Save simulation parameters for oscillating disk
    params_df = DataFrame(
        parameter = ["radius_mean", "radius_amp", "period", "x_0", "y_0", "D", "ν", "dt", "Tend"],
        value = [radius_mean, radius_amp, period, x_0, y_0, D, ν, Δt, Tend]
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
        R_t = radius_mean + radius_amp * sin(2π * t / period)
        R_prime = radius_amp * (2π / period) * cos(2π * t / period)
        
        disk_df = DataFrame(
            time = t,
            x_center = x_0,
            y_center = y_0,
            radius = R_t,
            radius_derivative = R_prime
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

# Function to generate snapshots from saved data - adapted for oscillating disk
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
        temp_min = min(temp_min, minimum(skipmissing(temp_df.temperature)))
        temp_max = max(temp_max, maximum(skipmissing(temp_df.temperature)))
    end
    
    temp_limits = (temp_min, temp_max)
    
    # Define snapshot times - choose specific phases of oscillation
    phases = [0.0, 0.25, 0.5, 0.75]  # t/T for phases
    phase_names = ["t=0 (R=R₀)", "t=T/4 (R=R₀+A)", "t=T/2 (R=R₀)", "t=3T/4 (R=R₀-A)"]
    
    # Calculate actual times for these phases
    period = params[:period]
    snapshot_times = phases .* period
    
    # Create figure with 4 panels
    fig = Figure(resolution=(1500, 400), fontsize=12)
    
    # Create common colorbar
    Colorbar(fig[1:1, 5], limits=temp_limits, colormap=:viridis, 
             label="Temperature", labelsize=14, size=25, ticklabelsize=12)
    
    for (i, (t_target, name)) in enumerate(zip(snapshot_times, phase_names))
        # Find closest timestep
        closest_idx = argmin(abs.(timesteps_df.time .- t_target))
        timestep = timesteps_df.timestep[closest_idx]
        t = timesteps_df.time[closest_idx]
        
        # Load temperature data for this timestep
        temp_df = CSV.read(joinpath(data_dir, "temperature_t$(timestep).csv"), DataFrame)
        
        # Get disk parameters at this time
        disk_row = disk_params_df[abs.(disk_params_df.time .- t) .< 1e-10, :]
        x_center = params[:x_0]  # Fixed for oscillating disk
        y_center = params[:y_0]  # Fixed for oscillating disk
        current_radius = disk_row.radius[1]
        
        # Create subplot
        ax = Axis(fig[1, i], 
                  title="t=$(round(t, digits=3))T, R=$(round(current_radius, digits=2))",
                  xlabel="x", ylabel="y",
                  aspect=DataAspect(),
                  limits=(x_center-2, x_center+2, y_center-2, y_center+2))
        
        # Reshape temperature data for heatmap
        T_mat = reshape(temp_df.temperature, (length(x_unique), length(y_unique)))'

        # Replace zeros with NaN for better visualization
        T_mat[T_mat .== 0.0] .= NaN
        
        # Plot temperature field
        hm = heatmap!(ax, x_unique, y_unique, T_mat, colormap=:viridis, colorrange=temp_limits)
        
        # Draw the interface at current radius
        n_interface_points = 200
        theta = range(0, 2π, length=n_interface_points)
        
        interface_x = x_center .+ current_radius .* cos.(theta) .- Δx/2
        interface_y = y_center .+ current_radius .* sin.(theta) .- Δy/2
        
        # Plot interface with higher visibility
        lines!(ax, interface_x, interface_y, color=:black, linewidth=2.5)
        
        # Draw min and max radius circles for reference
        min_radius = params[:radius_mean] - params[:radius_amp]
        max_radius = params[:radius_mean] + params[:radius_amp]
        
        # Draw min radius circle (dashed)
        min_x = x_center .+ min_radius .* cos.(theta) .- Δx/2
        min_y = y_center .+ min_radius .* sin.(theta) .-  Δy/2
        lines!(ax, min_x, min_y, color=:black, linewidth=1.0, linestyle=:dash)
        
        # Draw max radius circle (dotted)
        max_x = x_center .+ max_radius .* cos.(theta)
        max_y = y_center .+ max_radius .* sin.(theta)
        lines!(ax, max_x, max_y, color=:black, linewidth=1.0, linestyle=:dot)
        
        # Draw normal vectors showing radial direction
        normal_points = 8
        normal_theta = range(0, 2π, length=normal_points+1)[1:end-1]
        normal_length = 0.2
        
        for θ in normal_theta
            nx_point = cos(θ)
            ny_point = sin(θ)
            
            start_x = x_center + current_radius * nx_point
            start_y = y_center + current_radius * ny_point
            
            arrows!(ax, [start_x], [start_y], [normal_length * nx_point], [normal_length * ny_point], 
                   color=:red, linewidth=1.5, arrowsize=10)
        end
        
       
    end
    
    # Adjust layout
    fig[1, 1:4] = [GridLayout() for _ in 1:4]
    colgap!(fig.layout, 10)
    
    # Save the figure
    save(joinpath(pub_dir, "oscillating_disk_snapshots.pdf"), fig, pt_per_unit=2)
    save(joinpath(pub_dir, "oscillating_disk_snapshots.png"), fig, px_per_unit=4)
    
    # Create radius vs time figure
    time_fig = Figure(resolution=(800, 600), fontsize=12)
    ax_time = Axis(time_fig[1, 1], 
                  title="Disk Radius Over Time",
                  xlabel="Time (t)",
                  ylabel="Radius")
                  
    # Plot radius evolution
    times = disk_params_df.time
    radii = disk_params_df.radius
    
    lines!(ax_time, times, radii, color=:blue, linewidth=2.0)
    hlines!(ax_time, [params[:radius_mean]], color=:black, linestyle=:dash, linewidth=1.0, label="Mean radius")
    
    # Mark snapshot times
    for t in snapshot_times
        closest_idx = argmin(abs.(times .- t))
        t_exact = times[closest_idx]
        r = radii[closest_idx]
        
        scatter!(ax_time, [t_exact], [r], color=:red, markersize=8)
    end
    
    axislegend(ax_time)
    
    save(joinpath(pub_dir, "radius_vs_time.pdf"), time_fig)
    
    println("Publication figures from CSV data saved to $(pub_dir)")
    return fig
end

# After running your simulation
#data_dir = save_simulation_data(solver, mesh, oscillating_body)

# Either right away or in a separate session
plot_snapshots_from_csv()  # Uses default data directory
