using Penguin
using IterativeSolvers, SpecialFunctions
using Roots
using CairoMakie
using CSV, DataFrames

### 3D Test Case: Monophasic Unsteady Diffusion Equation inside an oscillating Sphere with Manufactured Solution
# Define the mesh - larger domain to accommodate oscillation
nx, ny, nz = 32, 32, 32  # Lower resolution for 3D
lx, ly, lz = 4.0, 4.0, 4.0  
x0, y0, z0 = 0.0, 0.0, 0.0
domain = ((x0, lx), (y0, ly), (z0, lz))
mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))

# Define oscillation parameters
radius_mean = 1.0    # mean radius of sphere
radius_amp = 0.5     # amplitude of oscillation 
period = 1.0         # oscillation period T
x_0 = lx/2           # center x-position (fixed)
y_0 = ly/2           # center y-position (fixed)
z_0 = lz/2           # center z-position (fixed)
center = [x_0, y_0, z_0]  # center position vector
ν = 1.0              # damping parameter for analytical solution
D = 1.0              # diffusion coefficient

# Define the oscillating body as a level set function
function oscillating_body(x, y, z, t)
    # Calculate oscillating radius
    R_t = radius_mean + radius_amp * sin(2π * t / period)
    
    # Return signed distance function to sphere
    return (sqrt((x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2) - R_t)
end

# Analytical solution for 3D
function Φ_ana(x, y, z, t)
    # Calculate radius from center
    r = sqrt((x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2)
    
    # Calculate oscillating radius at this time
    R_t = radius_mean + radius_amp * sin(2π * t / period)
    
    # Handle points outside the domain
    if r > R_t
        return 0.0
    end
    
    # Calculate analytical solution for 3D
    return (1 + 0.5 * sin(2π * t / period)) * cos(π * x) * cos(π * y) * cos(π * z)
end

# Implement source term for 3D manufactured solution
function source_term(x, y, z, t)
    # Calculate radius from center
    r = sqrt((x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2)
    
    # Calculate oscillating radius
    R_t = radius_mean + radius_amp * sin(2π * t / period)
    
    # Handle points outside domain
    if r > R_t
        return 0.0
    end
    
    # Calculate source term for 3D manufactured solution
    # Time derivative term
    term1 = (π / period) * cos(π * x) * cos(π * y) * cos(π * z) * cos(2π * t / period)
    
    # Laplacian term (3 dimensions)
    term2 = 3 * π^2 * D * (1 + 0.5 * sin(2π * t / period)) * cos(π * x) * cos(π * y) * cos(π * z)
    
    return term1 + term2
end

# Define the Space-Time mesh
Δt = 0.01
Tstart = 0.01  # Start at small positive time to avoid t=0 singularity
Tend = 0.04
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(oscillating_body, STmesh; method="ImplicitIntegration")

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
# Homogeneous Dirichlet on domain boundaries
bc = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :left => bc, :right => bc, 
    :top => bc, :bottom => bc,
    :front => bc, :back => bc
))

# Dirichlet boundary condition for the sphere interface
robin_bc = Dirichlet((x, y, z, t) -> Φ_ana(x, y, z, t))

# Define the phase with source term from manufactured solution
Fluide = Phase(capacity, operator, source_term, (x,y,z) -> D)

# Initialize with analytical solution at t=Tstart
function init_condition(x, y, z)
    return Φ_ana(x, y, z, Tstart)
end

# Create initial condition arrays
u0ₒ = zeros((nx+1)*(ny+1)*(nz+1))
u0ᵧ = zeros((nx+1)*(ny+1)*(nz+1))

# Fill with initial analytical solution
for i in 1:nx+1
    for j in 1:ny+1
        for k in 1:nz+1
            idx = (k-1)*(nx+1)*(ny+1) + (j-1)*(nx+1) + i
            x = mesh.nodes[1][i]
            y = mesh.nodes[2][j]
            z = mesh.nodes[3][k]
            u0ₒ[idx] = init_condition(x, y, z)
        end
    end
end

u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
solver = MovingDiffusionUnsteadyMono(Fluide, bc_b, robin_bc, Δt, u0, mesh, "BE")

# Solve the problem
solve_MovingDiffusionUnsteadyMono!(solver, Fluide, oscillating_body, Δt, Tstart, Tend, bc_b, robin_bc, mesh, "BE"; method=Base.:\)

# Check errors based on last body 
body_tend = (x, y, z, _=0) ->  begin
    # Calculate oscillating radius at Tend
    R_t = radius_mean + radius_amp * sin(2π * Tend / period)
    return sqrt((x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2) - R_t
end

capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)
Φ_ana_tend(x, y, z) = Φ_ana(x, y, z, Tend)
u_ana1, u_num1, global_err1, full_err1, cut_err1, empty_err1 = check_convergence(Φ_ana_tend, solver, capacity_tend, 2, false)

# Visualization function for 3D data with slices
function visualize_3d_solution(solver, mesh, time_index=1)
    # Extract mesh dimensions
    xi = mesh.nodes[1]
    yi = mesh.nodes[2]
    zi = mesh.nodes[3]
    nx = length(xi) - 1
    ny = length(yi) - 1
    nz = length(zi) - 1
    npts = (nx+1) * (ny+1) * (nz+1)
    
    # Get current time
    t = Tstart + (time_index-1) * Δt
    
    # Extract solution at current time
    state = solver.states[time_index]
    Tw = state[1:npts]
    
    # Reshape temperature data to 3D
    T3D = reshape(Tw, (nx+1, ny+1, nz+1))
    
    # Calculate current sphere radius
    R_t = radius_mean + radius_amp * sin(2π * t / period)
    
    # Create slices through the center
    fig = Figure(resolution=(1200, 400))
    
    # Calculate center indices
    mid_x_idx = div(nx+1, 2) + 1
    mid_y_idx = div(ny+1, 2) + 1
    mid_z_idx = div(nz+1, 2) + 1
    
    mid_x = xi[mid_x_idx]
    mid_y = yi[mid_y_idx]
    mid_z = zi[mid_z_idx]
    
    # Extract slices
    xy_slice = T3D[:, :, mid_z_idx]
    yz_slice = T3D[mid_x_idx, :, :]
    xz_slice = T3D[:, mid_y_idx, :]
    
    # Find temperature limits
    temp_limits = (minimum(filter(!iszero, Tw)), maximum(filter(!iszero, Tw)))
    
    # Create slices through the center
    fig = Figure(resolution=(1200, 400))
    
    # XY slice plot
    ax_xy = Axis(fig[1, 1], 
                 title="XY Plane (z = $(round(mid_z, digits=2)))", 
                 aspect=DataAspect(), 
                 xlabel="x", ylabel="y")
    
    hm_xy = heatmap!(ax_xy, xi, yi, xy_slice', colormap=:viridis, colorrange=temp_limits)
    
    # Draw sphere intersection
    theta = range(0, 2π, length=100)
    circle_x = center[1] .+ R_t .* cos.(theta)
    circle_y = center[2] .+ R_t .* sin.(theta)
    lines!(ax_xy, circle_x, circle_y, color=:white, linewidth=2)
    
    # YZ slice plot
    ax_yz = Axis(fig[1, 2], 
                 title="YZ Plane (x = $(round(mid_x, digits=2)))", 
                 aspect=DataAspect(), 
                 xlabel="y", ylabel="z")
    
    hm_yz = heatmap!(ax_yz, yi, zi, yz_slice', colormap=:viridis, colorrange=temp_limits)
    
    # Draw sphere intersection
    circle_y = center[2] .+ R_t .* cos.(theta)
    circle_z = center[3] .+ R_t .* sin.(theta)
    lines!(ax_yz, circle_y, circle_z, color=:white, linewidth=2)
    
    # XZ slice plot
    ax_xz = Axis(fig[1, 3], 
                 title="XZ Plane (y = $(round(mid_y, digits=2)))", 
                 aspect=DataAspect(), 
                 xlabel="x", ylabel="z")
    
    hm_xz = heatmap!(ax_xz, xi, zi, xz_slice', colormap=:viridis, colorrange=temp_limits)
    
    # Draw sphere intersection
    circle_x = center[1] .+ R_t .* cos.(theta)
    circle_z = center[3] .+ R_t .* sin.(theta)
    lines!(ax_xz, circle_x, circle_z, color=:white, linewidth=2)
    
    # Common colorbar
    Colorbar(fig[1, 4], hm_xy, label="Temperature")
    
    # Add title with time information
    fig_title = "Oscillating Sphere at t=$(round(t, digits=3)), Radius=$(round(R_t, digits=2))"
    Label(fig[0, :], fig_title, fontsize=14)
    
    return fig
end

# Function to save simulation data to CSV files for 3D oscillating sphere
function save_simulation_data(solver, mesh, body_func)
    # Create directory for data
    data_dir = joinpath(pwd(), "simulation_data_3d")
    mkpath(data_dir)
    
    # Extract mesh dimensions
    xi = mesh.centers[1]
    yi = mesh.centers[2]
    zi = mesh.centers[3]
    nx = length(xi)
    ny = length(yi)
    nz = length(zi)
    npts = (nx+1) * (ny+1) * (nz+1)
    
    # Save simulation parameters
    params_df = DataFrame(
        parameter = ["radius_mean", "radius_amp", "period", "x_0", "y_0", "z_0", "D", "ν", "dt", "Tend"],
        value = [radius_mean, radius_amp, period, x_0, y_0, z_0, D, ν, Δt, Tend]
    )
    CSV.write(joinpath(data_dir, "parameters.csv"), params_df)
    
    # Calculate middle indices for slices
    mid_x_idx = div(nx+1, 2) + 1
    mid_y_idx = div(ny+1, 2) + 1
    mid_z_idx = div(nz+1, 2) + 1
    
    # Save center coordinates
    center_df = DataFrame(
        center_point = ["x_center", "y_center", "z_center"],
        value = [mesh.nodes[1][mid_x_idx], mesh.nodes[2][mid_y_idx], mesh.nodes[3][mid_z_idx]]
    )
    CSV.write(joinpath(data_dir, "center_points.csv"), center_df)
    
    # Save X coordinates
    x_df = DataFrame(x = mesh.nodes[1])
    CSV.write(joinpath(data_dir, "x_coords.csv"), x_df)
    
    # Save Y coordinates
    y_df = DataFrame(y = mesh.nodes[2])
    CSV.write(joinpath(data_dir, "y_coords.csv"), y_df)
    
    # Save Z coordinates
    z_df = DataFrame(z = mesh.nodes[3])
    CSV.write(joinpath(data_dir, "z_coords.csv"), z_df)
    
    # Save temperature data for each timestep (only slices to save space)
    for (i, state) in enumerate(solver.states)
        t = Tstart + (i-1) * Δt
        
        # Extract temperature data
        Tw = state[1:npts]
        
        # Reshape to 3D array
        T3D = reshape(Tw, (nx+1, ny+1, nz+1))
        
        # Extract slices
        xy_slice = T3D[:, :, mid_z_idx]
        yz_slice = T3D[mid_x_idx, :, :]
        xz_slice = T3D[:, mid_y_idx, :]
        
        # Save slices to CSV
        writedlm(joinpath(data_dir, "xy_slice_t$(i-1).csv"), xy_slice, ',')
        writedlm(joinpath(data_dir, "yz_slice_t$(i-1).csv"), yz_slice, ',')
        writedlm(joinpath(data_dir, "xz_slice_t$(i-1).csv"), xz_slice, ',')
        
        # Save sphere parameters at this time
        R_t = radius_mean + radius_amp * sin(2π * t / period)
        R_prime = radius_amp * (2π / period) * cos(2π * t / period)
        
        sphere_df = DataFrame(
            time = t,
            x_center = x_0,
            y_center = y_0,
            z_center = z_0,
            radius = R_t,
            radius_derivative = R_prime
        )
        
        # Append to sphere positions file
        if i == 1
            CSV.write(joinpath(data_dir, "sphere_parameters.csv"), sphere_df)
        else
            CSV.write(joinpath(data_dir, "sphere_parameters.csv"), sphere_df, append=true)
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

# Plot snapshots at different time steps
function create_sphere_animation(solver, mesh, num_frames=4)
    # Create directory for figures
    pub_dir = joinpath(pwd(), "publication_figures_3d")
    mkpath(pub_dir)
    
    # Select frames evenly spaced in time
    frame_indices = round.(Int, range(1, length(solver.states), length=num_frames))
    
    # Generate plots for each frame
    for (i, frame_idx) in enumerate(frame_indices)
        fig = visualize_3d_solution(solver, mesh, frame_idx)
        save(joinpath(pub_dir, "sphere_frame_$(i).png"), fig, px_per_unit=4)
        save(joinpath(pub_dir, "sphere_frame_$(i).pdf"), fig)
        println("Generated frame $i of $num_frames")
    end
    
    println("Animation frames saved to $(pub_dir)")
end

# After solving, run visualization and save data
fig = visualize_3d_solution(solver, mesh, length(solver.states))
display(fig)

save_simulation_data(solver, mesh, oscillating_body)
create_sphere_animation(solver, mesh, 4)  # Create 4 frames showing different times

# Define a benchmark function for convergence study
function run_convergence_study(resolutions=[16, 32, 48])
    results = []
    
    for res in resolutions
        println("Running test with resolution $res")
        
        # Define mesh with current resolution
        mesh = Penguin.Mesh((res, res, res), (lx, ly, lz), (x0, y0, z0))
        
        # Define space-time mesh
        STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
        
        # Rest of setup (same as above)
        capacity = Capacity(oscillating_body, STmesh)
        operator = DiffusionOps(capacity)
        
        # Initialize solution
        u0ₒ = zeros((res+1)*(res+1)*(res+1))
        u0ᵧ = zeros((res+1)*(res+1)*(res+1))
        
        for i in 1:res+1
            for j in 1:res+1
                for k in 1:res+1
                    idx = (k-1)*(res+1)*(res+1) + (j-1)*(res+1) + i
                    x = mesh.nodes[1][i]
                    y = mesh.nodes[2][j]
                    z = mesh.nodes[3][k]
                    u0ₒ[idx] = init_condition(x, y, z)
                end
            end
        end
        
        u0 = vcat(u0ₒ, u0ᵧ)
        
        # Create solver
        solver = MovingDiffusionUnsteadyMono(Fluide, bc_b, robin_bc, Δt, u0, mesh, "BE")
        
        # Solve
        solve_MovingDiffusionUnsteadyMono!(solver, Fluide, oscillating_body, Δt, Tstart, Tend, bc_b, robin_bc, mesh, "BE"; method=Base.:\)
        
        # Check errors
        body_tend = (x, y, z, _=0) -> begin
            R_t = radius_mean + radius_amp * sin(2π * Tend / period)
            return sqrt((x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2) - R_t
        end
        
        capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)
        u_ana, u_num, global_err, full_err, cut_err, _ = check_convergence(Φ_ana_tend, solver, capacity_tend, 2, false)
        
        push!(results, (res, global_err, full_err, cut_err))
    end
    
    # Print convergence results
    println("\nConvergence Results:")
    println("Resolution | Global Error | Full Cell Error | Cut Cell Error")
    println("---------------------------------------------------------")
    
    for (res, global_err, full_err, cut_err) in results
        println("$res x $res x $res | $(round(global_err, digits=8)) | $(round(full_err, digits=8)) | $(round(cut_err, digits=8))")
    end
    
    return results
end

# Uncomment to run convergence study
# convergence_results = run_convergence_study([16, 24, 32])