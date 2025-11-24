using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations
using Statistics

### 2D Test Case : Two-phase Stefan Problem : Growing Interface
# Define the spatial mesh
nx, ny = 64, 64
lx, ly = 1.0, 1.0
x0, y0 = 0.0, 0.0
Δx, Δy = lx/(nx), ly/(ny)
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the initial interface shape
sₙ(y) = 0.5 * ly #+ 0.05 * ly * sin(2π*y)

# Define the body for each phase
body1 = (x,y,t,_=0) -> (x - sₙ(y))          # Phase 1 (left)
body2 = (x,y,t,_=0) -> -(x - sₙ(y))         # Phase 2 (right)

# Define the Space-Time mesh
Δt = 0.005
Tend = 0.08
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity for both phases
capacity1 = Capacity(body1, STmesh)
capacity2 = Capacity(body2, STmesh)

# Initial Height Vₙ₊₁ and Vₙ for phase 1
Vₙ₊₁_1 = capacity1.A[3][1:end÷2, 1:end÷2]
Vₙ_1 = capacity1.A[3][end÷2+1:end, end÷2+1:end]
Vₙ_1 = diag(Vₙ_1)
Vₙ₊₁_1 = diag(Vₙ₊₁_1)
Vₙ_1 = reshape(Vₙ_1, (nx+1, ny+1))
Vₙ₊₁_1 = reshape(Vₙ₊₁_1, (nx+1, ny+1))

# Get the column-wise height for the interface (initial position)
Hₙ⁰ = collect(vec(sum(Vₙ_1, dims=1)))
Hₙ₊₁⁰ = collect(vec(sum(Vₙ₊₁_1, dims=1)))

# Initial interface position
Interface_position = x0 .+ Hₙ⁰./Δy
println("Initial interface position: $(Interface_position)")

# Define the diffusion operators
operator1 = DiffusionOps(capacity1)
operator2 = DiffusionOps(capacity2)

# Define the boundary conditions
bc_hot = Dirichlet(1.0)    # bottom boundary (hot)
bc_cold = Dirichlet(0.0)   # top boundary (cold)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :bottom => bc_cold,
    :top => bc_hot
))

# Phase properties
ρ, L = 1.0, 1.0               # Density and latent heat
D1, D2 = 1.0, 1.0             # Diffusion coefficients
Tm = 0.3                      # Melting temperature

# Stefan condition
stef_cond = InterfaceConditions(
    ScalarJump(1.0, 1.0, Tm),  # Temperature jump condition (T₁ = T₂ = Tm at interface)
    FluxJump(1.0, 1.0, ρ*L)    # Flux jump condition (latent heat release)
)

# Define the source terms (no internal heat generation)
f1 = (x,y,z,t) -> 0.0
f2 = (x,y,z,t) -> 0.0

# Define diffusion coefficients
K1 = (x,y,z) -> D1
K2 = (x,y,z) -> D2

# Define the phases
Fluide1 = Phase(capacity1, operator1, f1, K1)
Fluide2 = Phase(capacity2, operator2, f2, K2)

# Initial condition
# Phase 1 (left) - hot side
u1ₒ = ones((nx+1)*(ny+1))    # Bulk initial temperature = 1.0
u1ᵧ = Tm*ones((nx+1)*(ny+1)) # Interface temperature = Tm (melting temperature)

# Phase 2 (right) - cold side
u2ₒ = Tm*ones((nx+1)*(ny+1))   # Bulk initial temperature = 0.0
u2ᵧ = Tm*ones((nx+1)*(ny+1)) # Interface temperature = Tm (melting temperature)

# Combine all initial values
u0 = vcat(u1ₒ, u1ᵧ, u2ₒ, u2ᵧ)

# Newton parameters
max_iter = 100
tol = 1e-5
reltol = 1e-5
α = 1.0  # Relaxation factor
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyDiph(Fluide1, Fluide2, bc_b, stef_cond, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log, reconstruct= solve_MovingLiquidDiffusionUnsteadyDiph2D!(solver, Fluide1, Fluide2, Interface_position, Hₙ⁰,sₙ, Δt, Tend, bc_b, stef_cond, mesh, "BE"; interpo="linear", Newton_params=Newton_params, method=Base.:\)

# Plot the position of the interface
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Interface position")
for i in 1:length(xf_log)
    lines!(ax, xf_log[i][1:end-1])
end
display(fig)

# Plot the residuals for one column
residuals = filter(x -> !isempty(x), residuals)

figure = Figure()
ax = Axis(figure[1,1], xlabel = "Newton Iterations", ylabel = "Residuals", title = "Residuals")
for i in 1:length(residuals)
    lines!(ax, log10.(residuals[i]), label = "Time = $(i*Δt)")
end
#axislegend(ax)
display(figure)

# Plot the position of one column
# Collect the interface position for column 5 from each time step in xf_log
column_vals = [xf[5] for xf in xf_log]

# save xf_log
open("xf_log_$nx.txt", "w") do io
    for i in 1:length(column_vals)
        println(io, column_vals[i])
    end
end


# Create a time axis (assuming each entry in xf_log corresponds to a time step; adjust if needed)
time_axis = Δt * collect(1:length(xf_log))

# Plot the time series
fig = Figure()
ax = Axis(fig[1,1], xlabel = "Time", ylabel = "Interface position", title = "Interface position (Column 5)")
lines!(ax, time_axis, column_vals, color=:blue)
display(fig)

# Animation
animate_solution(solver, mesh, body1)


function animate_stefan_diphasic(
    solver, 
    mesh,
    xf_log,
    Δt,
    nx, ny, lx, ly, x0, y0;
    filename="stefan_diphasic_animation.mp4",
    fps=10,
    title="Stefan Problem - Diphasic Heat Transfer",
    colorrange_bulk1=(0, 1),
    colorrange_interface1=(0, 1),
    colorrange_bulk2=(0, 1),
    colorrange_interface2=(0, 1),
    colormap1=:thermal,
    colormap2=:viridis,
    interpo="linear"
)
    # Create meshgrid for plotting
    xrange = range(x0, stop=x0+lx, length=nx+1)
    yrange = range(y0, stop=y0+ly, length=ny+1)
    
    # Create grid for visualization
    xs = range(x0, stop=x0+lx, length=5*nx)
    ys = range(y0, stop=y0+ly, length=5*ny)
    X = repeat(reshape(xs, 1, :), length(ys), 1)
    Y = repeat(ys, 1, length(xs))
    
    # Number of frames = number of saved states
    num_frames = length(solver.states)
    
    # Create a figure with 2x2 layout
    fig = Figure(resolution=(1200, 900))
    
    # Create titles for each subplot
    titles = [
        "Bulk Field - Phase 1", 
        "Interface Field - Phase 1",
        "Bulk Field - Phase 2", 
        "Interface Field - Phase 2"
    ]
    
    # Create axes for each subplot
    ax_bulk1 = Axis3(fig[1, 1], 
                    title=titles[1],
                    xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_interface1 = Axis3(fig[1, 2], 
                        title=titles[2], 
                        xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_bulk2 = Axis3(fig[2, 1], 
                    title=titles[3],
                    xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_interface2 = Axis3(fig[2, 2], 
                        title=titles[4],
                        xlabel="x", ylabel="y", zlabel="Temperature")
    
    # Add a main title
    Label(fig[0, :], title, fontsize=20)
    
    # Create colorbar for each phase
    Colorbar(fig[1, 3], colormap=colormap1, limits=colorrange_bulk1, label="Temperature (Phase 1)")
    Colorbar(fig[2, 3], colormap=colormap2, limits=colorrange_bulk2, label="Temperature (Phase 2)")
    
    # Set common view angles for 3D plots
    
    viewangle = (0.4pi, pi/8)
    for ax in [ax_bulk1, ax_interface1, ax_bulk2, ax_interface2]
        ax.azimuth = viewangle[1]
        ax.elevation = viewangle[2]
    end
    
    
    # Create time label and interface position indicator
    time_label = Label(fig[3, 1:2], "t = 0.00", fontsize=16)
    interface_label = Label(fig[3, 3], "Interface at x = 0.00", fontsize=16)
    
    # Create initial surface plots - will be updated in the animation
    bulk1_surface = surface!(ax_bulk1, xrange, yrange, zeros(ny+1, nx+1), 
                          colormap=colormap1, colorrange=colorrange_bulk1)
    
    interface1_surface = surface!(ax_interface1, xrange, yrange, zeros(ny+1, nx+1), 
                               colormap=colormap1, colorrange=colorrange_interface1)
    
    bulk2_surface = surface!(ax_bulk2, xrange, yrange, zeros(ny+1, nx+1), 
                          colormap=colormap2, colorrange=colorrange_bulk2)
    
    interface2_surface = surface!(ax_interface2, xrange, yrange, zeros(ny+1, nx+1), 
                               colormap=colormap2, colorrange=colorrange_interface2)
    
    # Create record of the animation
    println("Creating animation with $num_frames frames...")
    record(fig, filename, 1:num_frames; framerate=fps) do frame_idx
        # Extract the state at the current frame
        state = solver.states[frame_idx]
        
        # Get interface position for the current frame
        if frame_idx <= length(xf_log)
            current_xf = xf_log[frame_idx]
        else
            current_xf = xf_log[end]  # Use last position if we have more states than interface positions
        end
        
        # Determine interface position from height function
        centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
        if interpo == "linear"
            sₙ = linear_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
        elseif interpo == "quad"
            sₙ = extrapolate(scale(interpolate(current_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
        elseif interpo == "cubic"
            sₙ = cubic_spline_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
        else
            println("Interpolation method not supported")
            sₙ = y -> x0[1] + current_xf[1]  # Fallback to constant interface
        end
        
        # Create body function for domain splitting
        body1 = (x,y) -> (x - sₙ(y))
        body2 = (x,y) -> -(x - sₙ(y))
        
        # Extract solutions for each field
        u1_bulk = reshape(state[1:(nx+1)*(ny+1)], (ny+1, nx+1))
        u1_interface = reshape(state[(nx+1)*(ny+1)+1:2*(nx+1)*(ny+1)], (ny+1, nx+1))
        u2_bulk = reshape(state[2*(nx+1)*(ny+1)+1:3*(nx+1)*(ny+1)], (ny+1, nx+1))
        u2_interface = reshape(state[3*(nx+1)*(ny+1)+1:end], (ny+1, nx+1))
        
        # Compute phase indicators for masking
        phase1_indicator = zeros(ny+1, nx+1)
        phase2_indicator = zeros(ny+1, nx+1)
        
        # Compute mask for each phase
        for i in 1:nx+1
            for j in 1:ny+1
                phase1_indicator[j,i] = body1(xrange[i], yrange[j]) <= 0 ? 1.0 : NaN
                phase2_indicator[j,i] = body2(xrange[i], yrange[j]) <= 0 ? 1.0 : NaN
            end
        end
        
        # Apply masks to the solutions
        u1_bulk_masked = u1_bulk #.* phase1_indicator
        u1_interface_masked = u1_interface #.* phase1_indicator
        u2_bulk_masked = u2_bulk #.* phase2_indicator
        u2_interface_masked = u2_interface #.* phase2_indicator
        
        # Update surface plots with current data
        bulk1_surface[3] = u1_bulk_masked
        interface1_surface[3] = u1_interface_masked
        bulk2_surface[3] = u2_bulk_masked
        interface2_surface[3] = u2_interface_masked
        
        # Plot interface curve as a line on each surface plot
        interface_y = collect(ys)
        interface_x = sₙ.(interface_y)
        
        # Update time and interface labels
        time_t = round((frame_idx-1)*(Δt), digits=3)
        time_label.text = "t = $time_t"
        avg_pos = round(mean(current_xf), digits=3)
        interface_label.text = "Interface at x ≈ $avg_pos"
        
        # Progress indicator
        if frame_idx % 10 == 0
            println("Processing frame $frame_idx / $num_frames")
        end
    end
    
    println("Animation saved to $filename")
end

# Function to add interface contour to existing plots
function add_interface_contour!(
    ax, 
    xf, 
    mesh; 
    interpo="linear", 
    color=:white, 
    linewidth=2, 
    linestyle=:solid,
    z_height=nothing
)
    # Get y coordinates
    centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
    
    # Create interpolation function for interface position
    if interpo == "linear"
        sₙ = linear_interpolation(centroids, xf, extrapolation_bc=Interpolations.Periodic())
    elseif interpo == "quad"
        sₙ = extrapolate(scale(interpolate(xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
    elseif interpo == "cubic"
        sₙ = cubic_spline_interpolation(centroids, xf, extrapolation_bc=Interpolations.Periodic())
    else
        println("Interpolation method not supported")
        return
    end
    
    # Create array of interface points
    y_points = range(mesh.nodes[2][1], mesh.nodes[2][end], length=100)
    x_points = sₙ.(y_points)
    
    # Determine appropriate z-value
    if isnothing(z_height)
        # Default to 1.0 if no specific height provided
        z_values = ones(length(y_points)) * 1.0
    else
        z_values = ones(length(y_points)) * z_height
    end
    
    # Plot interface contour
    lines!(ax, x_points, y_points, z_values, 
           color=color, linewidth=linewidth, linestyle=linestyle)
end

# Function to visualize a single frame with interface
function plot_stefan_diphasic_frame(
    solver,
    mesh,
    xf_log,
    frame_idx,
    nx, ny, lx, ly, x0, y0;
    interpo="linear",
    colorrange_bulk1=(0, 1),
    colorrange_interface1=(0, 1),
    colorrange_bulk2=(0, 1),
    colorrange_interface2=(0, 1),
    colormap1=:thermal,
    colormap2=:viridis
)

    # Create meshgrid for plotting
    xrange = range(x0, stop=x0+lx, length=nx+1)
    yrange = range(y0, stop=y0+ly, length=ny+1)

    # Get current state
    state = solver.states[frame_idx]
    
    # Get interface position
    if frame_idx <= length(xf_log)
        current_xf = xf_log[frame_idx]
    else
        current_xf = xf_log[end]
    end
    
    # Create interpolation function for interface
    centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
    if interpo == "linear"
        sₙ = linear_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
    elseif interpo == "quad"
        sₙ = extrapolate(scale(interpolate(current_xf, BSpline(Quadratic())), centroids), Interpolations.Periodic())
    elseif interpo == "cubic"
        sₙ = cubic_spline_interpolation(centroids, current_xf, extrapolation_bc=Interpolations.Periodic())
    else
        println("Interpolation method not supported")
        sₙ = y -> x0[1] + current_xf[1]
    end
    
    # Create body functions
    body1 = (x,y) -> (x - sₙ(y))
    body2 = (x,y) -> -(x - sₙ(y))
    
    # Extract solutions for each field
    u1_bulk = reshape(state[1:(nx+1)*(ny+1)], (ny+1, nx+1))
    u1_interface = reshape(state[(nx+1)*(ny+1)+1:2*(nx+1)*(ny+1)], (ny+1, nx+1))
    u2_bulk = reshape(state[2*(nx+1)*(ny+1)+1:3*(nx+1)*(ny+1)], (ny+1, nx+1))
    u2_interface = reshape(state[3*(nx+1)*(ny+1)+1:end], (ny+1, nx+1))
    
    # Compute phase masks
    phase1_indicator = zeros(ny+1, nx+1)
    phase2_indicator = zeros(ny+1, nx+1)
    
    for i in 1:nx+1
        for j in 1:ny+1
            phase1_indicator[j,i] = body1(xrange[i], yrange[j]) <= 0 ? 1.0 : NaN
            phase2_indicator[j,i] = body2(xrange[i], yrange[j]) <= 0 ? 1.0 : NaN
        end
    end
    
    # Apply masks
    u1_bulk_masked = u1_bulk .* phase1_indicator
    u1_interface_masked = u1_interface .* phase1_indicator
    u2_bulk_masked = u2_bulk .* phase2_indicator
    u2_interface_masked = u2_interface .* phase2_indicator
    
    # Create figure with 2x2 layout
    fig = Figure(resolution=(1200, 900))
    
    # Create titles for each subplot
    titles = [
        "Bulk Field - Phase 1", 
        "Interface Field - Phase 1",
        "Bulk Field - Phase 2", 
        "Interface Field - Phase 2"
    ]
    
    # Create axes for each subplot
    ax_bulk1 = Axis3(fig[1, 1], 
                    title=titles[1],
                    xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_interface1 = Axis3(fig[1, 2], 
                        title=titles[2], 
                        xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_bulk2 = Axis3(fig[2, 1], 
                    title=titles[3],
                    xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_interface2 = Axis3(fig[2, 2], 
                        title=titles[4],
                        xlabel="x", ylabel="y", zlabel="Temperature")
    
    # Add a main title with frame info
    Label(fig[0, :], "Stefan Problem - Frame $frame_idx", fontsize=20)
    
    # Create surface plots
    surface!(ax_bulk1, xrange, yrange, u1_bulk_masked, 
             colormap=colormap1, colorrange=colorrange_bulk1)
    
    surface!(ax_interface1, xrange, yrange, u1_interface_masked, 
             colormap=colormap1, colorrange=colorrange_interface1)
    
    surface!(ax_bulk2, xrange, yrange, u2_bulk_masked, 
             colormap=colormap2, colorrange=colorrange_bulk2)
    
    surface!(ax_interface2, xrange, yrange, u2_interface_masked, 
             colormap=colormap2, colorrange=colorrange_interface2)
    
    # Add interface contours with specific z-height
    add_interface_contour!(ax_bulk1, current_xf, mesh, interpo=interpo, z_height=1.0)
    add_interface_contour!(ax_interface1, current_xf, mesh, interpo=interpo, z_height=1.0)
    add_interface_contour!(ax_bulk2, current_xf, mesh, interpo=interpo, z_height=1.0)
    add_interface_contour!(ax_interface2, current_xf, mesh, interpo=interpo, z_height=1.0)
    
    # Add colorbars
    Colorbar(fig[1, 3], colormap=colormap1, limits=colorrange_bulk1, label="Temperature (Phase 1)")
    Colorbar(fig[2, 3], colormap=colormap2, limits=colorrange_bulk2, label="Temperature (Phase 2)")
    
    # Set common view angles
    viewangle = (0.4pi, pi/8)
    for ax in [ax_bulk1, ax_interface1, ax_bulk2, ax_interface2]
        ax.azimuth = viewangle[1]
        ax.elevation = viewangle[2]
    end

    return fig
end

# Example use at the end of the stefan_2d_2ph.jl file:
# Create animation
animate_stefan_diphasic(
    solver, 
    mesh, 
    xf_log, 
    Δt,
    nx, ny, lx, ly, x0, y0;
    filename="stefan_diphasic_animation.mp4",
    fps=15,
    title="Stefan Problem - Diphasic Heat Transfer",
    colorrange_bulk1=(0, 1.0),
    colorrange_interface1=(0.4, 0.6),
    colorrange_bulk2=(0, 1.0),
    colorrange_interface2=(0.4, 0.6),
    colormap1=:viridis,
    colormap2=:viridis,
    interpo="linear"
)

# Plot last frame for detailed visualization
last_frame_fig = plot_stefan_diphasic_frame(
    solver,
    mesh,
    xf_log,
    length(solver.states),
    nx, ny, lx, ly, x0, y0,
    interpo="linear",
    colorrange_bulk1=(0, 1.0),
    colorrange_interface1=(0.4, 0.6),
    colorrange_bulk2=(0, 1.0),
    colorrange_interface2=(0.4, 0.6),
    colormap1=:viridis,
    colormap2=:viridis
)
display(last_frame_fig)