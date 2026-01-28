using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CairoMakie

### 1D Test Case : Diphasic Unsteady Diffusion Equation inside a moving body
# Define the spatial mesh
nx = 40
lx = 40.0
x0 = 0.0
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
xf = 0.1*lx   # Interface position
body = (x,t, _=0)->(x - xf)
body_c = (x,t, _=0)->-(x - xf)

# Define the Space-Time mesh
Δt = 0.00001
Tend = 0.1
Tinit = 0.0
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)
capacity_c = Capacity(body_c, STmesh)

# Define the diffusion operator
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
ρ, L = 1.0, 1.0
ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.5), FluxJump(1.0, 1.0, ρ*L))

# Define the source term
f1 = (x,y,z,t)->0.0
f2 = (x,y,z,t)->0.0

K1= (x,y,z)->1.0
K2= (x,y,z)->0.05

# Define the phase
Fluide1 = Phase(capacity, operator, f1, K1)
Fluide2 = Phase(capacity_c, operator_c, f2, K2)

# Initial condition
u0ₒ1 = ones(nx+1)
u0ᵧ1 = ones(nx+1)*0.5
u0ₒ2 = zeros(nx+1)
u0ᵧ2 = ones(nx+1)*0.5

u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Newton parameters
max_iter = 100
tol = eps()
reltol = eps()
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyDiph(Fluide1, Fluide2, bc_b, ic, Δt, u0, mesh, "CN")

# Solve the problem
solver, residuals, xf_log = solve_MovingLiquidDiffusionUnsteadyDiph!(solver, Fluide1, Fluide2, xf, Δt, Tend, Tinit, bc_b, ic, mesh, "CN"; Newton_params=Newton_params, method=Base.:\)

# Animation
#animate_solution(solver, mesh, body)

# Plot residuals   
#residuals[i] might be empty, remove them
residuals = filter(x -> !isempty(x), residuals)

figure = Figure()
ax = Axis(figure[1,1], xlabel = "Newton Iterations", ylabel = "Residuals", title = "Residuals")
for i in 1:length(residuals)
    lines!(ax, log10.(residuals[i]), label = "Time = $(i*Δt)")
end
#axislegend(ax)
display(figure)

# Plot the position
time_series = 0:Δt:Tend
figure = Figure()
ax = Axis(figure[1,1], xlabel = "Time", ylabel = "Interface position", title = "Interface position")
lines!(ax, time_series, xf_log, label = "Interface position")
display(figure)

# save xf_log
open("xf_log_$nx.txt", "w") do io
    for i in 1:length(xf_log)
        println(io, xf_log[i])
    end
end

# Plot
state_i = 2
u1ₒ = solver.states[state_i][1:nx+1]
u1ᵧ = solver.states[state_i][nx+2:2*(nx+1)]
u2ₒ = solver.states[state_i][2*(nx+1)+1:3*(nx+1)]
u2ᵧ = solver.states[state_i][3*(nx+1)+1:end]

x = range(x0, stop = lx, length = nx+1)
using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="u", title="Diphasic Unsteady Diffusion Equation")

for (idx, i) in enumerate(1:10:length(solver.states))
    u1ₒ = solver.states[i][1:nx+1]
    u1ᵧ = solver.states[i][nx+2:2*(nx+1)]
    u2ₒ = solver.states[i][2*(nx+1)+1:3*(nx+1)]
    u2ᵧ = solver.states[i][3*(nx+1)+1:end]

    # Display labels only for first iteration
    scatter!(ax, x, u1ₒ, color=:blue,  markersize=3,
        label = (idx == 1 ? "Bulk Field - Phase 1" : nothing))
    scatter!(ax, x, u1ᵧ, color=:red,   markersize=3,
        label = (idx == 1 ? "Interface Field - Phase 1" : nothing))
    scatter!(ax, x, u2ₒ, color=:green, markersize=3,
        label = (idx == 1 ? "Bulk Field - Phase 2" : nothing))
    scatter!(ax, x, u2ᵧ, color=:orange, markersize=3,
        label = (idx == 1 ? "Interface Field - Phase 2" : nothing))
end

axislegend(ax, position=:rt)
display(fig)

using Penguin
using CairoMakie
using Printf
using DelimitedFiles

function animate_temperature_profile(
    solver, 
    xf_log, 
    mesh;
    filename="temperature_evolution.mp4",
    fps=30
)
    # Get domain parameters from the mesh

    
    # Create spatial grid
    x_grid = range(x0, stop=x0+lx, length=nx+1)
    
    # Time parameters
    total_frames = length(solver.states)
    time_points = range(0, step=Δt, length=total_frames)
    
    # Choose which frames to render
    frame_indices = 1:total_frames
    
    # Create a figure with wider layout and better aspect ratio
    fig = Figure(resolution=(1000, 700), fontsize=18)
    
    # Use the full width of the figure for both panels
    # Top panel: Temperature profile (main view)
    temp_ax = Axis(fig[1, 1:3], 
                  xlabel = "Position (x)", 
                  ylabel = "Temperature",
                  title = "Temperature Profile",
                  titlesize = 24,
                  xlabelsize = 20,
                  ylabelsize = 20,
                  xticklabelsize = 16,
                  yticklabelsize = 16)
    
    # Bottom panel: Interface position over time
    interface_ax = Axis(fig[2, 1:3], 
                      xlabel = "Time", 
                      ylabel = "Interface Position",
                      title = "Interface Position",
                      titlesize = 24,
                      xlabelsize = 20,
                      ylabelsize = 20,
                      xticklabelsize = 16,
                      yticklabelsize = 16)
    
    # Adjust horizontal gap between panels
    rowgap!(fig.layout, 15)  # Increase gap between rows
    
    # Plot complete interface position history
    lines!(interface_ax, time_points, xf_log[1:total_frames], 
           color = :blue, linewidth = 3)
    
    # Set up time indicator with larger font
    time_label = Label(fig[0, 1:3], "Time: 0.000", fontsize=24, tellwidth=false)
    
    # Record animation
    record(fig, filename, frame_indices; framerate=fps) do frame_idx
        # Update time indicator
        current_time = (frame_idx-1) * Δt
        time_label.text = "Time: $(round(current_time, digits=3))"
        
        # Get current state
        state = solver.states[frame_idx]
        
        # Extract temperature fields
        u1ₒ = state[1:nx+1]                  # Bulk Field - Phase 1
        u2ₒ = state[2*(nx+1)+1:3*(nx+1)]      # Bulk Field - Phase 2
        
        # Get current interface position
        xf = xf_log[frame_idx]
        
        # Clear the temperature axis for the new frame
        empty!(temp_ax)
        
        # Create a continuous temperature profile that avoids the glitch
        temp = zeros(nx+1)
        
        for i in 1:nx+1
            if x_grid[i] < xf - 1e-10
                # Safely in phase 1
                temp[i] = u1ₒ[i]
            elseif x_grid[i] > xf + 1e-10
                # Safely in phase 2
                temp[i] = u2ₒ[i]
            else
                # At or very close to interface - use interface temperature (0.5)
                temp[i] = 0.5
            end
        end
        
        # Plot the continuous temperature profile with thicker line
        lines!(temp_ax, x_grid, temp, color=:red, linewidth=4)
        
        # Mark the interface on temperature plot
        vlines!(temp_ax, xf, color=:black, linewidth=2.5)
        scatter!(temp_ax, [xf], [0.5], color=:black, markersize=12)
        
        # Add annotation for interface position
        text!(temp_ax, "Interface: x = $(round(xf, digits=4))",
             position = (0.7*lx, 0.9),
             fontsize = 18,
             color = :black)
        
        # Mark current position on interface plot
        scatter!(interface_ax, [current_time], [xf], 
                color=:red, marker=:circle, markersize=10)
        
        # Set y-axis limits
        limits!(temp_ax, x0, x0+lx, 0, 1.1)
        upper_limit = ceil(maximum(xf_log) * 10) / 10  # Round up to nearest 0.1
        limits!(interface_ax, 0, time_points[end], 0, upper_limit+0.05)
    end
    
    println("Animation saved to $filename")
    return fig
end

# Version with just temperature profile (wide format)
function animate_temperature_wide(
    solver, 
    xf_log, 
    mesh;
    filename="temperature_wide.mp4",
    fps=30
)
    # Get domain parameters from the mesh

    # Create spatial grid
    x_grid = range(x0, stop=x0+lx, length=nx+1)
    
    # Time parameters
    total_frames = length(solver.states)
    
    # Create a wide figure for the temperature profile
    fig = Figure(resolution=(1200, 500), fontsize=18)
    
    # Create a single wide axis
    ax = Axis(fig[1, 1], 
             xlabel = "Position (x)", 
             ylabel = "Temperature",
             title = "Temperature Profile Evolution",
             titlesize = 24,
             xlabelsize = 20,
             ylabelsize = 20)
    
    # Time label
    time_label = Label(fig[0, 1], "Time: 0.000", fontsize=24)
    
    # Record animation
    record(fig, filename, 1:total_frames; framerate=fps) do frame_idx
        # Update time indicator
        current_time = (frame_idx-1) * Δt
        time_label.text = "Time: $(round(current_time, digits=3))"
        
        # Get current state
        state = solver.states[frame_idx]
        
        # Extract temperature fields
        u1ₒ = state[1:nx+1]                  # Bulk Field - Phase 1
        u2ₒ = state[2*(nx+1)+1:3*(nx+1)]      # Bulk Field - Phase 2
        
        # Get current interface position
        xf = xf_log[frame_idx]
        
        # Clear the axis
        empty!(ax)
        
        # Create a continuous temperature profile
        temp = zeros(nx+1)
        
        for i in 1:nx+1
            if x_grid[i] < xf - 1e-10
                temp[i] = u1ₒ[i]
            elseif x_grid[i] > xf + 1e-10
                temp[i] = u2ₒ[i]
            else
                temp[i] = 0.5
            end
        end
        
        # Plot temperature profile
        lines!(ax, x_grid, temp, color=:red, linewidth=4)
        
        # Mark interface
        vlines!(ax, xf, color=:black, linewidth=2.5)
        scatter!(ax, [xf], [0.5], color=:black, markersize=12)
        
        # Add interface position text
        text!(ax, "Interface: x = $(round(xf, digits=4))",
             position = (0.7*lx, 0.9),
             fontsize = 18,
             color = :black)
        
        # Set axis limits
        limits!(ax, x0, x0+lx, 0, 1.1)
    end
    
    println("Animation saved to $filename")
    return fig
end

# Call one of these functions
animate_temperature_profile(solver, xf_log, mesh; filename="temperature_evolution_wide.mp4", fps=30)
# Or use this version for just the temperature profile
animate_temperature_wide(solver, xf_log, mesh; filename="temperature_profile_wide.mp4", fps=30)