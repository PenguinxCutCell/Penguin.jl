using Penguin
using CairoMakie
using LinearAlgebra
using IterativeSolvers

"""
Unsteady 1D Stokes solution with a prescribed oscillating interface.
The interface oscillates with:
    x_interface(t) = x0 + A * sin(ω * t)

We start from rest, advance with Backward Euler (or Crank-Nicolson),
and visualize the velocity field around the moving boundary.
"""

###########
# Geometry
###########
nx = 64
Lx = 4.0
x0 = -2.0

# Interface parameters
x_interface0 = 0.0   # Initial interface position
A_osc = 0.5          # Oscillation amplitude
ω_osc = 2.0 * π      # Angular frequency (one period per unit time)
φ_osc = -π/2         # Phase shift so velocity is zero at t=0 (starts at lowest point)

# Body function: oscillating interface (x, t) -> level set
# Positive inside the body (left of interface), negative outside
body = (x, t) -> begin
    x_int = x_interface0 + A_osc * sin(ω_osc * t + φ_osc)
    return x_int - x
end

###########
# Meshes
###########
mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5*dx,))

###########
# Initial capacities & operators (t=0)
# Note: These are used only to initialize the Fluid object structure.
# The solve function recreates capacities at each time step using SpaceTimeMesh.
###########
capacity_u = Capacity((x,_=0) -> body(x,0.0), mesh_u)
capacity_p = Capacity((x,_=0) -> body(x,0.0), mesh_p)

operator_u = DiffusionOps(capacity_u)
operator_p = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
# Far-field: zero velocity (quiescent flow)
u_zero = Dirichlet(0.0)

bc_u = BorderConditions(Dict(
    :bottom => u_zero,  # left boundary
    :top    => u_zero   # right boundary
))

pressure_gauge = PinPressureGauge()

# Cut-cell boundary condition: velocity on the moving interface
# match the interface velocity
bc_cut = Dirichlet((x,t, _=0) -> A_osc * ω_osc * cos(ω_osc * t + φ_osc))

###########
# Physics
###########
μ = 1.0    # Dynamic viscosity
ρ = 1.0    # Density
fᵤ = (x, y=0.0, z=0.0) -> 0.0   # No body force
fₚ = (x, y=0.0, z=0.0) -> 0.0   # No pressure source

fluid = Fluid(mesh_u,
              capacity_u,
              operator_u,
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

###########
# Time integration setup
###########
Δt = 0.01
T_end = 1.0  # Short simulation for testing
scheme = :BE  # Backward Euler

# Initialize solver
nu = prod(operator_u.size)
np = prod(operator_p.size)
Ntot = 2 * nu + np
x0_vec = zeros(Ntot)

solver = MovingStokesUnsteadyMono(fluid, (bc_u,), pressure_gauge, bc_cut;
                                   scheme=scheme, x0=x0_vec)

println("Running prescribed motion Stokes (oscillating interface)")
println("  Δt=$(Δt), T_end=$(T_end), scheme=$(scheme)")
println("  Interface: oscillation A=$(A_osc), ω=$(ω_osc)")

times, states = solve_MovingStokesUnsteadyMono!(solver, body, mesh_p,
                                                 Δt, 0.0, T_end,
                                                 bc_u, bc_cut;
                                                 scheme=scheme,
                                                 method=Base.:\,
                                                 geometry_method="VOFI",
                                                 integration_method=:vofijul,
                                                 compute_centroids=true)

println("Completed $(length(times)-1) time steps")

###########
# Visualization
###########
function visualize_oscillating_interface(times, states, mesh_u, 
                                        body, nu, np;
                                        frames=[1, length(times)])
    xs = mesh_u.nodes[1]
    
    fig = Figure(size=(1200, 400))
    
    for (col, frame_idx) in enumerate(frames)
        t = times[frame_idx]
        state = states[frame_idx]
        
        uω = state[1:nu]
        
        ax = Axis(fig[1, col], 
                  xlabel="x", ylabel="u", 
                  title="t = $(round(t, digits=3))")
        
        lines!(ax, xs, uω, color=:blue, linewidth=2, label="velocity")
        
        # Mark interface position
        x_int = x_interface0 + A_osc * sin(ω_osc * t + φ_osc)
        vlines!(ax, [x_int], color=:red, linewidth=2, linestyle=:dash, label="interface")
        
        axislegend(ax, position=:lt)
    end
    
    save("oscillating_interface_stokes_1d.png", fig)
    println("Saved visualization to oscillating_interface_stokes_1d.png")
    return fig
end

# Visualize at start and end
fig = visualize_oscillating_interface(times, states, mesh_u,
                                     body, nu, np)
display(fig)

###########
# Simple diagnostics: maximum velocity over time
###########
max_velocities = Float64[]
for state in states
    uω = state[1:nu]
    push!(max_velocities, maximum(abs, uω))
end

println("Maximum velocity over time: $(max_velocities)")
println("Final max velocity: $(max_velocities[end])")

###########
# Animation of time evolution (GIF)
###########
println("Creating animation...")

function animate_oscillating_interface(times, states, mesh_u, nu;
                                       n_frames::Int=50, framerate::Int=8)
    xs = mesh_u.nodes[1]

    # Pick a subset of frames to keep file size manageable
    n_frames = min(n_frames, length(states))
    frame_indices = round.(Int, range(1, length(states), length=n_frames))

    # Helper to get velocity from state
    function state_to_velocity(state)
        return state[1:nu]
    end

    # Fix y-axis using min/max over all stored states
    extrema_pairs = map(states) do st
        u = state_to_velocity(st)
        return (minimum(u), maximum(u))
    end
    global_min = minimum(first, extrema_pairs)
    global_max = maximum(last, extrema_pairs)

    vel_obs = Observable(state_to_velocity(states[frame_indices[1]]))
    title_obs = Observable("Velocity at t = $(round(times[frame_indices[1]]; digits=3))")
    
    x_int_obs = Observable([x_interface0 + A_osc * sin(ω_osc * times[frame_indices[1]] + φ_osc)])

    fig_anim = Figure(resolution=(900, 520))
    ax_anim = Axis(fig_anim[1,1], xlabel="x", ylabel="u", title=title_obs)

    lines!(ax_anim, xs, vel_obs, color=:blue, linewidth=2)
    vlines!(ax_anim, x_int_obs, color=:red, linewidth=2, linestyle=:dash)
    ylims!(ax_anim, global_min - 0.1 * abs(global_max - global_min), 
                    global_max + 0.1 * abs(global_max - global_min))

    gif_path = "oscillating_interface_stokes_1d.gif"
    record(fig_anim, gif_path, 1:n_frames; framerate=framerate) do frame
        idx = frame_indices[frame]
        vel_obs[] = state_to_velocity(states[idx])
        t = times[idx]
        title_obs[] = "Velocity at t = $(round(t; digits=3))"
        
        x_int = x_interface0 + A_osc * sin(ω_osc * t + φ_osc)
        x_int_obs[] = [x_int]
    end

    println("Animation saved as $(gif_path)")
    return gif_path
end

# Create the animation using all stored states (subsampled inside)
animate_oscillating_interface(times, states, mesh_u, nu)
