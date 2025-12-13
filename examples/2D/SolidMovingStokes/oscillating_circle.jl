using Penguin
using CairoMakie
using LinearAlgebra
#using Observables
using IterativeSolvers

"""
Unsteady Stokes solution with a prescribed oscillating circle.
The circle oscillates vertically with:
    y_center(t) = y0 + A * sin(ω * t)

We start from rest, advance with Backward Euler (or Crank-Nicolson),
and visualize the velocity field around the moving body.
"""

###########
# Geometry
###########
nx, ny = 32, 32
Lx, Ly = 4.0, 4.0
x0, y0 = -2.0, -2.0

# Circle parameters
radius = 0.5
center_x = 0.0
center_y0 = 0.0     # Initial center position
A_osc = 0.5          # Oscillation amplitude
ω_osc = 2.0 * π      # Angular frequency (one period per unit time)
φ_osc = -π/2        # Phase shift so velocity is zero at t=0 (starts at lowest point)

# Body function: oscillating circle (x, y, t) -> level set
# Positive inside the body, negative outside
body = (x, y, t) -> begin
    cy = center_y0 + A_osc * sin(ω_osc * t + φ_osc)
    return radius - sqrt((x - center_x)^2 + (y - cy)^2)
end

###########
# Meshes
###########
mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

###########
# Initial capacities & operators (t=0)
# Note: These are used only to initialize the Fluid object structure.
# The solve function recreates capacities at each time step using SpaceTimeMesh.
###########
capacity_ux = Capacity((x,y,_=0) -> body(x,y,0.0), mesh_ux)
capacity_uy = Capacity((x,y,_=0) -> body(x,y,0.0), mesh_uy)
capacity_p  = Capacity((x,y,_=0) -> body(x,y,0.0), mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
# Far-field: zero velocity (quiescent flow)
ux_zero = Dirichlet(0.0)
uy_zero = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(
    :left   => ux_zero,
    :right  => ux_zero,
    :bottom => ux_zero,
    :top    => ux_zero
))
bc_uy = BorderConditions(Dict(
    :left   => uy_zero,
    :right  => uy_zero,
    :bottom => uy_zero,
    :top    => uy_zero
))

pressure_gauge = PinPressureGauge()

# Cut-cell boundary condition: velocity on the moving body surface
# match the body velocity
bc_cut = (
    Dirichlet((x,y,t) -> 0.0),  # u_x = 0 on body surface
    Dirichlet((x,y,t) -> A_osc * ω_osc * cos(ω_osc * t + φ_osc))  # u_y = dy_center/dt
)

###########
# Physics
###########
μ = 1.0    # Dynamic viscosity
ρ = 1.0    # Density
fᵤ = (x, y, z=0.0) -> 0.0   # No body force
fₚ = (x, y, z=0.0) -> 0.0   # No pressure source

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

###########
# Time integration setup
###########
Δt = 0.02
T_end = 0.1  # Short simulation for testing
scheme = :BE  # Backward Euler
geometry_method = "VOFI"
capacity_kwargs = (; method=geometry_method,
                    integration_method=:vofijul,
                    compute_centroids=true)

# Initialize solver
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
Ntot = 2 * (nu_x + nu_y) + np
x0_vec = zeros(Ntot)

solver = MovingStokesUnsteadyMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut;
                                   scheme=scheme, x0=x0_vec)

println("Running prescribed motion Stokes (oscillating circle)")
println("  Δt=$(Δt), T_end=$(T_end), scheme=$(scheme)")
println("  Circle: radius=$(radius), oscillation A=$(A_osc), ω=$(ω_osc)")

times, states = solve_MovingStokesUnsteadyMono!(solver, body, mesh_p,
                                                 Δt, 0.0, T_end,
                                                 (bc_ux, bc_uy), bc_cut;
                                                 scheme=scheme,
                                                 method=Base.:\,
                                                 #method=IterativeSolvers.gmres,
                                                 geometry_method=geometry_method,
                                                 integration_method=capacity_kwargs.integration_method,
                                                 compute_centroids=capacity_kwargs.compute_centroids)

println("Completed $(length(times)-1) time steps")

###########
# Force diagnostics (drag/lift at final time)
###########
if length(times) >= 2
    t_prev, t_last = times[end-1], times[end]
    STmesh_ux = Penguin.SpaceTimeMesh(mesh_ux, [t_prev, t_last], tag=mesh_p.tag)
    STmesh_uy = Penguin.SpaceTimeMesh(mesh_uy, [t_prev, t_last], tag=mesh_p.tag)
    STmesh_p = Penguin.SpaceTimeMesh(mesh_p, [t_prev, t_last], tag=mesh_p.tag)

    capacity_ux_last = Capacity(body, STmesh_ux; capacity_kwargs...)
    capacity_uy_last = Capacity(body, STmesh_uy; capacity_kwargs...)
    capacity_p_last = Capacity(body, STmesh_p; capacity_kwargs...)

    operator_ux_last = DiffusionOps(capacity_ux_last)
    operator_uy_last = DiffusionOps(capacity_uy_last)
    operator_p_last = DiffusionOps(capacity_p_last)

    block_data = Penguin.stokes2D_moving_blocks(solver.fluid,
                                                (operator_ux_last, operator_uy_last),
                                                (capacity_ux_last, capacity_uy_last),
                                                operator_p_last, capacity_p_last,
                                                scheme)

    force_diag = compute_navierstokes_force_diagnostics(solver, block_data)
    body_force = navierstokes_reaction_force_components(force_diag; acting_on=:body)
    pressure_body = .-force_diag.integrated_pressure
    viscous_body = .-force_diag.integrated_viscous
    U_ref = A_osc * ω_osc
    coeffs = drag_lift_coefficients(force_diag; ρ=ρ, U_ref=U_ref, length_ref=2 * radius, acting_on=:body)

    println("Forces on the circle at t=$(round(t_last; digits=3)):")
    println("  Drag  = $(round(body_force[1]; sigdigits=6)) (pressure=$(round(pressure_body[1]; sigdigits=6)), viscous=$(round(viscous_body[1]; sigdigits=6)))")
    println("  Lift  = $(round(body_force[2]; sigdigits=6)) (pressure=$(round(pressure_body[2]; sigdigits=6)), viscous=$(round(viscous_body[2]; sigdigits=6)))")
    println("  Cd = $(round(coeffs.Cd; sigdigits=6)), Cl = $(round(coeffs.Cl; sigdigits=6)) (U_ref=$(U_ref))")
else
    println("Not enough time samples to compute drag/lift diagnostics.")
end

###########
# Visualization
###########
function visualize_oscillating_circle(times, states, mesh_ux, 
                                       body, nu_x, nu_y, np;
                                       frames=[1, length(times)])
    xs = mesh_ux.nodes[1]
    ys = mesh_ux.nodes[2]
    
    fig = Figure(size=(800, 400))
    
    for (col, frame_idx) in enumerate(frames)
        t = times[frame_idx]
        state = states[frame_idx]
        
        uωx = state[1:nu_x]
        uωy = state[2*nu_x+1:2*nu_x+nu_y]
        
        Ux = reshape(uωx, (length(xs), length(ys)))
        Uy = reshape(uωy, (length(xs), length(ys)))
        
        speed = sqrt.(Ux.^2 .+ Uy.^2)
        
        ax = Axis(fig[1, col], 
                  xlabel="x", ylabel="y", 
                  title="t = $(round(t, digits=3))",
                  aspect=DataAspect())
        
        hm = heatmap!(ax, xs, ys, speed; colormap=:viridis)
        
        # Draw interface
        θ_circle = range(0, 2π, length=100)
        cy = center_y0 + A_osc * sin(ω_osc * t + φ_osc)
        circle_x = center_x .+ radius .* cos.(θ_circle)
        circle_y = cy .+ radius .* sin.(θ_circle)
        lines!(ax, circle_x, circle_y, color=:white, linewidth=2)
        
        if col == length(frames)
            Colorbar(fig[1, col+1], hm, label="Velocity magnitude")
        end
    end
    
    save("oscillating_circle_stokes.png", fig)
    println("Saved visualization to oscillating_circle_stokes.png")
    return fig
end

# Visualize at start and end
fig = visualize_oscillating_circle(times, states, mesh_ux,
                                    body, nu_x, nu_y, np)
display(fig)

###########
# Simple diagnostics: maximum velocity over time
###########
max_velocities = Float64[]
for state in states
    uωx = state[1:nu_x]
    uωy = state[2*nu_x+1:2*nu_x+nu_y]
    push!(max_velocities, max(maximum(abs, uωx), maximum(abs, uωy)))
end

println("Maximum velocity over time: $(max_velocities)")
println("Final max velocity: $(max_velocities[end])")

###########
# Animation of time evolution (GIF)
###########
println("Creating animation...")

function animate_oscillating_circle(times, states, mesh_ux, nu_x, nu_y;
                                   n_frames::Int=50, framerate::Int=8)
    xs = mesh_ux.nodes[1]
    ys = mesh_ux.nodes[2]

    # Pick a subset of frames to keep file size manageable
    n_frames = min(n_frames, length(states))
    frame_indices = round.(Int, range(1, length(states), length=n_frames))

    # Helper to reshape state -> speed field
    function state_to_speed(state)
        uωx = state[1:nu_x]
        uωy = state[2*nu_x+1:2*nu_x+nu_y]
        Ux = reshape(uωx, (length(xs), length(ys)))
        Uy = reshape(uωy, (length(xs), length(ys)))
        return sqrt.(Ux.^2 .+ Uy.^2)
    end

    # Fix colorbar using min/max over all stored states
    extrema_pairs = map(states) do st
        s = state_to_speed(st)
        return (minimum(s), maximum(s))
    end
    global_min = minimum(first, extrema_pairs)
    global_max = maximum(last, extrema_pairs)

    speed_obs = Observable(state_to_speed(states[frame_indices[1]]))
    title_obs = Observable("Velocity magnitude at t = $(round(times[frame_indices[1]]; digits=3))")

    θ_circle = range(0, 2π, length=120)
    cy0 = center_y0 + A_osc * sin(ω_osc * times[frame_indices[1]] + φ_osc)
    circle_x_obs = Observable(center_x .+ radius .* cos.(θ_circle))
    circle_y_obs = Observable(cy0 .+ radius .* sin.(θ_circle))

    fig_anim = Figure(resolution=(900, 520))
    ax_anim = Axis(fig_anim[1,1], xlabel="x", ylabel="y", title=title_obs, aspect=DataAspect())

    hm_anim = heatmap!(ax_anim, xs, ys, speed_obs; colormap=:plasma, colorrange=(global_min, global_max))
    lines!(ax_anim, circle_x_obs, circle_y_obs, color=:white, linewidth=2)
    Colorbar(fig_anim[1,2], hm_anim, label="Velocity magnitude")

    gif_path = "oscillating_circle_stokes.gif"
    record(fig_anim, gif_path, 1:n_frames; framerate=framerate) do frame
        idx = frame_indices[frame]
        speed_obs[] = state_to_speed(states[idx])
        t = times[idx]
        title_obs[] = "Velocity magnitude at t = $(round(t; digits=3))"

        cy = center_y0 + A_osc * sin(ω_osc * t + φ_osc)
        circle_x_obs[] = center_x .+ radius .* cos.(θ_circle)
        circle_y_obs[] = cy .+ radius .* sin.(θ_circle)
    end

    println("Animation saved as $(gif_path)")
    return gif_path
end

# Create the animation using all stored states (subsampled inside)
animate_oscillating_circle(times, states, mesh_ux, nu_x, nu_y)
