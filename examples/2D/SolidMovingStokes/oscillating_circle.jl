using Penguin
using CairoMakie
using LinearAlgebra

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
A_osc = 0.3          # Oscillation amplitude
ω_osc = 2.0 * π      # Angular frequency (one period per unit time)

# Body function: oscillating circle (x, y, t) -> level set
# Positive inside the body, negative outside
body = (x, y, t) -> begin
    cy = center_y0 + A_osc * sin(ω_osc * t)
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
ux_zero = Dirichlet((x, y, t=0.0) -> 0.0)
uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)

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
# For a solid body with prescribed motion, the velocity at the interface
# should match the body velocity.
# Body velocity: v_body = d(y_center)/dt = A_osc * ω_osc * cos(ω_osc * t)
bc_cut = Dirichlet((x, y, t=0.0) -> A_osc * ω_osc * cos(ω_osc * t))

###########
# Physics
###########
μ = 0.1    # Dynamic viscosity
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
T_end = 1.0
scheme = :BE  # Backward Euler

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
                                                 geometry_method="VOFI",
                                                 compute_centroids=false)

println("Completed $(length(times)-1) time steps")

###########
# Visualization
###########
function visualize_oscillating_circle(times, states, mesh_p, mesh_ux, 
                                       body, nu_x, nu_y, np, Δt;
                                       frames=[1, div(length(times),2), length(times)])
    xs = mesh_ux.nodes[1]
    ys = mesh_ux.nodes[2]
    
    fig = Figure(size=(1200, 400))
    
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
        θ = range(0, 2π, length=100)
        cy = center_y0 + A_osc * sin(ω_osc * t)
        circle_x = center_x .+ radius .* cos.(θ)
        circle_y = cy .+ radius .* sin.(θ)
        lines!(ax, circle_x, circle_y, color=:white, linewidth=2)
        
        if col == length(frames)
            Colorbar(fig[1, col+1], hm, label="Velocity magnitude")
        end
    end
    
    save("oscillating_circle_stokes.png", fig)
    println("Saved visualization to oscillating_circle_stokes.png")
    return fig
end

# Visualize at start, middle, and end
fig = visualize_oscillating_circle(times, states, mesh_p, mesh_ux,
                                    body, nu_x, nu_y, np, Δt)
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

fig_vel = Figure(size=(600, 300))
ax_vel = Axis(fig_vel[1,1], xlabel="time", ylabel="max |u|", 
              title="Maximum velocity magnitude over time")
lines!(ax_vel, times, max_velocities)
save("oscillating_circle_max_velocity.png", fig_vel)
println("Saved max velocity plot to oscillating_circle_max_velocity.png")
display(fig_vel)
