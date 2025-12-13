using Penguin
using CairoMakie
using LinearAlgebra
using Statistics

"""
Coupled moving Stokes and heat transfer problem.
A hot circle oscillates vertically in a quiescent fluid, 
releasing heat into the surrounding fluid.

The circle has a prescribed temperature on its surface and oscillates with:
    y_center(t) = y0 + A * sin(ω * t)

The temperature field is advected by the Stokes flow and diffuses through the fluid.
"""

###########
# Geometry and discretization
###########
nx, ny = 24, 24
Lx, Ly = 4.0, 4.0
x0, y0 = -2.0, -2.0

# Circle parameters
radius = 0.4
center_x = 0.0
center_y0 = 0.0     # Initial center position
A_osc = 0.4         # Oscillation amplitude
ω_osc = 2.0 * π     # Angular frequency (one period per unit time)
φ_osc = -π/2        # Phase shift so velocity is zero at t=0

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
mesh_T = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))  # Temperature on same mesh as pressure

dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

###########
# Initial capacities (t=0) - for structure initialization only
###########
capacity_ux = Capacity((x,y,_=0) -> body(x,y,0.0), mesh_ux)
capacity_uy = Capacity((x,y,_=0) -> body(x,y,0.0), mesh_uy)
capacity_p  = Capacity((x,y,_=0) -> body(x,y,0.0), mesh_p)
capacity_T  = Capacity((x,y,_=0) -> body(x,y,0.0), mesh_T)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Velocity boundary conditions
###########
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

# Cut-cell velocity BC: match body velocity
bc_cut_velocity = (
    Dirichlet((x,y,t) -> 0.0),  # u_x = 0 on body
    Dirichlet((x,y,t) -> A_osc * ω_osc * cos(ω_osc * t + φ_osc))  # u_y = dy_center/dt
)

###########
# Temperature boundary conditions
###########
T_ambient = 0.0      # Ambient temperature
T_body = 1.0         # Hot body temperature

# Border conditions: ambient temperature at far field
bc_T = BorderConditions(Dict(
    :left   => Dirichlet(T_ambient),
    :right  => Dirichlet(T_ambient),
    :bottom => Dirichlet(T_ambient),
    :top    => Dirichlet(T_ambient)
))

# Cut-cell temperature BC: hot body surface
bc_cut_T = Dirichlet((x, y, t) -> T_body)

###########
# Physical parameters
###########
# Fluid properties
μ = 1.0    # Dynamic viscosity
ρ = 1.0    # Density
fᵤ = (x, y, z=0.0) -> 0.0   # No body force (buoyancy handled separately if needed)
fₚ = (x, y, z=0.0) -> 0.0   # No pressure source

# Thermal properties
κ = 0.1    # Thermal diffusivity
f_T = (x, y, z=0.0) -> 0.0  # No internal heat source (heat comes from body)

# Buoyancy parameters (optional - set β=0 for passive coupling)
β = 0.0          # Thermal expansion coefficient (set to 0 for no buoyancy)
gravity = (0.0, -1.0)  # Gravity vector
T_ref = T_ambient      # Reference temperature

###########
# Create fluid object
###########
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
Δt = 0.025
T_end = 0.2  # Short simulation for testing
scheme = :BE  # Backward Euler
geometry_method = "VOFI"

# Initialize Stokes solver
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
Ntot_stokes = 2 * (nu_x + nu_y) + np
x0_stokes = zeros(Ntot_stokes)

stokes_solver = MovingStokesUnsteadyMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut_velocity;
                                         scheme=scheme, x0=x0_stokes)

# Initialize scalar field (starts at ambient temperature)
N_T_nodes = prod(size(mesh_T))
T0 = vcat(fill(T_ambient, N_T_nodes), zeros(N_T_nodes))  # (Tω, Tγ)

# Create coupled solver
coupled_solver = MovingStokesScalarCoupler(stokes_solver, capacity_T, κ, f_T, bc_T, bc_cut_T;
                                          β=β, gravity=gravity, T_ref=T_ref, T0=T0)

println("Running coupled moving Stokes-heat problem (oscillating thermal circle)")
println("  Δt=$(Δt), T_end=$(T_end), scheme=$(scheme)")
println("  Circle: radius=$(radius), oscillation A=$(A_osc), ω=$(ω_osc)")
println("  Thermal: T_body=$(T_body), T_ambient=$(T_ambient), κ=$(κ)")
println("  Buoyancy: β=$(β)")

# Solve the coupled problem
times, velocity_states, scalar_states = solve_MovingStokesScalarCoupler!(
    coupled_solver, body, mesh_p,
    Δt, 0.0, T_end,
    (bc_ux, bc_uy), bc_cut_velocity;
    scheme=scheme,
    method=Base.:\,
    geometry_method=geometry_method,
    coupling=:passive,
    integration_method=:vofijul,
    compute_centroids=true
)

println("Completed $(length(times)-1) time steps")

###########
# Visualization
###########
function visualize_coupled_solution(times, velocity_states, scalar_states,
                                   mesh_ux, mesh_T, body, nu_x, nu_y;
                                   frames=[1, length(times)])
    xs_u = mesh_ux.nodes[1]
    ys_u = mesh_ux.nodes[2]
    xs_T = mesh_T.nodes[1]
    ys_T = mesh_T.nodes[2]
    
    fig = Figure(size=(1200, 800))
    
    for (col, frame_idx) in enumerate(frames)
        t = times[frame_idx]
        vel_state = velocity_states[frame_idx]
        scalar_state = scalar_states[frame_idx]
        
        # Extract velocity
        uωx = vel_state[1:nu_x]
        uωy = vel_state[2*nu_x+1:2*nu_x+nu_y]
        
        Ux = reshape(uωx, (length(xs_u), length(ys_u)))
        Uy = reshape(uωy, (length(xs_u), length(ys_u)))
        speed = sqrt.(Ux.^2 .+ Uy.^2)
        
        # Extract temperature
        N_T_nodes = length(xs_T) * length(ys_T)
        Tω = scalar_state[1:N_T_nodes]
        T_field = reshape(Tω, (length(xs_T), length(ys_T)))
        
        # Velocity plot
        ax1 = Axis(fig[1, col], 
                  xlabel="x", ylabel="y", 
                  title="Velocity (t = $(round(t, digits=3)))",
                  aspect=DataAspect())
        
        hm1 = heatmap!(ax1, xs_u, ys_u, speed; colormap=:viridis)
        
        # Draw circle
        θ_circle = range(0, 2π, length=100)
        cy = center_y0 + A_osc * sin(ω_osc * t + φ_osc)
        circle_x = center_x .+ radius .* cos.(θ_circle)
        circle_y = cy .+ radius .* sin.(θ_circle)
        lines!(ax1, circle_x, circle_y, color=:white, linewidth=2)
        
        # Temperature plot
        ax2 = Axis(fig[2, col], 
                  xlabel="x", ylabel="y", 
                  title="Temperature (t = $(round(t, digits=3)))",
                  aspect=DataAspect())
        
        hm2 = heatmap!(ax2, xs_T, ys_T, T_field; colormap=:thermal)
        lines!(ax2, circle_x, circle_y, color=:white, linewidth=2)
        
        if col == length(frames)
            Colorbar(fig[1, col+1], hm1, label="Velocity magnitude")
            Colorbar(fig[2, col+1], hm2, label="Temperature")
        end
    end
    
    save("oscillating_thermal_circle.png", fig)
    println("Saved visualization to oscillating_thermal_circle.png")
    return fig
end

# Visualize at start and end
fig = visualize_coupled_solution(times, velocity_states, scalar_states,
                                 mesh_ux, mesh_T, body, nu_x, nu_y)
display(fig)

###########
# Diagnostics
###########
println("\nDiagnostics:")
for (i, t) in enumerate(times)
    vel_state = velocity_states[i]
    scalar_state = scalar_states[i]
    
    max_vel = maximum(abs.(vel_state[1:2*(nu_x+nu_y)]))
    N_T_nodes = length(scalar_state) ÷ 2
    T_bulk = scalar_state[1:N_T_nodes]
    max_T = maximum(T_bulk)
    min_T = minimum(T_bulk)
    mean_T = mean(T_bulk)
    
    println("t=$(round(t, digits=4)): max_vel=$(round(max_vel, digits=5)), " *
            "T∈[$(round(min_T, digits=4)), $(round(max_T, digits=4))], " *
            "mean_T=$(round(mean_T, digits=4))")
end

###########
# Animation
###########
println("\nCreating animation...")

function animate_coupled_solution(times, velocity_states, scalar_states,
                                 mesh_ux, mesh_T, nu_x, nu_y;
                                 n_frames::Int=30, framerate::Int=8)
    xs_u = mesh_ux.nodes[1]
    ys_u = mesh_ux.nodes[2]
    xs_T = mesh_T.nodes[1]
    ys_T = mesh_T.nodes[2]
    
    # Pick subset of frames
    n_frames = min(n_frames, length(times))
    frame_indices = round.(Int, range(1, length(times), length=n_frames))
    
    # Helper functions
    function state_to_speed(vel_state)
        uωx = vel_state[1:nu_x]
        uωy = vel_state[2*nu_x+1:2*nu_x+nu_y]
        Ux = reshape(uωx, (length(xs_u), length(ys_u)))
        Uy = reshape(uωy, (length(xs_u), length(ys_u)))
        return sqrt.(Ux.^2 .+ Uy.^2)
    end
    
    function state_to_temp(scalar_state)
        N_T_nodes = length(xs_T) * length(ys_T)
        Tω = scalar_state[1:N_T_nodes]
        return reshape(Tω, (length(xs_T), length(ys_T)))
    end
    
    # Fixed color ranges
    speeds = [state_to_speed(velocity_states[i]) for i in frame_indices]
    temps = [state_to_temp(scalar_states[i]) for i in frame_indices]
    
    speed_min = minimum(minimum, speeds)
    speed_max = maximum(maximum, speeds)
    temp_min = minimum(minimum, temps)
    temp_max = maximum(maximum, temps)
    
    # Observables
    speed_obs = Observable(speeds[1])
    temp_obs = Observable(temps[1])
    title_obs = Observable("t = $(round(times[frame_indices[1]]; digits=3))")
    
    θ_circle = range(0, 2π, length=120)
    cy0 = center_y0 + A_osc * sin(ω_osc * times[frame_indices[1]] + φ_osc)
    circle_x_obs = Observable(center_x .+ radius .* cos.(θ_circle))
    circle_y_obs = Observable(cy0 .+ radius .* sin.(θ_circle))
    
    fig_anim = Figure(resolution=(1000, 900))
    
    # Velocity panel
    ax1 = Axis(fig_anim[1,1], xlabel="x", ylabel="y", 
               title=@lift($title_obs * " - Velocity"),
               aspect=DataAspect())
    hm1 = heatmap!(ax1, xs_u, ys_u, speed_obs; 
                  colormap=:viridis, colorrange=(speed_min, speed_max))
    lines!(ax1, circle_x_obs, circle_y_obs, color=:white, linewidth=2)
    Colorbar(fig_anim[1,2], hm1, label="Velocity magnitude")
    
    # Temperature panel
    ax2 = Axis(fig_anim[2,1], xlabel="x", ylabel="y", 
               title=@lift($title_obs * " - Temperature"),
               aspect=DataAspect())
    hm2 = heatmap!(ax2, xs_T, ys_T, temp_obs; 
                  colormap=:thermal, colorrange=(temp_min, temp_max))
    lines!(ax2, circle_x_obs, circle_y_obs, color=:white, linewidth=2)
    Colorbar(fig_anim[2,2], hm2, label="Temperature")
    
    gif_path = "oscillating_thermal_circle.gif"
    record(fig_anim, gif_path, 1:n_frames; framerate=framerate) do frame
        idx = frame_indices[frame]
        speed_obs[] = speeds[frame]
        temp_obs[] = temps[frame]
        
        t = times[idx]
        title_obs[] = "t = $(round(t; digits=3))"
        
        cy = center_y0 + A_osc * sin(ω_osc * t + φ_osc)
        circle_x_obs[] = center_x .+ radius .* cos.(θ_circle)
        circle_y_obs[] = cy .+ radius .* sin.(θ_circle)
    end
    
    println("Animation saved as $(gif_path)")
    return gif_path
end

animate_coupled_solution(times, velocity_states, scalar_states,
                        mesh_ux, mesh_T, nu_x, nu_y)

println("\n✓ Example completed successfully!")
