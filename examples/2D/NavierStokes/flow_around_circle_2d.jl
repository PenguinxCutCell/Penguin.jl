using Penguin
using CairoMakie
using LinearAlgebra
using FFTW
using Statistics
"""
Unsteady Navier–Stokes flow past a circular obstacle using the staggered
velocity / cell-centered pressure discretization. Convection is handled with an
explicit Adams–Bashforth extrapolation while viscosity/pressure use an implicit
θ-scheme (Crank–Nicolson by default).
"""

###########
# Geometry
###########
nx, ny = 128, 64
channel_length = 4.0
channel_height = 2.0
x0, y0 = -1.0, -1.0

circle_center = (0.0, 0.0)
circle_radius = 0.5
diameter = 2 * circle_radius
circle_body = (x, y, _=0.0) -> circle_radius - sqrt((x - circle_center[1])^2 + (y - circle_center[2])^2)

###########
# Meshes
###########
mesh_p  = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0 - 0.5 * dy))

###########
# Capacities & operators
###########
capacity_ux = Capacity(circle_body, mesh_ux)
capacity_uy = Capacity(circle_body, mesh_uy)
capacity_p  = Capacity(circle_body, mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
Umax = 1.0
parabolic = (x, y, t=0.0) -> begin
    ξ = (y - (y0 + channel_height / 2)) / (channel_height / 2)
    Umax * (1 - ξ^2)
end

ux_left   = Dirichlet((x, y, t=0.0) -> parabolic(x, y, t))
ux_right  = Dirichlet((x, y, t=0.0) -> parabolic(x, y, t))
ux_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
ux_top    = Dirichlet((x, y, t=0.0) -> 0.0)

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>Outflow(), :bottom=>ux_bottom, :top=>ux_top
))
bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero
))
pressure_gauge = PinPressureGauge()

interface_bc = Dirichlet(0.0)

###########
# Physics
###########
μ = 0.01 
ρ = 1.0
println("Re=", ρ * Umax * (2 * circle_radius) / μ)
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

###########
# Solver setup
###########
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)

x0_vec = zeros(2 * (nu_x + nu_y) + np)

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)

Δt = 0.005
T_end = 0.5

println("Running Navier–Stokes unsteady simulation ")
times, histories =solve_NavierStokesMono_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=:CN)


println("Simulation finished. Stored states = ", length(histories))
println("Final time step = ", times[end])

###########
# Force diagnostics
###########
force_diag = compute_navierstokes_force_diagnostics(solver)
body_force = navierstokes_reaction_force_components(force_diag; acting_on=:body)
pressure_body = -force_diag.integrated_pressure
viscous_body = -force_diag.integrated_viscous
coeffs = drag_lift_coefficients(force_diag; ρ=ρ, U_ref=Umax, length_ref=diameter, acting_on=:body)
println("Integrated forces on the cylinder (body reaction):")
println("  Fx = $(round(body_force[1]; sigdigits=6)) (pressure=$(round(pressure_body[1]; sigdigits=6)), viscous=$(round(viscous_body[1]; sigdigits=6)))")
println("  Fy = $(round(body_force[2]; sigdigits=6)) (pressure=$(round(pressure_body[2]; sigdigits=6)), viscous=$(round(viscous_body[2]; sigdigits=6)))")
println("  Drag coefficient Cd = $(round(coeffs.Cd; sigdigits=6)), lift coefficient Cl = $(round(coeffs.Cl; sigdigits=6))")

pressure_trace = pressure_trace_on_cut(solver; center=circle_center)


###########
# Visualization of final velocity and pressure
###########
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
Xp = mesh_p.nodes[1];  Yp = mesh_p.nodes[2]

uωx = solver.x[1:nu_x]
uωy = solver.x[2nu_x+1:2nu_x+nu_y]
pω  = solver.x[2*(nu_x + nu_y)+1:end]

Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))
P  = reshape(pω, (length(Xp), length(Yp)))

speed = sqrt.(Ux.^2 .+ Uy.^2)

nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))
velocity_field(x, y) = Point2f(Ux[nearest_index(xs, x), nearest_index(ys, y)],
                               Uy[nearest_index(xs, x), nearest_index(ys, y)])

fig = Figure(resolution=(1200, 700))

ax_speed = Axis(fig[1,1], xlabel="x", ylabel="y", title="Speed magnitude")
hm_speed = heatmap!(ax_speed, xs, ys, speed; colormap=:plasma)
contour!(ax_speed, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig[1,2], hm_speed)

ax_pressure = Axis(fig[1,3], xlabel="x", ylabel="y", title="Pressure field")
hm_pressure = heatmap!(ax_pressure, Xp, Yp, P; colormap=:balance)
contour!(ax_pressure, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig[1,4], hm_pressure)

ax_stream = Axis(fig[2,1], xlabel="x", ylabel="y", title="Velocity streamlines")
streamplot!(ax_stream, velocity_field, xs[1]..xs[end], ys[1]..ys[end]; colormap=:thermal)
contour!(ax_stream, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:red, linewidth=2)

ax_contours = Axis(fig[2,3], xlabel="x", ylabel="y", title="Velocity magnitude contours")
contour!(ax_contours, xs, ys, speed; levels=10, color=:navy)
contour!(ax_contours, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:black, linewidth=2)

save("navierstokes2d_circle_flow_.png", fig)
display(fig)

if !isempty(pressure_trace.θ)
    fig_pressure = Figure(resolution=(700, 400))
    ax_pressure_trace = Axis(fig_pressure[1,1], xlabel="θ [rad]", ylabel="p", title="Pressure around the cylinder")
    lines!(ax_pressure_trace, pressure_trace.θ, pressure_trace.p; color=:royalblue)
    scatter!(ax_pressure_trace, pressure_trace.θ, pressure_trace.p; color=:orange, markersize=6)
    display(fig_pressure)
    save("navierstokes2d_circle_pressure_trace_.png", fig_pressure)

    order = sortperm(pressure_trace.θ)
    θ_sorted = pressure_trace.θ[order]
    p_surface = pressure_trace.p_from_stress[order]
    θ_deg = rad2deg.(θ_sorted)
    p_ref = mean(@view P[:, 1])
    dynamic_pressure = 0.5 * ρ * Umax^2
    cp_surface = (p_surface .- p_ref) ./ dynamic_pressure
    pressure_coefficient = hcat(θ_deg, cp_surface)
    println("Stored $(size(pressure_coefficient, 1)) pressure coefficient samples around the cylinder.")

    fig_cp = Figure(resolution=(700, 400))
    ax_cp = Axis(fig_cp[1,1], xlabel="θ [deg]", ylabel="Cp", title="Pressure coefficient around the cylinder")
    lines!(ax_cp, θ_deg, cp_surface; color=:forestgreen)
    scatter!(ax_cp, θ_deg, cp_surface; color=:firebrick, markersize=6)
    display(fig_cp)
    save("navierstokes2d_circle_pressure_coefficient_.png", fig_cp)

    # save pressure coefficient data to CSV
    using CSV, DataFrames
    df_cp = DataFrame(θ_deg=θ_deg, Cp=cp_surface)
    CSV.write("navierstokes2d_circle_pressure_coefficient_.csv", df_cp)
    
end

###########
# Animation of time evolution
###########
println("Creating animation...")

# Select subset of time steps for animation (to keep file size reasonable)
n_frames = min(50, length(histories))
frame_indices = round.(Int, range(1, length(histories), length=n_frames))

fig_anim = Figure(resolution=(800, 600))
ax_anim = Axis(fig_anim[1,1], xlabel="x", ylabel="y", title="Velocity magnitude evolution")

# Observable for updating the plot
speed_obs = Observable(zeros(length(xs), length(ys)))

# Create the heatmap and contours
hm_anim = heatmap!(ax_anim, xs, ys, speed_obs; colormap=:plasma, colorrange=(0, 1.5))
contour!(ax_anim, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig_anim[1,2], hm_anim)

# Function to update frame
function update_frame!(frame_idx)
    # Extract velocity from history
    hist = histories[frame_indices[frame_idx]]
    ux_hist = hist[1:nu_x]
    uy_hist = hist[2nu_x+1:2nu_x+nu_y]
    
    # Reshape and compute speed
    Ux_hist = reshape(ux_hist, (length(xs), length(ys)))
    Uy_hist = reshape(uy_hist, (length(xs), length(ys)))
    speed_hist = sqrt.(Ux_hist.^2 .+ Uy_hist.^2)
    
    # Update observable
    speed_obs[] = speed_hist
    
    # Update title with time
    ax_anim.title = "Velocity magnitude at t = $(round(times[frame_indices[frame_idx]], digits=3))"
end

# Create animation
record(fig_anim, "navierstokes2d_vonkarman_.gif", 1:n_frames; framerate=8) do frame
    update_frame!(frame)
end

println("Animation saved as navierstokes2d_vonkarman.gif")

# Also create a streamline animation
println("Creating streamline animation...")

fig_stream = Figure(resolution=(800, 600))
ax_stream_anim = Axis(fig_stream[1,1], xlabel="x", ylabel="y", title="Velocity streamlines evolution")

# Function to create velocity field function for streamlines
function make_velocity_field(ux_data, uy_data)
    Ux_data = reshape(ux_data, (length(xs), length(ys)))
    Uy_data = reshape(uy_data, (length(xs), length(ys)))
    
    # Create interpolation functions for better coverage
    return function(x, y)
        # Clamp coordinates to domain bounds
        x_clamped = clamp(x, xs[1], xs[end])
        y_clamped = clamp(y, ys[1], ys[end])
        
        # Find nearest indices
        i = nearest_index(xs, x_clamped)
        j = nearest_index(ys, y_clamped)
        
        # Return velocity components
        return Point2f(Ux_data[i, j], Uy_data[i, j])
    end
end

using GeometryBasics

record(fig_stream, "navierstokes2d_streamlines_.gif", 1:n_frames; framerate=8) do frame
    empty!(ax_stream_anim)
    
    # Extract velocity from history
    hist = histories[frame_indices[frame]]
    ux_hist = hist[1:nu_x]
    uy_hist = hist[2nu_x+1:2nu_x+nu_y]
    
    # Create velocity field function
    vel_field = make_velocity_field(ux_hist, uy_hist)
    
    # Plot streamlines with better coverage
    x_min, x_max = xs[1], xs[end]
    y_min, y_max = ys[1], ys[end]
    stepsize = 0.01
    maxsteps = 1000
    density = 2.0
    color_func = (p) -> norm(p)  # Color by speed magnitude
    velocity_func = (x, y) -> vel_field(x, y)
    rect = Rect(x_min, y_min, x_max - x_min, y_max - y_min)
    streamplot!(ax_stream_anim, velocity_func, rect, stepsize=stepsize, maxsteps=maxsteps, density=density, color=color_func)
    contour!(ax_stream_anim, xs, ys, [circle_body(x,y) for x in xs, y in ys]; 
             levels=[0.0], color=:red, linewidth=2)
    
    # Set axis limits explicitly
    xlims!(ax_stream_anim, xs[1], xs[end])
    ylims!(ax_stream_anim, ys[1], ys[end])
    
    ax_stream_anim.title = "Streamlines at t = $(round(times[frame_indices[frame]], digits=3))"
end

println("Streamline animation saved as navierstokes2d_streamlines.gif")
