using Penguin
using CairoMakie
using LinearAlgebra
using IterativeSolvers

"""
Unsteady 3D Stokes solution with a prescribed oscillating sphere.
The sphere oscillates vertically with:
    z_center(t) = z0 + A * sin(ω * t)

We start from rest, advance with Backward Euler (or Crank-Nicolson),
and visualize the velocity field around the moving body.

NOTE: This is a 3D example and may be computationally expensive.
      Use small grid sizes for testing.
"""

###########
# Geometry
###########
nx, ny, nz = 16, 16, 16  # Small grid for demonstration
Lx, Ly, Lz = 4.0, 4.0, 4.0
x0, y0, z0 = -2.0, -2.0, -2.0

# Sphere parameters
radius = 0.5
center_x = 0.0
center_y = 0.0
center_z0 = 0.0      # Initial center position
A_osc = 0.5          # Oscillation amplitude
ω_osc = 2.0 * π      # Angular frequency (one period per unit time)
φ_osc = -π/2         # Phase shift so velocity is zero at t=0 (starts at lowest point)

# Body function: oscillating sphere (x, y, z, t) -> level set
# Positive inside the body, negative outside
body = (x, y, z, t) -> begin
    cz = center_z0 + A_osc * sin(ω_osc * t + φ_osc)
    return radius - sqrt((x - center_x)^2 + (y - center_y)^2 + (z - cz)^2)
end

###########
# Meshes
###########
mesh_p = Penguin.Mesh((nx, ny, nz), (Lx, Ly, Lz), (x0, y0, z0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
dz = mesh_p.nodes[3][2] - mesh_p.nodes[3][1]
mesh_ux = Penguin.Mesh((nx, ny, nz), (Lx, Ly, Lz), (x0 - 0.5*dx, y0, z0))
mesh_uy = Penguin.Mesh((nx, ny, nz), (Lx, Ly, Lz), (x0, y0 - 0.5*dy, z0))
mesh_uz = Penguin.Mesh((nx, ny, nz), (Lx, Ly, Lz), (x0, y0, z0 - 0.5*dz))

###########
# Initial capacities & operators (t=0)
# Note: These are used only to initialize the Fluid object structure.
# The solve function recreates capacities at each time step using SpaceTimeMesh.
###########
capacity_ux = Capacity((x,y,z,_=0) -> body(x,y,z,0.0), mesh_ux)
capacity_uy = Capacity((x,y,z,_=0) -> body(x,y,z,0.0), mesh_uy)
capacity_uz = Capacity((x,y,z,_=0) -> body(x,y,z,0.0), mesh_uz)
capacity_p  = Capacity((x,y,z,_=0) -> body(x,y,z,0.0), mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_uz = DiffusionOps(capacity_uz)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
# Far-field: zero velocity (quiescent flow)
u_zero = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(
    :left   => u_zero,
    :right  => u_zero,
    :bottom => u_zero,
    :top    => u_zero,
    :front  => u_zero,
    :back   => u_zero
))
bc_uy = BorderConditions(Dict(
    :left   => u_zero,
    :right  => u_zero,
    :bottom => u_zero,
    :top    => u_zero,
    :front  => u_zero,
    :back   => u_zero
))
bc_uz = BorderConditions(Dict(
    :left   => u_zero,
    :right  => u_zero,
    :bottom => u_zero,
    :top    => u_zero,
    :front  => u_zero,
    :back   => u_zero
))

pressure_gauge = PinPressureGauge()

# Cut-cell boundary condition: velocity on the moving body surface
# match the body velocity
bc_cut = (
    Dirichlet((x,y,z,t) -> 0.0),  # u_x = 0 on body surface
    Dirichlet((x,y,z,t) -> 0.0),  # u_y = 0 on body surface
    Dirichlet((x,y,z,t) -> A_osc * ω_osc * cos(ω_osc * t + φ_osc))  # u_z = dz_center/dt
)

###########
# Physics
###########
μ = 1.0    # Dynamic viscosity
ρ = 1.0    # Density
fᵤ = (x, y, z) -> 0.0   # No body force
fₚ = (x, y, z) -> 0.0   # No pressure source

fluid = Fluid((mesh_ux, mesh_uy, mesh_uz),
              (capacity_ux, capacity_uy, capacity_uz),
              (operator_ux, operator_uy, operator_uz),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

###########
# Time integration setup
###########
Δt = 0.05
T_end = 0.2  # Very short simulation for testing (3D is expensive)
scheme = :BE  # Backward Euler

# Initialize solver
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
nu_z = prod(operator_uz.size)
np = prod(operator_p.size)
Ntot = 2 * (nu_x + nu_y + nu_z) + np
x0_vec = zeros(Ntot)

solver = MovingStokesUnsteadyMono(fluid, (bc_ux, bc_uy, bc_uz), pressure_gauge, bc_cut;
                                   scheme=scheme, x0=x0_vec)

println("Running prescribed motion Stokes (oscillating sphere - 3D)")
println("  Δt=$(Δt), T_end=$(T_end), scheme=$(scheme)")
println("  Sphere: radius=$(radius), oscillation A=$(A_osc), ω=$(ω_osc)")
println("  Grid size: $(nx)×$(ny)×$(nz)")
println("  WARNING: This is a 3D simulation and may be computationally expensive!")

times, states = solve_MovingStokesUnsteadyMono!(solver, body, mesh_p,
                                                 Δt, 0.0, T_end,
                                                 (bc_ux, bc_uy, bc_uz), bc_cut;
                                                 scheme=scheme,
                                                 method=Base.:\,
                                                 geometry_method="VOFI",
                                                 integration_method=:vofijul,
                                                 compute_centroids=true)

println("Completed $(length(times)-1) time steps")

###########
# Visualization
###########
function visualize_oscillating_sphere_3d(times, states, mesh_ux, mesh_uy, mesh_uz,
                                         body, nu_x, nu_y, nu_z, np;
                                         frames=[1, length(times)])
    xs = mesh_ux.nodes[1]
    ys = mesh_ux.nodes[2]
    zs = mesh_ux.nodes[3]
    
    # Take mid-plane slice
    mid_z = Int(ceil(length(zs) / 2))
    
    fig = Figure(size=(1200, 500))
    
    for (col, frame_idx) in enumerate(frames)
        t = times[frame_idx]
        state = states[frame_idx]
        
        uωx = state[1:nu_x]
        uωy = state[2*nu_x+1:2*nu_x+nu_y]
        uωz = state[2*nu_x+2*nu_y+1:2*nu_x+2*nu_y+nu_z]
        
        Ux = reshape(uωx, (length(xs), length(ys), length(zs)))
        Uy = reshape(uωy, (length(xs), length(ys), length(zs)))
        Uz = reshape(uωz, (length(xs), length(ys), length(zs)))
        
        # Take slice at mid z
        Ux_slice = Ux[:, :, mid_z]
        Uy_slice = Uy[:, :, mid_z]
        Uz_slice = Uz[:, :, mid_z]
        
        speed_slice = sqrt.(Ux_slice.^2 .+ Uy_slice.^2 .+ Uz_slice.^2)
        
        ax = Axis(fig[1, col], 
                  xlabel="x", ylabel="y", 
                  title="t = $(round(t, digits=3)) (z-slice)",
                  aspect=DataAspect())
        
        hm = heatmap!(ax, xs, ys, speed_slice; colormap=:viridis)
        
        # Draw sphere cross-section at this z-slice
        cz = center_z0 + A_osc * sin(ω_osc * t + φ_osc)
        z_slice = zs[mid_z]
        r_at_slice = sqrt(max(0, radius^2 - (z_slice - cz)^2))
        
        if r_at_slice > 0
            θ_circle = range(0, 2π, length=100)
            circle_x = center_x .+ r_at_slice .* cos.(θ_circle)
            circle_y = center_y .+ r_at_slice .* sin.(θ_circle)
            lines!(ax, circle_x, circle_y, color=:white, linewidth=2)
        end
        
        if col == length(frames)
            Colorbar(fig[1, col+1], hm, label="Velocity magnitude")
        end
    end
    
    save("oscillating_sphere_stokes_3d.png", fig)
    println("Saved visualization to oscillating_sphere_stokes_3d.png")
    return fig
end

# Visualize at start and end
fig = visualize_oscillating_sphere_3d(times, states, mesh_ux, mesh_uy, mesh_uz,
                                      body, nu_x, nu_y, nu_z, np)
display(fig)

###########
# Simple diagnostics: maximum velocity over time
###########
max_velocities = Float64[]
for state in states
    uωx = state[1:nu_x]
    uωy = state[2*nu_x+1:2*nu_x+nu_y]
    uωz = state[2*nu_x+2*nu_y+1:2*nu_x+2*nu_y+nu_z]
    push!(max_velocities, max(maximum(abs, uωx), maximum(abs, uωy), maximum(abs, uωz)))
end

println("Maximum velocity over time: $(max_velocities)")
println("Final max velocity: $(max_velocities[end])")
