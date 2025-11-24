using Penguin
using CairoMakie
using LinearAlgebra

# Domain and discretization
nx, ny = 64, 32
width, height = 1.0, 1.0
x0, y0 = 0.0, 0.0

println("="^60)
println("Rayleigh-Bénard Convection Setup")
println("="^60)
println("Domain: $width × $height")
println("Grid: $nx × $ny")

# Create meshes for pressure (cell centers) and velocities (staggered)
mesh_p = Penguin.Mesh((nx, ny), (width, height), (x0, y0))
dx = width / nx
dy = height / ny
mesh_ux = Penguin.Mesh((nx, ny), (width, height), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (width, height), (x0, y0 - 0.5 * dy))
mesh_T = mesh_p  # Temperature on same grid as pressure

# Body function (no immersed solid, return -1 everywhere)
body = (x, y, _=0.0) -> -1.0

# Build capacities
capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
capacity_T  = Capacity(body, mesh_T)

# Build operators
operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

println("\nBoundary Conditions:")
println("  Velocity: no-slip on all walls")
println("  Temperature: hot bottom, cold top, insulated sides")


# Velocity boundary conditions (no-slip on all walls)
zero_dirichlet = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>zero_dirichlet,
    :right=>zero_dirichlet,
    :bottom=>zero_dirichlet,
    :top=>zero_dirichlet
))
bc_uy = BorderConditions(Dict(
    :left=>zero_dirichlet,
    :right=>zero_dirichlet,
    :bottom=>zero_dirichlet,
    :top=>zero_dirichlet
))
pressure_gauge = MeanPressureGauge()
interface_bc = Dirichlet(0.0)

# Fluid properties
μ = 1.0  # Dynamic viscosity
ρ = 1.0     # Density
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0  # No external forcing
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

println("\nFluid Properties:")
println("  Viscosity μ = $μ")
println("  Density ρ = $ρ")
println("  Prandtl number Pr = $(μ / (1.0e-3)) (assuming κ = 1e-3)")

# Create fluid and Navier-Stokes solver
fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
x0_vec = zeros(2 * (nu_x + nu_y) + np)
ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)

# Temperature boundary conditions
T_hot = 1.0
T_cold = -1.0
bc_T = BorderConditions(Dict(
    :top=>Dirichlet(T_hot),
    :bottom=>Dirichlet(T_cold),
))
bc_T_cut = Dirichlet(0.0)

println("\nTemperature BCs:")
println("  Bottom (hot): T = $T_hot")
println("  Top (cold): T = $T_cold")
println("  Sides: insulated (∂T/∂n = 0)")

# Initialize temperature with linear stratification + small perturbation
nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_temp = Nx_T * Ny_T

T0ω = zeros(Float64, N_temp)
y_min = nodes_Ty[1]
y_max = nodes_Ty[end]
span_y = y_max - y_min

# Linear stratification with small random perturbation to trigger instability
for j in 1:Ny_T
    y = nodes_Ty[j]
    frac = span_y ≈ 0 ? 0.0 : (y - y_min) / span_y
    val = T_hot + (T_cold - T_hot) * frac
    for i in 1:Nx_T
        x = nodes_Tx[i]
        # Add small perturbation in the middle region
        pert = 0.0
        if 0.2 < frac < 0.8
            pert = 0.01 * sin(2π * x / width) * sin(π * frac)
        end
        idx = i + (j - 1) * Nx_T
        T0ω[idx] = val + pert
    end
end
T0γ = copy(T0ω)
T0 = vcat(T0ω, T0γ)

println("\nInitial temperature: linear + perturbation")
println("  T_ω ∈ [$(round(minimum(T0ω); digits=3)), $(round(maximum(T0ω); digits=3))]")

# Thermal properties
κ = 1.0e-2  # Thermal diffusivity
heat_source = (x, y, z=0.0, t=0.0) -> 0.0

println("\nThermal Properties:")
println("  Thermal diffusivity κ = $κ")
println("  Thermal expansion β = 1.0")
println("  Gravity g = (0, -1)")

# Create coupled solver
coupled = NavierStokesHeat2D(ns_solver, capacity_T, κ, heat_source,
                             bc_T, bc_T_cut;
                             β=1.0,
                             gravity=(0.0, -1.0),
                             T_ref=0.0,
                             T0=T0)

# Time stepping
Δt = 0.005
T_end = 1.0

println("\nTime Integration:")
println("  Time step Δt = $Δt")
println("  End time T_end = $T_end")
println("  Number of steps ≈ $(Int(ceil(T_end / Δt)))")
println("  Scheme: Crank-Nicolson (θ = 0.5)")
println("="^60)

println("\nRunning coupled Navier–Stokes / heat simulation...")
times, velocity_hist, temperature_hist = solve_NavierStokesHeat2D_unsteady!(coupled;
                                                                            Δt=Δt,
                                                                            T_end=T_end,
                                                                            scheme=:BE)
println("\nSimulation complete!")
println("Stored $(length(temperature_hist)) snapshots")
println("="^60)

# Extract final solution for visualization
println("\nPost-processing final state...")

uωx = coupled.momentum.x[1:nu_x]
uωy = coupled.momentum.x[2nu_x+1:2nu_x+nu_y]

xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]
Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))
speed = sqrt.(Ux.^2 .+ Uy.^2)

Tω_final = coupled.temperature[1:N_temp]
Temperature = reshape(Tω_final, (Nx_T, Ny_T))

println("  Velocity: |u| ∈ [$(round(minimum(speed); digits=6)), $(round(maximum(speed); digits=6))]")
println("  Temperature: T ∈ [$(round(minimum(Temperature); digits=4)), $(round(maximum(Temperature); digits=4))]")

# Helper function for streamplot
nearest_index(vec::AbstractVector{<:Real}, val::Real) = begin
    idx = searchsortedfirst(vec, val)
    if idx <= 1
        return 1
    elseif idx > length(vec)
        return length(vec)
    else
        prev_val = vec[idx - 1]
        curr_val = vec[idx]
        return abs(val - prev_val) <= abs(curr_val - val) ? idx - 1 : idx
    end
end

velocity_field(x, y) = Point2f(Ux[nearest_index(xs, x), nearest_index(ys, y)],
                               Uy[nearest_index(xs, x), nearest_index(ys, y)])

# Create visualization
println("\nCreating visualization...")
fig = Figure(resolution=(1200, 600))

ax_temp = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Temperature",
               aspect=DataAspect())
hm_temp = heatmap!(ax_temp, nodes_Tx, nodes_Ty, Temperature'; colormap=:thermal)
Colorbar(fig[1, 2], hm_temp, label="T")

ax_speed = Axis(fig[1, 3], xlabel="x", ylabel="y", title="Velocity magnitude",
                aspect=DataAspect())
hm_speed = heatmap!(ax_speed, xs, ys, speed; colormap=:viridis)
Colorbar(fig[1, 4], hm_speed, label="|u|")

ax_stream = Axis(fig[2, 1:4], xlabel="x", ylabel="y", title="Velocity streamlines",
                 aspect=DataAspect())
streamplot!(ax_stream, velocity_field, xs[1]..xs[end], ys[1]..ys[end]; 
            colormap=:plasma, arrow_size=10, density=1.0)

save("rayleigh_benard_snapshot.png", fig)
println("Saved: rayleigh_benard_snapshot.png")
display(fig)

# Create animation
println("\nCreating temperature animation...")
n_frames = min(80, length(temperature_hist))
frame_indices = collect(unique(round.(Int, range(1, length(temperature_hist), length=n_frames))))

fig_anim = Figure(resolution=(800, 600))
ax_anim = Axis(fig_anim[1, 1], xlabel="x", ylabel="y", 
               title="Temperature evolution", aspect=DataAspect())
temp_obs = Observable(Temperature')
hm_anim = heatmap!(ax_anim, nodes_Tx, nodes_Ty, temp_obs; 
                   colormap=:thermal, colorrange=(T_cold, T_hot))
Colorbar(fig_anim[1, 2], hm_anim, label="T")

record(fig_anim, "rayleigh_benard_temperature.gif", 1:length(frame_indices); framerate=10) do frame
    hist = temperature_hist[frame_indices[frame]]
    Tω_hist = hist[1:N_temp]
    temp_obs[] = reshape(Tω_hist, (Nx_T, Ny_T))'
    ax_anim.title = "Temperature at t = $(round(times[frame_indices[frame]], digits=3))"
end

println("Saved: rayleigh_benard_temperature.gif")
println("\n" * "="^60)
println("Analysis complete!")
println("="^60)
