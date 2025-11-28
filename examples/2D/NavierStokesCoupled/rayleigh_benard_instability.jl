using Penguin
using LinearAlgebra
using Statistics
try
   using CairoMakie
catch
   @warn "CairoMakie not available; visualization disabled."
end

# ---------------------------------------------------------------------------
# Rayleigh-Bénard Instability (2D)
# ---------------------------------------------------------------------------
# This example simulates the classic Rayleigh-Bénard convection problem where
# a horizontal fluid layer is heated from below and cooled from above. When the
# temperature difference exceeds a critical value (determined by the Rayleigh
# number), the fluid becomes unstable and develops convection cells.
#
# Key physics:
# - Fluid heated from bottom (T_hot), cooled from top (T_cold)
# - Buoyancy-driven flow with Boussinesq approximation
# - Critical Rayleigh number Ra_c ≈ 1708 for rigid boundaries
# - Above Ra_c, convection cells (rolls) develop
#
# Dimensionless parameters:
# - Ra = (g * β * ΔT * H³) / (ν * α) : Rayleigh number
# - Pr = ν / α : Prandtl number
#
# Where:
# - g: gravitational acceleration
# - β: thermal expansion coefficient
# - ΔT: temperature difference (T_hot - T_cold)
# - H: domain height
# - ν: kinematic viscosity
# - α: thermal diffusivity
# ---------------------------------------------------------------------------

# Physical parameters
Ra = 5.0e3         # Rayleigh number (above critical Ra_c ≈ 1708)
Pr = 0.71          # Prandtl number (typical for air)

# Temperature boundary conditions
T_hot = 1.0        # Bottom temperature
T_cold = 0.0       # Top temperature
ΔT = T_hot - T_cold
T_ref = (T_hot + T_cold) / 2  # Reference temperature for buoyancy

# Domain geometry
Lx = 1.0           # Domain width (aspect ratio 2:1)
Ly = 2.0           # Domain height
origin = (0.0, 0.0)

# Mesh resolution
nx, ny = 64, 32    # Grid points

# Derived physical quantities (non-dimensional formulation)
# With L = Ly = 1 as reference length and appropriate scaling:
ν = sqrt(Pr / Ra)  # Kinematic viscosity
α = ν / Pr         # Thermal diffusivity
β = 1.0            # Thermal expansion coefficient
gravity = (0.0, -1.0)  # Gravity pointing downward

println("=== Rayleigh-Bénard Instability Simulation ===")
println("Rayleigh number Ra = $Ra (critical ≈ 1708)")
println("Prandtl number Pr = $Pr")
println("Grid: $nx × $ny")
println("Domain: $Lx × $Ly")

# ---------------------------------------------------------------------------
# Mesh setup (staggered grid)
# ---------------------------------------------------------------------------
mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), origin)
dx = Lx / nx
dy = Ly / ny
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (origin[1] - 0.5 * dx, origin[2]))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (origin[1], origin[2] - 0.5 * dy))
mesh_T = mesh_p

# ---------------------------------------------------------------------------
# Capacity and operators (full domain, no embedded boundaries)
# ---------------------------------------------------------------------------
body = (x, y, _=0.0) -> -1.0  # Negative value fills entire domain

capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
capacity_T  = Capacity(body, mesh_T)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# ---------------------------------------------------------------------------
# Boundary conditions for velocity: no-slip walls all around
# ---------------------------------------------------------------------------
zero_dirichlet = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
   :left   => zero_dirichlet,
   :right  => zero_dirichlet,
   :bottom => zero_dirichlet,
   :top    => zero_dirichlet
))

bc_uy = BorderConditions(Dict(
   :left   => zero_dirichlet,
   :right  => zero_dirichlet,
   :bottom => zero_dirichlet,
   :top    => zero_dirichlet
))

pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)

# ---------------------------------------------------------------------------
# Fluid setup
# ---------------------------------------------------------------------------
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0  # No external forcing
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
             (capacity_ux, capacity_uy),
             (operator_ux, operator_uy),
             mesh_p,
             capacity_p,
             operator_p,
             ν, 1.0, fᵤ, fₚ)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np   = prod(operator_p.size)
initial_state = zeros(2 * (nu_x + nu_y) + np)

ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut; x0=initial_state)

# ---------------------------------------------------------------------------
# Temperature boundary conditions: hot bottom, cold top, insulated sides
# ---------------------------------------------------------------------------
bc_T = BorderConditions(Dict(
   :right => Dirichlet(T_hot),
   :left    => Dirichlet(T_cold),
   # Left and right are Neumann (zero flux) by default (adiabatic walls)
))
bc_T_cut = Dirichlet(0.0)

# ---------------------------------------------------------------------------
# Initial temperature field: linear profile with small perturbation
# ---------------------------------------------------------------------------
nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_scalar = Nx_T * Ny_T

T0ω = zeros(Float64, N_scalar)
for j in 1:Ny_T
   y = nodes_Ty[j]
   # Linear temperature profile (conduction solution)
   frac_y = (y - first(nodes_Ty)) / (last(nodes_Ty) - first(nodes_Ty))
   T_linear = T_hot - (T_hot - T_cold) * frac_y
   
   for i in 1:Nx_T
       x = nodes_Tx[i]
       # Add small sinusoidal perturbation to trigger instability
       # This mimics small initial disturbances that grow into convection rolls
       perturb = 0.05 * sinpi(4 * x / Lx) * sinpi(y / Ly)
       idx = i + (j - 1) * Nx_T
       T0ω[idx] = T_linear + perturb
   end
end
T0γ = copy(T0ω)
T0 = vcat(T0ω, T0γ)

# ---------------------------------------------------------------------------
# Create the coupled solver with Picard iterations
# ---------------------------------------------------------------------------
coupler = NavierStokesScalarCoupler(ns_solver,
                                   capacity_T,
                                   α,
                                   (x, y, z=0.0, t=0.0) -> 0.0,  # No heat source
                                   bc_T,
                                   bc_T_cut;
                                   strategy=PicardCoupling(tol_T=1e-6, tol_U=1e-6, maxiter=5, relaxation=0.9),
                                   β=β,
                                   gravity=gravity,
                                   T_ref=T_ref,
                                   T0=T0,
                                   store_states=true)

# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
Δt = 5.0e-3
T_end = 0.5

println("\nStarting time integration (Δt=$Δt, T_end=$T_end)...")
times, velocity_hist, scalar_hist = solve_NavierStokesScalarCoupling!(coupler;
                                                                     Δt=Δt,
                                                                     T_end=T_end,
                                                                     scheme=:CN)
println("Simulation complete: $(length(times)) time levels stored.")

# ---------------------------------------------------------------------------
# Extract final fields
# ---------------------------------------------------------------------------
u_state = coupler.velocity_state
T_state = coupler.scalar_state

# Velocity components
ux_nodes = mesh_ux.nodes
uy_nodes = mesh_uy.nodes
uωx = view(u_state, 1:nu_x)
uωy = view(u_state, 2 * nu_x + 1:2 * nu_x + nu_y)
Ux_grid = reshape(uωx, length(ux_nodes[1]), length(ux_nodes[2]))'
Uy_grid = reshape(uωy, length(uy_nodes[1]), length(uy_nodes[2]))'
speed_grid = sqrt.(Ux_grid.^2 .+ Uy_grid.^2)

# Temperature field
temp_grid = reshape(view(T_state, 1:N_scalar), Nx_T, Ny_T)'

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
vel_max = maximum(speed_grid)
temp_range = extrema(temp_grid)
mean_temp = mean(temp_grid)

println("\n=== Diagnostics ===")
println("Maximum velocity magnitude: $vel_max")
println("Temperature range: $(temp_range[1]) to $(temp_range[2])")
println("Mean temperature: $mean_temp")

# Check for convection: if max velocity is significant, convection has developed
if vel_max > 1e-4
   println("Status: Convection has developed (Ra > Ra_c)")
else
   println("Status: Flow is approximately stagnant (Ra < Ra_c or simulation needs more time)")
end

# Compute approximate Nusselt number at the bottom wall
# Nu = -H * (dT/dy)|_{y=0} / ΔT
Δy = nodes_Ty[2] - nodes_Ty[1]
Nu_local = zeros(Nx_T)
for i in 1:Nx_T
   # Second-order one-sided derivative at y=0
   T1 = temp_grid[1, i]   # y = nodes_Ty[1]
   T2 = temp_grid[2, i]   # y = nodes_Ty[2]
   T3 = temp_grid[3, i]   # y = nodes_Ty[3]
   dTdy = (-3*T1 + 4*T2 - T3) / (2*Δy)
   Nu_local[i] = -Ly * dTdy / ΔT
end
Nu_mean = mean(Nu_local)
println("Mean Nusselt number at bottom wall: $Nu_mean")
println("(Nu = 1 for pure conduction, Nu > 1 indicates convective enhancement)")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
if @isdefined CairoMakie
   fig = Figure(size=(1200, 800))
   
   # Temperature field
   ax_T = Axis(fig[1, 1], xlabel="x", ylabel="y",
               title="Temperature Field (t = $(round(times[end], digits=3)))",
               aspect=DataAspect())
   hm_T = heatmap!(ax_T, nodes_Tx, nodes_Ty, temp_grid; colormap=:thermal)
   Colorbar(fig[1, 2], hm_T; label="T")
   
   # Velocity magnitude
   ax_u = Axis(fig[1, 3], xlabel="x", ylabel="y",
               title="Velocity Magnitude",
               aspect=DataAspect())
   hm_u = heatmap!(ax_u, ux_nodes[1], ux_nodes[2], speed_grid; colormap=:viridis)
   Colorbar(fig[1, 4], hm_u; label="|u|")
   
   # Vertical velocity component (shows convection rolls)
   ax_uy = Axis(fig[2, 1], xlabel="x", ylabel="y",
                title="Vertical Velocity (uy)",
                aspect=DataAspect())
   hm_uy = heatmap!(ax_uy, uy_nodes[1], uy_nodes[2], Uy_grid; colormap=:balance)
   Colorbar(fig[2, 2], hm_uy; label="uy")
   
   # Temperature profile at mid-height
   mid_y_idx = div(Ny_T, 2)
   ax_profile = Axis(fig[2, 3], xlabel="x", ylabel="T",
                     title="Temperature at y = $(round(nodes_Ty[mid_y_idx], digits=2))")
   lines!(ax_profile, nodes_Tx, temp_grid[mid_y_idx, :])
   
   display(fig)
   
   output_path = joinpath(@__DIR__, "rayleigh_benard_instability.png")
   save(output_path, fig)
   println("\nFigure saved to: $output_path")
   
   # Animation of temperature evolution
   if !isempty(scalar_hist) && length(scalar_hist) > 1
       temp_min = minimum(map(s -> minimum(view(s, 1:N_scalar)), scalar_hist))
       temp_max = maximum(map(s -> maximum(view(s, 1:N_scalar)), scalar_hist))
       clim = (temp_min, temp_max)
       
       anim_fig = Figure(size=(800, 400))
       anim_ax = Axis(anim_fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y",
                      title="Temperature (t = $(round(times[1], digits=3)))")
       
       scalar_frame = Observable(reshape(view(scalar_hist[1], 1:N_scalar), Nx_T, Ny_T)')
       anim_hm = heatmap!(anim_ax, nodes_Tx, nodes_Ty, scalar_frame;
                         colormap=:thermal, colorrange=clim)
       Colorbar(anim_fig[1, 2], anim_hm; label="T")
       
       anim_path = joinpath(@__DIR__, "rayleigh_benard_instability.mp4")
       println("Recording animation → $anim_path")
       
       record(anim_fig, anim_path, eachindex(times); framerate=30) do idx
           scalar_frame[] = reshape(view(scalar_hist[idx], 1:N_scalar), Nx_T, Ny_T)'
           anim_ax.title = "Rayleigh-Bénard (t = $(round(times[idx], digits=3)))"
       end
       
       println("Animation saved to: $anim_path")
   end
end

println("\n=== Rayleigh-Bénard Instability Simulation Complete ===")