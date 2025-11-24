using Penguin
using LinearAlgebra
using Statistics
using CairoMakie

# ---------------------------------------------------------------------------
# Differentially heated cavity with Picard coupling
# ---------------------------------------------------------------------------
# This script demonstrates the two-way (Picard) strategy of
# `NavierStokesScalarCoupler`. Buoyancy couples the momentum and temperature
# equations, and a small number of outer iterations is taken every time step.
# ---------------------------------------------------------------------------

# Domain and discretisation ---------------------------------------------------
nx, ny = 64, 32
width, height = 1.0, 1.0
origin = (0.0, 0.0)

mesh_p = Penguin.Mesh((nx, ny), (width, height), origin)
dx = width / nx
dy = height / ny
mesh_ux = Penguin.Mesh((nx, ny), (width, height), (origin[1] - 0.5 * dx, origin[2]))
mesh_uy = Penguin.Mesh((nx, ny), (width, height), (origin[1], origin[2] - 0.5 * dy))
mesh_T = mesh_p

body = (x, y, _=0.0) -> -1.0

# Capacities and operators ----------------------------------------------------
capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
capacity_T  = Capacity(body, mesh_T)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# Velocity boundary conditions: no-slip walls, stationary top/bottom
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

# Fluid properties ------------------------------------------------------------
μ = 1.0
ρ = 1.0
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np   = prod(operator_p.size)
initial_state = zeros(2 * (nu_x + nu_y) + np)

ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut; x0=initial_state)

# Scalar boundary conditions and initial condition ----------------------------
T_hot = 0.5
T_cold = -0.5
bc_T = BorderConditions(Dict(
    :right    => Dirichlet(T_hot),
    :left => Dirichlet(T_cold)
))
bc_T_cut = Dirichlet(0.0)

nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_scalar = Nx_T * Ny_T

T0ω = zeros(Float64, N_scalar)
for j in 1:Ny_T
    y = nodes_Ty[j]
    frac = (y - first(nodes_Ty)) / (last(nodes_Ty) - first(nodes_Ty))
    base = T_cold + (T_hot - T_cold) * frac
    for i in 1:Nx_T
        x = nodes_Tx[i]
        perturb = 0.05 * sinpi(x / width) * sinpi(frac)
        idx = i + (j - 1) * Nx_T
        T0ω[idx] = base + perturb
    end
end
T0γ = copy(T0ω)
T0 = vcat(T0ω, T0γ)

# Couple Navier–Stokes and scalar with Picard iterations ----------------------
κ = 1.0e-2
heat_source = (x, y, z=0.0, t=0.0) -> 0.0

coupler = NavierStokesScalarCoupler(ns_solver,
                                    capacity_T,
                                    κ,
                                    heat_source,
                                    bc_T,
                                    bc_T_cut;
                                    strategy=PicardCoupling(tol_T=1e-6, tol_U=1e-6, maxiter=4, relaxation=0.9),
                                    β=1.0,
                                    gravity=(-1.0, 0.0),
                                    T_ref=0.0,
                                    T0=T0,
                                    store_states=true)

# Time integration parameters -------------------------------------------------
Δt = 2.5e-3
T_end = 0.25

println("Starting differentially heated cavity Picard run (Δt=$(Δt), T_end=$(T_end))")
times, velocity_hist, scalar_hist = solve_NavierStokesScalarCoupling!(coupler;
                                                                      Δt=Δt,
                                                                      T_end=T_end,
                                                                      scheme=:CN)
println("Simulation complete: stored $(length(times)) time levels.")

# Diagnostics -----------------------------------------------------------------
u_state = coupler.velocity_state
T_state = coupler.scalar_state

vel_max = maximum(abs, u_state)
temp_range = extrema(T_state)

println("Velocity max magnitude: $(vel_max)")
println("Scalar range: $(temp_range)")

if !isempty(scalar_hist)
    mean_temperature = mean(view(T_state, 1:N_scalar))
    println("Mean interior temperature: $(mean_temperature)")
end

println("Picard iterations reached: $(length(coupler.momentum.residual_history)) samples recorded.")

# ------------------------------------------------------------------------------
# Visualization and animation (optional, requires CairoMakie)
# ------------------------------------------------------------------------------

# Velocity magnitude snapshot
ux_nodes = mesh_ux.nodes
uy_nodes = mesh_uy.nodes
uωx = view(u_state, 1:nu_x)
uωy = view(u_state, 2 * nu_x + 1:2 * nu_x + nu_y)
Ux_grid = reshape(uωx, length(ux_nodes[1]), length(ux_nodes[2]))'
Uy_grid = reshape(uωy, length(uy_nodes[1]), length(uy_nodes[2]))'
speed_grid = sqrt.(Ux_grid.^2 .+ Uy_grid.^2)
temp_grid = reshape(view(T_state, 1:N_scalar), Nx_T, Ny_T)'

fig = Figure(resolution=(900, 450))
ax_u = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y",
            title="Velocity magnitude (t = $(round(times[end]; digits=3)))")
hm_u = heatmap!(ax_u, ux_nodes[1], ux_nodes[2], speed_grid; colormap=:viridis)
Colorbar(fig[1, 2], hm_u; label="|u|")

ax_T = Axis(fig[1, 3], aspect=DataAspect(), xlabel="x", ylabel="y",
            title="Temperature (t = $(round(times[end]; digits=3)))")
hm_T = heatmap!(ax_T, nodes_Tx, nodes_Ty, temp_grid; colormap=:thermal)
Colorbar(fig[1, 4], hm_T; label="T")

display(fig)

# Animation of the scalar field
if !isempty(scalar_hist)
    temp_min = minimum(map(state -> minimum(view(state, 1:N_scalar)), scalar_hist))
    temp_max = maximum(map(state -> maximum(view(state, 1:N_scalar)), scalar_hist))
    clim = (temp_min, temp_max)

    anim_fig = Figure(resolution=(600, 500))
    anim_ax = Axis(anim_fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y",
                    title="Temperature (t = $(round(times[1]; digits=3)))")
    scalar_frame = Observable(reshape(view(scalar_hist[1], 1:N_scalar), Nx_T, Ny_T)')
    anim_hm = heatmap!(anim_ax, nodes_Tx, nodes_Ty, scalar_frame;
                        colormap=:thermal, colorrange=clim)
    Colorbar(anim_fig[1, 2], anim_hm; label="T")

    output_path = joinpath(@__DIR__, "picard_differential_cavity.mp4")
    println("Recording Picard coupling animation → $(output_path)")

    record(anim_fig, output_path, eachindex(times)) do idx
            scalar_frame[] = reshape(view(scalar_hist[idx], 1:N_scalar), Nx_T, Ny_T)'
            anim_ax.title = "Temperature (t = $(round(times[idx]; digits=3)))"
    end

    println("Animation saved to $(output_path)")
end
