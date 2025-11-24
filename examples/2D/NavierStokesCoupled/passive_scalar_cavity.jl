using Penguin
using LinearAlgebra

# ------------------------------------------------------------------------------
# Passive scalar transport in a lid-driven cavity
#
# This example exercises the NavierStokesScalarCoupler using the PassiveCoupling
# strategy, which advances the Navier–Stokes solver first and then transports a
# passive scalar with the updated velocity field. The scalar does not feed back
# on the flow (β = 0).
# ------------------------------------------------------------------------------

# Domain ----------------------------------------------------------------------
nx, ny = 48, 48
width, height = 1.0, 1.0
origin = (0.0, 0.0)

mesh_p = Penguin.Mesh((nx, ny), (width, height), origin)
dx = width / nx
dy = height / ny
mesh_ux = Penguin.Mesh((nx, ny), (width, height), (origin[1] - 0.5 * dx, origin[2]))
mesh_uy = Penguin.Mesh((nx, ny), (width, height), (origin[1], origin[2] - 0.5 * dy))
mesh_T = mesh_p

body = (x, y, _=0.0) -> -1.0

capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
capacity_T  = Capacity(body, mesh_T)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# Boundary conditions ---------------------------------------------------------
zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left   => zero,
    :right  => zero,
    :bottom => zero,
    :top    => Dirichlet((x, y, t=0.0) -> 2.0)  # lid velocity
))

bc_uy = BorderConditions(Dict(
    :left   => zero,
    :right  => zero,
    :bottom => zero,
    :top    => zero
))

pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)

# Fluid setup -----------------------------------------------------------------
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

# Scalar setup ----------------------------------------------------------------
κ = 1.0e-3  # scalar diffusivity
scalar_source = (x, y, z=0.0, t=0.0) -> 0.0

nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_scalar = Nx_T * Ny_T

# Initial scalar: localized Gaussian blob near the bottom-center
function gaussian_blob(x, y; x0=0.5, y0=0.5, σ=0.07)
    dx = x - x0
    dy = y - y0
    return exp(-(dx^2 + dy^2) / (2σ^2))
end

T0ω = zeros(Float64, N_scalar)
for j in 1:Ny_T
    y = nodes_Ty[j]
    for i in 1:Nx_T
        x = nodes_Tx[i]
        idx = i + (j - 1) * Nx_T
        T0ω[idx] = gaussian_blob(x, y)
    end
end
T0γ = copy(T0ω)
T0 = vcat(T0ω, T0γ)

bc_T = BorderConditions(Dict(
    :left   => zero,
    :right  => zero,
    :bottom => zero,
    :top    => zero
))
bc_T_cut = Dirichlet(0.0)

coupler = NavierStokesScalarCoupler(ns_solver,
                                    capacity_T,
                                    κ,
                                    scalar_source,
                                    bc_T,
                                    bc_T_cut;
                                    strategy=PassiveCoupling(),
                                    β=0.0,
                                    gravity=(0.0, 0.0),
                                    T_ref=0.0,
                                    T0=T0,
                                    store_states=true)

# Time integration ------------------------------------------------------------
Δt = 5.0e-3
T_end = 1.0

println("Starting passive scalar cavity run (Δt=$(Δt), T_end=$(T_end))")
times, velocity_hist, scalar_hist = solve_NavierStokesScalarCoupling!(coupler;
                                                                      Δt=Δt,
                                                                      T_end=T_end,
                                                                      scheme=:CN)
println("Simulation complete: stored $(length(times)) time levels.")

# Diagnostics -----------------------------------------------------------------
u_state = coupler.velocity_state
T_state = coupler.scalar_state

max_u = maximum(abs, u_state[1:nu_x + 2nu_y])
min_T = minimum(T_state)
max_T = maximum(T_state)

println("Final velocity max norm: ", max_u)
println("Final scalar range: ", (min_T, max_T))

# Visualization ----------------------------------------------------------------
using CairoMakie

fig = Figure(resolution=(800, 400))
ax1 = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y", title="Velocity magnitude at t=$(round(times[end], digits=3))")
u_x = u_state[1:nu_x]
u_y = u_state[2 * nu_x + 1:2 * nu_x + nu_y]
u_mag = sqrt.(u_x .^ 2 .+ u_y .^ 2)
hm = heatmap!(ax1, reshape(u_mag, (nx+1, ny+1))'; colormap=:viridis)
Colorbar(fig[1, 2], hm; label="|u|")
ax2 = Axis(fig[2, 1], aspect=DataAspect(), xlabel="x", ylabel="y", title="Scalar field at t=$(round(times[end], digits=3))")
T_ω = T_state[1:N_scalar]
hm1 = heatmap!(ax2, reshape(T_ω, (Nx_T, Ny_T))'; colormap=:plasma)
Colorbar(fig[2, 2], hm1; label="T")
display(fig)

# Record animation of scalar transport
if isempty(scalar_hist)
    @warn "No scalar snapshots available; skipping animation."
else
    scalar_min = minimum(map(state -> minimum(view(state, 1:N_scalar)), scalar_hist))
    scalar_max = maximum(map(state -> maximum(view(state, 1:N_scalar)), scalar_hist))
    clim = (scalar_min, scalar_max)

    anim_fig = Figure(resolution=(600, 500))
    anim_ax = Axis(anim_fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y",
                   title="Passive scalar (t = $(round(times[1]; digits=3)))")

    scalar_frame = Observable(reshape(view(scalar_hist[1], 1:N_scalar), Nx_T, Ny_T)')
    anim_hm = heatmap!(anim_ax, nodes_Tx, nodes_Ty, scalar_frame; colormap=:plasma, colorrange=clim)
    Colorbar(anim_fig[1, 2], anim_hm; label="T")

    output_path = joinpath(@__DIR__, "passive_scalar_cavity.mp4")
    println("Recording animation to $(output_path)")

    record(anim_fig, output_path, eachindex(times)) do idx
        scalar_frame[] = reshape(view(scalar_hist[idx], 1:N_scalar), Nx_T, Ny_T)'
        anim_ax.title = "Passive scalar (t = $(round(times[idx]; digits=3)))"
    end

    println("Animation saved to $(output_path)")
end
