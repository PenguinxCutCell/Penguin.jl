using Penguin
using CairoMakie
using Random
using Statistics

"""
Decaying two-dimensional turbulence in a unit square with periodic boundaries.
The initial velocity is spatially uncorrelated noise in [-1, 1]; viscosity
causes turbulent structures to diffuse away over time.
"""

Random.seed!(2024)

# ---------------------------------------------------------------------------
# Domain and discretization
# ---------------------------------------------------------------------------
Lx = 1.0
Ly = 1.0
nx = 128
ny = 128

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (-0.5 * dx, 0.0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (0.0, -0.5 * dy))

# Fully filled domain (no internal obstacles)
solid_indicator = (x, y, _=0.0) -> -1.0
capacity_ux = Capacity(solid_indicator, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(solid_indicator, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(solid_indicator, mesh_p;  compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# ---------------------------------------------------------------------------
# Boundary conditions (periodic in both directions)
# ---------------------------------------------------------------------------
periodic_bc = BorderConditions(Dict(
    :left=>Dirichlet(0.0),
    :right=>Dirichlet(0.0),
    :bottom=>Dirichlet(0.0),
    :top=>Dirichlet(0.0)
))

bc_ux = periodic_bc
bc_uy = periodic_bc
pressure_gauge = MeanPressureGauge()
interface_bc = Dirichlet(0.0)

# ---------------------------------------------------------------------------
# Fluid properties and forcing
# ---------------------------------------------------------------------------
ρ = 1.0
μ = 1e-3
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

# ---------------------------------------------------------------------------
# Initial condition: divergence-uncorrected noise
# ---------------------------------------------------------------------------
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
Ntot = 2 * (nu_x + nu_y) + np

x0_vec = zeros(Float64, Ntot)

noise_x = 2 .* rand(Float64, nu_x) .- 1.0
noise_x .-= mean(noise_x)
noise_y = 2 .* rand(Float64, nu_y) .- 1.0
noise_y .-= mean(noise_y)

off_uωx = 0
off_uγx = nu_x
off_uωy = 2 * nu_x
off_uγy = 2 * nu_x + nu_y
off_p   = 2 * (nu_x + nu_y)

x0_vec[off_uωx+1:off_uωx+nu_x] .= noise_x
x0_vec[off_uγx+1:off_uγx+nu_x] .= noise_x
x0_vec[off_uωy+1:off_uωy+nu_y] .= noise_y
x0_vec[off_uγy+1:off_uγy+nu_y] .= noise_y

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)

# ---------------------------------------------------------------------------
# Time integration parameters
# ---------------------------------------------------------------------------
Δt = 0.005
T_end = 1.0
println("[Decaying Turbulence] domain=$(Lx)x$(Ly), nx=$(nx), ny=$(ny)")
println("[Decaying Turbulence] Δt=$(Δt), T_end=$(T_end), viscosity=$(μ)")

kinetic_energy(state) = begin
    ux = view(state, off_uωx+1:off_uωx+nu_x)
    uy = view(state, off_uωy+1:off_uωy+nu_y)
    0.5 * (sum(abs2, ux) + sum(abs2, uy))
end

times, histories = solve_NavierStokesMono_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=:CN)
println("[Decaying Turbulence] time steps = $(length(times) - 1)")

energies = Float64[kinetic_energy(state) for state in histories]
println("[Decaying Turbulence] final energy=$(energies[end])")

# ---------------------------------------------------------------------------
# Visualization: final speed field and kinetic energy decay
# ---------------------------------------------------------------------------
xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]

Ux = reshape(solver.x[off_uωx+1:off_uωx+nu_x], (length(xs), length(ys)))
Uy = reshape(solver.x[off_uωy+1:off_uωy+nu_y], (length(xs), length(ys)))
speed = sqrt.(Ux.^2 .+ Uy.^2)

fig = Figure(resolution=(1000, 450))
ax_speed = Axis(fig[1, 1], xlabel="x", ylabel="y",
                title="Speed magnitude at t = $(round(times[end]; digits=3))")
hm_speed = heatmap!(ax_speed, xs, ys, speed; colormap=:plasma)
Colorbar(fig[1, 2], hm_speed, label="|u|")

ax_energy = Axis(fig[1, 3], xlabel="time", ylabel="Kinetic energy",
                 title="Energy decay")
lines!(ax_energy, times, energies; color=:navy, linewidth=2)

display(fig)
save("navierstokes2d_decaying_turbulence.png", fig)

println("[Decaying Turbulence] snapshot saved to navierstokes2d_decaying_turbulence.png")

# ---------------------------------------------------------------------------
# Animation of velocity magnitude
# ---------------------------------------------------------------------------
println("[Decaying Turbulence] creating animation...")
n_frames = min(60, length(histories))
frame_indices = round.(Int, range(1, length(histories), length=n_frames))

fig_anim = Figure(resolution=(600, 500))
ax_anim = Axis(fig_anim[1, 1], xlabel="x", ylabel="y",
               title="Velocity magnitude")

speed_obs = Observable(zeros(length(xs), length(ys)))
hm_anim = heatmap!(ax_anim, xs, ys, speed_obs; colormap=:plasma)
Colorbar(fig_anim[1, 2], hm_anim, label="|u|")

function update_frame!(frame_idx)
    hist = histories[frame_indices[frame_idx]]
    ux_hist = hist[off_uωx+1:off_uωx+nu_x]
    uy_hist = hist[off_uωy+1:off_uωy+nu_y]
    Ux_hist = reshape(ux_hist, (length(xs), length(ys)))
    Uy_hist = reshape(uy_hist, (length(xs), length(ys)))
    speed_obs[] = sqrt.(Ux_hist.^2 .+ Uy_hist.^2)
    ax_anim.title = "Velocity magnitude at t = $(round(times[frame_indices[frame_idx]]; digits=3))"
end

record(fig_anim, "navierstokes2d_decaying_turbulence.gif", 1:n_frames; framerate=12) do frame
    update_frame!(frame)
end

println("[Decaying Turbulence] animation saved to navierstokes2d_decaying_turbulence.gif")
