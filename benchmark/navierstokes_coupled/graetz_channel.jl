using Penguin
using LinearAlgebra
using Statistics
try
    using CairoMakie
catch
    @warn "CairoMakie not available; visualization will be skipped."
end

"""
Forced-convection (Graetz) benchmark in a 2-D channel.

- Laminar Poiseuille inflow with prescribed parabolic profile.
- Cold fluid enters a hot-wall channel; buoyancy disabled.
- Compares near-outlet Nusselt number against fully-developed theory (Nu ≈ 7.54).
NOT COMPLETED YET
"""

# Geometry and mesh ---------------------------------------------------------
nx, ny = 120, 40
lengthh, height = 1.0, 1.0
origin = (0.0, 0.0)

mesh_p = Penguin.Mesh((nx, ny), (lengthh, height), origin)
dx = lengthh / nx
dy = height / ny
mesh_ux = Penguin.Mesh((nx, ny), (lengthh, height), (origin[1] - 0.5 * dx, origin[2]))
mesh_uy = Penguin.Mesh((nx, ny), (lengthh, height), (origin[1], origin[2] - 0.5 * dy))
mesh_T = mesh_p

body = (x, y, _=0.0) -> -1.0

capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
capacity_T  = Capacity(body, mesh_T)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# Velocity boundary conditions
umax = 1.5
channel_height = last(mesh_T.nodes[2]) - first(mesh_T.nodes[2])
u_profile = (x, y, t=0.0) -> begin
    y0 = y - first(mesh_T.nodes[2])
    umax * 4.0 * y0 * (channel_height - y0) / channel_height^2
end

zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left   => Dirichlet(u_profile),
    :right  => Outflow(),
    :bottom => zero,
    :top    => zero
))
bc_uy = BorderConditions(Dict(
    :left   => zero,
    :right  => zero,
    :bottom => zero,
    :top    => zero
))

pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)

# Fluid properties ----------------------------------------------------------
μ = 0.02
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

ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut)

# Pre-compute steady laminar profile ---------------------------------------
println("=== Graetz channel benchmark: computing base flow ===")
solve_NavierStokesMono_steady!(ns_solver; tol=1e-10, maxiter=30)

# Scalar boundary conditions ------------------------------------------------
T_inlet = 0.0
T_wall = 1.0

bc_T = BorderConditions(Dict(
    :left   => Dirichlet(T_inlet),
    :right  => Neumann(0.0),
    :top    => Dirichlet(T_wall),
    :bottom => Dirichlet(T_wall)
))
bc_T_cut = Dirichlet(0.0)

nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_scalar = Nx_T * Ny_T

initial_scalar = vcat(fill(T_inlet, N_scalar), fill(T_inlet, N_scalar))

# Coupler setup -------------------------------------------------------------
κ = 1.0e-3
heat_source = (x, y, z=0.0, t=0.0) -> 0.0

coupler = NavierStokesScalarCoupler(ns_solver,
                                    capacity_T,
                                    κ,
                                    heat_source,
                                    bc_T,
                                    bc_T_cut;
                                    strategy=PassiveCoupling(),
                                    β=0.0,
                                    gravity=(0.0, -1.0),
                                    T_ref=0.0,
                                    T0=initial_scalar,
                                    store_states=true)

Δt = 1.0e-2
T_end = 1.0

println("=== Advecting thermal field ===")
times, velocity_hist, scalar_hist = solve_NavierStokesScalarCoupling!(coupler;
                                                                      Δt=Δt,
                                                                      T_end=T_end,
                                                                      scheme=:CN)

@assert !isempty(scalar_hist) "Scalar history not stored; set store_states=true."

# Extract final fields ------------------------------------------------------
u_state = coupler.velocity_state
T_state = coupler.scalar_state

# Reshape temperature to 2D grid (cell-centered)
Tω = reshape(view(T_state, 1:N_scalar), Nx_T, Ny_T)

# Compute local and average Nusselt numbers ---------------------------------
function compute_local_nusselt(field, nodes_x::Vector{Float64}, nodes_y::Vector{Float64}, wall_T::Float64)
    Nx, Ny = size(field)
    dy = nodes_y[2] - nodes_y[1]
    Nu = zeros(Float64, Nx)
    for i in 1:Nx
        T_wall_cell = field[i, Ny]
        T_adjacent = field[i, Ny-1]
        bulk = sum(field[i, :]) / Ny
        denom = bulk - wall_T
        if abs(denom) < 1e-10
            Nu[i] = NaN
        else
            Nu[i] = abs(-(T_wall_cell - T_adjacent) / dy / denom)
        end
    end
    return Nu
end

Nu_local = compute_local_nusselt(Tω, nodes_Tx, nodes_Ty, T_wall)

valid_indices = findall(!isnan, Nu_local)
Nu_local = Nu_local[valid_indices]
x_positions = nodes_Tx[valid_indices]

tail_len = min(length(Nu_local), 50)
Nu_tail = mean(Nu_local[end-tail_len+1:end])

println("Mean tail Nusselt ≈ ", Nu_tail)
#@assert abs(Nu_tail - 7.54) / 7.54 ≤ 0.2 "Tail Nusselt deviates significantly from theory."

if !isempty(Nu_local)
    Nu_peak = maximum(Nu_local)
    println("Peak Nusselt near inlet ≈ ", Nu_peak)
    @assert Nu_peak > Nu_tail "Nusselt should decrease along the channel."
end

# Visualization --------------------------------------------------------------
if @isdefined CairoMakie
    xs = nodes_Tx
    ys = nodes_Ty
    fig = Figure(resolution=(900, 450))
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y",
              title="Temperature field (t = $(round(times[end]; digits=3)))",
              aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, Tω'; colormap=:thermal)
    Colorbar(fig[1, 2], hm; label="T")

    ax_nu = Axis(fig[2, 1], xlabel="x", ylabel="Nu",
                 title="Local Nusselt along the heated wall")
    lines!(ax_nu, x_positions, Nu_local; color=:blue)
    hlines!(ax_nu, [7.54]; color=:red, linestyle=:dash, label="Fully developed")
    axislegend(ax_nu)

    display(fig)

    # Animation of scalar evolution
    T_snapshots = map(state -> reshape(view(state, 1:N_scalar), Nx_T, Ny_T)', scalar_hist)
    temp_min = minimum(minimum.(T_snapshots))
    temp_max = maximum(maximum.(T_snapshots))

    anim_fig = Figure(resolution=(600, 400))
    anim_ax = Axis(anim_fig[1, 1], xlabel="x", ylabel="y",
                   title="Temperature evolution",
                   aspect=DataAspect())
    temp_obs = Observable(T_snapshots[1])
    anim_hm = heatmap!(anim_ax, xs, ys, temp_obs; colormap=:thermal,
                       colorrange=(temp_min, temp_max))

    output_path = joinpath(@__DIR__, "graetz_channel.mp4")
    println("Recording temperature animation → $(output_path)")
    record(anim_fig, output_path, eachindex(times)) do idx
        temp_obs[] = T_snapshots[idx]
        anim_ax.title = "Temperature (t = $(round(times[idx]; digits=3)))"
    end
    println("Animation saved to $(output_path)")
end
