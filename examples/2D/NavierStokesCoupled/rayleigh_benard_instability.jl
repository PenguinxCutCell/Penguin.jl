using Penguin
using LinearAlgebra
using Statistics
try
    using CairoMakie
catch
    @warn "CairoMakie not available; visualization disabled."
end

# ---------------------------------------------------------------------------
# Rayleigh–Bénard instability demonstration (Pr = 1.0, Ra = 2e4)
# ---------------------------------------------------------------------------
# This example sets up a rectangular cavity heated from below and cooled from
# above. For Ra above the critical value (~1708), convection cells emerge.
# We choose Ra = 2e4 to highlight the instability and visualize the flow.
# ---------------------------------------------------------------------------

# Geometry and mesh ---------------------------------------------------------
nx, ny = 96, 48
width, height = 2.0, 1.0
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

# Rayleigh–Bénard parameters -------------------------------------------------
Ra = 2.0e4
Pr = 1.0
ΔT = 1.0
T_bottom = 0.5
T_top = -0.5
Lz = height

ν = sqrt(Pr / Ra)
α = ν / Pr
μ = ν
κ = α
β = 1.0
gravity = (0.0, -10.0)

# Velocity boundary conditions: no-slip on all walls ------------------------
zero = Dirichlet((x, y, t=0.0) -> 0.0)
bc_ux = BorderConditions(Dict(
    :left=>zero,
    :right=>zero,
    :bottom=>zero,
    :top=>zero
))
bc_uy = BorderConditions(Dict(
    :left=>zero,
    :right=>zero,
    :bottom=>zero,
    :top=>zero
))

pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, 1.0, (x,y,z=0,t=0)->0.0, (x,y,z=0,t=0)->0.0)

ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut)

# Temperature boundary conditions -------------------------------------------
bc_T = BorderConditions(Dict(
    :bottom=>Dirichlet(T_bottom),
    :top=>Dirichlet(T_top),
    :left=>Neumann(0.0),
    :right=>Neumann(0.0)
))
bc_T_cut = Dirichlet(0.0)

# Initial temperature field: linear stratification plus small perturbation ---
nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_scalar = Nx_T * Ny_T

T_init_center = Vector{Float64}(undef, N_scalar)
for j in 1:Ny_T
    y = nodes_Ty[j]
    for i in 1:Nx_T
        x = nodes_Tx[i]
        frac = (x - first(nodes_Tx)) / (last(nodes_Tx) - first(nodes_Tx))  # linear in x now
        linear = T_bottom + (T_top - T_bottom) * frac
        perturb = 0.1 * sinpi(y / height) * sinpi(frac)
        idx = i + (j - 1) * Nx_T
        T_init_center[idx] = linear + perturb
    end
end
T_init_interface = copy(T_init_center)
T_init = vcat(T_init_center, T_init_interface)
T_init .= 0.0

# Coupler setup --------------------------------------------------------------
coupler = NavierStokesScalarCoupler(ns_solver,
                                    capacity_T,
                                    κ,
                                    (x, y, z=0.0, t=0.0) -> 0.0,
                                    bc_T,
                                    bc_T_cut;
                                    strategy=PicardCoupling(tol_T=1e-6, tol_U=1e-6, maxiter=8, relaxation=0.9),
                                    β=β,
                                    gravity=gravity,
                                    T_ref=0.0,
                                    T0=T_init,
                                    store_states=true)

# Time stepping --------------------------------------------------------------
Δt = 2.0e-3
T_end = 0.1

println("=== Rayleigh–Bénard instability demonstration ===")
println("Grid: $nx × $ny, Ra = $Ra, Pr = $Pr, Δt = $Δt, T_end = $T_end")

times, velocity_hist, scalar_hist = solve_NavierStokesScalarCoupling!(coupler;
                                                                      Δt=Δt,
                                                                      T_end=T_end,
                                                                      scheme=:CN)

# Extract final fields -------------------------------------------------------
u_state = coupler.velocity_state
T_state = coupler.scalar_state

data = Penguin.navierstokes2D_blocks(coupler.momentum)
nu_x = data.nu_x
nu_y = data.nu_y

Ux = reshape(view(u_state, 1:nu_x), length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2]))
Uy = reshape(view(u_state, 2 * nu_x + 1:2 * nu_x + nu_y), length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2]))
T_field = reshape(view(T_state, 1:N_scalar), Nx_T, Ny_T)

speed = sqrt.(Ux.^2 .+ Uy.^2)
max_speed = maximum(speed)
println("Final max velocity magnitude ≈ ", max_speed)

# Visualization --------------------------------------------------------------
if @isdefined CairoMakie
    xs = nodes_Tx
    ys = nodes_Ty

    fig = Figure(resolution=(960, 480))
    ax_T = Axis(fig[1, 1], xlabel="x", ylabel="y",
                title="Temperature field (t = $(round(times[end]; digits=3)))",
                aspect=DataAspect())
    hm = heatmap!(ax_T, xs, ys, T_field'; colormap=:thermal)
    Colorbar(fig[1, 2], hm; label="T")

    ax_u = Axis(fig[2, 1], xlabel="x", ylabel="y",
                title="Velocity magnitude",
                aspect=DataAspect())
    hm_u = heatmap!(ax_u, mesh_ux.nodes[1], mesh_ux.nodes[2], speed'; colormap=:viridis)
    Colorbar(fig[2, 2], hm_u; label="|u|")

    display(fig)

    # Animation of temperature evolution ------------------------------------
    temp_snapshots = map(state -> reshape(view(state, 1:N_scalar), Nx_T, Ny_T), scalar_hist)
    tmin = minimum(minimum.(temp_snapshots))
    tmax = maximum(maximum.(temp_snapshots))

    anim_fig = Figure(resolution=(640, 360))
    anim_ax = Axis(anim_fig[1, 1], xlabel="x", ylabel="y",
                   title="Temperature evolution",
                   aspect=DataAspect())
    temp_obs = Observable(temp_snapshots[1]')
    heatmap!(anim_ax, xs, ys, temp_obs; colormap=:thermal, colorrange=(tmin, tmax))

    output_path = joinpath(@__DIR__, "rayleigh_benard_instability.mp4")
    println("Recording animation → $(output_path)")
    record(anim_fig, output_path, eachindex(times)) do idx
        temp_obs[] = temp_snapshots[idx]'
        anim_ax.title = "Temperature (t = $(round(times[idx]; digits=3)))"
    end
    println("Animation saved to $(output_path)")
end
