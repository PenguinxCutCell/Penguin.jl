using Penguin
using LinearAlgebra
using Statistics
using Printf
try
    using CairoMakie
catch
    @warn "CairoMakie not available; skipping visualization." 
end

"""
Differentially heated cavity benchmark (Pr = 0.71, Ra = 1e3).

- Square cavity, left wall hot, right wall cold, horizontal walls adiabatic.
- Buoyancy-driven flow solved with Picard coupling.
- Compares mean hot-wall Nusselt number and peak velocities to de Vahl Davis (1983) reference data.
"""

# Problem parameters ---------------------------------------------------------
Ra = 1.0e3
Pr = 0.71
ΔT = 1.0
T_hot = 0.5
T_cold = -0.5
L = 1.0

# Material properties (dimensionless scaling)
ν = sqrt(Pr / Ra)
α = ν / Pr
μ = ν  # ρ = 1
κ = α
β = 1.0
gravity = (-1.0, 0.0)  # g magnitude = 1 with our scaling

# Discretisation -------------------------------------------------------------
nx, ny = 128, 128
origin = (0.0, 0.0)

mesh_p = Penguin.Mesh((nx, ny), (L, L), origin)
dx = L / nx
dy = L / ny
mesh_ux = Penguin.Mesh((nx, ny), (L, L), (origin[1] - 0.5 * dx, origin[2]))
mesh_uy = Penguin.Mesh((nx, ny), (L, L), (origin[1], origin[2] - 0.5 * dy))
mesh_T = mesh_p

body = (x, y, _=0.0) -> -1.0

capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
capacity_T  = Capacity(body, mesh_T)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# Velocity boundary conditions (no-slip)
zero = Dirichlet((x, y, t=0.0) -> 0.0)
bc_ux = BorderConditions(Dict(
    :left=>zero, :right=>zero,
    :bottom=>zero, :top=>zero
))
bc_uy = BorderConditions(Dict(
    :left=>zero, :right=>zero,
    :bottom=>zero, :top=>zero
))
pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)

# Fluid setup ----------------------------------------------------------------
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

# Temperature boundary conditions -------------------------------------------
bc_T = BorderConditions(Dict(
    :left=>Dirichlet(T_hot),
    :right=>Dirichlet(T_cold),
))
bc_T_cut = Dirichlet(0.0)

# Initial temperature: linear profile between hot/cold walls
nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_scalar = Nx_T * Ny_T

T_init = Vector{Float64}(undef, 2 * N_scalar)
for j in 1:Ny_T
    y = nodes_Ty[j]
    for i in 1:Nx_T
        x = nodes_Tx[i]
        frac = (x - first(nodes_Tx)) / (last(nodes_Tx) - first(nodes_Tx))
        T_val = T_hot + (T_cold - T_hot) * frac
        idx = i + (j - 1) * Nx_T
        T_init[idx] = T_val
        T_init[idx + N_scalar] = T_val
    end
end

# Coupler --------------------------------------------------------------------
coupler = NavierStokesScalarCoupler(ns_solver,
                                    capacity_T,
                                    κ,
                                    (x, y, z=0.0, t=0.0) -> 0.0,
                                    bc_T,
                                    bc_T_cut;
                                    strategy=PicardCoupling(tol_T=1e-6, tol_U=1e-6, maxiter=12, relaxation=0.8),
                                    β=β,
                                    gravity=gravity,
                                    T_ref=0.0,
                                    T0=T_init,
                                    store_states=false)

println("=== Differentially heated cavity benchmark ===")
println("Grid: $nx × $ny, Ra=$Ra, Pr=$Pr")

solve_NavierStokesScalarCoupling_steady!(coupler; tol=1e-6, maxiter=25, method=Base.:\)

u_state = coupler.velocity_state
T_state = coupler.scalar_state

# Helper accessors ----------------------------------------------------------
data = Penguin.navierstokes2D_blocks(coupler.momentum)
nu_x = data.nu_x
nu_y = data.nu_y

uωx = view(u_state, 1:nu_x)
uωy = view(u_state, 2 * nu_x + 1:2 * nu_x + nu_y)

# Convert to grids -----------------------------------------------------------
Ux_grid = reshape(uωx, length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2]))
Uy_grid = reshape(uωy, length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2]))
T_grid = reshape(view(T_state, 1:N_scalar), Nx_T, Ny_T)

# Average hot-wall Nusselt number -------------------------------------------
Δx = nodes_Tx[2] - nodes_Tx[1]
Nu_hot = zeros(Float64, Ny_T)
for j in 1:Ny_T
    T1 = T_grid[j, 1]
    T2 = T_grid[j, 2]
    Nu_hot[j] = -(L / ΔT) * (T2 - T1) / Δx
end
Nu_mean = mean(Nu_hot[1:end])

# Velocity metrics (dimensionless velocities scaled by α / H) ---------------
idx_mid_x = findmin(abs.(mesh_ux.nodes[1] .- 0.5))[2]
idx_mid_y = findmin(abs.(mesh_uy.nodes[2] .- 0.5))[2]

u_line = abs.(Ux_grid[idx_mid_x, :])
v_line = abs.(Uy_grid[:, idx_mid_y])

v_mid_dimless = maximum(u_line[2:end-1]) / (α / L)
u_mid_dimless = maximum(v_line[2:end-1]) / (α / L)

# Reference comparisons (de Vahl Davis 1983, Ra=1e3, Pr=0.71)
Nu_ref = 1.116
u_ref = 3.634  # scaled by α / H
v_ref = 3.7

@assert abs(Nu_mean - Nu_ref) / Nu_ref ≤ 0.05 "Mean Nusselt deviates from reference."
@assert abs(u_mid_dimless - u_ref) / u_ref ≤ 0.1 "Max horizontal velocity deviates from reference."
@assert abs(v_mid_dimless - v_ref) / v_ref ≤ 0.1 "Max vertical velocity deviates from reference."

println("Benchmark passed against reference tolerances.")

@printf("Reference Nu = %.3f, computed Nu = %.3f\n", Nu_ref, Nu_mean)
@printf("Reference max |u| = %.3f, computed max |u| = %.3f\n", u_ref, u_mid_dimless)
@printf("Reference max |v| = %.3f, computed max |v| = %.3f\n", v_ref, v_mid_dimless)

if @isdefined CairoMakie
    xs = nodes_Tx
    ys = nodes_Ty
    fig = Figure(resolution=(900, 450))
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y",
              title="Temperature field ",
              aspect=DataAspect())
    hm = heatmap!(ax, xs, ys, T_grid'; colormap=:thermal)
    Colorbar(fig[1, 2], hm; label="T")

    ax_stream = Axis(fig[2, 1], xlabel="x", ylabel="y",
                     title="Velocity magnitude")
    speed = sqrt.(Ux_grid.^2 .+ Uy_grid.^2)
    hm2 = heatmap!(ax_stream, mesh_ux.nodes[1], mesh_ux.nodes[2], speed'; colormap=:viridis)
    Colorbar(fig[2, 2], hm2; label="|u|")

    display(fig)
end
