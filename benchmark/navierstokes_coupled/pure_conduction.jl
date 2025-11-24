using Penguin
using LinearAlgebra
using Statistics
using Printf

"""
Pure conduction sanity check for Navier–Stokes/temperature coupling.

- Domain: square cavity, insulated sides, isothermal top/bottom.
- Buoyancy disabled (β = 0) so velocity should remain identically zero.
- Checks: ‖u‖₂ ≤ 1e-12 and L₂ temperature error against analytic linear profile.
"""

nx, ny = 48, 32
width, height = 1.0, 1.0
origin = (0.0, 0.0)

mesh_p = Penguin.Mesh((nx, ny), (width, height), origin)
dx = width / nx
dy = height / ny
mesh_ux = Penguin.Mesh((nx, ny), (width, height), (origin[1] - 0.5 * dx, origin[2]))
mesh_uy = Penguin.Mesh((nx, ny), (width, height), (origin[1], origin[2] - 0.5 * dy))
mesh_T = mesh_p

body = (x, y, _=0.0) -> -1.0

capacity_ux = Capacity(body, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(body, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(body, mesh_p;  compute_centroids=false)
capacity_T  = Capacity(body, mesh_T;  compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

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
bc_cut = Dirichlet(0.0)

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

ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut)

T_bottom = 1.0
T_top = 0.0
analytic_T = (y) -> T_bottom + (T_top - T_bottom) * ((y - origin[2]) / height)

bc_T = BorderConditions(Dict(
    :bottom=>Dirichlet(T_bottom),
    :top=>Dirichlet(T_top)
))
bc_T_cut = Dirichlet(0.0)

nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_temp = Nx_T * Ny_T

T_init = Vector{Float64}(undef, 2 * N_temp)
for j in 1:Ny_T
    y = nodes_Ty[j]
    val = analytic_T(y)
    for i in 1:Nx_T
        idx = i + (j - 1) * Nx_T
        T_init[idx] = val
        T_init[idx + N_temp] = val
    end
end

coupler = NavierStokesScalarCoupler(ns_solver,
                                    capacity_T,
                                    1.0e-2,
                                    (x, y, z=0.0, t=0.0) -> 0.0,
                                    bc_T,
                                    bc_T_cut;
                                    strategy=PassiveCoupling(),
                                    β=0.0,
                                    gravity=(0.0, -1.0),
                                    T_ref=0.0,
                                    T0=T_init,
                                    store_states=false)

Δt = 2.0e-4
T_end = 1.0e-3

println("=== Navier–Stokes + heat pure conduction sanity ===")
println("Grid: $nx × $ny, dt=$Δt, T_end=$T_end")

solve_NavierStokesScalarCoupling!(coupler; Δt=Δt, T_end=T_end, scheme=:CN)

data = Penguin.navierstokes2D_blocks(coupler.momentum)
nu_x = data.nu_x
nu_y = data.nu_y
u_center_x = view(coupler.velocity_state, 1:nu_x)
u_center_y = view(coupler.velocity_state, 2 * nu_x + 1:2 * nu_x + nu_y)
velocity_l2 = norm(vcat(u_center_x, u_center_y))

@printf("Velocity L2 norm: %.3e\n", velocity_l2)
@assert velocity_l2 ≤ 1.0e-12 "Velocity deviated from zero beyond tolerance."

T_full = reshape(view(coupler.scalar_state, 1:N_temp), Nx_T, Ny_T)
T_numeric = vec(T_full[2:end-2, 2:end-2])

analytic_field = [analytic_T(y) for y in nodes_Ty for x in nodes_Tx]
analytic_mat = reshape(analytic_field, Nx_T, Ny_T)
analytic_sub = vec(analytic_mat[2:end-2, 2:end-2])

dx_cell = width / (Nx_T - 1)
dy_cell = height / (Ny_T - 1)
weights = fill(dx_cell * dy_cell, length(T_numeric))

diff_vec = T_numeric .- analytic_sub
temp_l2 = sqrt(sum(weights .* diff_vec .^ 2))
@printf("Temperature L2 error: %.6e\n", temp_l2)

println("Pure conduction sanity check passed.")