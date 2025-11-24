using Penguin
using LinearAlgebra

"""
Hydrostatic stratification check.

- Domain: square cavity with no-slip, adiabatic walls.
- Prescribed linear temperature distribution T(y) = T₀ + ΔT * y / H.
- Buoyancy enabled, but the pressure field should balance it so velocity stays zero.
- Checks: ‖u‖₂ ≤ 1e-12 and linear system residual ≤ 1e-12.
"""

nx, ny = 32, 32
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

ΔT = 0.2
T0 = 0.5
linear_profile = (y) -> T0 + ΔT * (y - origin[2]) / height

bc_T = BorderConditions(Dict(
))
bc_T_cut = Dirichlet(0.0)

nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
N_scalar = Nx_T * Ny_T

initial_scalar = Vector{Float64}(undef, 2 * N_scalar)
for j in 1:Ny_T
    y = nodes_Ty[j]
    val = linear_profile(y)
    for i in 1:Nx_T
        idx = i + (j - 1) * Nx_T
        initial_scalar[idx] = val
        initial_scalar[idx + N_scalar] = val
    end
end

coupler = NavierStokesScalarCoupler(ns_solver,
                                    capacity_T,
                                    0.0,
                                    (x, y, z=0.0, t=0.0) -> 0.0,
                                    bc_T,
                                    bc_T_cut;
                                    strategy=PassiveCoupling(),
                                    β=1.0,
                                    gravity=(0.0, -1.0),
                                    T_ref=T0,
                                    T0=initial_scalar,
                                    store_states=false)

Δt = 1.0e-3
T_end = 5.0e-3

println("=== Hydrostatic stratification check ===")
println("Grid: $nx × $ny, dt=$Δt, T_end=$T_end")

solve_NavierStokesScalarCoupling!(coupler; Δt=Δt, T_end=T_end, scheme=:CN)

data = Penguin.navierstokes2D_blocks(coupler.momentum)
nu_x = data.nu_x
nu_y = data.nu_y
u_vec = vcat(view(coupler.velocity_state, 1:nu_x),
             view(coupler.velocity_state, 2 * nu_x + 1:2 * nu_x + nu_y))

vel_norm = norm(u_vec)
println("Velocity L2 norm: ", vel_norm)
@assert vel_norm ≤ 1e-12 "Hydrostatic test failed: non-zero velocity."

residual = coupler.momentum.A * coupler.velocity_state - coupler.momentum.b
res_norm = norm(residual)
println("Momentum linear system residual: ", res_norm)
@assert res_norm ≤ 1e-12 "Hydrostatic test failed: residual too large."
