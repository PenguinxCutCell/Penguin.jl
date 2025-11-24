using Penguin
using LinearAlgebra
using Statistics
try
    using CairoMakie
catch
    @warn "CairoMakie not available; visualization disabled."
end

# ---------------------------------------------------------------------------
# Monolithic coupling demo: differentially heated cavity (Ra = 1e3, Pr = 0.71)
# ---------------------------------------------------------------------------
# This example revisits the classic cavity but solves each time step with the
# fully coupled Newton strategy (MonolithicCoupling). The coupling drives a
# single Newton solve for velocity, pressure, and temperature simultaneously.
# ---------------------------------------------------------------------------

# Geometry and mesh ---------------------------------------------------------
nx, ny = 128, 128
L = 1.0
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

# Physical parameters -------------------------------------------------------
Ra = 1.0e5
Pr = 0.71
ΔT = 1.0
T_hot = 0.5
T_cold = -0.5

ν = sqrt(Pr / Ra)
α = ν / Pr
β = 1.0
gravity = (-1.0, 0.0)

# Velocity boundary conditions ----------------------------------------------
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

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              ν, 1.0,
              (x,y,z=0,t=0)->0.0,
              (x,y,z=0,t=0)->0.0)

ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut)

# Temperature boundary conditions -------------------------------------------
bc_T = BorderConditions(Dict(
    :left=>Dirichlet(T_hot),
    :right=>Dirichlet(T_cold),
))
bc_T_cut = Dirichlet(0.0)

# Initial temperature: linear gradation ------------------------------------
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
        frac = (x - first(nodes_Tx)) / (last(nodes_Tx) - first(nodes_Tx))
        T_val = T_hot + (T_cold - T_hot) * frac
        idx = i + (j - 1) * Nx_T
        T_init_center[idx] = T_val
    end
end
T_init_interface = copy(T_init_center)
T_init = vcat(T_init_center, T_init_interface)
T_init .= 0.0

# Coupler with monolithic strategy ------------------------------------------
coupler = NavierStokesScalarCoupler(ns_solver,
                                    capacity_T,
                                    α,
                                    (x, y, z=0.0, t=0.0) -> 0.0,
                                    bc_T,
                                    bc_T_cut;
                                    strategy=MonolithicCoupling(tol=1e-6, maxiter=100, damping=0.2, verbose=true),
                                    β=β,
                                    gravity=gravity,
                                    T_ref=0.0,
                                    T0=T_init,
                                    store_states=false)

println("=== Monolithic coupling: differentially heated cavity ===")
println("Grid: $nx × $ny, Ra = $Ra, Pr = $Pr")

solve_NavierStokesScalarCoupling_steady!(coupler; tol=1e-6, maxiter=25, method=Base.:\)

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
println("Final max velocity magnitude ≈ ", maximum(speed))

# Hot-wall Nusselt number ----------------------------------------------------
Δx = nodes_Tx[2] - nodes_Tx[1]
Nu_profile = zeros(Float64, Ny_T)
for j in 1:Ny_T
    T1 = T_field[j, 1]
    T2 = T_field[j, 2]
    T3 = T_field[j, 3]
    grad = (-3 * T1 + 4 * T2 - T3) / (2Δx)
    Nu_profile[j] = -(L) * grad / (T_hot - T_cold)
end
Nu_mean = mean(Nu_profile)
println("Mean hot-wall Nusselt ≈ ", Nu_mean)


# Mid-column velocity profile ------------------------------------------------
idx_mid_x = findmin(abs.(mesh_ux.nodes[1] .- 0.5))[2]
idx_mid_y = findmin(abs.(mesh_uy.nodes[2] .- 0.5))[2]

u_line = abs.(Ux_grid[idx_mid_x, :])
v_line = abs.(Uy_grid[:, idx_mid_y])
println(u_line)
println(v_line)

v_mid_dimless = maximum(u_line[2:end-1]) / (α / L)
u_mid_dimless = maximum(v_line[2:end-1]) / (α / L)
println("Max vertical mid-column velocity (dimensionless) ≈ ", v_mid_dimless)
println("Max horizontal mid-row velocity (dimensionless) ≈ ", u_mid_dimless)

# Visualization --------------------------------------------------------------
if @isdefined CairoMakie
    xs = nodes_Tx
    ys = nodes_Ty

    fig = Figure(resolution=(960, 480))
    ax_T = Axis(fig[1, 1], xlabel="x", ylabel="y",
                title="Temperature ",
                aspect=DataAspect())
    hm = heatmap!(ax_T, xs, ys, T_field'; colormap=:thermal)
    Colorbar(fig[1, 2], hm; label="T")

    ax_u = Axis(fig[2, 1], xlabel="x", ylabel="y",
                title="Velocity magnitude",
                aspect=DataAspect())
    hm_u = heatmap!(ax_u, mesh_ux.nodes[1], mesh_ux.nodes[2], speed'; colormap=:viridis)
    Colorbar(fig[2, 2], hm_u; label="|u|")

    display(fig)
    save("differential_cavity_monolithic.png", fig)

end
