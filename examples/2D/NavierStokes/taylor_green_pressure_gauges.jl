using Penguin
using CairoMakie
using LinearAlgebra
using IterativeSolvers
"""
Compare mass residual (div u) for `PinPressureGauge` vs `MeanPressureGauge`
on a Taylor-Green-like velocity field. Domain has no immersed body (`body=-1`).
"""

# Taylor-Green snapshot for boundary data
tg_u(x, y, t=0.0) =  sin(x) * cos(y)
tg_v(x, y, t=0.0) = -cos(x) * sin(y)

# Grid and meshes
nx, ny = 32, 32
Lx, Ly = 2 * pi, 2 * pi
x0, y0 = 0.0, 0.0

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = Lx / nx
dy = Ly / ny
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

# No body: fluid everywhere
body = (x, y, _t=0.0) -> -1.0
cap_ux = Capacity(body, mesh_ux; method="VOFI")
cap_uy = Capacity(body, mesh_uy; method="VOFI")
cap_p  = Capacity(body, mesh_p;  method="VOFI")

op_ux = DiffusionOps(cap_ux)
op_uy = DiffusionOps(cap_uy)
op_p  = DiffusionOps(cap_p)

bc_ux = BorderConditions(Dict(
    :left=>Dirichlet(tg_u), :right=>Dirichlet(tg_u),
    :bottom=>Dirichlet(tg_u), :top=>Dirichlet(tg_u)
))
bc_uy = BorderConditions(Dict(
    :left=>Dirichlet(tg_v), :right=>Dirichlet(tg_v),
    :bottom=>Dirichlet(tg_v), :top=>Dirichlet(tg_v)
))

μ = 1.0
ρ = 1.0
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0
bc_cut = Dirichlet(0.0)

fluid = Fluid((mesh_ux, mesh_uy),
              (cap_ux, cap_uy),
              (op_ux, op_uy),
              mesh_p, cap_p, op_p,
              μ, ρ, fᵤ, fₚ)

nu_x = prod(op_ux.size)
nu_y = prod(op_uy.size)
np = prod(op_p.size)

function pin_index(gauge::AbstractPressureGauge, cap_p)
    gauge isa PinPressureGauge || return nothing
    idx = gauge.index
    if idx === nothing
        diagV = diag(cap_p.V)
        tol = 1e-12
        idx = findfirst(x -> x > tol, diagV)
        idx === nothing && (idx = 1)
    end
    return idx
end

function run_gauge(gauge::AbstractPressureGauge)
    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), gauge, bc_cut)
    iters, res = solve_NavierStokesMono_steady!(solver; method=Base.:\,
                                                nlsolve_method=:picard, tol=1e-10, maxiter=100, relaxation=1.0)
    data = Penguin.navierstokes2D_blocks(solver)

    uωx = solver.x[1:nu_x]
    uγx = solver.x[nu_x+1:2nu_x]
    uωy = solver.x[2nu_x+1:2nu_x+nu_y]
    uγy = solver.x[2nu_x+nu_y+1:2*(nu_x+nu_y)]

    mass_residual = data.div_x_ω * Vector{Float64}(uωx) +
                    data.div_x_γ * Vector{Float64}(uγx) +
                    data.div_y_ω * Vector{Float64}(uωy) +
                    data.div_y_γ * Vector{Float64}(uγy)

    return solver, mass_residual, iters, res
end

pin_gauge = PinPressureGauge()
mean_gauge = MeanPressureGauge()

pin_solver, pin_mass, pin_iters, pin_res = run_gauge(pin_gauge)
mean_solver, mean_mass, mean_iters, mean_res = run_gauge(mean_gauge)

pin_idx = pin_index(pin_gauge, cap_p)
if pin_idx !== nothing
    pin_sub = CartesianIndices(op_p.size)[pin_idx]
    pin_center = (mesh_p.centers[1][pin_sub[1]], mesh_p.centers[2][pin_sub[2]])
    println("PinPressureGauge pins pressure at DOF ", pin_idx, " (i,j)=", Tuple(pin_sub), " center=", pin_center)
end

println("PinPressureGauge: iters=", pin_iters, " res=", pin_res, " max|div u|=", maximum(abs, pin_mass))
println("MeanPressureGauge: iters=", mean_iters, " res=", mean_res, " max|div u|=", maximum(abs, mean_mass))

pin_mass_mat = reshape(pin_mass, op_p.size)
mean_mass_mat = reshape(mean_mass, op_p.size)

xs = mesh_p.nodes[1]
ys = mesh_p.nodes[2]
vmax = maximum(abs, vcat(pin_mass, mean_mass))
vmax = vmax == 0 ? 1e-12 : vmax

fig = Figure(resolution=(1000, 420))
ax1 = Axis(fig[1,1], title="PinPressureGauge: div(u)", xlabel="x", ylabel="y")
hm1 = heatmap!(ax1, xs, ys, pin_mass_mat'; colormap=:balance, colorrange=(-vmax, vmax))
if pin_idx !== nothing
    scatter!(ax1, [pin_center[1]], [pin_center[2]]; color=:black, markersize=10, marker=:cross)
end
Colorbar(fig[1,2], hm1)

ax2 = Axis(fig[1,3], title="MeanPressureGauge: div(u)", xlabel="x")
hm2 = heatmap!(ax2, xs, ys, mean_mass_mat'; colormap=:balance, colorrange=(-vmax, vmax))
Colorbar(fig[1,4], hm2)

save("taylor_green_gauge_divu_border.png", fig)
println("Saved taylor_green_gauge_divu_border.png")
