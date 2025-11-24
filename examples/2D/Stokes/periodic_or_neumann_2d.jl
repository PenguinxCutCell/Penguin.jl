using Penguin
using CairoMakie

"""
2D Stokes: Demonstrate Periodic and Neumann velocity boundary conditions.

Case A (Periodic-x Poiseuille):
- Velocity BCs: left/right periodic for both ux, uy; top/bottom no-slip (Dirichlet 0)
- Pressure: gauge only (no BCs) — drive with uniform body force fᵤx
Expected: ux is parabolic in y and independent of x; uy≈0; left/right strips identical

Case B (Neumann-x Poiseuille):
- Velocity BCs: left/right Neumann(0) for both ux, uy; top/bottom no-slip (Dirichlet 0)
- Pressure: gauge only; uniform body force mimics the pressure gradient
Expected: ux parabolic in y, uy≈0; zero x-gradient at both ends
"""

###########
# Common grid/ops
###########
nx, ny = 40, 24
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, 0.0

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx, dy = Lx / nx, Ly / ny
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

body = (x, y, _t=0.0) -> -1.0
cap_ux = Capacity(body, mesh_ux; method="ImplicitIntegration")
cap_uy = Capacity(body, mesh_uy; method="ImplicitIntegration")
cap_p  = Capacity(body, mesh_p;  method="ImplicitIntegration")

op_ux = DiffusionOps(cap_ux)
op_uy = DiffusionOps(cap_uy)
op_p  = DiffusionOps(cap_p)

μ = 1.0
ρ = 1.0

###########
# Case A: Periodic-x Poiseuille (driven by body force)
###########
f0 = 1.0
fᵤ_A = (x, y, z=0.0) -> f0
fₚ_A = (x, y, z=0.0) -> 0.0

bc_ux_A = BorderConditions(Dict(
    :left=>Periodic(), :right=>Periodic(),
    :bottom=>Dirichlet(0.0), :top=>Dirichlet(0.0)
))
bc_uy_A = BorderConditions(Dict(
    :left=>Periodic(), :right=>Periodic(),
    :bottom=>Dirichlet(0.0), :top=>Dirichlet(0.0)
))
pressure_gauge_A = MeanPressureGauge()
bc_cut = Dirichlet(0.0)

fluid_A = Fluid((mesh_ux, mesh_uy),
                (cap_ux, cap_uy),
                (op_ux, op_uy),
                mesh_p, cap_p, op_p,
                μ, ρ, fᵤ_A, fₚ_A)

solver_A = StokesMono(fluid_A, bc_ux_A, bc_uy_A; pressure_gauge=pressure_gauge_A, bc_cut=bc_cut)
solve_StokesMono!(solver_A; method=Base.:\)

nu_x = prod(op_ux.size)
nu_y = prod(op_uy.size)
np   = prod(op_p.size)

uωx_A = solver_A.x[1:nu_x]
uωy_A = solver_A.x[2*nu_x + 1 : 2*nu_x + nu_y]
pω_A  = solver_A.x[2*(nu_x+nu_y) + 1 : 2*(nu_x+nu_y) + np]

Ux_A = reshape(uωx_A, op_ux.size)
Uy_A = reshape(uωy_A, op_uy.size)
P_A  = reshape(pω_A,  op_p.size)

println("[Case A] Periodic-x: |Uy|_max = ", maximum(abs, Uy_A))
println("[Case A] Periodic-x: periodic check max |Ux[:,1] - Ux[:,end-1]| = ", maximum(abs.(Ux_A[:,1] .- Ux_A[:,end-1])))

# Plot slice of ux vs y at mid-x
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
ix_mid = Int(clamp(round(length(xs)/2), 1, length(xs)))

figA = Figure(resolution=(800,400))
axA1 = Axis(figA[1,1], title="Case A: Ux field")
hmA = heatmap!(axA1, xs, ys, Ux_A'; colormap=:plasma)
Colorbar(figA[1,2], hmA)

axA2 = Axis(figA[1,3], title="Case A: Ux(y) at x=mid", xlabel="y", ylabel="Ux")
lines!(axA2, ys, Ux_A[ix_mid, :])

save("stokes2d_periodic_x.png", figA)
println("Saved stokes2d_periodic_x.png")

###########
# Case B: Neumann-x Poiseuille (driven by body force)
###########
ΔP_B = 10.0
gradP_B = ΔP_B / Lx
fᵤ_B = (x, y, z=0.0) -> gradP_B
fₚ_B = (x, y, z=0.0) -> 0.0

bc_ux_B = BorderConditions(Dict(
    :left=>Neumann(0.0), :right=>Neumann(0.0),
    :bottom=>Dirichlet(0.0), :top=>Dirichlet(0.0)
))
bc_uy_B = BorderConditions(Dict(
    :left=>Neumann(0.0), :right=>Neumann(0.0),
    :bottom=>Dirichlet(0.0), :top=>Dirichlet(0.0)
))

pressure_gauge_B = MeanPressureGauge()

fluid_B = Fluid((mesh_ux, mesh_uy),
                (cap_ux, cap_uy),
                (op_ux, op_uy),
                mesh_p, cap_p, op_p,
                μ, ρ, fᵤ_B, fₚ_B)

solver_B = StokesMono(fluid_B, bc_ux_B, bc_uy_B; pressure_gauge=pressure_gauge_B, bc_cut=bc_cut)
solve_StokesMono!(solver_B; method=Base.:\)

uωx_B = solver_B.x[1:nu_x]
uωy_B = solver_B.x[2*nu_x + 1 : 2*nu_x + nu_y]
pω_B  = solver_B.x[2*(nu_x+nu_y) + 1 : 2*(nu_x+nu_y) + np]

Ux_B = reshape(uωx_B, op_ux.size)
Uy_B = reshape(uωy_B, op_uy.size)
P_B  = reshape(pω_B,  op_p.size)

println("[Case B] Neumann-x: |Uy|_max = ", maximum(abs, Uy_B))

figB = Figure(resolution=(800,400))
axB1 = Axis(figB[1,1], title="Case B: Ux field")
hmB = heatmap!(axB1, xs, ys, Ux_B'; colormap=:plasma)
Colorbar(figB[1,2], hmB)

axB2 = Axis(figB[1,3], title="Case B: Ux(y) at x=mid", xlabel="y", ylabel="Ux")
lines!(axB2, ys, Ux_B[ix_mid, :])

save("stokes2d_neumann_x.png", figB)
println("Saved stokes2d_neumann_x.png")
