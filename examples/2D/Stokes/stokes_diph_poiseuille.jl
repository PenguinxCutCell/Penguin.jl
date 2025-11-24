using Penguin
using LinearSolve
using LinearAlgebra
using CairoMakie


# Geometry
Nx, Ny = 32, 32
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, -Ly/2

mesh_p  = Penguin.Mesh((Nx, Ny), (Lx, Ly), (x0, y0))
Δx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
Δy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((Nx, Ny), (Lx, Ly), (x0 - 0.5*Δx, y0))
mesh_uy = Penguin.Mesh((Nx, Ny), (Lx, Ly), (x0, y0 - 0.5*Δy))

# Level sets for each phase
interface_y = 0.0
fluid1_body = (x, y, _t=0.0) -> y - interface_y
fluid2_body = (x, y, _t=0.0) -> interface_y - y

cap_ux₁ = Capacity(fluid1_body, mesh_ux; compute_centroids=false)
cap_uy₁ = Capacity(fluid1_body, mesh_uy; compute_centroids=false)
cap_p₁  = Capacity(fluid1_body, mesh_p; compute_centroids=false)

cap_ux₂ = Capacity(fluid2_body, mesh_ux; compute_centroids=false)
cap_uy₂ = Capacity(fluid2_body, mesh_uy; compute_centroids=false)
cap_p₂  = Capacity(fluid2_body, mesh_p; compute_centroids=false)

op_ux1 = DiffusionOps(cap_ux₁)
op_uy1 = DiffusionOps(cap_uy₁)
op_p1  = DiffusionOps(cap_p₁)
op_ux2 = DiffusionOps(cap_ux₂)
op_uy2 = DiffusionOps(cap_uy₂)
op_p2  = DiffusionOps(cap_p₂)

fluid1 = Fluid((mesh_ux, mesh_uy),
               (cap_ux₁, cap_uy₁),
               (op_ux1, op_uy1),
               mesh_p,
               cap_p₁,
               op_p1,
               1.0, 1.0,
               (x,y,_=0) -> 0.0,
               (x,y,_=0) -> 0.0)

fluid2 = Fluid((mesh_ux, mesh_uy),
               (cap_ux₂, cap_uy₂),
               (op_ux2, op_uy2),
               mesh_p,
               cap_p₂,
               op_p2,
               1.0, 1.0,
               (x,y,_=0) -> 0.0,
               (x,y,_=0) -> 0.0)

Umax = 1.0
parabolic = (x,y) -> begin
    ξ = (y - (y0 + Ly/2)) / (Ly/2)
    Umax * (1 - ξ^2)
end

inlet = Dirichlet((x,y) -> parabolic(x,y))
outlet = Outflow()
noslip = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(:left=>inlet, :right=>outlet, :bottom=>noslip, :top=>noslip))
bc_uy = BorderConditions(Dict(:left=>noslip, :right=>noslip, :bottom=>noslip, :top=>noslip))

ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0),
                         FluxJump(1.0, 1.0, 0.0))

nu_x = prod(op_ux1.size)
nu_y = prod(op_uy1.size)
np = prod(op_p1.size)
xlen = 2 * (2*(nu_x + nu_y) + np)
x0 = zeros(xlen)

solver = StokesDiph(fluid1, fluid2, (bc_ux, bc_uy), (bc_ux, bc_uy), ic;
                    pressure_gauge_a=PinPressureGauge(), pressure_gauge_b=PinPressureGauge(),
                    x0=x0)

solve_StokesDiph!(solver; algorithm=UMFPACKFactorization())

println("Diphasic Poiseuille solved. Unknowns = ", length(solver.x))

off_u1ωx = 0
off_u1γx = nu_x
off_u1ωy = 2 * nu_x
off_u1γy = 2 * nu_x + nu_y
off_p1   = 2 * (nu_x + nu_y)

base2 = off_p1 + np
off_u2ωx = base2
off_u2ωy = base2 + 2 * nu_x

u1x = solver.x[off_u1ωx+1:off_u1ωx+nu_x]
u2x = solver.x[off_u2ωx+1:off_u2ωx+nu_x]

println("Phase 1 velocity range: ", extrema(u1x))
println("Phase 2 velocity range: ", extrema(u2x))

# Simple heatmap of u_x at uωx locations
ux_grid = reshape(u1x, length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2]))
fig = Figure(resolution=(800,400))
ax = Axis(fig[1,1], title="u_x", xlabel="x", ylabel="y")
hm = heatmap!(ax, mesh_ux.nodes[1], mesh_ux.nodes[2], ux_grid; colormap=:plasma)
Colorbar(fig[1,2], hm; label="u_x")

save("stokes_diph_poiseuille_velocity.png", fig)
println("Saved stokes_diph_poiseuille_velocity.png")

# Mid-column profile of averaged u_x
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
icol = Int(cld(length(xs), 2))
profile = ux_grid[icol, :]
fig_profile = Figure(resolution=(500,400))
axp = Axis(fig_profile[1,1], title="u_x profile at x[mid]", xlabel="u_x", ylabel="y")
lines!(axp, profile, ys)
scatter!(axp, profile, ys; markersize=4)
save("stokes_diph_poiseuille_profile.png", fig_profile)
println("Saved stokes_diph_poiseuille_profile.png at column ", icol)