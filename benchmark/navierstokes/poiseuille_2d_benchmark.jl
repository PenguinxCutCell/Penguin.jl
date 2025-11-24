using Penguin
using CairoMakie
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Statistics

"""
2D Navier–Stokes Poiseuille benchmark (steady) with cut-cell channel

Adapted from examples/2D/Stokes/poiseuille_2d_cut.jl but uses the
Navier–Stokes steady solver. The script computes profile errors against
an analytical parabolic profile on the mid-column and prints diagnostics.
"""

###########
# Grids
###########
nx, ny = 64, 64
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, 0.0

# Channel geometry (walls at y = 0.2 and y = 0.8)
y_wall_bot = 0.2
y_wall_top = 0.8
channel_height = y_wall_top - y_wall_bot

# Body function: distance to channel walls (negative = fluid, positive = solid)
body = (x, y, _=0) -> begin
    if y < y_wall_bot
        return y_wall_bot - y   # positive outside below
    elseif y > y_wall_top
        return y - y_wall_top   # positive outside above
    else
        return -min(y - y_wall_bot, y_wall_top - y)
    end
end

# Pressure grid (cell-centered)
mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))

# Component-wise staggered velocity grids
dx, dy = Lx/nx, Ly/ny
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

###########
# Capacities and operators
###########
capacity_ux = Capacity(body, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(body, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(body, mesh_p; compute_centroids=false)
operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# BCs
###########
Umax = 1.0
# Parabolic profile for the channel (referenced to channel walls)
parabola = (x, y) -> begin
    if y < y_wall_bot || y > y_wall_top
        return 0.0
    else
        y_local = y - y_wall_bot
        return 4Umax * y_local * (channel_height - y_local) / (channel_height^2)
    end
end

ux_left  = Dirichlet(parabola)
ux_right = Dirichlet(parabola)
ux_bot   = Dirichlet((x, y)-> 0.0)
ux_top   = Dirichlet((x, y)-> 0.0)
bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right, :bottom=>ux_bot, :top=>ux_top
))

uy_zero = Dirichlet((x, y)-> 0.0)
bc_uy = BorderConditions(Dict(:left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero))

# Pressure gauge
pressure_gauge = PinPressureGauge()

# Cut-cell / interface BC for uγ (no-slip on channel walls)
u_bc = Dirichlet(0.0)

###########
# Sources and material
###########
fᵤ = (x, y, z=0.0) -> 0.0
fₚ = (x, y, z=0.0) -> 0.0
μ = 1.0
ρ = 1.0

# Fluid with per-component (ux, uy) meshes/capacities/operators
fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

###########
# Initial guess
###########
nu = prod(operator_ux.size)
np = prod(operator_p.size)
x0 = zeros(4*nu + np)

###########
# Solver and solve (Navier–Stokes steady)
###########
solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, u_bc; x0=x0)
_, iters, res = solve_NavierStokesMono_steady!(solver; tol=1e-10, maxiter=200, relaxation=1.0)
println("Navier–Stokes steady solve finished: iterations=", iters, ", residual=", res)

# Extract components
uωx = solver.x[1:nu]; uγx = solver.x[nu+1:2nu]
uωy = solver.x[2nu+1:3nu]; uγy = solver.x[3nu+1:4nu]
pω  = solver.x[4nu+1:end]

###########
# Post-processing: mid-column profile and error checks
###########
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2] .+ 0.5*dy  # mid-column y-locations for ux
LIux = LinearIndices((length(xs), length(ys)))

# Mid-column index
icol = Int(cld(length(xs), 2))
ux_profile = [uωx[LIux[icol, j]] for j in 1:length(ys)]
ux_analytical = [parabola(0.0, y) for y in ys]

# Trim to fluid region
fluid_indices = findall(y -> y_wall_bot <= y <= y_wall_top, ys)
profile_err = ux_profile[fluid_indices] .- ux_analytical[fluid_indices]
ℓ2_profile = sqrt(sum(abs2, profile_err) / length(profile_err))
ℓinf_profile = maximum(abs, profile_err[2:end-1])  # ignore boundary points

println("Profile L2 error (fluid region) = ", ℓ2_profile, ", Linf = ", ℓinf_profile)

# Relative centerline error
j_center = argmin(abs.(ys .- (y_wall_bot + y_wall_top)/2))
ux_centerline = ux_profile[j_center]
ux_analytical_centerline = parabola(0.0, ys[j_center])
rel_center_err = abs(ux_centerline - ux_analytical_centerline) / (abs(ux_analytical_centerline) + eps())
println("Relative centerline error = ", rel_center_err)

# Simple checks
println("checks:")
println("- profile Linf < 1e-2? ", ℓinf_profile < 1e-2)
println("- centerline relative error < 1e-3? ", rel_center_err < 1e-2)
println("- residual < 1e-10? ", res < 1e-10)

# Optionally plot results (commented for CI)
fig = Figure(size=(1400,400))
ax1 = Axis(fig[1,1], xlabel="u_x", ylabel="y", title="Mid-column profile")
lines!(ax1, ux_profile, ys, label="numerical")
lines!(ax1, ux_analytical, ys, color=:red, linestyle=:dash, label="analytical")
axislegend(ax1, position=:rb)
display(fig)
# save("benchmark_poiseuille2d_profile.png", fig)

println("benchmark poiseuille 2D completed.")
