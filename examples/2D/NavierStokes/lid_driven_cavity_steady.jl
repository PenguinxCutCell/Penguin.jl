using Penguin
using CairoMakie
using LinearAlgebra
using DelimitedFiles

"""
Steady lid-driven cavity solved with either Picard or Newton nonlinear solver
for the Navier–Stokes equations. The setup mirrors the classic benchmark: 
unit square domain, no-slip walls, and a unit tangential velocity at the lid. 

Two solver options are available:
- Picard iteration: Linearizes convection using previous iterate (successive substitution)
- Newton method: Full Newton-Raphson with exact Jacobian (quadratic convergence)

The convection operator uses skew-symmetric flux form; viscosity is implicit.
Switch between methods by changing the nlsolve_method parameter below.
"""

###########
# Geometry
###########
nx, ny = 64, 64
Lx, Ly = 1.0, 1.0
x0, y0 = 0.0, 0.0

###########
# Meshes
###########
mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

###########
# Capacities & operators
###########
# Negative body value fills the entire domain.
full_body = (x, y, _=0.0) -> -1.0

capacity_ux = Capacity(full_body, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(full_body, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(full_body, mesh_p; compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
lid_speed = 100.0

ux_top    = Dirichlet((x, y, t=0.0) -> lid_speed)
ux_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
ux_left   = Dirichlet((x, y, t=0.0) -> 0.0)
ux_right  = Dirichlet((x, y, t=0.0) -> 0.0)

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left,
    :right=>ux_right,
    :bottom=>ux_bottom,
    :top=>ux_top
))

bc_uy = BorderConditions(Dict(
    :left=>uy_zero,
    :right=>uy_zero,
    :bottom=>uy_zero,
    :top=>uy_zero
))

pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)

###########
# Fluid properties and forcing
###########
Re = 100.0
μ = 1.0
ρ = 1.0
fᵤ = (x, y, z=0.0) -> 0.0
fₚ = (x, y, z=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

###########
# Solver setup
###########
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)

x0_vec = zeros(2 * (nu_x + nu_y) + np)

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut; x0=x0_vec)

tol = 1e-9
maxiter = 40
relaxation = 0.7

println("Refining solution with Newton iterations")
_, newton_iters, newton_res = solve_NavierStokesMono_steady!(solver; 
                                                            tol=1e-10,
                                                            maxiter=20,
                                                            nlsolve_method=:newton)
println("Newton iterations completed: iters=$(newton_iters), residual=$(newton_res)")



###########
# Diagnostics
###########
data = Penguin.navierstokes2D_blocks(solver)

nu_x = data.nu_x
nu_y = data.nu_y

# Extract velocity components
uωx = solver.x[1:nu_x]
uγx = solver.x[nu_x+1:2nu_x]
uωy = solver.x[2nu_x+1:2nu_x+nu_y]
uγy = solver.x[2nu_x+nu_y+1:2*(nu_x+nu_y)]

mass_residual = data.div_x_ω * Vector{Float64}(uωx) + data.div_x_γ * Vector{Float64}(uγx) +
                 data.div_y_ω * Vector{Float64}(uωy) + data.div_y_γ * Vector{Float64}(uγy)

println(mass_residual)
println("‖div(u)‖∞ = $(maximum(abs, mass_residual))")

ke = 0.5 * sum(abs2, uωx) + 0.5 * sum(abs2, uωy)
println("Kinetic energy = $(ke)")



###########
# Basic diagnostics
###########
uωx = solver.x[1:nu_x]
uωy = solver.x[2nu_x+1:2nu_x+nu_y]
pω  = solver.x[2*(nu_x + nu_y)+1:end]

println("Velocity magnitudes: max(|u_x|)=$(maximum(abs, uωx)), max(|u_y|)=$(maximum(abs, uωy))")
println("Pressure range: min=$(minimum(pω)), max=$(maximum(pω))")


###########
# Visualization (CairoMakie)
###########
Xs = mesh_p.nodes[1]
Ys = mesh_p.nodes[2]
xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]

# reshape into 2D fields (x fast index, y slow index matching other examples)
Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))
P  = reshape(pω, (length(Xs), length(Ys)))

speed = sqrt.(Ux.^2 .+ Uy.^2)

nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))
velocity_field(x, y) = Point2f(Ux[nearest_index(xs, x), nearest_index(ys, y)],
                               Uy[nearest_index(xs, x), nearest_index(ys, y)])

fig = Figure(resolution=(1200, 700))

ax_speed = Axis(fig[1,1], xlabel="x", ylabel="y", title="Speed magnitude (Navier–Stokes)")
hm_speed = heatmap!(ax_speed, xs, ys, speed; colormap=:viridis)
Colorbar(fig[1,2], hm_speed)

ax_pressure = Axis(fig[1,3], xlabel="x", ylabel="y", title="Pressure field")
hm_pressure = heatmap!(ax_pressure, Xs, Ys, P; colormap=:balance)
Colorbar(fig[1,4], hm_pressure)

ax_stream = Axis(fig[2,1], xlabel="x", ylabel="y", title="Velocity streamlines")
streamplot!(ax_stream, velocity_field, xs[1]..xs[end], ys[1]..ys[end]; colormap=:thermal)

ax_contours = Axis(fig[2,3], xlabel="x", ylabel="y", title="Velocity magnitude contours")
contour!(ax_contours, xs, ys, speed; levels=20, color=:navy)

display(fig)
save("navierstokes2d_lid_driven_cavity_speed.png", fig)

# Centerline profiles (exclude the last point which may be a ghost/duplicate)
icol = nearest_index(xs, x0 + Lx/2)
row  = nearest_index(ys, y0 + Ly/2)

# Trim the final entry from the coordinate vectors and profile data
ys_vis = ys[1:end-1]
xs_vis = xs[1:end-1]
u_center_vertical = Ux[icol, 1:end-1]
v_center_horizontal = Uy[1:end-1, row]


fig_profiles = Figure(resolution=(800, 350))
ax_vert = Axis(fig_profiles[1,1], xlabel="u_x", ylabel="y",
               title="Vertical centerline u_x(x=0.5)")
lines!(ax_vert, u_center_vertical, ys_vis, label="Numerical")

ax_horiz = Axis(fig_profiles[1,2], xlabel="x", ylabel="u_y",
                title="Horizontal centerline u_y(y=0.5)")
lines!(ax_horiz, xs_vis, v_center_horizontal, label="Numerical")

axislegend(ax_vert, position=:rb)
axislegend(ax_horiz, position=:rb)
display(fig_profiles)
save("navierstokes2d_lidcavity_profile.png", fig_profiles)

# Write profiles to CSV: two files (y,u_x) and (x,u_y)
try
    writedlm("navierstokes2d_lidcavity_vertical_profile.csv", hcat(ys_vis, u_center_vertical), ',')
    writedlm("navierstokes2d_lidcavity_horizontal_profile.csv", hcat(xs_vis, v_center_horizontal), ',')
    println("Wrote centerline profiles to CSV files")
catch e
    @warn "Failed to write CSV profiles" exception=(e, catch_backtrace())
end

# Convergence plot for both methods
nlsolve_method = :picard
if !isempty(solver.residual_history)
    fig_conv = Figure(resolution=(600, 400))
    method_name = nlsolve_method == :picard ? "Picard" : "Newton"
    ax_conv = Axis(fig_conv[1,1], xlabel="$(method_name) iteration", ylabel="Residual max|Δu|", 
                   title="$(method_name) Convergence", yscale=log10)
    lines!(ax_conv, 1:length(solver.residual_history), solver.residual_history, linewidth=2)
    scatter!(ax_conv, 1:length(solver.residual_history), solver.residual_history, color=:red, markersize=8)
    display(fig_conv)
    
    # Save with method-specific filename
    if nlsolve_method == :picard
        save("navierstokes2d_picard_convergence.png", fig_conv)
    else
        save("navierstokes2d_newton_convergence.png", fig_conv)
    end
end
