using Penguin
using CairoMakie
using LinearAlgebra
using LinearSolve
using VTKOutputs

"""
Steady Stokes flow past a spherical obstacle embedded in a 3D channel.
The sphere is represented implicitly via a level-set function feeding the
cut-cell capacity construction. The interface (cut) boundary is given a
homogeneous Dirichlet velocity (no-slip) enforcing u^γ = 0.

Domain: rectangular box of length Lx and square cross-section Ly×Lz.
Inlet: uniform plug profile in x-component, zero for y,z. Outlet: outflow.
Walls: no-slip.
Sphere: centered, radius R.
"""

###########
# Geometry
###########
# Grid & domain (center sphere inside channel)
Nx, Ny, Nz = 32, 32, 32
Lx, Ly, Lz = 16.0, 16.0, 16.0
x0, y0, z0 = -Lx/2, -Ly/2, -Lz/2   # domain centered at origin

sphere_center = (0.0, 0.0, 0.0)
R = 1.0
###########
sphere_body = (x, y, z, _t=0.0) -> -(sqrt((x - sphere_center[1])^2 +
                             (y - sphere_center[2])^2 +
                             (z - sphere_center[3])^2) - R)

###########
# Meshes (staggered velocity, collocated pressure)
###########
mesh_p  = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0, z0))
Δx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
Δy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
Δz = mesh_p.nodes[3][2] - mesh_p.nodes[3][1]
mesh_ux = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0 - 0.5*Δx, y0, z0))
mesh_uy = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0 - 0.5*Δy, z0))
mesh_uz = Penguin.Mesh((Nx, Ny, Nz), (Lx, Ly, Lz), (x0, y0, z0 - 0.5*Δz))

###########
# Capacities & Operators
###########
println("capacity ux...")
cap_ux = Capacity(sphere_body, mesh_ux; method="VOFI", integration_method=:vofijul)
println("capacity uy...")
cap_uy = Capacity(sphere_body, mesh_uy; method="VOFI", integration_method=:vofijul)
println("capacity uz...")
cap_uz = Capacity(sphere_body, mesh_uz; method="VOFI", integration_method=:vofijul)
println("capacity p...")
cap_p  = Capacity(sphere_body, mesh_p; method="VOFI", integration_method=:vofijul)
op_ux = DiffusionOps(cap_ux)
op_uy = DiffusionOps(cap_uy)
op_uz = DiffusionOps(cap_uz)
op_p  = DiffusionOps(cap_p)

###########
# Boundary Conditions
###########
Umax = 1.0
# Uniform plug profile in x-component at inlet; zero elsewhere
uniform_plug = (x,y,z) -> Umax

# For ux component
ux_left   = Dirichlet((x,y,z) -> uniform_plug(x,y,z))
ux_right  = Outflow()
ux_bottom = Dirichlet((x,y,z) -> uniform_plug(x,y,z))
ux_top    = Dirichlet((x,y,z) -> uniform_plug(x,y,z))
ux_front  = Dirichlet((x,y,z) -> uniform_plug(x,y,z))
ux_back   = Dirichlet((x,y,z) -> uniform_plug(x,y,z))

# For uy component (all walls no-slip, inlet/outlet zero cross-flow)
uy_zero = Dirichlet(0.0)

# For uz component
uz_zero = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right,
    :bottom=>ux_bottom, :top=>ux_top,
    :front=>ux_front, :back=>ux_back
))

bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero,
    :bottom=>uy_zero, :top=>uy_zero,
    :front=>uy_zero, :back=>uy_zero
))

bc_uz = BorderConditions(Dict(
    :left=>uz_zero, :right=>uz_zero,
    :bottom=>uz_zero, :top=>uz_zero,
    :front=>uz_zero, :back=>uz_zero
))

pressure_gauge = PinPressureGauge()

# Cut-cell interface (sphere) -> enforce u^γ = 0
bc_cut = Dirichlet(0.0)

###########
# Physics
###########
μ = 1.0; ρ = 1.0
fᵤ = (x,y,z) -> 0.0
fₚ = (x,y,z) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy, mesh_uz),
              (cap_ux, cap_uy, cap_uz),
              (op_ux, op_uy, op_uz),
              mesh_p,
              cap_p,
              op_p,
              μ, ρ, fᵤ, fₚ)

###########
# Solve (steady)
###########
nu_x = prod(op_ux.size); nu_y = prod(op_uy.size); nu_z = prod(op_uz.size)
np = prod(op_p.size)
# state length = 2*(nu_x+nu_y+nu_z)+np
xlen = 2*(nu_x+nu_y+nu_z) + np
x0_vec = zeros(xlen)

solver = StokesMono(fluid, (bc_ux, bc_uy, bc_uz), pressure_gauge, bc_cut; x0=x0_vec)
solve_StokesMono!(solver; method=Base.:\)
println("3D Stokes flow around sphere solved. Unknowns = ", length(solver.x))

# Extract fields
uωx = solver.x[1:nu_x]
uωy = solver.x[2*nu_x + nu_y + 1 - (nu_y):2*nu_x + nu_y]  # We'll reshape systematically below
# Better: reconstruct via offsets
off_uωx = 0
off_uγx = nu_x
off_uωy = 2*nu_x
off_uγy = 2*nu_x + nu_y
off_uωz = 2*nu_x + 2*nu_y
off_uγz = 2*nu_x + 2*nu_y + nu_z
off_p   = 2*(nu_x + nu_y + nu_z)

Ux = reshape(solver.x[off_uωx+1:off_uωx+nu_x],
             (length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2]), length(mesh_ux.nodes[3])))
Uy = reshape(solver.x[off_uωy+1:off_uωy+nu_y],
             (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2]), length(mesh_uy.nodes[3])))
Uz = reshape(solver.x[off_uωz+1:off_uωz+nu_z],
             (length(mesh_uz.nodes[1]), length(mesh_uz.nodes[2]), length(mesh_uz.nodes[3])))
P  = reshape(solver.x[off_p+1:off_p+np],
             (length(mesh_p.nodes[1]), length(mesh_p.nodes[2]), length(mesh_p.nodes[3])))

println("Velocity magnitude stats: min=", minimum(sqrt.(Ux.^2 .+ Uy.^2 .+ Uz.^2)),
        " max=", maximum(sqrt.(Ux.^2 .+ Uy.^2 .+ Uz.^2)))

###########
# Simple slice visualization
###########
xs = mesh_p.nodes[1]; ys = mesh_p.nodes[2]; zs = mesh_p.nodes[3]
mid_z_index = Int(clamp(round(length(zs)/2), 1, length(zs)))
Ux_slice = Ux[:,:,mid_z_index]
Uy_slice = Uy[:,:,mid_z_index]
Uz_slice = Uz[:,:,mid_z_index]
P_slice  = P[:,:,mid_z_index]

speed_slice = sqrt.(Ux_slice.^2 .+ Uy_slice.^2 .+ Uz_slice.^2)

fig = Figure(resolution=(1200,600))
ax_speed = Axis(fig[1,1], title="|u| slice (z=mid)", xlabel="x", ylabel="y")
hm = heatmap!(ax_speed, mesh_ux.nodes[1], mesh_ux.nodes[2], speed_slice; colormap=:plasma)
Colorbar(fig[1,2], hm)

ax_pressure = Axis(fig[1,3], title="Pressure slice (z=mid)", xlabel="x", ylabel="y")
hm2 = heatmap!(ax_pressure, xs, ys, P_slice; colormap=:balance)
Colorbar(fig[1,4], hm2)

save("stokes3d_sphere_slice.png", fig)
println("Saved stokes3d_sphere_slice.png")

###########
# Analytical Solution Comparison
###########
"""
Analytical Stokes flow solution around a sphere in uniform flow.
For a sphere of radius 'a' in uniform flow U in the x-direction:
  vr = U(1 − 3a/(2r) + a³/(2r³))cosθ
  vθ = U(−1 + 3a/(4r) + a³/(4r³))sinθ
  vφ = 0
where (r, θ, φ) are spherical coordinates with θ measured from x-axis.

Note: The analytical solution assumes an unbounded domain with uniform flow at infinity.
The current numerical simulation uses a bounded channel with a uniform plug inflow.
For closer agreement with the unbounded analytical solution, consider extending the domain to
~16 times the sphere diameter (Lx ~ 6.4 for R=0.2) and expanding the cross-section to reduce wall effects.
"""
function analytical_stokes_sphere(x, y, z, U, a, center=(0.0, 0.0, 0.0))
    # Translate to sphere center
    dx = x - center[1]
    dy = y - center[2]
    dz = z - center[3]
    
    # Convert to spherical coordinates (with x as polar axis)
    r = sqrt(dx^2 + dy^2 + dz^2)
    
    # Avoid singularity at sphere center
    if r < a * 1.01  # Inside or very close to sphere surface
        return (0.0, 0.0, 0.0)  # No-slip boundary condition
    end
    
    # θ is angle from x-axis (polar angle)
    cos_theta = dx / r
    sin_theta = sqrt(dy^2 + dz^2) / r
    
    # Radial and tangential velocities in spherical coordinates
    vr = U * (1.0 - 3.0*a/(2.0*r) + a^3/(2.0*r^3)) * cos_theta
    vtheta = -U * (1.0 - 3.0*a/(4.0*r) - a^3/(4.0*r^3)) * sin_theta
    
    # Convert back to Cartesian coordinates
    # Unit vectors: e_r = (dx, dy, dz)/r
    #               e_theta in x-y plane perpendicular to e_r
    if sin_theta > 1e-10
        # e_theta points in direction of decreasing θ
        # In Cartesian: e_theta = (-sin_theta, cos_theta*dy/sqrt(dy^2+dz^2), cos_theta*dz/sqrt(dy^2+dz^2))
        rho = sqrt(dy^2 + dz^2)
        vx = vr * cos_theta - vtheta * sin_theta
        vy = vr * dy/r + vtheta * cos_theta * dy/rho
        vz = vr * dz/r + vtheta * cos_theta * dz/rho
    else
        # On x-axis, theta = 0 or π
        vx = vr * cos_theta
        vy = 0.0
        vz = 0.0
    end
    
    return (vx, vy, vz)
end

# Use the inlet plug velocity as U_infinity for the analytical comparison
U_inf = Umax

# Small constant for numerical stability in relative error calculations
EPS = 1e-10

# Simple trilinear interpolation on structured grids (clamped at boundaries)
function trilinear_interp(nodes, field, x, y, z)
    xs, ys, zs = nodes
    nx, ny, nz = length(xs), length(ys), length(zs)

    ix = clamp(searchsortedlast(xs, x), 1, nx-1)
    iy = clamp(searchsortedlast(ys, y), 1, ny-1)
    iz = clamp(searchsortedlast(zs, z), 1, nz-1)

    x1, x2 = xs[ix], xs[ix+1]; y1, y2 = ys[iy], ys[iy+1]; z1, z2 = zs[iz], zs[iz+1]
    tx = (x2 == x1) ? 0.0 : (x - x1) / (x2 - x1)
    ty = (y2 == y1) ? 0.0 : (y - y1) / (y2 - y1)
    tz = (z2 == z1) ? 0.0 : (z - z1) / (z2 - z1)

    c000 = field[ix,   iy,   iz  ]
    c100 = field[ix+1, iy,   iz  ]
    c010 = field[ix,   iy+1, iz  ]
    c110 = field[ix+1, iy+1, iz  ]
    c001 = field[ix,   iy,   iz+1]
    c101 = field[ix+1, iy,   iz+1]
    c011 = field[ix,   iy+1, iz+1]
    c111 = field[ix+1, iy+1, iz+1]

    c00 = c000*(1-tx) + c100*tx
    c10 = c010*(1-tx) + c110*tx
    c01 = c001*(1-tx) + c101*tx
    c11 = c011*(1-tx) + c111*tx

    c0 = c00*(1-ty) + c10*ty
    c1 = c01*(1-ty) + c11*ty

    return c0*(1-tz) + c1*tz
end

println("\n" * "="^60)
println("Analytical Stokes Solution Comparison")
println("="^60)
println("Sphere radius: a = ", R)
println("Reference velocity: U = ", U_inf)
println("Sphere center: ", sphere_center)

# Sample points for comparison along centerline (y=0, z=0)
n_samples = 20
x_sample = range(sphere_center[1] + 1.5*R, stop=x0+Lx-0.5, length=n_samples)
y_sample = zeros(n_samples)
z_sample = zeros(n_samples)

# Interpolate numerical solution at sample points
# Note: Using nearest-neighbor sampling for simplicity. For more accurate
# comparisons, consider implementing proper interpolation (linear/cubic).
u_num_x = zeros(n_samples)
u_num_y = zeros(n_samples)
u_num_z = zeros(n_samples)
u_ana_x = zeros(n_samples)
u_ana_y = zeros(n_samples)
u_ana_z = zeros(n_samples)

for i in 1:n_samples
    xi, yi, zi = x_sample[i], y_sample[i], z_sample[i]

    # Trilinear interpolation on staggered grids
    u_num_x[i] = trilinear_interp(mesh_ux.nodes, Ux, xi, yi, zi)
    u_num_y[i] = trilinear_interp(mesh_uy.nodes, Uy, xi, yi, zi)
    u_num_z[i] = trilinear_interp(mesh_uz.nodes, Uz, xi, yi, zi)
    
    # Analytical solution
    vx, vy, vz = analytical_stokes_sphere(xi, yi, zi, U_inf, R, sphere_center)
    u_ana_x[i] = vx
    u_ana_y[i] = vy
    u_ana_z[i] = vz
end

# Compute errors
error_x = u_num_x .- u_ana_x
error_y = u_num_y .- u_ana_y
error_z = u_num_z .- u_ana_z
error_magnitude = sqrt.(error_x.^2 .+ error_y.^2 .+ error_z.^2)

# Compute error metrics
l2_error = sqrt(sum(error_magnitude.^2) / n_samples)
max_error = maximum(error_magnitude)
mean_error = sum(error_magnitude) / n_samples

println("\nError Metrics (centerline samples):")
println("  L2 error:   ", l2_error)
println("  Max error:  ", max_error)
println("  Mean error: ", mean_error)

# Relative errors
u_ana_magnitude = sqrt.(u_ana_x.^2 .+ u_ana_y.^2 .+ u_ana_z.^2)
relative_error = error_magnitude ./ (u_ana_magnitude .+ EPS)
mean_relative_error = sum(relative_error) / n_samples

println("  Mean relative error: ", mean_relative_error * 100, "%")

# Visualization: comparison plot
fig2 = Figure(resolution=(1200, 800))

# Plot velocity components along centerline
ax1 = Axis(fig2[1,1], title="Velocity vx along centerline (y=0, z=0)", 
           xlabel="x", ylabel="vx")
lines!(ax1, x_sample, u_num_x, label="Numerical", linewidth=2)
lines!(ax1, x_sample, u_ana_x, label="Analytical", linewidth=2, linestyle=:dash)
axislegend(ax1, position=:rb)

ax2 = Axis(fig2[1,2], title="Velocity magnitude along centerline", 
           xlabel="x", ylabel="|v|")
u_num_mag = sqrt.(u_num_x.^2 .+ u_num_y.^2 .+ u_num_z.^2)
u_ana_mag = sqrt.(u_ana_x.^2 .+ u_ana_y.^2 .+ u_ana_z.^2)
lines!(ax2, x_sample, u_num_mag, label="Numerical", linewidth=2)
lines!(ax2, x_sample, u_ana_mag, label="Analytical", linewidth=2, linestyle=:dash)
axislegend(ax2, position=:rb)

ax3 = Axis(fig2[2,1], title="Error in velocity magnitude", 
           xlabel="x", ylabel="Error |v|")
lines!(ax3, x_sample, error_magnitude, linewidth=2, color=:red)

ax4 = Axis(fig2[2,2], title="Relative error", 
           xlabel="x", ylabel="Relative error")
lines!(ax4, x_sample, relative_error .* 100, linewidth=2, color=:red)
ax4.ylabel = "Relative error (%)"

save("stokes3d_sphere_comparison.png", fig2)
println("\nSaved comparison plot: stokes3d_sphere_comparison.png")

# Additional comparison: radial profile at 45 degrees
println("\nAdditional radial comparison at θ=45° (in x-y plane):")
n_radial = 15
r_values = range(1.5*R, stop=minimum([Lx/2, Ly/2])*0.9, length=n_radial)
theta_deg = 45.0
theta_rad = theta_deg * π / 180.0

u_rad_num_x = zeros(n_radial)
u_rad_num_y = zeros(n_radial)
u_rad_num_z = zeros(n_radial)
u_rad_ana_x = zeros(n_radial)
u_rad_ana_y = zeros(n_radial)
u_rad_ana_z = zeros(n_radial)

for i in 1:n_radial
    r = r_values[i]
    xi = sphere_center[1] + r * cos(theta_rad)
    yi = sphere_center[2] + r * sin(theta_rad)
    zi = sphere_center[3]
    
    # Numerical solution (interpolated)
    u_rad_num_x[i] = trilinear_interp(mesh_ux.nodes, Ux, xi, yi, zi)
    u_rad_num_y[i] = trilinear_interp(mesh_uy.nodes, Uy, xi, yi, zi)
    u_rad_num_z[i] = trilinear_interp(mesh_uz.nodes, Uz, xi, yi, zi)
    
    # Analytical solution
    vx, vy, vz = analytical_stokes_sphere(xi, yi, zi, U_inf, R, sphere_center)
    u_rad_ana_x[i] = vx
    u_rad_ana_y[i] = vy
    u_rad_ana_z[i] = vz
end

rad_error = sqrt.((u_rad_num_x .- u_rad_ana_x).^2 .+ 
                  (u_rad_num_y .- u_rad_ana_y).^2 .+ 
                  (u_rad_num_z .- u_rad_ana_z).^2)
println("  Mean error along radial: ", sum(rad_error) / n_radial)
println("  Max error along radial: ", maximum(rad_error))

println("="^60)

###########
# Drag Force Calculation
###########
println("\n" * "="^60)
println("Drag Force Calculation")
println("="^60)

force_diag = compute_navierstokes_force_diagnostics(solver)
body_force = navierstokes_reaction_force_components(force_diag; acting_on=:body)
pressure_body = .-force_diag.integrated_pressure
viscous_body = .-force_diag.integrated_viscous
coeffs = drag_lift_coefficients(force_diag; ρ=ρ, U_ref=Umax, length_ref=2*R, acting_on=:body)

println("Integrated forces on the sphere (body reaction):")
println("  Fx = $(round(body_force[1]; sigdigits=6)) (pressure=$(round(pressure_body[1]; sigdigits=6)), viscous=$(round(viscous_body[1]; sigdigits=6)))")
println("  Fy = $(round(body_force[2]; sigdigits=6)) (pressure=$(round(pressure_body[2]; sigdigits=6)), viscous=$(round(viscous_body[2]; sigdigits=6)))")
println("  Fz = $(round(body_force[3]; sigdigits=6)) (pressure=$(round(pressure_body[3]; sigdigits=6)), viscous=$(round(viscous_body[3]; sigdigits=6)))")
println("  Drag coefficient Cd = $(round(coeffs.Cd; sigdigits=6))")
println("  Lift coefficient Cl = $(round(coeffs.Cl; sigdigits=6))")

# Analytical Drag (Stokes Law for unbounded domain)
# F_drag = 6 * pi * mu * R * U
F_drag_ana = 6 * π * μ * R * Umax
println("\nAnalytical Drag (Stokes Law, unbounded): ", round(F_drag_ana; sigdigits=6))
println("Relative Error in Drag: ", round(abs(body_force[1] - F_drag_ana)/F_drag_ana * 100; digits=2), "%")
println("(Note: Wall effects in a channel will increase drag significantly compared to unbounded Stokes flow)")
