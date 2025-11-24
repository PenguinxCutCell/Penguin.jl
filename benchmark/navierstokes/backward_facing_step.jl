using Penguin
using Statistics
using Printf
using CairoMakie
using GeometryBasics
using DelimitedFiles

"""
Backward-facing step benchmark for steady incompressible Navier–Stokes.

Geometry:
  * Upstream channel height h₁, downstream channel height h₂ = 2 h₁.
  * Step located at x = 0 lowering the bottom wall by h_s = h₂ - h₁.
  * Computational domain spans x ∈ [-L_in, L_out], y ∈ [-h_s, h₁].

Boundary conditions:
  * Inlet (x = -L_in): laminar parabolic profile u(y) = 4 Uₘ y (h₁ - y) / h₁², v = 0.
  * Walls: no-slip (u = v = 0).
  * Outflow (x = L_out): traction-free (Outflow()).
  * Pressure reference enforced by PinPressureGauge().

Diagnostics:
  * Reattachment length based on sign change of τ_w ≈ μ ∂u/∂y at the lower wall.
  * Velocity profiles at x/h₁ = 2, 4, 6, 8, 10.
  * Pressure distribution along the bottom wall.
  * Mass flux balance between inlet and outlet.

References :
- J. Kim P. Moin, "Application of a fractional-step method to incompressible Navier–Stokes equations",
  J. Comput. Phys., 59(2), 308-323 (1985).
- B. F. Armaly, F. Durst, J. C. F. Pereira, B. Schönung,
  "Experimental and theoretical investigation of backward-facing step flow",
  J. Fluid Mech., 127, 473-496 (1983).
"""

###########
# Parameters
###########
h1 = 1.0
ratio = 2.0
h2 = ratio * h1
h_step = h2 - h1
L_in = 4h1
L_out = 20h1
Lx = L_in + L_out
x0 = -L_in

ny = 80
nx = 240

ρ = 1.0
U_mean = 1.0
Re = 200.0
ν = U_mean * h1 / Re
μ = ρ * ν

Um = 1.5 * U_mean  # Peak velocity giving mean U_mean for parabolic profile

println("=== Backward-facing step benchmark ===")
println(@sprintf("Grid: %d × %d, Re = %.1f, ν = %.3e", nx, ny, Re, ν))

###########
# Meshes & Capacity
###########
top_y = h1
bottom_level(x) = x < 0 ? 0.0 : -h_step
step_body = (x, y, _=0.0) -> bottom_level(x) - y

mesh_p  = Penguin.Mesh((nx, ny), (Lx, top_y + h_step), (x0, -h_step))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, top_y + h_step), (x0 - 0.5dx, -h_step))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, top_y + h_step), (x0, -h_step - 0.5dy))

capacity_ux = Capacity(step_body, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(step_body, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(step_body, mesh_p; compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
parabolic_inlet = (x, y, t=0.0) -> begin
    if y < 0 || y > h1
        return 0.0
    end
    4Um * y * (h1 - y) / h1^2
end

no_slip = (x, y, t=0.0) -> 0.0

bc_ux = BorderConditions(Dict(
    :left   => Dirichlet(parabolic_inlet),
    :right  => Outflow(),
    :bottom => Dirichlet(no_slip),
    :top    => Dirichlet(no_slip)
))

bc_uy = BorderConditions(Dict(
    :left   => Dirichlet((x, y, t=0.0) -> 0.0),
    :right  => Outflow(),
    :bottom => Dirichlet(no_slip),
    :top    => Dirichlet(no_slip)
))

pressure_gauge = PinPressureGauge()
interface_bc = Dirichlet(0.0)

fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

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

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc)

println("Solving steady Navier–Stokes...")
_, picard_iters, picard_res = solve_NavierStokesMono_steady!(
    solver;
    tol=1e-6,
    maxiter=8,
    relaxation=0.8,
    nlsolve_method=:picard,
)
println(@sprintf("Picard iterations: %d (residual %.3e)", picard_iters, picard_res))

_, newton_iters, newton_res = solve_NavierStokesMono_steady!(
    solver;
    tol=5e-9,
    maxiter=10,
    nlsolve_method=:newton,
)
println(@sprintf("Newton iterations: %d (residual %.3e)", newton_iters, newton_res))

###########
# Extract fields
###########
xs_ux = mesh_ux.nodes[1]
ys_ux = mesh_ux.nodes[2]
xs_uy = mesh_uy.nodes[1]
ys_uy = mesh_uy.nodes[2]
xs_p = mesh_p.nodes[1]
ys_p = mesh_p.nodes[2]

uωx = solver.x[1:nu_x]
uωy = solver.x[2nu_x+1:2nu_x+nu_y]
pω  = solver.x[2*(nu_x + nu_y)+1:end]

Ux = reshape(uωx, Tuple(operator_ux.size))
Uy = reshape(uωy, Tuple(operator_uy.size))
P  = reshape(pω, Tuple(operator_p.size))

trim_copy(A) = size(A, 1) > 1 && size(A, 2) > 1 ? copy(@view A[1:end-1, 1:end-1]) : copy(A)
trim_coords(v) = length(v) > 1 ? copy(v[1:end-1]) : copy(v)

Ux_trim = trim_copy(Ux)
Uy_trim = trim_copy(Uy)
P_trim  = trim_copy(P)
xs_ux_trim = trim_coords(xs_ux)
ys_ux_trim = trim_coords(ys_ux)
xs_uy_trim = trim_coords(xs_uy)
ys_uy_trim = trim_coords(ys_uy)
xs_p_trim = trim_coords(xs_p)
ys_p_trim = trim_coords(ys_p)

###########
# Diagnostics: reattachment length
###########
function bottom_y(x)
    x < 0 ? 0.0 : -h_step
end

function nearest_index(vec::AbstractVector{<:Real}, val::Real)
    clamp(argmin(abs.(vec .- val)), 1, length(vec))
end

shear_samples = Float64[]
shear_x = Float64[]
μ_inv_dy = μ / dy

for (i, x) in enumerate(xs_ux_trim)
    x < 0 && continue
    by = bottom_y(x)
    j = nearest_index(ys_ux_trim, by + dy)
    u_wall = Ux_trim[i, j]
    push!(shear_x, x)
    push!(shear_samples, μ_inv_dy * u_wall)
end

reattach_x = NaN
for k in 2:length(shear_samples)
    if shear_samples[k-1] < 0 && shear_samples[k] ≥ 0
        t = shear_samples[k-1] / (shear_samples[k-1] - shear_samples[k] + eps())
        global reattach_x = shear_x[k-1] + t * (shear_x[k] - shear_x[k-1])
        break
    end
end

if isnan(reattach_x)
    println("Reattachment length not detected (shear remained negative).")
else
    println(@sprintf("Reattachment length L_r ≈ %.3f (L_r/h₁ = %.3f)",
                     reattach_x, reattach_x / h1))
end

###########
# Velocity profiles
###########
function bilinear(xs, ys, field, xp, yp)
    x = clamp(xp, xs[1], xs[end])
    y = clamp(yp, ys[1], ys[end])
    ix_hi = searchsortedfirst(xs, x)
    ix_lo = clamp(ix_hi - 1, 1, length(xs))
    ix_hi = clamp(ix_hi, 1, length(xs))
    iy_hi = searchsortedfirst(ys, y)
    iy_lo = clamp(iy_hi - 1, 1, length(ys))
    iy_hi = clamp(iy_hi, 1, length(ys))
    x1, x2 = xs[ix_lo], xs[ix_hi]
    y1, y2 = ys[iy_lo], ys[iy_hi]
    tx = x2 ≈ x1 ? 0.0 : (x - x1) / (x2 - x1)
    ty = y2 ≈ y1 ? 0.0 : (y - y1) / (y2 - y1)
    f11 = field[ix_lo, iy_lo]
    f21 = field[ix_hi, iy_lo]
    f12 = field[ix_lo, iy_hi]
    f22 = field[ix_hi, iy_hi]
    (1 - tx) * (1 - ty) * f11 +
    tx * (1 - ty) * f21 +
    (1 - tx) * ty * f12 +
    tx * ty * f22
end

sample_positions = [2, 4, 6, 8, 10]
profile_data = Dict{Float64, Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}()

for ξ in sample_positions
    x_pos = ξ * h1
    if x_pos > L_out
        continue
    end
    u_vals = Float64[]
    v_vals = Float64[]
    for y in ys_p_trim
        push!(u_vals, bilinear(xs_ux_trim, ys_ux_trim, Ux_trim, x_pos, y))
        push!(v_vals, bilinear(xs_uy_trim, ys_uy_trim, Uy_trim, x_pos, y))
    end
    profile_data[ξ] = (ys_p_trim, u_vals, v_vals)
end

###########
# Pressure along bottom wall
###########
bottom_x = Float64[]
bottom_pressure = Float64[]

for x in xs_p_trim
    if x < 0
        continue
    end
    by = bottom_y(x) + 0.5dy
    p_val = bilinear(xs_p_trim, ys_p_trim, P_trim, x, by)
    push!(bottom_x, x)
    push!(bottom_pressure, p_val)
end

###########
# Mass flux balance
###########
function integrate_face_flux(Ufield, xs, ys; column)
    values = Ufield[column, :]
    mask = (ys .>= 0) .| (xs[column] ≥ 0)
    dy_weights = diff(vcat(ys, ys[end] + dy))
    return sum(values .* dy_weights .* mask)
end

flux_in = integrate_face_flux(Ux_trim, xs_ux_trim, ys_ux_trim; column=1)
flux_out = integrate_face_flux(Ux_trim, xs_ux_trim, ys_ux_trim; column=size(Ux_trim, 1))
imbalance = flux_out - flux_in
println(@sprintf("Mass flux: inflow %.6f, outflow %.6f, imbalance %.3e",
                 flux_in, flux_out, imbalance))

###########
# Output: velocity profiles CSV
###########
profile_rows = Vector{Vector{Float64}}()
header = ["y_over_h1"]

for (ξ, (y_vals, u_vals, v_vals)) in sort(profile_data; by=first)
    push!(header, @sprintf("u_x/h1=%.0f", ξ))
    push!(header, @sprintf("v_x/h1=%.0f", ξ))
end

max_len = maximum(length.(first.(values(profile_data))))
for i in 1:max_len
    row = Float64[]
    y = i <= length(first(first(values(profile_data)))) ? first(first(values(profile_data)))[i] : NaN
    push!(row, y / h1)
    for (_, (y_vals, u_vals, v_vals)) in sort(profile_data; by=first)
        if i <= length(y_vals)
            push!(row, u_vals[i] / U_mean)
            push!(row, v_vals[i] / U_mean)
        else
            push!(row, NaN, NaN)
        end
    end
    push!(profile_rows, row)
end

writedlm("navierstokes_step_velocity_profiles.csv", [header; profile_rows])

###########
# Plotting
###########
fig_profiles = Figure(resolution=(900, 500))
ax_prof = Axis(fig_profiles[1, 1], xlabel="u / Uₘ", ylabel="y / h₁",
               title="Velocity profiles downstream of step")

for (ξ, (y_vals, u_vals, _)) in sort(profile_data; by=first)
    lines!(ax_prof, u_vals ./ U_mean, y_vals ./ h1; label=@sprintf("x/h₁ = %.0f", ξ))
end
axislegend(ax_prof; position=:rb)
save("navierstokes_step_velocity_profiles.png", fig_profiles)
display(fig_profiles)

fig_pressure = Figure(resolution=(900, 400))
ax_pwall = Axis(fig_pressure[1, 1], xlabel="x / h₁", ylabel="p",
                title="Pressure along bottom wall")
lines!(ax_pwall, bottom_x ./ h1, bottom_pressure; color=:royalblue)
scatter!(ax_pwall, bottom_x ./ h1, bottom_pressure; color=:orange, markersize=5)
save("navierstokes_step_pressure.png", fig_pressure)
display(fig_pressure)

fig_shear = Figure(resolution=(900, 400))
ax_shear = Axis(fig_shear[1,1], xlabel="x / h₁", ylabel="τ_w",
                title="Wall shear downstream of step")
lines!(ax_shear, shear_x ./ h1, shear_samples; color=:crimson)
hlines!(ax_shear, [0.0]; linewidth=1, color=:black, linestyle=:dash)
save("navierstokes_step_shear.png", fig_shear)
display(fig_shear)

fig_stream = Figure(resolution=(1000, 450))
ax_stream = Axis(fig_stream[1,1], xlabel="x / h₁", ylabel="y / h₁",
                 title="Streamlines over backward-facing step")

x_min, x_max = xs_ux_trim[1], xs_ux_trim[end]
y_min, y_max = ys_ux_trim[1], ys_ux_trim[end]
stepsize = 0.015
maxsteps = 1500
density = 2.5
color_func = p -> norm(p)
velocity_func = (x, y) -> begin
    Point2f(
        bilinear(xs_ux_trim, ys_ux_trim, Ux_trim, x, y),
        bilinear(xs_uy_trim, ys_uy_trim, Uy_trim, x, y)
    )
end
rect = Rect(x_min, y_min, x_max - x_min, y_max - y_min)
streamplot!(ax_stream, velocity_func, rect, stepsize=stepsize, maxsteps=maxsteps,
            density=density, color=color_func)
contour!(ax_stream, xs_ux_trim, ys_ux_trim,
         [step_body(x,y) for x in xs_ux_trim, y in ys_ux_trim];
         levels=[0.0], color=:white, linewidth=2)
xlims!(ax_stream, x_min, x_max)
ylims!(ax_stream, y_min, y_max)
save("navierstokes_step_streamlines.png", fig_stream)
display(fig_stream)

println("Diagnostics saved: reattachment shear plot, pressure curve, velocity profiles, streamlines, CSV data.")

###########
# Multi-Reynolds reattachment sweep
###########
reference_ranges = Dict(
    100.0 => (3.5, 5.0),
    200.0 => (4.0, 6.0),
    300.0 => (5.5, 8.0),
)

function run_step_case(Re_case)
    ν_case = U_mean * h1 / Re_case
    μ_case = ρ * ν_case

    fluid_case = Fluid((mesh_ux, mesh_uy),
                       (capacity_ux, capacity_uy),
                       (operator_ux, operator_uy),
                       mesh_p,
                       capacity_p,
                       operator_p,
                       μ_case, ρ, fᵤ, fₚ)

    solver_case = NavierStokesMono(fluid_case, (bc_ux, bc_uy), pressure_gauge, interface_bc)
    solve_NavierStokesMono_steady!(solver_case; tol=1e-6, maxiter=5, relaxation=0.8, nlsolve_method=:picard)
    solve_NavierStokesMono_steady!(solver_case; tol=5e-9, maxiter=8, nlsolve_method=:newton)

    uωx_case = solver_case.x[1:nu_x]
    Ux_case = reshape(uωx_case, Tuple(operator_ux.size))
    shear = Float64[]
    xvals = Float64[]
    μ_inv = μ_case / dy
    for (i, x) in enumerate(xs_ux_trim)
        x < 0 && continue
        by = bottom_y(x)
        j = nearest_index(ys_ux_trim, by + dy)
        u_wall = Ux_case[i, j]
        isfinite(u_wall) || continue
        push!(xvals, x)
        push!(shear, μ_inv * u_wall)
    end
    return xvals, shear
end

results_table = IOBuffer()
println(results_table, "Re,L_r/h1,Reference_min,Reference_max,Status")

for Re_case in sort(collect(keys(reference_ranges)))
    xvals, shear = run_step_case(Re_case)
    reattach = NaN
    for k in 2:length(shear)
        if shear[k-1] < 0 && shear[k] ≥ 0
            t = shear[k-1] / (shear[k-1] - shear[k] + eps())
            reattach = xvals[k-1] + t * (xvals[k] - xvals[k-1])
            break
        end
    end
    (ref_min, ref_max) = reference_ranges[Re_case]
    status = isnan(reattach) ? "not-detected" :
             (ref_min <= reattach/h1 <= ref_max ? "within-range" : "outside-range")
    println(results_table, @sprintf("%.0f,%.3f,%.2f,%.2f,%s",
                                    Re_case, isnan(reattach) ? NaN : reattach / h1,
                                    ref_min, ref_max, status))
end

write("navierstokes_step_reattachment_sweep.csv", String(take!(results_table)))
println("Saved multi-Re reattachment sweep to navierstokes_step_reattachment_sweep.csv")
