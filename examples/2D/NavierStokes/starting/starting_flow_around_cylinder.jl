using Penguin
using CairoMakie
using DelimitedFiles
using LinearAlgebra
using Printf

"""
Starting flow around a cylinder (impulsively started uniform flow).
Adapted from the Basilisk starting.c example with a uniform grid and the
embedded-boundary Navier-Stokes solver in Penguin.

Outputs:
- Drag/lift history file in this folder
- Drag coefficient comparison plot (with reference data if available)
- Final speed magnitude plot
"""

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Re = 1000.0
U_ref = 1.0
diameter = 1.0
radius = diameter / 2

rho = 1.0
mu = rho * U_ref * diameter / Re

L0 = 18.0
nx, ny = 64, 64
x0, y0 = -L0 / 2, -L0 / 2

circle_body = (x, y, _=0.0) -> radius - sqrt(x^2 + y^2)

# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------
mesh_p = Penguin.Mesh((nx, ny), (L0, L0), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (L0, L0), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (L0, L0), (x0, y0 - 0.5 * dy))

println("Re=$(Re), grid=$(nx)x$(ny), dx=$(dx), points_per_diameter=$(diameter / dx)")

# ---------------------------------------------------------------------------
# Capacities and operators
# ---------------------------------------------------------------------------
capacity_ux = Capacity(circle_body, mesh_ux)
capacity_uy = Capacity(circle_body, mesh_uy)
capacity_p = Capacity(circle_body, mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p = DiffusionOps(capacity_p)

# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
ux_left = Dirichlet((x, y, t=0.0) -> U_ref)
ux_right = Outflow()
ux_bottom = Dirichlet((x, y, t=0.0) -> U_ref)
ux_top = Dirichlet((x, y, t=0.0) -> U_ref)

uy_left = Dirichlet((x, y, t=0.0) -> 0.0)
uy_right = Outflow()
uy_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
uy_top = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left => ux_left,
    :right => ux_right,
    :bottom => ux_bottom,
    :top => ux_top
))
bc_uy = BorderConditions(Dict(
    :left => uy_left,
    :right => uy_right,
    :bottom => uy_bottom,
    :top => uy_top
))

pressure_gauge = PinPressureGauge()
interface_bc = Dirichlet(0.0)

# ---------------------------------------------------------------------------
# Physics and solver setup
# ---------------------------------------------------------------------------
f_u = (x, y, z=0.0, t=0.0) -> 0.0
f_p = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              mu, rho, f_u, f_p)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)

Ntot = 2 * (nu_x + nu_y) + np
x0_vec = zeros(Ntot)
x0_vec[1:nu_x] .= U_ref
x0_vec[nu_x + 1:2 * nu_x] .= U_ref

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)

# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
dt = 0.2 * min(dx, dy) / max(U_ref, 1e-12)
t_end = 3.0

times = Float64[]
dt_hist = Float64[]
cd_hist = Float64[]
cl_hist = Float64[]
drag_hist = Float64[]
lift_hist = Float64[]
pressure_x = Float64[]
pressure_y = Float64[]
viscous_x = Float64[]
viscous_y = Float64[]

function record_forces!(t, dt_step)
    force_diag = compute_navierstokes_force_diagnostics(solver)
    coeffs = drag_lift_coefficients(force_diag; ρ=rho, U_ref=U_ref, length_ref=diameter, acting_on=:body)
    body_force = navierstokes_reaction_force_components(force_diag; acting_on=:body)
    pressure_body = -force_diag.integrated_pressure
    viscous_body = -force_diag.integrated_viscous

    push!(times, t)
    push!(dt_hist, dt_step)
    push!(cd_hist, coeffs.Cd)
    push!(cl_hist, coeffs.Cl)
    push!(drag_hist, body_force[1])
    push!(lift_hist, body_force[2])
    push!(pressure_x, pressure_body[1])
    push!(pressure_y, pressure_body[2])
    push!(viscous_x, viscous_body[1])
    push!(viscous_y, viscous_body[2])
end

println("Starting unsteady Newton solve up to t_end=$(t_end) with dt=$(dt)")
times_raw, histories = solve_NavierStokesMono_unsteady_newton!(solver; Δt=dt, T_end=t_end, scheme=:BE, store_states=true)
dt_hist_raw = vcat(0.0, diff(times_raw))
for (idx, state) in enumerate(histories)
    solver.x = state
    record_forces!(times_raw[idx], dt_hist_raw[idx])
end

solver.x = histories[end]

# ---------------------------------------------------------------------------
# Save force history
# ---------------------------------------------------------------------------
output_path = joinpath(@__DIR__, "starting_drag_history_Re$(Int(round(Re))).dat")
open(output_path, "w") do io
    println(io, "# t dt Cd Cl drag_x drag_y pressure_x pressure_y viscous_x viscous_y")
    for i in eachindex(times)
        @printf(io, "%g %g %g %g %g %g %g %g %g %g\n",
                times[i], dt_hist[i], cd_hist[i], cl_hist[i],
                drag_hist[i], lift_hist[i],
                pressure_x[i], pressure_y[i],
                viscous_x[i], viscous_y[i])
    end
end

println("Saved force history to $(output_path)")

# ---------------------------------------------------------------------------
# Drag coefficient comparison plot
# ---------------------------------------------------------------------------
data_dir = joinpath(@__DIR__, "data")

function read_reference_drag(path)
    if !isfile(path)
        return nothing
    end
    raw = readdlm(path)
    if ndims(raw) == 1
        raw = reshape(raw, 1, :)
    end
    return (t=raw[:, 1] ./ 2.0, Cd=raw[:, 2])
end

sim_path = Re < 2000 ? joinpath(data_dir, "fig1a.SIM") : joinpath(data_dir, "fig1b.SIM")
kl_path = Re < 2000 ? joinpath(data_dir, "fig1a.KL") : joinpath(data_dir, "fig1b.KL")

ref_sim = read_reference_drag(sim_path)
ref_kl = read_reference_drag(kl_path)

t_scaled = times .* (U_ref / diameter)

fig_drag = Figure(resolution=(900, 450))
ax_drag = Axis(fig_drag[1, 1], xlabel="t/(D/U)", ylabel="Cd",
               title="Starting flow around a cylinder (Re=$(Int(round(Re))))")
lines!(ax_drag, t_scaled, cd_hist; color=:royalblue, linewidth=2, label="Penguin")
if ref_sim !== nothing
    scatter!(ax_drag, ref_sim.t, ref_sim.Cd; color=:black, markersize=5,
             label="Mohaghegh et al. 2017 (SIM)")
end
if ref_kl !== nothing
    scatter!(ax_drag, ref_kl.t, ref_kl.Cd; color=:gray35, markersize=5,
             label="Koumoutsakos & Leonard 1995")
end
axislegend(ax_drag, position=:rb)

drag_plot_path = joinpath(@__DIR__, "starting_drag_Re$(Int(round(Re))).png")
save(drag_plot_path, fig_drag)
display(fig_drag)

println("Saved drag comparison plot to $(drag_plot_path)")

# ---------------------------------------------------------------------------
# Final speed magnitude plot
# ---------------------------------------------------------------------------
xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]
uox = solver.x[1:nu_x]
uoy = solver.x[2 * nu_x + 1:2 * nu_x + nu_y]

Ux = reshape(uox, (length(xs), length(ys)))
Uy = reshape(uoy, (length(xs), length(ys)))
speed = sqrt.(Ux.^2 .+ Uy.^2)

fig_speed = Figure(resolution=(900, 600))
ax_speed = Axis(fig_speed[1, 1], xlabel="x", ylabel="y",
                title="Speed magnitude at t=$(round(times[end]; digits=3))")
hm_speed = heatmap!(ax_speed, xs, ys, speed; colormap=:plasma)
contour!(ax_speed, xs, ys, [circle_body(x, y) for x in xs, y in ys];
         levels=[0.0], color=:white, linewidth=2)
Colorbar(fig_speed[1, 2], hm_speed)

speed_plot_path = joinpath(@__DIR__, "starting_speed_Re$(Int(round(Re))).png")
save(speed_plot_path, fig_speed)
display(fig_speed)

println("Saved final speed plot to $(speed_plot_path)")
