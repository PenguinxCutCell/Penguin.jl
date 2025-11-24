using Penguin
using CairoMakie
using Statistics
using CairoMakie

"""
Unsteady 2D Stokes Poiseuille flow solved with Crank–Nicolson.
Velocity starts from rest and relaxes toward the analytical parabola imposed at the walls.
"""

############
# Geometry
############
nx, ny = 64, 32
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, 0.0

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx, dy = Lx / nx, Ly / ny
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

############
# Capacities/operators
############
body = (x, y, _=0) -> -1.0
capacity_ux = Capacity(body, mesh_ux)
capacity_uy = Capacity(body, mesh_uy)
capacity_p  = Capacity(body, mesh_p)
operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

############
# Boundary conditions
############
Umax = 1.0
parabola = (x, y, t) -> 4Umax * (y - y0) * (Ly - (y - y0)) / (Ly^2)

ux_left  = Dirichlet(parabola)
ux_right = Dirichlet(parabola)
ux_bot   = Dirichlet((x, y, t) -> 0.0)
ux_top   = Dirichlet((x, y, t) -> 0.0)
bc_ux = BorderConditions(Dict(
    :left => ux_left,
    :right => ux_right,
    :bottom => ux_bot,
    :top => ux_top,
))

uy_zero = Dirichlet((x, y, t) -> 0.0)
bc_uy = BorderConditions(Dict(
    :left => uy_zero,
    :right => uy_zero,
    :bottom => uy_zero,
    :top => uy_zero,
))

pressure_gauge = PinPressureGauge()
u_bc = Dirichlet(0.0)

############
# Fluid and initial state
############
μ = 1.0
ρ = 1.0
fᵤ = (x, y, z=0.0) -> 0.0
fₚ = (x, y, z=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p, capacity_p, operator_p,
              μ, ρ, fᵤ, fₚ)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
x0_vec = zeros(2 * (nu_x + nu_y) + np)

solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, u_bc; x0=x0_vec)

############
# Time integration
############
Δt = 0.01
T_end = 0.4
scheme = :CN

println("Running unsteady Stokes with Δt=$(Δt), T_end=$(T_end), scheme=$(scheme)")
times, states = solve_StokesMono_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=scheme, method=Base.:\)

############
# Post-processing final state
############
final_state = states[end]

uωx = final_state[1:nu_x]
uγx = final_state[nu_x+1:2nu_x]

offset_uωy = 2nu_x
uωy = final_state[offset_uωy+1:offset_uωy+nu_y]
uγy = final_state[offset_uωy+nu_y+1:offset_uωy+2nu_y]

offset_p = 2 * (nu_x + nu_y)
pω = final_state[offset_p+1:end]

xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
xp = mesh_p.nodes[1];  yp = mesh_p.nodes[2]
LIux = LinearIndices((length(xs), length(ys)))

icol = Int(cld(length(xs), 2))
ux_profile = [uωx[LIux[icol, j]] for j in 1:length(ys)]
ux_analytical = [parabola(0.0, y, 0.0) for y in ys]
profile_err = ux_profile .- ux_analytical
ℓ2_profile = sqrt(mean(abs2, profile_err))
ℓinf_profile = maximum(abs, profile_err)
println("Final profile errors: L2=$(ℓ2_profile), Linf=$(ℓinf_profile)")

fig_profile = Figure(resolution=(950, 420))
axp = Axis(fig_profile[1, 1], xlabel="u_x", ylabel="y", title="Poiseuille: CN final profile vs analytical")
lines!(axp, ux_profile, ys, label="numerical")
lines!(axp, ux_analytical, ys, color=:red, linestyle=:dash, label="analytical")
axislegend(axp, position=:rb)

display(fig_profile)
save("stokes2d_poiseuille_unsteady_profile.png", fig_profile)

############
# Time evolution diagnostic
############
jcol = Int(cld(length(xs), 2))
jrow = Int(cld(length(ys), 2))
center_idx = LIux[jcol, jrow]
ux_center_history = [state[center_idx] for state in states]

fig_hist = Figure(resolution=(900, 360))
axh = Axis(fig_hist[1, 1], xlabel="time", ylabel="u_x(center)", title="Centerline speed evolution")
lines!(axh, times, ux_center_history)
display(fig_hist)
save("stokes2d_poiseuille_unsteady_center.png", fig_hist)

println("Saved figures: stokes2d_poiseuille_unsteady_profile.png, stokes2d_poiseuille_unsteady_center.png")

############
# Animation of the relaxation
############

to_Ux(state) = reshape(state[1:nu_x], (length(xs), length(ys)))

Ux_first = to_Ux(states[1])
heatmap_data = Observable(Ux_first')

profile_data = Observable([Ux_first[icol, j] for j in 1:length(ys)])
time_obs = Observable(times[1])

fig_anim = Figure(resolution=(1100, 420))
ax_anim = Axis(fig_anim[1, 1], xlabel="x", ylabel="y",
               title=lift(t -> "uₓ field, t=$(round(t; digits=3))", time_obs))
hm = heatmap!(ax_anim, xs, ys, heatmap_data; colormap=:viridis)
Colorbar(fig_anim[1, 2], hm)

ax_profile = Axis(fig_anim[1, 3], xlabel="uₓ", ylabel="y",
                  title="Mid-column evolution")
lines!(ax_profile, profile_data, ys)

record(fig_anim, "stokes2d_poiseuille_unsteady.gif", eachindex(states)) do i
    Ux = to_Ux(states[i])
    heatmap_data[] = Ux'
    profile_data[] = [Ux[icol, j] for j in 1:length(ys)]
    time_obs[] = times[i]
end

println("Saved animation: stokes2d_poiseuille_unsteady.gif")
