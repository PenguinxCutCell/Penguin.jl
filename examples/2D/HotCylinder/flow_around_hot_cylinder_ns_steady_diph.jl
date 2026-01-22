using Penguin
using CairoMakie
using LinearAlgebra

"""
Two-stage demo: (1) solve steady Navier–Stokes flow past a circular cylinder
using a Picard nonlinear solve; (2) feed the converged velocity into an
unsteady diphasic advection–diffusion run so the cylinder interior
temperature is explicitly resolved. The script saves velocity and temperature
snapshots, a simple animation, and a VTK dump for inspection.
"""

###########
# Geometry
###########
nx, ny = 192, 96
channel_length = 4.0
channel_height = 1.0
x0, y0 = -0.5, -0.5

circle_center = (0.5, 0.0)
circle_radius = 0.2
circle_body = (x, y, _=0) -> circle_radius - sqrt((x - circle_center[1])^2 + (y - circle_center[2])^2)
circle_out = (x, y, _=0) -> circle_body(x, y)
circle_in = (x, y, _=0) -> -circle_out(x, y)

###########
# Meshes
###########
mesh_p  = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0 - 0.5*dy))

###########
# Capacities & operators (cut cell aware)
###########
capacity_ux = Capacity(circle_body, mesh_ux)
capacity_uy = Capacity(circle_body, mesh_uy)
capacity_p  = Capacity(circle_body, mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions for Navier–Stokes step
###########
Umax = 1.0
parabolic = (x, y, t=0.0) -> begin
    ξ = (y - (y0 + channel_height/2)) / (channel_height/2)
    Umax * (1 - ξ^2)
end

ux_left   = Dirichlet((x, y, t=0.0) -> parabolic(x, y, t))
ux_right  = Dirichlet((x, y, t=0.0) -> parabolic(x, y, t))
ux_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
ux_top    = Dirichlet((x, y, t=0.0) -> 0.0)

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right, :bottom=>ux_bottom, :top=>ux_top
))
bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero
))
pressure_gauge = MeanPressureGauge()

u_bc = Dirichlet(0.0)  # enforce u^γ = 0 on the interface

###########
# Fluid properties
###########
μ = 0.01
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
# Solve steady Navier–Stokes (Picard)
###########
nu = prod(operator_ux.size)
np = prod(operator_p.size)
x0_vec = zeros(4*nu + np)

ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, u_bc; x0=x0_vec)
iters, final_res = solve_NavierStokesMono_steady!(ns_solver; nlsolve_method=:picard, tol=1e-8, maxiter=40, relaxation=1.0)
println("Picard iterations = ", iters, ", final residual = ", final_res)

println("Navier–Stokes flow converged. Unknowns = ", length(ns_solver.x))

uωx = ns_solver.x[1:nu]
uωy = ns_solver.x[2nu+1:3nu]
pω  = ns_solver.x[4nu+1:end]
println("Velocity sanity check: max |u_y| = ", maximum(abs, uωy))

###########
# Map velocity to the scalar mesh
###########
xs_ux, ys_ux = mesh_ux.nodes
xs_uy, ys_uy = mesh_uy.nodes
Xp, Yp = mesh_p.nodes

Ux = reshape(uωx, (length(xs_ux), length(ys_ux)))
Uy = reshape(uωy, (length(xs_uy), length(ys_uy)))

nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))

function map_field_to_mesh(src_xs, src_ys, field, dst_xs, dst_ys)
    mapped = zeros(length(dst_xs) * length(dst_ys))
    for (j, y) in enumerate(dst_ys), (i, x) in enumerate(dst_xs)
        idx = i + (j - 1) * length(dst_xs)
        mapped[idx] = field[nearest_index(src_xs, x), nearest_index(src_ys, y)]
    end
    mapped
end

uₒx = map_field_to_mesh(xs_ux, ys_ux, Ux, Xp, Yp)
uₒy = map_field_to_mesh(xs_uy, ys_uy, Uy, Xp, Yp)
uᵧ = zeros(2 * length(Xp) * length(Yp))  # interface velocity (unused here)

###########
# Diphasic advection–diffusion setup
###########
capacity_out = Capacity(circle_out, mesh_p)
capacity_in = Capacity(circle_in, mesh_p)
operator_adv_out = ConvectionOps(capacity_out, (uₒx, uₒy), uᵧ)
uₒx_zero = zeros(length(uₒx))
uₒy_zero = zeros(length(uₒy))
operator_adv_in = ConvectionOps(capacity_in, (uₒx_zero, uₒy_zero), uᵧ)

T_cold = 0.0
T_hot = 1.0
D_in, D_out = 1.0e-2, 1.0e-1
κ_out = (x, y, _=0.0) -> D_out
κ_in = (x, y, _=0.0) -> D_in
f_heat = (x, y, z, t) -> 0.0

phase_out = Phase(capacity_out, operator_adv_out, f_heat, κ_out)
phase_in = Phase(capacity_in, operator_adv_in, f_heat, κ_in)

ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(D_in, D_out, 0.0))
bc_b = BorderConditions(Dict(
    :left=>Dirichlet(T_cold),
    :right=>Dirichlet(T_cold),
    :bottom=>Dirichlet(T_cold),
    :top=>Dirichlet(T_cold)
))

Nnodes = length(Xp) * length(Yp)
T0ₒ_out = fill(T_cold, Nnodes)
T0ᵧ_out = fill(T_hot, Nnodes)
T0ₒ_in = fill(T_hot, Nnodes)
T0ᵧ_in = fill(T_hot, Nnodes)

T0 = vcat(T0ₒ_out, T0ᵧ_out, T0ₒ_in, T0ᵧ_in)

Δt = 0.01
Tend = 2.0
advdiff_solver = AdvectionDiffusionUnsteadyDiph(phase_out, phase_in, bc_b, ic, Δt, T0, "BE")
solve_AdvectionDiffusionUnsteadyDiph!(advdiff_solver, phase_out, phase_in, Δt, Tend, bc_b, ic, "CN"; method=Base.:\)

###########
# Outputs: VTK, snapshots, animation
###########
#write_vtk("hot_cylinder_advdiff", mesh_p, advdiff_solver)

# Velocity magnitude snapshot on scalar mesh (used for advection)
u_speed = sqrt.(reshape(uₒx, (length(Xp), length(Yp))).^2 .+ reshape(uₒy, (length(Xp), length(Yp))).^2)

fig_vel = Figure(resolution=(1100, 450))
ax_vel = Axis(fig_vel[1, 1], xlabel="x", ylabel="y", title="Navier–Stokes speed (used for advection)", aspect=DataAspect())
hm_vel = heatmap!(ax_vel, Xp, Yp, u_speed; colormap=:plasma)
contour!(ax_vel, Xp, Yp, [circle_body(x,y) for x in Xp, y in Yp]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig_vel[1, 2], hm_vel, label="|u|")
save("hot_cylinder_velocity_ns_steady_diph.png", fig_vel)

# Temperature snapshots (initial / mid / final) Axis position top to bottom +1 colorbar for all
states = advdiff_solver.states
times_to_show = [1, max(1, length(states) ÷ 2), length(states)]

circle_vals = [circle_body(x,y) for x in Xp, y in Yp]
inside_mask = circle_vals .> 0

function combined_temperature(state, Nnodes, nx_nodes, ny_nodes, inside_mask)
    T_out = reshape(state[1:Nnodes], (nx_nodes, ny_nodes))
    T_in = reshape(state[2Nnodes+1:3Nnodes], (nx_nodes, ny_nodes))
    T_full = copy(T_out)
    T_full[inside_mask] .= T_in[inside_mask]
    return T_full
end

fig_T = Figure(resolution=(1400, 480))
for (i, t_idx) in enumerate(times_to_show)
    ax_T = Axis(fig_T[1, i], xlabel="x", ylabel="y",
                title="Temperature at t = $(round((t_idx-1)*Δt, digits=2))", aspect=DataAspect())
    T_snapshot = combined_temperature(states[t_idx], Nnodes, length(Xp), length(Yp), inside_mask)
    global hm_T = heatmap!(ax_T, Xp, Yp, T_snapshot; colormap=:viridis, colorrange=(0.0, 1.0))
    contour!(ax_T, Xp, Yp, T_snapshot; levels=0.1:0.2:0.9, color=:black, linewidth=1.0)
    contour!(ax_T, Xp, Yp, circle_vals; levels=[0.0], color=:white, linewidth=2)
end
Colorbar(fig_T[1, end+1], hm_T, label="Temperature")
save("hot_cylinder_temperature_snapshots_ns_steady_diph.png", fig_T)

# Animation of the temperature field
fig_anim = Figure(resolution=(900, 450))
ax_anim = Axis(fig_anim[1, 1], xlabel="x", ylabel="y",
               title="Hot cylinder advection–diffusion (diphasic)", aspect=DataAspect())
T_first = combined_temperature(states[1], Nnodes, length(Xp), length(Yp), inside_mask)
heatmap!(ax_anim, Xp, Yp, T_first; colormap=:viridis, colorrange=(0.0, 1.0))
contour!(ax_anim, Xp, Yp, T_first; levels=0.1:0.2:0.9, color=:black, linewidth=1.0)
contour!(ax_anim, Xp, Yp, circle_vals; levels=[0.0], color=:white, linewidth=2)


record(fig_anim, "hot_cylinder_temperature_ns_steady_diph.mp4", 1:length(states); framerate=12) do frame
    T_frame = combined_temperature(states[frame], Nnodes, length(Xp), length(Yp), inside_mask)
    heatmap!(ax_anim, Xp, Yp, T_frame; colormap=:viridis, colorrange=(0.0, 1.0))
    contour!(ax_anim, Xp, Yp, T_frame; levels=0.1:0.2:0.9, color=:black, linewidth=1.0)
end


###########
# Nusselt number diagnostics
###########
outer_state = vcat(states[end][1:Nnodes], states[end][Nnodes+1:2Nnodes])
θ, Nu_local, Nu_mean = nusselt_profile(operator_adv_out, capacity_out, outer_state;
    center=circle_center, k=1.0e-2, Tw=T_hot, Tinf=T_cold, Lchar=2*circle_radius)
perm = sortperm(θ)
θ_sorted = θ[perm]
Nu_sorted = Nu_local[perm]
Nu_theta = (θ=θ_sorted, Nu=Nu_sorted)
println("Mean Nusselt number on the cylinder ≈ ", Nu_mean)
