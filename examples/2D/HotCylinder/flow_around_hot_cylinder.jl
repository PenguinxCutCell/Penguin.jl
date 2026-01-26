using Penguin
using CairoMakie
using LinearAlgebra

"""
Two-stage demo: (1) solve steady Stokes flow past a circular cylinder in a
rectangular channel; (2) feed the converged velocity into an unsteady
advection–diffusion run with a hot cylinder wall. The script saves velocity
and temperature snapshots, a simple animation, and a VTK dump for inspection.
"""

###########
# Geometry
###########
nx, ny = 64, 64
channel_length = 10.0
channel_height = 10.0
x0, y0 = -0.5, -0.5

circle_center = (0.5, 0.0)
circle_radius = 0.2
circle_body = (x, y, _=0) -> circle_radius - sqrt((x - circle_center[1])^2 + (y - circle_center[2])^2)

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
# Boundary conditions for Stokes step
###########
Umax = 1.0
parabolic = (x, y) -> begin
    ξ = (y - (y0 + channel_height/2)) / (channel_height/2)
    Umax * (1 - ξ^2)
end

ux_left   = Dirichlet((x, y) -> parabolic(x, y))
ux_right  = Dirichlet((x, y) -> parabolic(x, y))
ux_bottom = Dirichlet((x, y) -> 0.0)
ux_top    = Dirichlet((x, y) -> 0.0)

uy_zero = Dirichlet((x, y) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right, :bottom=>ux_bottom, :top=>ux_top
))
bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero
))
pressure_gauge = PinPressureGauge()

u_bc = Dirichlet(0.0)  # enforce u^γ = 0 on the interface

###########
# Fluid properties
###########
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
# Solve Stokes
###########
nu = prod(operator_ux.size)
np = prod(operator_p.size)
x0_vec = zeros(4*nu + np)

stokes_solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, u_bc; x0=x0_vec)
solve_StokesMono!(stokes_solver; method=Base.:\)

println("Stokes flow converged. Unknowns = ", length(stokes_solver.x))

uωx = stokes_solver.x[1:nu]
uωy = stokes_solver.x[2nu+1:3nu]
pω  = stokes_solver.x[4nu+1:end]
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
# Advection–diffusion setup
###########
capacity_T = Capacity(circle_body, mesh_p)
operator_adv = ConvectionOps(capacity_T, (uₒx, uₒy), uᵧ)

T_cold = 0.0
T_hot = 1.0
k = 1.0e-2  # thermal diffusivity
κ = (x, y, _=0.0) -> k
f_heat = (x, y, z, t) -> 0.0

phase_heat = Phase(capacity_T, operator_adv, f_heat, κ)

ic = Dirichlet(T_hot)  # hot cylinder interface
bc_b = BorderConditions(Dict(
    :left=>Dirichlet(T_cold),
    :right=>Dirichlet(T_cold),
    :bottom=>Dirichlet(T_cold),
    :top=>Dirichlet(T_cold)
))

Nnodes = length(Xp) * length(Yp)
T0ₒ = fill(T_cold, Nnodes)
T0ᵧ = fill(T_hot, Nnodes)

# Warm up the interior of the cylinder
for (j, y) in enumerate(Yp), (i, x) in enumerate(Xp)
    if circle_body(x, y) > 0   # inside the cylinder
        idx = i + (j - 1) * length(Xp)
        T0ₒ[idx] = T_hot
    end
end

T0 = vcat(T0ₒ, T0ᵧ)

Δt = 0.5 * min(dx, dy)^2
Tend = 10.0
advdiff_solver = AdvectionDiffusionUnsteadyMono(phase_heat, bc_b, ic, Δt, T0, "BE")
solve_AdvectionDiffusionUnsteadyMono!(advdiff_solver, phase_heat, Δt, Tend, bc_b, ic, "CN"; method=Base.:\)


n = prod(operator_adv.size)
Tω = @view advdiff_solver.states[end][1:n]
Tγ = @view advdiff_solver.states[end][n+1:2n]

# Discrete volume interface contribution : volume integrated normal heat flux on the interface : ∫_{Γ} q_n ds
Q = operator_adv.H' * operator_adv.Wꜝ * (operator_adv.G * Tω + operator_adv.H * Tγ)

# Extract interface lengths
Γ = diag(capacity_T.Γ)

# Compute mean interfacial fluxes : ∫_{Γ} q_n ds. It is already integrated over the interface length
q_nds = Q 

# Divide by total interface length to get mean flux
q_n_mean = sum(q_nds[.!isnan.(q_nds)])
q_n_mean /= sum(Γ)

# Multiply by diffusivity to get flux
q_n_mean *= k

println("Mean interfacial heat flux: $q_n_mean")

# Compute mean temperature at the interface
T_interface = 0.0
for i in 1:length(Tγ)
    if !isnan(Γ[i]) && Γ[i] > 0.0
        global T_interface += Tγ[i] * Γ[i]
    end
end
T_interface /= sum(Γ[.!isnan.(Γ)])
println("Mean interfacial temperature: $T_interface")

# compute heat transfer coefficient
T_ref = 0.0  # reference temperature in liquid far from interface
h = q_n_mean / (T_interface - T_ref)
println("Heat transfer coefficient h: $h")

# Characteristic length
L = 2 * circle_radius

# Compute Nusselt number
Nu = h * L / k
println("Nusselt number Nu: $Nu")




###########
# Nusselt number diagnostics
###########
θ, Nu_local, Nu_mean = nusselt_profile(operator_adv, capacity_T, advdiff_solver.states[end]; Tγ_override=T_hot, center=circle_center, k=k, Lchar=2*circle_radius)
perm = sortperm(θ)
θ_sorted = θ[perm]
Nu_sorted = Nu_local[perm]
Nu_theta = (θ=θ_sorted, Nu=Nu_sorted)
println("Mean Nusselt number on the cylinder ≈ ", Nu_mean)


###########
# Outputs: VTK, snapshots, animation
###########
#write_vtk("hot_cylinder_advdiff", mesh_p, advdiff_solver)

# Velocity magnitude snapshot on scalar mesh (used for advection)
u_speed = sqrt.(reshape(uₒx, (length(Xp), length(Yp))).^2 .+ reshape(uₒy, (length(Xp), length(Yp))).^2)

fig_vel = Figure(resolution=(1100, 450))
ax_vel = Axis(fig_vel[1, 1], xlabel="x", ylabel="y", title="Stokes speed (used for advection)", aspect=DataAspect())
hm_vel = heatmap!(ax_vel, Xp, Yp, u_speed; colormap=:plasma)
contour!(ax_vel, Xp, Yp, [circle_body(x,y) for x in Xp, y in Yp]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig_vel[1, 2], hm_vel, label="|u|")
save("hot_cylinder_velocity.png", fig_vel)

# Temperature snapshots (initial / mid / final) Axis position top to bottom +1 colorbar for all
states = advdiff_solver.states
times_to_show = [1, max(1, length(states) ÷ 2), length(states)]

fig_T = Figure(resolution=(1400, 480))
for (i, t_idx) in enumerate(times_to_show)
    ax_T = Axis(fig_T[1, i], xlabel="x", ylabel="y",
                title="Temperature at t = $(round((t_idx-1)*Δt, digits=2))", aspect=DataAspect())
    T_snapshot = reshape(states[t_idx][1:Nnodes], (length(Xp), length(Yp)))
    # Enforce hot cylinder boundary
    circle_vals = [circle_body(x,y) for x in Xp, y in Yp]
    T_snapshot[circle_vals .> 0] .= T_hot
    global hm_T = heatmap!(ax_T, Xp, Yp, T_snapshot; colormap=:viridis, colorrange=(0.0, 1.0))
    contour!(ax_T, Xp, Yp, T_snapshot; levels=0.1:0.2:0.9, color=:black, linewidth=1.0)
    contour!(ax_T, Xp, Yp, circle_vals; levels=[0.0], color=:white, linewidth=2)
end
Colorbar(fig_T[1, end+1], hm_T, label="Temperature")
save("hot_cylinder_temperature_snapshots.png", fig_T)

# Animation of the temperature field
Tmin = minimum([minimum(state[1:Nnodes]) for state in states])
Tmax = maximum([maximum(state[1:Nnodes]) for state in states])

fig_anim = Figure(resolution=(900, 450))
ax_anim = Axis(fig_anim[1, 1], xlabel="x", ylabel="y",
               title="Hot cylinder advection–diffusion", aspect=DataAspect())
T_first = reshape(states[1][1:Nnodes], (length(Xp), length(Yp)))
circle_vals = [circle_body(x,y) for x in Xp, y in Yp]
T_first[circle_vals .> 0] .= 1.0
heatmap!(ax_anim, Xp, Yp, T_first; colormap=:viridis, colorrange=(0.0, 1.0))
contour!(ax_anim, Xp, Yp, T_first; levels=0.1:0.2:0.9, color=:black, linewidth=1.0)
contour!(ax_anim, Xp, Yp, circle_vals; levels=[0.0], color=:white, linewidth=2)


record(fig_anim, "hot_cylinder_temperature.mp4", 1:length(states); framerate=12) do frame
    T_frame = reshape(states[frame][1:Nnodes], (length(Xp), length(Yp)))
    T_frame[circle_vals .> 0] .= 1.0
    heatmap!(ax_anim, Xp, Yp, T_frame; colormap=:viridis, colorrange=(0.0, 1.0))
    contour!(ax_anim, Xp, Yp, T_frame; levels=0.1:0.2:0.9, color=:black, linewidth=1.0)
end