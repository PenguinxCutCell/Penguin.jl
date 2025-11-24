using Penguin
using CairoMakie
using LinearAlgebra

"""
Oscillatory inflow around a circular cylinder to trigger a von Kármán vortex
street. The inflow is parabolic with a small sinusoidal modulation. The case is
run with the Navier–Stokes prototype solver; viscosity is chosen to forecast a
moderate Reynolds number where periodic shedding happens.
"""

# ---------------------------------------------------------------------------
# Geometry and meshes
# ---------------------------------------------------------------------------
Lx, Ly = 6.0, 2.0
x0, y0 = -1.0, -1.0
nx, ny = 256, 128

mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))

circle_center = (0.0, 0.0)
circle_radius = 0.2
circle_body = (x, y, _=0.0) -> circle_radius - sqrt((x - circle_center[1])^2 + (y - circle_center[2])^2)

capacity_ux = Capacity(circle_body, mesh_ux)
capacity_uy = Capacity(circle_body, mesh_uy)
capacity_p  = Capacity(circle_body, mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

# ---------------------------------------------------------------------------
# Physical parameters and inflow definition
# ---------------------------------------------------------------------------
ρ = 1.0
Re = 150.0                    # moderate Reynolds number
U_ref = 1.0
μ = ρ * U_ref * (2 * circle_radius) / Re

ω = 2π * 0.5                  # forcing frequency
β = 0.1                       # amplitude of oscillation compared to base flow

function inflow_profile(x, y, t)
    ym = y0 + Ly / 2
    η = (y - ym) / (Ly / 2)
    base = U_ref * (1 - η^2)
    return base * (1 + β * sin(ω * t))
end

# impose a small transverse oscillation as well to jump-start asymmetry
function inflow_transverse(x, y, t)
    ym = y0 + Ly / 2
    return 0.05 * U_ref * sin(ω * t) * exp(-((y - ym)/(Ly/4))^2)
end

# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
ux_left   = Dirichlet(inflow_profile)
ux_right  = Periodic()
ux_bottom = Dirichlet((x, y, t)->0.0)
ux_top    = Dirichlet((x, y, t)->0.0)

uy_left   = Dirichlet(inflow_transverse)
uy_right  = Dirichlet((x, y, t)->0.0)
uy_bottom = Dirichlet((x, y, t)->0.0)
uy_top    = Dirichlet((x, y, t)->0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right, :bottom=>ux_bottom, :top=>ux_top
))
bc_uy = BorderConditions(Dict(
    :left=>uy_left, :right=>uy_right, :bottom=>uy_bottom, :top=>uy_top
))

pressure_gauge = MeanPressureGauge()
interface_bc = Dirichlet(0.0)  # no-slip on the cylinder

# body forces (none)
fᵤ = (x, y, z=0.0) -> 0.0
fₚ = (x, y, z=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

# initial state (start from rest)
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
Ntot = 2 * (nu_x + nu_y) + np
x0_vec = zeros(Float64, Ntot)

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)

# ---------------------------------------------------------------------------
# Time integration parameters
# ---------------------------------------------------------------------------
Δt = 0.01
T_end = 0.05

println("Running oscillatory cylinder flow up to T=$(T_end) (Δt=$(Δt))")

probe = ComplexF64[]
probe_times = Float64[]
probe_point = (1.0, 0.0)
probe_ix = nearest_index(mesh_ux.nodes[1], probe_point[1])
probe_iy = nearest_index(mesh_ux.nodes[2], probe_point[2])
nx_nodes = length(mesh_ux.nodes[1])

store_states = true
if store_states
    times, states = solve_NavierStokesMono_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=:CN)
    for (ti, st) in zip(times, states)
        ux = st[1:nu_x]
        push!(probe, Complex(ux[(probe_iy-1)*nx_nodes + probe_ix], 0.0))
        push!(probe_times, ti)
    end
else
    θ = Penguin.scheme_to_theta(:CN)
    data = Penguin.navierstokes2D_blocks(solver)
    conv_prev = solver.prev_conv
    x_prev = copy(solver.x)
    p_offset = 2 * (data.nu_x + data.nu_y)
    np = data.np
    p_half_prev = zeros(np)
    t = 0.0
    while t < T_end - 1e-12
        dt_step = min(Δt, T_end - t)
        t_next = t + dt_step
        conv_curr = Penguin.assemble_navierstokes2D_unsteady!(solver, data, dt_step, x_prev, p_half_prev, t, t_next, θ, conv_prev)
        Penguin.solve_navierstokes_linear_system!(solver; method=Base.:\)
        x_prev = copy(solver.x)
        p_half_prev .= solver.x[p_offset+1:p_offset+np]
        conv_prev = (copy(conv_curr[1]), copy(conv_curr[2]))

        ux = solver.x[1:nu_x]
        # project probe velocity (use stored indices)
        push!(probe, Complex(ux[(probe_iy-1)*nx_nodes + probe_ix], 0.0))
        push!(probe_times, t_next)

        t = t_next
    end
    solver.prev_conv = conv_prev
end

println("Simulation finished.")

# ---------------------------------------------------------------------------
# Visualisation (final state)
# ---------------------------------------------------------------------------
xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]
Xp = mesh_p.nodes[1]
Yp = mesh_p.nodes[2]

uωx = solver.x[1:nu_x]
uωy = solver.x[2nu_x+1:2nu_x+nu_y]
pω  = solver.x[2*(nu_x + nu_y)+1:end]

Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))
P  = reshape(pω, (length(Xp), length(Yp)))

speed = sqrt.(Ux.^2 .+ Uy.^2)

fig = Figure(resolution=(1200, 700))

ax_speed = Axis(fig[1,1], xlabel="x", ylabel="y", title="Speed magnitude")
hm_speed = heatmap!(ax_speed, xs, ys, speed; colormap=:plasma)
contour!(ax_speed, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig[1,2], hm_speed)

ax_pressure = Axis(fig[1,3], xlabel="x", ylabel="y", title="Pressure field")
hm_pressure = heatmap!(ax_pressure, Xp, Yp, P; colormap=:balance)
contour!(ax_pressure, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig[1,4], hm_pressure)

ax_stream = Axis(fig[2,1], xlabel="x", ylabel="y", title="Velocity streamlines")
velocity_field(x, y) = Point2f(Ux[nearest_index(xs, x), nearest_index(ys, y)],
                               Uy[nearest_index(xs, x), nearest_index(ys, y)])
streamplot!(ax_stream, velocity_field, xs[1]..xs[end], ys[1]..ys[end]; colormap=:thermal)
contour!(ax_stream, xs, ys, [circle_body(x,y) for x in xs, y in ys]; levels=[0.0], color=:red, linewidth=2)

ax_probe = Axis(fig[2,3], xlabel="time", ylabel="u_x(probe)", title="Probe history at x=1")
lines!(ax_probe, probe_times, real.(probe))

save("navierstokes2d_vonkarman.png", fig)
display(fig)
