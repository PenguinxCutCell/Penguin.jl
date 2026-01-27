using Penguin
using SpecialFunctions
using CairoMakie
using Roots

# ------------------------------------------------------------
# Favier test case: Two-phase Stefan problem with specific initial profile
# Domain: [0, 1]
# Bottom temperature: 1.0, Top temperature: 0.0
# Initial interface at x = 1/5, Melting temperature T_m = 1/5
# Initial temperature profile: (exp(-β(x-1))-1)/(exp(-β)-1) with β = 8.041
# Stefan number = 1.0
# Expected steady state: T = 1 - x, interface at x = 4/5
# ------------------------------------------------------------

# ------------------------------------------------------------
# Problem definition
# ------------------------------------------------------------
nx = 32
lx = 1.0 + 1/nx  # Extend domain slightly to avoid boundary issues
x0 = 0.0
mesh = Penguin.Mesh((nx,), (lx,), (x0,))
x_offset = mesh.nodes[1][1] - x0
Δx = lx / nx

Δt = 0.5 * Δx^2
Tstart = 0.0
Tend = 1.0
STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Tstart + Δt], tag=mesh.tag)

# Material/thermal parameters
α1 = 1.0
α2 = 1.0
T_bottom = 1.0  # Bottom boundary temperature
T_top = 0.0     # Top boundary temperature
T_m = 1.0/5.0   # Melting temperature
ρ = 1.0
Ste = 1.0       # Stefan number
Lh = (T_bottom - T_m) / Ste  # Latent heat computed from Stefan number

# Initial interface position
xf0_phys = 1.0/5.0
xf0 = xf0_phys + x_offset

# Initial temperature profile parameter
β = 8.041

# Function for initial temperature profile
function initial_temperature(x)
    return (exp(-β * (x - 1.0)) - 1.0) / (exp(β) - 1.0)
end

# plot initial profile for verification
x_fine = range(x0, x0 + lx, length=200)
u_initial = initial_temperature.(x_fine)
fig_init = Figure(resolution=(800, 500), fontsize=14)
ax_init = Axis(fig_init[1, 1], xlabel="Position (x)", ylabel="Temperature",
    title="Initial Temperature Profile for Favier Test Case")
lines!(ax_init, x_fine, u_initial, color=:blue, linewidth=3, label="Initial Profile")
hlines!(ax_init, T_m, color=:red, linewidth=2, linestyle=:dash, label="Melting Temperature T_m")
axislegend(ax_init, position=:rt, framevisible=true, backgroundcolor=(:white, 0.7))
display(fig_init)

# Steady state profile (expected at t → ∞)
steady_state_temperature(x) = 1.0 - x
steady_state_interface = 4.0/5.0

body = (x, t, _=0) -> (x - xf0)
body_c = (x, t, _=0) -> -(x - xf0)

capacity = Capacity(body, STmesh)
capacity_c = Capacity(body_c, STmesh)
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

f1 = (x, y, z, t) -> 0.0
f2 = (x, y, z, t) -> 0.0
K1 = (x, y, z) -> α1
K2 = (x, y, z) -> α2

phase1 = Phase(capacity, operator, f1, K1)
phase2 = Phase(capacity_c, operator_c, f2, K2)

# Boundary and interface conditions
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(T_top), :bottom => Dirichlet(T_bottom)))
ic = InterfaceConditions(ScalarJump(1.0, 1.0, T_m), FluxJump(1.0, 1.0, ρ * Lh))

# Initial condition from specified profile
x_phys = mesh.nodes[1] .- x_offset
u0ₒ1 = initial_temperature.(x_phys)
u0ᵧ1 = fill(T_m, nx + 1)
u0ₒ2 = initial_temperature.(x_phys)
u0ᵧ2 = fill(T_m, nx + 1)
u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Print initial conditions
println("="^60)
println("Favier Test Case - Two-Phase Stefan Problem")
println("="^60)
println("Domain: [0, 1]")
println("Initial interface position: x = $(xf0_phys)")
println("Melting temperature: T_m = $(T_m)")
println("Bottom temperature: T = $(T_bottom)")
println("Top temperature: T = $(T_top)")
println("Stefan number: Ste = $(Ste)")
println("Latent heat: L = $(Lh)")
println("Initial profile parameter: β = $(β)")
println("Expected steady-state interface: x = $(steady_state_interface)")
println("Expected steady-state profile: T(x) = 1 - x")
println("="^60)

# Solver setup
solver = MovingLiquidDiffusionUnsteadyDiph(phase1, phase2, bc_b, ic, Δt, u0, mesh, "BE")
println("Solving Favier two-phase Stefan problem...")
solver, residuals, xf_log = solve_MovingLiquidDiffusionUnsteadyDiph!(
    solver, phase1, phase2, xf0, Δt, Tstart, Tend, bc_b, ic, mesh, "BE";
    Newton_params=(1, 1e-10, 1e-10, 1.0),
    method=Base.:\,
    adaptive_timestep=false,
    cfl_target=0.5
)
println("Simulation complete!")

# ------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------
xf_log_phys = xf_log .- x_offset
xf_final = xf_log_phys[end]
x_num = x_phys

# Build piecewise numerical field (phase1 on left, phase2 on right)
u1_bulk = solver.x[1:(nx + 1)]
u2_bulk = solver.x[2 * (nx + 1) + 1:3 * (nx + 1)]
u_num = similar(x_num)
mask_left = x_num .<= xf_final
u_num[mask_left] .= u1_bulk[mask_left]
u_num[.!mask_left] .= u2_bulk[.!mask_left]

# Plot temperature comparison with steady state
function plot_temperature_favier(u_num, x_num, xf_num, xf_steady, T_steady; save_path="temperature_favier.png")
    x_fine = range(minimum(x_num), maximum(x_num), 2000)
    u_steady = T_steady.(x_fine)

    fig = Figure(resolution=(900, 600), fontsize=14)
    ax = Axis(fig[1, 1], xlabel="Position (x)", ylabel="Temperature",
        title="Favier Test: Temperature at t = $(Tend)")

    lines!(ax, x_fine, u_steady, color=:red, linewidth=3, label="Steady State (1-x)")
    scatter!(ax, x_num, u_num, color=:blue, markersize=6, label="Numerical")
    vlines!(ax, xf_steady, color=:black, linewidth=2, linestyle=:dash, label="Steady Interface (x=4/5)")
    vlines!(ax, xf_num, color=:orange, linewidth=2, linestyle=:dot, label="Current Interface")
    hlines!(ax, T_m, color=:green, linewidth=2, linestyle=:dot, label="T = T_m (1/5)")
    axislegend(ax, position=:rt, framevisible=true, backgroundcolor=(:white, 0.7))

    !isempty(save_path) && save(save_path, fig, px_per_unit=4)
    display(fig)
    return fig
end

# Plot interface trajectory
function plot_interface_favier(xf_vals, Δt, Tstart, xf_steady; save_path="interface_position_favier.png")
    times = Tstart .+ collect(1:length(xf_vals)) .* Δt

    fig = Figure(resolution=(900, 600), fontsize=14)
    ax = Axis(fig[1, 1], xlabel="Time (t)", ylabel="Interface position",
        title="Favier Test: Interface position vs time")
    scatter!(ax, times, xf_vals, color=:blue, markersize=8, label="Numerical")
    hlines!(ax, xf_steady, color=:red, linewidth=3, linestyle=:dash, label="Steady State (x=4/5)")
    axislegend(ax, position=:rb, framevisible=true, backgroundcolor=(:white, 0.7))

    !isempty(save_path) && save(save_path, fig, px_per_unit=4)
    display(fig)
    return fig
end

println("\nPlotting temperature comparison with steady state...")
plot_temperature_favier(u_num, x_num, xf_final, steady_state_interface, steady_state_temperature)

println("Plotting interface position evolution...")
plot_interface_favier(xf_log_phys, Δt, Tstart, steady_state_interface)

# Diagnostics
pos_err = abs(xf_final - steady_state_interface)
println("\n" * "="^60)
println("DIAGNOSTICS")
println("="^60)
println("Final interface position: x = $xf_final")
println("Steady-state interface: x = $steady_state_interface")
println("Interface error: $(pos_err) ($(100*pos_err/steady_state_interface)%)")

# Compare with steady state temperature profile
u_steady = steady_state_temperature.(x_num)
L2_error = sqrt(sum((u_num .- u_steady).^2) / length(u_num))
println("L2 temperature error vs steady state: $L2_error")
println("="^60)

println("\nDone.")
