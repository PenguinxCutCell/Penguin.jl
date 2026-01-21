using Penguin
using SpecialFunctions
using CairoMakie
using Roots

# ------------------------------------------------------------
# Analytical helpers for the two-phase Stefan problem
# ------------------------------------------------------------

# Solve for λ in the two-phase Neumann solution:
# sqrt(π) λ = Ste₁ * exp(-λ²) / erf(λ) + Ste₂ * β * exp(-(βλ)²) / erfc(βλ)
# with β = sqrt(α₁ / α₂). Here we default to α₁ = α₂ = 1 so β = 1.
function find_lambda_two_phase(Ste1, Ste2; beta::Float64=1.0)
    f = λ -> sqrt(pi) * λ - Ste1 * exp(-λ^2) / erf(λ) - Ste2 * beta * exp(-(beta * λ)^2) / erfc(beta * λ)
    return find_zero(f, (1e-6, 10.0), Bisection())
end

analytical_interface(t, α1, λ) = 2 * λ * sqrt(α1 * t)

function analytical_temperature_left(x, t, T_L, T_m, α1, λ)
    t <= 0 && return T_L
    return T_L - (T_L - T_m) * erf(x / (2 * sqrt(α1 * t))) / erf(λ)
end

function analytical_temperature_right(x, t, T_R, T_m, α1, α2, λ)
    t <= 0 && return T_R
    β = sqrt(α1 / α2)
    ξ = (x - analytical_interface(t, α1, λ)) / (2 * sqrt(α2 * t))
    return T_R + (T_m - T_R) * erfc(ξ) / erfc(β * λ)
end

# ------------------------------------------------------------
# Problem definition
# ------------------------------------------------------------
nx = 32
lx = 1.0
x0 = 0.0
mesh = Penguin.Mesh((nx,), (lx,), (x0,))
x_offset = mesh.nodes[1][1] - x0
Δx = lx / nx

Δt = 0.5 * Δx^2
Tstart = 0.03
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Tstart + Δt], tag=mesh.tag)

# Material/thermal parameters (symmetric case)
α1 = 1.0
α2 = 1.0
T_L, T_R, T_m = 1.0, 0.0, 0.0
ρ, Lh = 1.0, 1.0
Ste1 = (T_L - T_m) / Lh
Ste2 = (T_m - T_R) / Lh
beta = sqrt(α1 / α2)
lambda = find_lambda_two_phase(Ste1, Ste2; beta=beta)

xf0_phys = analytical_interface(Tstart, α1, lambda)
xf0 = xf0_phys + x_offset

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
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(T_R), :bottom => Dirichlet(T_L)))
ic = InterfaceConditions(ScalarJump(1.0, 1.0, T_m), FluxJump(1.0, 1.0, ρ * Lh))

# Initial condition from analytical profiles at Tstart
x_phys = mesh.nodes[1] .- x_offset
u0ₒ1 = analytical_temperature_left.(x_phys, Tstart, T_L, T_m, α1, lambda)
u0ᵧ1 = fill(T_m, nx + 1)
u0ₒ2 = analytical_temperature_right.(x_phys, Tstart, T_R, T_m, α1, α2, lambda)
u0ᵧ2 = fill(T_m, nx + 1)
u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Solver setup
solver = MovingLiquidDiffusionUnsteadyDiph(phase1, phase2, bc_b, ic, Δt, u0, mesh, "BE")
println("Solving two-phase Stefan problem (1D, symmetric) with analytical comparison...")
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
T_exact_left  = x -> analytical_temperature_left(x, Tend, T_L, T_m, α1, lambda)
T_exact_right = x -> analytical_temperature_right(x, Tend, T_R, T_m, α1, α2, lambda)

function piecewise_exact(x, xf, T_left, T_right)
    map(xi -> xi <= xf ? T_left(xi) : T_right(xi), x)
end

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

# Plot temperature comparison
function plot_temperature(T_left, T_right, u_num, x_num, xf_num, xf_exact; save_path="temperature_comparison_2ph_simple.png")
    x_fine = range(minimum(x_num), maximum(x_num), 2000)
    u_exact = piecewise_exact(x_fine, xf_exact, T_left, T_right)

    fig = Figure(resolution=(900, 600), fontsize=14)
    ax = Axis(fig[1, 1], xlabel="Position (x)", ylabel="Temperature",
        title="Two-phase Stefan: Temperature at t = $(Tend)")

    lines!(ax, x_fine, u_exact, color=:red, linewidth=3, label="Analytical")
    scatter!(ax, x_num, u_num, color=:blue, markersize=6, label="Numerical")
    vlines!(ax, xf_exact, color=:black, linewidth=2, linestyle=:dash, label="Interface (analytical)")
    vlines!(ax, xf_num, color=:orange, linewidth=2, linestyle=:dot, label="Interface (numerical)")
    hlines!(ax, T_m, color=:green, linewidth=2, linestyle=:dot, label="T = T_m")
    axislegend(ax, position=:rt, framevisible=true, backgroundcolor=(:white, 0.7))

    !isempty(save_path) && save(save_path, fig, px_per_unit=4)
    display(fig)
    return fig
end

# Plot interface trajectory
function plot_interface(xf_vals, Δt, Tstart; lambda=1.0, α1=1.0, save_path="interface_position_2ph_simple.png")
    times = Tstart .+ collect(1:length(xf_vals)) .* Δt
    t_exact = range(0.0, maximum(times), 500)
    xf_exact = analytical_interface.(t_exact, α1, lambda)

    fig = Figure(resolution=(900, 600), fontsize=14)
    ax = Axis(fig[1, 1], xlabel="Time (t)", ylabel="Interface position",
        title="Two-phase Stefan: Interface position vs time")
    lines!(ax, t_exact, xf_exact, color=:red, linewidth=3, label="Analytical")
    scatter!(ax, times, xf_vals, color=:blue, markersize=8, label="Numerical")
    axislegend(ax, position=:lt, framevisible=true, backgroundcolor=(:white, 0.7))

    !isempty(save_path) && save(save_path, fig, px_per_unit=4)
    display(fig)
    return fig
end

println("Plotting temperature comparison...")
xf_exact_Tend = analytical_interface(Tend, α1, lambda)
plot_temperature(T_exact_left, T_exact_right, u_num, x_num, xf_final, xf_exact_Tend)

println("Plotting interface position...")
plot_interface(xf_log_phys, Δt, Tstart; lambda=lambda, α1=α1)

# Simple diagnostics
xf_exact = xf_exact_Tend
pos_err = abs(xf_final - xf_exact)
println("Final interface position: numerical = $xf_final, analytical = $xf_exact, abs error = $pos_err", "rel error = $(pos_err / abs(xf_exact))")

mask = x_num .<= xf_final
if any(mask)
    u_exact_below = T_exact_left.(x_num[mask])
    u_num_below = u_num[mask]
    L2_error = sqrt(sum((u_num_below .- u_exact_below).^2) / length(u_num_below))
    println("L2 temperature error on left side at Tend: $L2_error")
end

println("Done.")
