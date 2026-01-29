using Penguin
using CairoMakie
using SpecialFunctions
using Roots

# Analytical helpers for the one-phase Stefan problem
function find_lambda(Stefan_number)
    f = (λ) -> sqrt(pi) * λ * exp(λ^2) * erf(λ) - 1 / Stefan_number
    return find_zero(f, (1e-6, 10.0), Bisection())
end

function analytical_temperature(x, t, T₀, k, lambda)
    t <= 0 && return T₀
    return T₀ - (T₀ / erf(lambda)) * erf(x / (2 * sqrt(k * t)))
end

analytical_position(t, k, lambda) = 2 * lambda * sqrt(t)

# Mesh and initial geometry
nx = 32
lx = 1.0  
x0 = 0.0
mesh = Penguin.Mesh((nx,), (lx,), (x0,))
x_offset = mesh.nodes[1][1] - x0  # mesh nodes start at x0 + Δx/2
Δx = lx / nx

# Time setup
Δt = 0.5 * Δx^2
Tstart = 0.03
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Tstart + Δt], tag=mesh.tag)

# Stefan parameters and initial interface
Stefan_number = 1.0
lambda = find_lambda(Stefan_number)
xf0_phys = analytical_position(Tstart, 1.0, lambda)
xf0 = xf0_phys + x_offset
body = (x, t, _=0) -> (x - xf0)

# Physics
capacity = Capacity(body, STmesh)
operator = DiffusionOps(capacity)
f = (x, y, z, t) -> 0.0
K = (x, y, z) -> 1.0
phase = Phase(capacity, operator, f, K)

# Boundary and interface conditions
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
bc_i = Dirichlet(0.0)
ρ, L = 1.0, 1.0
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ * L))

# Initial temperature from the analytical solution at Tstart
x_nodes_phys = mesh.nodes[1] .- x_offset
u0ₒ = analytical_temperature.(x_nodes_phys, Tstart, 1.0, 1.0, lambda)
u0ᵧ = zeros(nx + 1)
u0 = vcat(u0ₒ, u0ᵧ)

# Build solver and run the least-squares Stefan update
solver = StefanMono1D(phase, bc_b, bc_i, Δt, u0, mesh, "BE")
println("Solving the Stefan problem with 1D least-squares interface update...")
solver, residuals, xf_log, timestep_history = solve_StefanMono1D!(
    solver, phase, xf0, Δt, Tstart, Tend, bc_b, bc_i, stef_cond, mesh, "BE";
    method=Base.:\,
    Newton_params=(10, eps(), eps(), 1.0),
    fd_eps=1e-6,
    step_max=Δx
)
println("Simulation complete!")

# Analytical function at Tend
T_analytical = x -> analytical_temperature(x, Tend, 1.0, 1.0, lambda)

# Plot temperature at final time
function plot_temperature(
    T_exact,
    u_num,
    x_num,
    xf_final;
    save_path::String="temperature_comparison_1d_ls.png"
)
    x_fine = range(minimum(x_num), maximum(x_num), 2000)
    u_exact = T_exact.(x_fine)

    fig = Figure(resolution=(900, 600), fontsize=14)
    ax = Axis(fig[1, 1], xlabel="Position (x)", ylabel="Temperature",
        title="Stefan 1D (least squares): temperature at t = $(Tend)")

    lines!(ax, x_fine, u_exact, color=:red, linewidth=3, label="Analytical")
    scatter!(ax, x_num, u_num, color=:blue, markersize=6, label="Numerical")
    vlines!(ax, xf_final, color=:black, linewidth=2, linestyle=:dash, label="Interface")
    hlines!(ax, 0.0, color=:green, linewidth=2, linestyle=:dot, label="T=0")
    axislegend(ax, position=:rt, framevisible=true, backgroundcolor=(:white, 0.7))

    if !isempty(save_path)
        save(save_path, fig, px_per_unit=4)
    end
    display(fig)
    return fig
end

# Plot interface trajectory
function plot_interface(times, xf_vals; lambda=1.0, save_path::String="interface_position_1d_ls.png")
    t_exact = range(0.0, maximum(times), 500)
    xf_exact = analytical_position.(t_exact, 1.0, lambda)

    fig = Figure(resolution=(900, 600), fontsize=14)
    ax = Axis(fig[1, 1], xlabel="Time (t)", ylabel="Interface position",
        title="Stefan 1D (least squares): interface position")
    lines!(ax, t_exact, xf_exact, color=:red, linewidth=3, label="Analytical")
    scatter!(ax, times, xf_vals, color=:blue, markersize=8, label="Numerical")
    axislegend(ax, position=:lt, framevisible=true, backgroundcolor=(:white, 0.7))

    if !isempty(save_path)
        save(save_path, fig, px_per_unit=4)
    end
    display(fig)
    return fig
end

# Time stamps for logged interfaces (skip the initial entry in timestep_history)
times = [tp[1] for tp in timestep_history[2:end]]
xf_log_phys = xf_log .- x_offset
x_num_phys = mesh.nodes[1] .- x_offset

println("Plotting temperature comparison...")
plot_temperature(T_analytical, solver.x[1:(nx + 1)], x_num_phys, xf_log_phys[end])

println("Plotting interface position...")
plot_interface(times, xf_log_phys; lambda=lambda)

# Basic diagnostics
xf_num = xf_log_phys[end]
xf_exact = analytical_position(times[end], 1.0, lambda)
println("Final interface position: numerical = $xf_num, analytical = $xf_exact")

mask = x_num_phys .<= xf_num
if any(mask)
    u_num_below = solver.x[1:(nx + 1)][mask]
    u_exact_below = analytical_temperature.(x_num_phys[mask], Tend, 1.0, 1.0, lambda)
    L2_error = sqrt(sum((u_num_below .- u_exact_below).^2) / length(u_num_below))
    println("L2 temperature error below interface at Tend: $L2_error")
end
