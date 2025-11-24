using Penguin
using CairoMakie
using FFTW
using Statistics
using Printf
"""
Unsteady Navier–Stokes benchmark: vortex shedding past a circular cylinder at Re = 100.
The setup mirrors the DFG 2D-3 benchmark geometry (channel 2.2 × 0.41 with a cylinder of
radius 0.05 at (0.2, 0.2)). A parabolic inflow with peak velocity 1.5 generates a mean
velocity of 1.0, yielding Re = 100 with kinematic viscosity ν = 1e-3. The script advances
the unsteady Navier–Stokes equations, samples drag/lift coefficients and pressure
difference, and compares the resulting statistics against published benchmark ranges.
"""
###########
# Geometry and discretisation
###########
nx, ny = 128, 64
Lx, Ly = 2.2, 0.41
x0, y0 = 0.0, 0.0
radius = 0.05
diameter = 2radius
cylinder_center = (0.2, 0.2)
circle_levelset = (x, y, _=0.0) -> radius - sqrt((x - cylinder_center[1])^2 + (y - cylinder_center[2])^2)
mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))
###########
# Capacities and operators
###########
capacity_ux = Capacity(circle_levelset, mesh_ux)
capacity_uy = Capacity(circle_levelset, mesh_uy)
capacity_p  = Capacity(circle_levelset, mesh_p)
operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)
###########
# Boundary conditions
###########
Umax = 1.5  # gives mean velocity 1.0 for parabolic profile
channel_height = Ly
parabolic = (x, y, t=0.0) -> begin
    ξ = y / channel_height
    Umax * 4 * ξ * (1 - ξ)
end
ux_left   = Dirichlet((x, y, t=0.0) -> parabolic(x, y, t))
ux_right  = Outflow()
ux_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
ux_top    = Dirichlet((x, y, t=0.0) -> 0.0)
uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)
bc_ux = BorderConditions(Dict(
    :left=>ux_left,
    :right=>ux_right,
    :bottom=>ux_bottom,
    :top=>ux_top,
))
bc_uy = BorderConditions(Dict(
    :left=>uy_zero,
    :right=>uy_zero,
    :bottom=>uy_zero,
    :top=>uy_zero,
))
pressure_gauge = PinPressureGauge()
cut_bc = Dirichlet(0.0)
###########
# Fluid properties and forcing
###########
ρ = 1.0
ν = 1e-3
μ = ρ * ν
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
x0_vec = zeros(2 * (nu_x + nu_y) + np)
solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, cut_bc; x0=x0_vec)
Δt = 0.002
T_end = 8.0
println("=== Cylinder vortex-shedding benchmark (Re=100) ===")
println("Grid: $(nx) × $(ny), Δt=$(Δt), T_end=$(T_end)")
###########
# Time integration
###########
times, histories = solve_NavierStokesMono_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=:CN)
println("Time steps: $(length(times) - 1)")
###########
# Diagnostics over time
###########
function dominant_frequency(signal::AbstractVector{<:Real}, dt::Real)
    n = length(signal)
    n < 16 && return NaN, NaN
    sig = signal .- mean(signal)
    spec = abs.(fft(sig))
    freqs = (0:n-1) .* (1 / (dt * n))
    half = 2:clamp(div(n, 2), 2, n)
    idx = half[argmax(spec[half])]
    return freqs[idx], spec[idx]
end
function extract_forces!(solver, state)
    solver.x .= state
    force_diag = compute_navierstokes_force_diagnostics(solver)
    coeffs = drag_lift_coefficients(force_diag; ρ=ρ, U_ref=1.0, length_ref=diameter, acting_on=:body)
    p_force = navierstokes_reaction_force_components(force_diag; acting_on=:body)
    return coeffs.Cd, coeffs.Cl, p_force
end
pressure_offset = 2 * (nu_x + nu_y)
nx_p = length(mesh_p.nodes[1])
ny_p = length(mesh_p.nodes[2])
function sample_pressure(points)
    pω = view(solver.x, pressure_offset + 1:length(solver.x))
    p_field = reshape(pω, (nx_p, ny_p))
    xs = mesh_p.nodes[1]
    ys = mesh_p.nodes[2]
    pressures = Float64[]
    for (xp, yp) in points
        i = clamp(argmin(abs.(xs .- xp)), 1, length(xs))
        j = clamp(argmin(abs.(ys .- yp)), 1, length(ys))
        push!(pressures, p_field[i, j])
    end
    return pressures
end
Cd_series = Float64[]
Cl_series = Float64[]
pdrop_series = Float64[]
point_A = (0.15, 0.2)
point_B = (0.25, 0.2)
for state in histories
    Cd, Cl, _ = extract_forces!(solver, state)
    push!(Cd_series, Cd)
    push!(Cl_series, Cl)
    p_vals = sample_pressure((point_A, point_B))
    push!(pdrop_series, p_vals[1] - p_vals[2])
end
dt = times[2] - times[1]
analysis_start = 2.0  # start time for statistics
start_idx = searchsortedfirst(times, analysis_start)
Cd_window = Cd_series[start_idx:end]
Cl_window = Cl_series[start_idx:end]
pdrop_window = pdrop_series[start_idx:end]
if isempty(Cd_window)
    Cd_window = Cd_series
    Cl_window = Cl_series
    pdrop_window = pdrop_series
    analysis_start = times[1]
    start_idx = 1
end
Cd_mean = -mean(Cd_window)
Cd_max = -maximum(Cd_window)
Cl_mean = mean(Cl_window)
Cl_rms = std(Cl_window)
Cl_peak_to_peak = maximum(Cl_window) - minimum(Cl_window)
St_freq, _ = dominant_frequency(Cl_window, dt)
Strouhal = St_freq * diameter / 1.0
pdrop_amp = maximum(pdrop_window) - minimum(pdrop_window)
###########
# Reference comparisons
###########
targets = Dict(
    :Cd_mean    => (3.1, 3.6, "Mean drag coefficient"),
    :Cl_mean    => (-0.08, 0.08, "Mean lift coefficient"),
    :Strouhal   => (0.29, 0.32, "Strouhal number (vortex shedding)")
)

values = Dict(
    :Cd_mean   => Cd_mean,
    :Cl_mean   => Cl_mean,
    :Strouhal  => Strouhal
)

println("\n=== Benchmark statistics (times ≥ $(analysis_start)) ===")
for (key, val) in values
    low, high, label = targets[key]
    within = low <= val <= high
    println(@sprintf("%-25s: %8.4f  target[%5.3f, %5.3f]  -> %s",
                     label, val, low, high, within ? "OK" : "out"))
end

###########
# Visualization / diagnostic plots
###########
times = times[start_idx:end]
Cd_series = Cd_series[start_idx:end]
Cl_series = Cl_series[start_idx:end]
pdrop_series = pdrop_series[start_idx:end]
fig = Figure(resolution=(1200, 700))
ax_cd = Axis(fig[1, 1], xlabel="time", ylabel="C_D", title="Drag coefficient")
lines!(ax_cd, times, Cd_series; color=:navy)
ax_cl = Axis(fig[2, 1], xlabel="time", ylabel="C_L", title="Lift coefficient")
lines!(ax_cl, times, Cl_series; color=:darkorange)
ax_pd = Axis(fig[3, 1], xlabel="time", ylabel="Δp", title="Pressure difference (A-B)")
lines!(ax_pd, times, pdrop_series; color=:green)
xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]
Ux_final = reshape(histories[end][1:nu_x], (length(xs), length(ys)))
Uy_final = reshape(histories[end][2nu_x+1:2nu_x+nu_y], (length(xs), length(ys)))
speed_final = sqrt.(Ux_final.^2 .+ Uy_final.^2)
ax_speed = Axis(fig[1:3, 2], xlabel="x", ylabel="y", title="Speed magnitude (final state)")
heatmap!(ax_speed, xs, ys, speed_final; colormap=:plasma)
contour!(ax_speed, xs, ys, [circle_levelset(x, y) for x in xs, y in ys]; levels=[0.0], color=:white, linewidth=2)
fig_path = joinpath(@__DIR__, "navierstokes2d_cylinder_re100_benchmark.png")
save(fig_path, fig)
println("Saved diagnostics to $(fig_path)")
