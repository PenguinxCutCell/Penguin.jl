using Penguin
using CairoMakie
using Statistics
using Printf

"""
Steady Navier–Stokes flow around a circular cylinder benchmark.

For a set of subcritical Reynolds numbers the script solves the steady
Navier–Stokes equations in a channel with a centered cylinder, measures
the wake (recirculation) length behind the body, and compares the results
against literature data:
    • Cuntanceau and Bouard
    • Russell and Wang
    • Calhoun et al.
    • Ye et al.
    • Fornberg
    • Guerrero

The wake length is defined as the distance from the rear stagnation point
to the first downstream location on the centerline where the streamwise
velocity changes sign (reattachment).
"""

###########
# Geometry and discretisation
###########
nx, ny = 256, 128
channel_length = 4.0
channel_height = 1.0
x0, y0 = -0.5, -0.5

circle_center = (0.5, 0.0)
circle_radius = 0.2
diameter = 2circle_radius

circle_body = (x, y, _=0.0) -> circle_radius - sqrt((x - circle_center[1])^2 + (y - circle_center[2])^2)

mesh_p  = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0 - 0.5 * dy))

capacity_ux = Capacity(circle_body, mesh_ux)
capacity_uy = Capacity(circle_body, mesh_uy)
capacity_p  = Capacity(circle_body, mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
U∞ = 1.0
uniform_inlet = (x, y, t=0.0) -> U∞

ux_left   = Dirichlet((x, y, t=0.0) -> uniform_inlet(x, y, t))
ux_right  = Outflow()
ux_bottom = Symmetry()
ux_top    = Symmetry()

uy_zero   = Dirichlet((x, y, t=0.0) -> 0.0)

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
interface_bc = Dirichlet(0.0)

###########
# Solver helpers
###########
ρ = 1.0
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
x_template = zeros(2 * (nu_x + nu_y) + np)

function build_fluid(μ::Float64)
    Fluid((mesh_ux, mesh_uy),
          (capacity_ux, capacity_uy),
          (operator_ux, operator_uy),
          mesh_p,
          capacity_p,
          operator_p,
          μ, ρ, fᵤ, fₚ)
end

function solve_steady_state(μ; picard_tol=1e-9, picard_maxiter=120, relaxation=1.0)
    fluid = build_fluid(μ)
    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=copy(x_template))

    _, picard_iters, picard_res = solve_NavierStokesMono_steady!(
        solver; nlsolve_method=:picard, tol=picard_tol, maxiter=picard_maxiter, relaxation=relaxation)

    blocks = Penguin.navierstokes2D_blocks(solver)
    mass_residual = blocks.div_x_ω * Vector{Float64}(solver.x[1:nu_x]) +
                    blocks.div_x_γ * Vector{Float64}(solver.x[nu_x+1:2nu_x]) +
                    blocks.div_y_ω * Vector{Float64}(solver.x[2nu_x+1:2nu_x+nu_y]) +
                    blocks.div_y_γ * Vector{Float64}(solver.x[2nu_x+nu_y+1:2*(nu_x+nu_y)])

    return solver, picard_iters, picard_res, maximum(abs, mass_residual)
end

###########
# Wake-length utilities
###########
xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]
x_trailing = circle_center[1] + circle_radius

nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))

function recirculation_length(Ux::Matrix{Float64})
    j_center = nearest_index(ys, circle_center[2])
    start_idx = searchsortedfirst(xs, x_trailing + dx)

    start_idx > length(xs) && return (0.0, missing)

    first_neg = nothing
    for i in start_idx:length(xs)
        if Ux[i, j_center] < 0
            first_neg = i
            break
        end
    end
    first_neg === nothing && return (0.0, missing)

    last_neg = first_neg
    next_pos = nothing
    for i in (first_neg + 1):length(xs)
        if Ux[i, j_center] < 0
            last_neg = i
        else
            next_pos = i
            break
        end
    end

    next_pos === nothing && return (NaN, missing)

    x_neg = xs[last_neg]
    x_pos = xs[next_pos]
    u_neg = Ux[last_neg, j_center]
    u_pos = Ux[next_pos, j_center]
    x_zero = x_neg - u_neg * (x_pos - x_neg) / (u_pos - u_neg)

    wake_len = max(0.0, x_zero - x_trailing)
    return (wake_len, x_zero)
end

###########
# Reference data
###########
wake_brackets = Dict(
    20 => (0.73, 0.94),
    40 => (1.60, 2.29),
)

###########
# Benchmark sweep
###########
Re_list = [20, 40]
results = Dict{Int, Dict{Symbol, Any}}()

println("=== Steady flow around a circular cylinder benchmark ===")

for Re in Re_list
    μ = ρ * U∞ * diameter / Re
    println()
    println(@sprintf("Re = %d  -> viscosity μ = %.4e", Re, μ))

    solver, picard_iters, picard_res, mass_inf =
        solve_steady_state(μ)
    println(@sprintf("  Picard:  iters=%d  residual=%.3e", picard_iters, picard_res))
    println(@sprintf("  max|div u| ≈ %.3e", mass_inf))

    Ux = reshape(solver.x[1:nu_x], (length(xs), length(ys)))
    uy_offset = 2nu_x
    Uy = reshape(solver.x[uy_offset+1:uy_offset+nu_y], (length(xs), length(ys)))

    wake_len, x_reattach = recirculation_length(Ux)
    wake_len_D = wake_len / diameter
    println(@sprintf("  wake length L_r = %.3e (%.3f D)", wake_len, wake_len_D))

    bracket = get(wake_brackets, Re, (NaN, NaN))
    within_bracket = isfinite(bracket[1]) ? (bracket[1] <= wake_len_D <= bracket[2]) : false
    results[Re] = Dict(
        :solver => solver,
        :Ux => Ux,
        :Uy => Uy,
        :wake_length => wake_len,
        :wake_length_D => wake_len_D,
        :reattach_x => x_reattach,
        :mass_inf => mass_inf,
        :picard_iters => picard_iters,
        :bracket => bracket,
        :within_bracket => within_bracket,
    )

    if isfinite(bracket[1])
        println(@sprintf("  bracket check: [%.2f, %.2f] D -> %s",
                         bracket[1], bracket[2], within_bracket ? "inside" : "outside"))
    else
        println("  no wake bracket available")
    end
end

println()
println("=== Summary ===")
println("Re    L_r/D(num)   bracket_min   bracket_max   inside?   max|div u|   Picard iters")
println("--------------------------------------------------------------------------------")
for Re in Re_list
    data = results[Re]
    @printf("%3d   %9.4f   %11.2f   %11.2f   %7s   %.2e   %4d\n",
            Re,
            data[:wake_length_D],
            data[:bracket][1],
            data[:bracket][2],
            data[:within_bracket] ? "yes" : "no",
            data[:mass_inf],
            data[:picard_iters])
end

###########
# Visualization for highest Reynolds number
###########
Re_plot = minimum(Re_list)
Ux_plot = results[Re_plot][:Ux]
Uy_plot = results[Re_plot][:Uy]
x_reattach = results[Re_plot][:reattach_x]

Xs = mesh_p.nodes[1]
Ys = mesh_p.nodes[2]
P = reshape(results[Re_plot][:solver].x[2 * (nu_x + nu_y) + 1:end], (length(Xs), length(Ys)))

speed = sqrt.(Ux_plot.^2 .+ Uy_plot.^2)

fig = Figure(resolution=(1100, 450))
ax_speed = Axis(fig[1, 1], xlabel="x", ylabel="y",
                title=@sprintf("Speed magnitude (Re=%d)", Re_plot))
hm = heatmap!(ax_speed, xs, ys, speed; colormap=:viridis)
Colorbar(fig[1, 2], hm, label="|u|")

j_center = nearest_index(ys, circle_center[2])
lines!(ax_speed, [x_trailing, x_trailing], [minimum(ys), maximum(ys)]; color=:white, linewidth=1.0, linestyle=:dash)
if x_reattach !== missing
    lines!(ax_speed, [x_reattach, x_reattach], [minimum(ys), maximum(ys)]; color=:red, linewidth=1.5)
end

ax_center = Axis(fig[1, 3], xlabel="x", ylabel="u_x(x, y=0)",
                 title="Centerline streamwise velocity")
ux_center_profile = Ux_plot[:, j_center]
lines!(ax_center, xs, ux_center_profile; color=:navy, linewidth=2)
hlines!(ax_center, [0.0]; color=:black, linestyle=:dash)
if x_reattach !== missing
    scatter!(ax_center, [x_reattach], [0.0]; color=:red, markersize=8)
end
ux_min, ux_max = extrema(ux_center_profile)
lines!(ax_center, [x_trailing, x_trailing], [ux_min, ux_max]; color=:gray, linestyle=:dashdot)

fig_path = joinpath(@__DIR__, "navierstokes2d_cylinder_wake_steady.png")
save(fig_path, fig)
println()
println("Saved diagnostic figure to $(fig_path)")
