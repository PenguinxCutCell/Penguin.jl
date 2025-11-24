using Penguin
using LinearAlgebra
using CairoMakie
using Statistics
using SpecialFunctions

"""
Simple regression-style example that exercises the geometric interface update
for the front-tracking Stefán problem.  The setup mirrors the self-similar
Frank sphere growth but keeps the script focused on the new algorithm.
"""

# -----------------------------------------------------------------------------
# Physical parameters and analytical reference
# -----------------------------------------------------------------------------
const ρ = 1.0
const L = 1.0
const c = 1.0
const T∞ = -0.5
const TM = 0.0

Ste = (c * (TM - T∞)) / L
println("Stefan number: $Ste")

S = 1.56  # similarity parameter for the Frank solution

function interface_position(t)
    return S * sqrt(t)
end

function analytical_temperature(r, t)
    s = r / sqrt(t)
    return s < S ? TM : T∞ * (1 - expint(s^2 / 4) / expint(S^2 / 4))
end

# -----------------------------------------------------------------------------
# Numerical setup
# -----------------------------------------------------------------------------
t_init = 1.0
t_final = 1.1

nx, ny = 48, 48
lx, ly = 12.0, 12.0
domain_origin = (-lx/2, -ly/2)
mesh = Penguin.Mesh((nx, ny), (lx, ly), domain_origin)

Δx = lx / nx
Δt = 0.15 * Δx^2  # explicit-ish CFL for diffusion
t_final = t_init + 5Δt

println("Mesh: $(nx)x$(ny), Δx = $(round(Δx, digits=3))")
println("Time step Δt = $(round(Δt, digits=5))")

# Front construction
nmarkers = 60
front = FrontTracker()
create_circle!(front, 0.0, 0.0, interface_position(t_init), nmarkers)

body = (x, y, t, _=0) -> -sdf(front, x, y)
STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + Δt], tag=mesh.tag)
capacity = Capacity(body, STmesh; compute_centroids=false)
operator = DiffusionOps(capacity)
bc_outer = Dirichlet(T∞)
bc_interface = Dirichlet(TM)
bc_b = BorderConditions(Dict(:left => bc_outer, :right => bc_outer, :top => bc_outer, :bottom => bc_outer))
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ * L))

K = (x, y, z) -> 1.0
f = (x, y, z, t) -> 0.0
phase = Phase(capacity, operator, f, K)

# -----------------------------------------------------------------------------
# Initial condition compatible with the reference solution
# -----------------------------------------------------------------------------
function radial_temperature_field(front::FrontTracker, mesh, t)
    body_init = (x, y, _=0) -> -sdf(front, x, y)
    cap_init = Capacity(body_init, mesh; compute_centroids=false)
    centroids = cap_init.C_ω
    temps = zeros(length(centroids))
    for (idx, centroid) in enumerate(centroids)
        x, y = centroid
        r = hypot(x, y)
        temps[idx] = analytical_temperature(r, t)
    end
    return temps
end

bulk_ic = radial_temperature_field(front, mesh, t_init)
ghost_ic = fill(TM, length(bulk_ic))
u0 = vcat(bulk_ic, ghost_ic)

# -----------------------------------------------------------------------------
# Solve with geometric update strategy
# -----------------------------------------------------------------------------
solver = StefanMono2D(phase, bc_b, bc_interface, Δt, u0, mesh, "BE")
Newton_params = (10, 5e-7, 5e-7, 1.0)

solver, residuals, xf_log, timestep_history, phase, position_increments = solve_StefanMono2D_geom!(
    solver, phase, front, Δt, t_init, t_final, bc_b, bc_interface, stef_cond, mesh, "BE";
    Newton_params=Newton_params,
    adaptive_timestep=false,
    smooth_factor=1.0,
    window_size=10,
    method=Base.:\)

println("Simulation finished. Stored $(length(solver.states)) states.")

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
function plot_geometric_interface_history(xf_log, timestep_history; samples=5)
    keys_sorted = sort(collect(keys(xf_log)))
    if isempty(keys_sorted)
        @warn "No interface snapshots recorded."
        return nothing
    end

    picks = unique(floor.(Int, range(1, stop=length(keys_sorted), length=min(samples, length(keys_sorted)))))
    fig = Figure(size=(800, 700))
    ax = Axis(fig[1, 1], title="Interface evolution (geometric update)", xlabel="x", ylabel="y", aspect=DataAspect())

    cmap = cgrad(:viridis, length(picks))
    for (idx, k) in enumerate(keys_sorted[picks])
        markers = xf_log[k]
        xs = [m[1] for m in markers]
        ys = [m[2] for m in markers]
        lines!(ax, xs, ys, color=cmap[idx], linewidth=2)
        t_plot = timestep_history[min(k, length(timestep_history))][1]
        text!(ax, mean(xs), mean(ys), text="t=$(round(t_plot, digits=3))", color=cmap[idx], align=(:center, :center))
    end

    Colorbar(fig[1, 2], limits=(1, length(picks)), colormap=:viridis, label="snapshot index")
    display(fig)
    save("stefan_geometric_interface_history.png", fig)
    return fig
end

function plot_geometric_residuals(residuals)
    if isempty(residuals)
        @warn "Residual history is empty"
        return nothing
    end

    fig = Figure(size=(700, 500))
    ax = Axis(fig[1, 1], title="Geometric residual norm", xlabel="Iteration", ylabel="‖G‖₂", yscale=log10)
    for (step, res_hist) in sort(collect(residuals))
        lines!(ax, 1:length(res_hist), res_hist, label="Δt #$step")
    end
    axislegend(ax, position=:rb)
    display(fig)
    save("stefan_geometric_residuals.png", fig)
    return fig
end

function compare_radius_history(xf_log, timestep_history)
    times = [entry[1] for entry in timestep_history]
    radii = Float64[]

    for key in sort(collect(keys(xf_log)))
        markers = xf_log[key]
        xs = [m[1] for m in markers]
        ys = [m[2] for m in markers]
        center = (mean(xs), mean(ys))
        push!(radii, mean(hypot.(xs .- center[1], ys .- center[2])))
    end

    analytic_times = range(times[1], stop=times[end], length=200)
    analytic_radii = interface_position.(analytic_times)

    fig = Figure(size=(700, 500))
    ax = Axis(fig[1, 1], title="Radius comparison", xlabel="time", ylabel="mean radius")
    lines!(ax, analytic_times, analytic_radii, color=:black, linestyle=:dash, label="analytic")
    scatter!(ax, times[1:length(radii)], radii, color=:red, markersize=7, label="geometric")
    axislegend(ax, position=:lt)
    display(fig)
    save("stefan_geometric_radius_comparison.png", fig)
    return fig
end

# Trigger diagnostics
plot_geometric_interface_history(xf_log, timestep_history)
plot_geometric_residuals(residuals)
compare_radius_history(xf_log, timestep_history)

println("Geometric front-tracking example complete.")
