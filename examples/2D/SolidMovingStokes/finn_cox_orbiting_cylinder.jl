using Penguin
using LinearAlgebra
using Printf
using CairoMakie
using ForwardDiff
using IterativeSolvers

"""
Finn–Cox benchmark: translating (non-rotating) cylinder orbiting inside a
stationary circular container. The analytical streamfunction of Finn & Cox
(*J. Eng. Math.*, 2001) is used for boundary conditions, field comparison,
and the closed-form hydrodynamic force on the inner cylinder.
"""

# -----------------
# Geometry & motion
# -----------------
struct FinnCoxParams
    a_in::Float64   # inner cylinder radius
    a_out::Float64  # outer cylinder radius
    eps::Float64    # orbital radius of the inner center
    T::Float64      # orbital period
    μ::Float64
    ρ::Float64
end

FinnCoxParams(; a_in = 0.2, a_out = 1.0, eps = 0.5, T = 1.0, μ = 1.0, ρ = 1.0) =
    FinnCoxParams(a_in, a_out, eps, T, μ, ρ)

# Slower orbit for better resolution of the flow
ω_orbit(p::FinnCoxParams) = 2π / p.T / 2

orbit_center(p::FinnCoxParams, t) = (p.eps * cos(ω_orbit(p) * t),
                                     p.eps * sin(ω_orbit(p) * t))

orbit_velocity(p::FinnCoxParams, t) = (-(p.eps * ω_orbit(p)) * sin(ω_orbit(p) * t),
                                        (p.eps * ω_orbit(p)) * cos(ω_orbit(p) * t))

interval_min(a, b) = 0.5 * (a + b - abs(a - b))

# -----------------
# Finn–Cox helper formulas
# -----------------
"""Return (s, σ, L) from Finn & Cox eqs. (45)–(47)."""
function fc_geometry(ain, aout, ξ)
    Δ = sqrt((aout + ain + ξ) * (aout + ain - ξ) * (aout - ain + ξ) * (aout - ain - ξ))
    σ = (aout^2 - ain^2 + ξ^2 + Δ) / (2 * ξ)
    s = aout^2 / σ
    L = log((σ * (s - ξ)) / (s * (σ - ξ)))
    return s, σ, L
end

"""Coefficients for x-translation (eqs. 55–56)."""
function fc_coeffs_x(U, ain, aout, ξ, s, σ, L)
    denom = 2 * ξ * (σ - s) + (ain^2 + aout^2 - ξ^2) * L
    B = 2 * U * (σ - ξ) * s / denom
    β = -2 * U * (s - ξ) * σ / denom
    return B, β
end

"""Coefficients for y-translation (eqs. 60–63). Ω_in = 0 here."""
function fc_coeffs_y(V, ain, aout, ξ, s, σ, L; Ω_in = 0.0)
    geom = (aout^2 - ain^2 + ξ^2)
    common = 2 * (σ - s) * ξ + (ain^2 + aout^2) * L

    A = s * ξ * (ain^2 * (2 * ξ - L * σ) * Ω_in - 2 * V * (σ^2 + ain^2)) /
        (geom * common)
    α = -σ * ξ * (ain^2 * (2 * ξ + L * s) * Ω_in - 2 * V * (s^2 + ain^2)) /
         (geom * common)

    num_rot = ain^2 * common * Ω_in - 2 * V * ξ * (σ - s)
    C = -aout^2 * num_rot / (2 * ξ * (σ - s) * common)
    γ =  aout^2 * num_rot / (2 * ξ * (σ - s) * common)
    return A, α, C, γ
end

"""
Streamfunction ψ(x, y) in the Finn–Cox canonical frame (inner center on x-axis
at ξ), using complex form eq. (68). Rotation to/from lab frame is handled
outside this function.
"""
function fc_streamfunction(x, y, ain, aout, ξ, U, V; Ω_in = 0.0, Ω_out = 0.0)
    s, σ, L = fc_geometry(ain, aout, ξ)
    B, β = fc_coeffs_x(U, ain, aout, ξ, s, σ, L)
    A, α, C, γ = fc_coeffs_y(V, ain, aout, ξ, s, σ, L; Ω_in = Ω_in)

    z = complex(x, y)
    z̄ = conj(z)
    r2 = abs2(z)

    term_log_pref = 0.25 * (A - α) * (z + z̄) - 0.25im * (B - β) * (z - z̄) + 0.5 * (C - γ)
    term_log = log(σ * abs2(z - s) / (s * abs2(z - σ)))

    term_frac = 0.25 * (r2 - aout^2) * (
        (A / (2s)) * ((z + s) / (z - σ) + (z̄ + s) / (z̄ - σ)) +
        (α / (2σ)) * ((z + σ) / (z - s) + (z̄ + σ) / (z̄ - s))
    )

    term_B = 1im * B * (z - z̄) * (σ - s) / (2s * abs2(z - σ))
    term_β = -1im * β * (z - z̄) * (σ - s) / (2σ * abs2(z - s))
    term_C = C * (r2 - σ^2) / (2 * aout^2 * abs2(z - σ))
    term_γ = γ * (r2 - s^2) / (2 * aout^2 * abs2(z - s))

    ψ = term_log_pref * term_log + term_frac + term_B + term_β + term_C + term_γ - Ω_out
    return real(ψ), (; A, α, B, β, C, γ, s, σ)
end

"""Velocity (u, v) in canonical frame via ∂ψ/∂y and -∂ψ/∂x using AD."""
function fc_velocity_canonical(x, y, ain, aout, ξ, U, V; Ω_in = 0.0, Ω_out = 0.0)
    ψfun = (xy) -> begin
        ψ, _ = fc_streamfunction(xy[1], xy[2], ain, aout, ξ, U, V; Ω_in = Ω_in, Ω_out = Ω_out)
        return ψ
    end
    grad = ForwardDiff.gradient(ψfun, [x, y])
    u = grad[2]
    v = -grad[1]
    ψ, meta = fc_streamfunction(x, y, ain, aout, ξ, U, V; Ω_in = Ω_in, Ω_out = Ω_out)
    return u, v, ψ, meta
end

"""
Analytical force on inner cylinder (eq. 72) in the canonical frame, then
rotated to the lab frame.
"""
function fc_force_lab(p::FinnCoxParams, t)
    xc, yc = orbit_center(p, t)
    ξ = hypot(xc, yc)
    θ = atan(yc, xc)
    cθ, sθ = cos(θ), sin(θ)
    Uc, Vc = orbit_velocity(p, t)
    U = cθ * Uc + sθ * Vc
    V = -sθ * Uc + cθ * Vc

    _, meta = fc_streamfunction(0.0, 0.0, p.a_in, p.a_out, ξ, U, V)
    A, α, B, β = meta.A, meta.α, meta.B, meta.β

    Fx_c = 4π * p.μ * (β - B)
    Fy_c = 4π * p.μ * (A - α)

    Fx = cθ * Fx_c - sθ * Fy_c
    Fy = sθ * Fx_c + cθ * Fy_c
    return Fx, Fy
end

"""Analytical velocity in the lab frame at (x, y, t)."""
function fc_velocity_lab(p::FinnCoxParams, x, y, t)
    xc, yc = orbit_center(p, t)
    ξ = hypot(xc, yc)
    θ = atan(yc, xc)
    cθ, sθ = cos(θ), sin(θ)

    # Rotate point to canonical frame
    xp = cθ * x + sθ * y
    yp = -sθ * x + cθ * y

    Uc, Vc = orbit_velocity(p, t)
    U = cθ * Uc + sθ * Vc
    V = -sθ * Uc + cθ * Vc

    u′, v′, ψ, _ = fc_velocity_canonical(xp, yp, p.a_in, p.a_out, ξ, U, V)

    u = cθ * u′ - sθ * v′
    v = sθ * u′ + cθ * v′
    return u, v, ψ
end

# -----------------
# Geometry and grids
# -----------------
p = FinnCoxParams()

nx = ny = 64
L = 2.4
x0 = y0 = -0.5 * L

mesh_p = Penguin.Mesh((nx, ny), (L, L), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (L, L), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (L, L), (x0, y0 - 0.5 * dy))

"""Signed distance: positive in solids (inner cylinder or outside outer container)."""
body = (x, y, t) -> begin
    xc, yc = orbit_center(p, t)
    r_in = hypot(x - xc, y - yc)
    r_out = hypot(x, y)
    ϕ_inner = r_in - p.a_in            # positive outside the inner cylinder
    ϕ_outer_ext = p.a_out - r_out      # positive inside the outer cylinder
    levelset = interval_min(ϕ_inner, ϕ_outer_ext)
    return -levelset                   # positive in solids, negative in fluid
end

# Initial capacities/operators (t = 0)
cap_ux0 = Capacity((x, y, _ = 0.0) -> body(x, y, 0.0), mesh_ux; method = "VOFI",
                   integration_method = :vofijul, compute_centroids = true)
cap_uy0 = Capacity((x, y, _ = 0.0) -> body(x, y, 0.0), mesh_uy; method = "VOFI",
                   integration_method = :vofijul, compute_centroids = true)
cap_p0 = Capacity((x, y, _ = 0.0) -> body(x, y, 0.0), mesh_p; method = "VOFI",
                  integration_method = :vofijul, compute_centroids = true)

op_ux0 = DiffusionOps(cap_ux0)
op_uy0 = DiffusionOps(cap_uy0)
op_p0 = DiffusionOps(cap_p0)

# -----------------
# Boundary conditions
# -----------------
zero_bc = Dirichlet(0.0)
bc_ux = BorderConditions(Dict(:left => zero_bc, :right => zero_bc,
                              :bottom => zero_bc, :top => zero_bc))
bc_uy = BorderConditions(Dict(:left => zero_bc, :right => zero_bc,
                              :bottom => zero_bc, :top => zero_bc))
pressure_gauge = PinPressureGauge()

tol_cut = 2.0 * min(dx, dy)

function finn_cox_cut(component::Symbol)
    return (x, y, t) -> begin
        xc, yc = orbit_center(p, t)
        r_in = hypot(x - xc, y - yc)
        r_out = hypot(x, y)
        if abs(r_in - p.a_in) <= tol_cut
            Uc, Vc = orbit_velocity(p, t)
            return component === :ux ? Uc : Vc
        elseif abs(r_out - p.a_out) <= tol_cut
            return 0.0
        else
            return 0.0
        end
    end
end

bc_cut = (Dirichlet(finn_cox_cut(:ux)), Dirichlet(finn_cox_cut(:uy)))

# -----------------
# Fluid and solver
# -----------------
fᵤ = (x, y, z = 0.0) -> 0.0
fₚ = (x, y, z = 0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (cap_ux0, cap_uy0),
              (op_ux0, op_uy0),
              mesh_p, cap_p0, op_p0,
              p.μ, p.ρ, fᵤ, fₚ)

nu_x = prod(op_ux0.size)
nu_y = prod(op_uy0.size)
np = prod(op_p0.size)
Ntot = 2 * (nu_x + nu_y) + np
x0_vec = zeros(Ntot)

Δt = p.T / 64
T_end = p.T
scheme = :BE
geometry_method = "VOFI"
capacity_kwargs = (; method = geometry_method,
                    integration_method = :vofijul,
                    compute_centroids = true)

solver = MovingStokesUnsteadyMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut;
                                   scheme = scheme, x0 = x0_vec)

println("Running Finn–Cox orbiting cylinder benchmark")
println(@sprintf("  a_in=%.2f, a_out=%.2f, eps=%.2f, Δt=%.4f, steps=%d",
                 p.a_in, p.a_out, p.eps, Δt, Int(round(T_end / Δt))))

times, states = solve_MovingStokesUnsteadyMono!(solver, body, mesh_p,
                                                 Δt, 0.0, T_end,
                                                 (bc_ux, bc_uy), bc_cut;
                                                 scheme = scheme,
                                                 method = IterativeSolvers.gmres,
                                                 geometry_method = geometry_method,
                                                 integration_method = capacity_kwargs.integration_method,
                                                 compute_centroids = capacity_kwargs.compute_centroids)

println("Completed $(length(times) - 1) time steps")

# -----------------
# Force diagnostics at stored times
# -----------------
force_num = Vector{NTuple{2, Float64}}()

for k in 2:length(times)
    t_prev, t_now = times[k-1], times[k]
    STmesh_ux = Penguin.SpaceTimeMesh(mesh_ux, [t_prev, t_now], tag = mesh_p.tag)
    STmesh_uy = Penguin.SpaceTimeMesh(mesh_uy, [t_prev, t_now], tag = mesh_p.tag)
    STmesh_p = Penguin.SpaceTimeMesh(mesh_p, [t_prev, t_now], tag = mesh_p.tag)

    cap_ux = Capacity(body, STmesh_ux; capacity_kwargs...)
    cap_uy = Capacity(body, STmesh_uy; capacity_kwargs...)
    cap_p = Capacity(body, STmesh_p; capacity_kwargs...)

    op_ux = DiffusionOps(cap_ux)
    op_uy = DiffusionOps(cap_uy)
    op_p = DiffusionOps(cap_p)

    block_data = Penguin.stokes2D_moving_blocks(solver.fluid,
                                                (op_ux, op_uy),
                                                (cap_ux, cap_uy),
                                                op_p, cap_p, scheme)

    force_diag = compute_navierstokes_force_diagnostics(solver, block_data)
    body_force = navierstokes_reaction_force_components(force_diag; acting_on = :body)
    push!(force_num, (body_force[1], body_force[2]))
end

# Exact forces at the same times (using mid-point time for each interval)
force_exact = [fc_force_lab(p, 0.5 * (times[k-1] + times[k])) for k in 2:length(times)]

println("Sample force comparisons (Fx_num, Fy_num) vs exact (Fx_exact, Fy_exact):")
for idx in round.(Int, range(1, length(force_num), length = min(length(force_num), 8)))
    Fx_num, Fy_num = force_num[idx]
    Fx_ex, Fy_ex = force_exact[idx]
    t_mid = 0.5 * (times[idx] + times[idx + 1])
    println(@sprintf("  t/T=%.3f  num=(%8.3f,%8.3f)  exact=(%8.3f,%8.3f)",
                    t_mid / p.T, Fx_num, Fy_num, Fx_ex, Fy_ex))
end

# -----------------
# Field error at final time
# -----------------
final_state = states[end]
uωx = final_state[1:nu_x]
uωy = final_state[2 * nu_x + 1:2 * nu_x + nu_y]

xs = mesh_ux.nodes[1]
ys = mesh_ux.nodes[2]
LI = LinearIndices((length(xs), length(ys)))

global sample_count = 0
global sum_sq = 0.0
global max_err = 0.0

for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
    if body(x, y, T_end) >= 0
        continue
    end
    idx = LI[i, j]
    ux_num = uωx[idx]
    uy_num = uωy[idx]
    ux_ex, uy_ex, _ = fc_velocity_lab(p, x, y, T_end)
    du = ux_num - ux_ex
    dv = uy_num - uy_ex
    err = sqrt(du^2 + dv^2)
    global sample_count += 1
    global sum_sq += err^2
    global max_err = max(max_err, err)
end

if sample_count > 0
    l2 = sqrt(sum_sq / sample_count)
    println(@sprintf("Final-time velocity error: L2=%.4e, Linf=%.4e (samples=%d)", l2, max_err, sample_count))
end

# -----------------
# Visualization: snapshots and animation
# -----------------
function speed_field(state)
    uωx = state[1:nu_x]
    uωy = state[2 * nu_x + 1:2 * nu_x + nu_y]
    Ux_field = reshape(uωx, (length(xs), length(ys)))
    Uy_field = reshape(uωy, (length(xs), length(ys)))
    return sqrt.(Ux_field .^ 2 .+ Uy_field .^ 2)
end

mask_at_time(t) = [body(x, y, t) < 0 ? 1.0 : NaN for x in xs, y in ys]

nearest_index(vec, val) = findmin(abs.(vec .- val))[2]

function velocity_sampler(state)
    uωx = state[1:nu_x]
    uωy = state[2 * nu_x + 1:2 * nu_x + nu_y]
    Ux_field = reshape(uωx, (length(xs), length(ys)))
    Uy_field = reshape(uωy, (length(xs), length(ys)))
    return (x, y) -> begin
        i = nearest_index(xs, x)
        j = nearest_index(ys, y)
        return Point2f(Ux_field[i, j], Uy_field[i, j])
    end
end

function plot_snapshots(times, states; frame_indices = nothing, outfile = "finn_cox_snapshots.png")
    if frame_indices === nothing
        frame_indices = round.(Int, range(1, length(states), length = 4))
    end
    speeds = [speed_field(states[i]) .* mask_at_time(times[i]) for i in frame_indices]
    vmax = maximum(map(s -> maximum(skipmissing(replace(vec(s), NaN => missing))), speeds))

    fig = Figure(size = (1200, 650))
    θs = range(0, 2π; length = 160)

    for (plot_idx, state_idx) in enumerate(frame_indices)
        t = times[state_idx]
        sp = speeds[plot_idx]
        cx, cy = orbit_center(p, t)
        inner_x = cx .+ p.a_in .* cos.(θs)
        inner_y = cy .+ p.a_in .* sin.(θs)
        outer_x = p.a_out .* cos.(θs)
        outer_y = p.a_out .* sin.(θs)

        row = div(plot_idx - 1, 2) + 1
        col = mod(plot_idx - 1, 2) + 1
        ax = Axis(fig[row, col], xlabel = "x", ylabel = "y",
                  title = @sprintf("t/T=%.3f", t / p.T))
        heatmap!(ax, xs, ys, sp; colormap = :viridis, colorrange = (0.0, vmax))
        lines!(ax, inner_x, inner_y; color = :white, linewidth = 2)
        lines!(ax, outer_x, outer_y; color = :white, linewidth = 2)
    end
    Colorbar(fig[:, 3], label = "|u|", colorrange = (0.0, vmax), colormap = :viridis)

    save(outfile, fig)
    println("Saved snapshots to $(outfile)")
    return fig
end

function plot_streamlines(times, states; frame_indices = nothing, outfile = "finn_cox_streamlines.png")
    if frame_indices === nothing
        frame_indices = round.(Int, range(1, length(states), length = 4))
    end

    fig = Figure(size = (1200, 650))
    θs = range(0, 2π; length = 160)

    for (plot_idx, state_idx) in enumerate(frame_indices)
        t = times[state_idx]
        sampler = velocity_sampler(states[state_idx])
        cx, cy = orbit_center(p, t)
        inner_x = cx .+ p.a_in .* cos.(θs)
        inner_y = cy .+ p.a_in .* sin.(θs)
        outer_x = p.a_out .* cos.(θs)
        outer_y = p.a_out .* sin.(θs)

        row = div(plot_idx - 1, 2) + 1
        col = mod(plot_idx - 1, 2) + 1
        ax = Axis(fig[row, col], xlabel = "x", ylabel = "y",
                  title = @sprintf("Streamlines t/T=%.3f", t / p.T))
        streamplot!(ax, sampler, xs[1]..xs[end], ys[1]..ys[end];
                    density = 2.0, color=(p)->norm(p))
        lines!(ax, inner_x, inner_y; color = :white, linewidth = 2)
        lines!(ax, outer_x, outer_y; color = :white, linewidth = 2)
    end

    save(outfile, fig)
    println("Saved streamline snapshots to $(outfile)")
    return fig
end

function animate_orbiting(times, states; n_frames = 40, framerate = 10, outfile = "finn_cox_orbit.gif")
    frame_indices = round.(Int, range(1, length(states), length = min(n_frames, length(states))))
    θs = range(0, 2π; length = 160)

    speeds = [speed_field(states[i]) .* mask_at_time(times[i]) for i in frame_indices]
    vmax = maximum(map(s -> maximum(skipmissing(replace(vec(s), NaN => missing))), speeds))

    speed_obs = Observable(first(speeds))
    cx0, cy0 = orbit_center(p, times[frame_indices[1]])
    inner_x_obs = Observable(cx0 .+ p.a_in .* cos.(θs))
    inner_y_obs = Observable(cy0 .+ p.a_in .* sin.(θs))

    fig = Figure(size = (600, 520))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Finn–Cox orbit")
    hm = heatmap!(ax, xs, ys, speed_obs; colormap = :plasma, colorrange = (0.0, vmax))
    lines!(ax, inner_x_obs, inner_y_obs; color = :white, linewidth = 2)
    lines!(ax, p.a_out .* cos.(θs), p.a_out .* sin.(θs); color = :white, linewidth = 2)
    Colorbar(fig[1, 2], hm, label = "|u|")

    record(fig, outfile, 1:length(frame_indices); framerate = framerate) do frame
        idx = frame_indices[frame]
        speed_obs[] = speeds[frame]
        cx, cy = orbit_center(p, times[idx])
        inner_x_obs[] = cx .+ p.a_in .* cos.(θs)
        inner_y_obs[] = cy .+ p.a_in .* sin.(θs)
        ax.title = @sprintf("t/T = %.3f", times[idx] / p.T)
    end

    println("Saved animation to $(outfile)")
    return outfile
end

# Final-time plot (for convenience)
speed_T = speed_field(final_state) .* mask_at_time(T_end)
fig_final = Figure(size = (520, 520))
ax_final = Axis(fig_final[1, 1], xlabel = "x", ylabel = "y",
                title = "Finn–Cox orbit: |u| at t=T")
heatmap!(ax_final, xs, ys, speed_T; colormap = :viridis)
θs = range(0, 2π; length = 160)
cx, cy = orbit_center(p, T_end)
inner_x = cx .+ p.a_in .* cos.(θs)
inner_y = cy .+ p.a_in .* sin.(θs)
outer_x = p.a_out .* cos.(θs)
outer_y = p.a_out .* (-sin.(θs))
lines!(ax_final, inner_x, inner_y; color = :white, linewidth = 2)
lines!(ax_final, outer_x, outer_y; color = :white, linewidth = 2)
Colorbar(fig_final[1, 2], label = "|u|")
save("finn_cox_orbiting_cylinder.png", fig_final)
println("Saved visualization to finn_cox_orbiting_cylinder.png")

# Also produce snapshots and animation
plot_snapshots(times, states)
plot_streamlines(times, states)
animate_orbiting(times, states)
