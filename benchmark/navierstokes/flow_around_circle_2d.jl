using Penguin
using CairoMakie
using LinearAlgebra
using Statistics
using FFTW

"""
Benchmark: Flow around a circular cylinder (2D Navier–Stokes)

This script runs an unsteady Navier–Stokes simulation for several Reynolds
numbers and extracts wake properties (dominant shedding frequency → Strouhal
number) from a point probe in the wake. It prints comparisons with a simple
reference value (Strouhal ≈ 0.2 for Re in the range ~50–200).
"""

###########
# Geometry (same as examples)
###########
nx, ny = 128, 64
channel_length = 4.0
channel_height = 1.0
x0, y0 = -0.5, -0.5

circle_center = (0.5, 0.0)
circle_radius = 0.2
diameter = 2 * circle_radius

circle_body = (x, y, _=0.0) -> circle_radius - sqrt((x - circle_center[1])^2 + (y - circle_center[2])^2)

###########
# Meshes and operators
###########
mesh_p  = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (channel_length, channel_height), (x0, y0 - 0.5*dy))

capacity_ux = Capacity(circle_body, mesh_ux)
capacity_uy = Capacity(circle_body, mesh_uy)
capacity_p  = Capacity(circle_body, mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
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
uy_zero   = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(:left=>ux_left, :right=>Outflow(), :bottom=>Symmetry(), :top=>Symmetry()))
bc_uy = BorderConditions(Dict(:left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero))

pressure_gauge = PinPressureGauge()
interface_bc = Dirichlet(0.0)

###########
# Solver factory
###########
function make_solver(μ, ρ; x0_vec=nothing)
    fluid = Fluid((mesh_ux, mesh_uy), (capacity_ux, capacity_uy), (operator_ux, operator_uy),
                  mesh_p, capacity_p, operator_p, μ, ρ, (x,y,z=0.0,t=0.0)->0.0, (x,y,z=0.0,t=0.0)->0.0)
    nu_x = prod(operator_ux.size)
    nu_y = prod(operator_uy.size)
    np = prod(operator_p.size)
    if x0_vec === nothing
        x0_vec = zeros(2*(nu_x+nu_y) + np)
    end
    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)
    return solver
end

###########
# Probe and analysis helpers
###########
probe_x, probe_y = 1.0, 0.0
nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))

function estimate_strouhal_from_series(series, dt)
    # Remove mean, window a bit
    s = series .- mean(series)
    N = length(s)
    if N < 8
        return (NaN, NaN)
    end
    # FFT
    spec = abs.(fft(s))
    freqs = (0:N-1) .* (1.0 / (dt * N))
    # ignore zero-frequency; search positive range up to Nyquist
    pos = 2:clamp(div(N,2), 2, N)
    idx = pos[argmax(spec[pos])]
    fdom = freqs[idx]
    # power of dominant (for heuristic of convergence)
    return fdom, spec[idx]
end

###########
# Reynolds numbers to test
###########
Re_list = [50, 100, 200]
ρ = 1.0

results = Dict()

for Re in Re_list
    println("\n=== Running Re = $Re ===")
    μ = ρ * Umax * diameter / Re
    println("viscosity μ = ", μ)

    solver = make_solver(μ, ρ)

    # Time stepping choices: longer for lower Re to capture shedding
    Δt = 0.005
    T_end = if Re <= 50
        20.0
    elseif Re <= 100
        15.0
    else
        10.0
    end

    println("Running unsteady simulation: dt=", Δt, ", T_end=", T_end)
    times, histories = solve_NavierStokesMono_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=:CN)

    println("Finished: stored states = ", length(histories))

    # Extract probe series (transverse velocity) from histories
    nu_x = prod(operator_ux.size)
    nu_y = prod(operator_uy.size)
    np = prod(operator_p.size)

    xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
    ixp = nearest_index(xs, probe_x)
    iyp = nearest_index(ys, probe_y)
    LIux = LinearIndices((length(xs), length(ys)))
    probe_vals = Float64[]

    for hist in histories
        ux_hist = hist[1:nu_x]
        uy_hist = hist[2nu_x+1:2nu_x+nu_y]
        # Build local arrays (avoid expensive reshape each time if needed, but okay here)
        Ux = reshape(ux_hist, (length(xs), length(ys)))
        Uy = reshape(uy_hist, (length(xs), length(ys)))
        push!(probe_vals, Uy[ixp, iyp])
    end

    dt = times[2] - times[1]
    fdom, power = estimate_strouhal_from_series(probe_vals, dt)
    Str = fdom * diameter / Umax

    # Compute some basic statistics on final state
    final = histories[end]
    ux_f = final[1:nu_x]
    uy_f = final[2nu_x+1:2nu_x+nu_y]
    Ux_f = reshape(ux_f, (length(xs), length(ys)))
    Uy_f = reshape(uy_f, (length(xs), length(ys)))
    speed = sqrt.(Ux_f.^2 .+ Uy_f.^2)
    max_speed = maximum(speed)
    mean_speed = mean(speed)

    # crude vorticity estimate (∂u_y/∂x - ∂u_x/∂y) via central differences on grid
    ω = zeros(size(Ux_f))
    for j in 2:(size(Ux_f,2)-1), i in 2:(size(Ux_f,1)-1)
        dudy = (Ux_f[i, j+1] - Ux_f[i, j-1]) / (2*dy)
        dvdx = (Uy_f[i+1, j] - Uy_f[i-1, j]) / (2*dx)
        ω[i,j] = dvdx - dudy
    end
    max_vort = maximum(abs, ω)

    # Surface forces on the cylinder
    force_diag = compute_navierstokes_force_diagnostics(solver)
    body_force = navierstokes_reaction_force_components(force_diag; acting_on=:body)
    coeffs = drag_lift_coefficients(force_diag; ρ=ρ, U_ref=Umax, length_ref=diameter, acting_on=:body)

    # Compare to reference Strouhal (rough experimental/empirical reference)
    ref_Str = 0.2
    rel_err_Str = isfinite(Str) ? abs(Str - ref_Str) / ref_Str : NaN

    println("Re = $Re summary:")
    println("  f_dom = ", round(fdom, sigdigits=6), "  Str = ", round(Str, sigdigits=6), "  rel_err_Str = ", round(rel_err_Str, sigdigits=6))
    println("  max_speed = ", round(max_speed, sigdigits=6), ", mean_speed = ", round(mean_speed, sigdigits=6))
    println("  max_vorticity = ", round(max_vort, sigdigits=6))
    println("  Fx_body = ", round(body_force[1]; sigdigits=6), ", Fy_body = ", round(body_force[2]; sigdigits=6),
            ", Cd = ", round(coeffs.Cd; sigdigits=6), ", Cl = ", round(coeffs.Cl; sigdigits=6))

    # Simple checks and pass/fail heuristics
    pass_strouhal = isfinite(Str) && (rel_err_Str < 0.3)
    pass_vorticity = max_vort > 0.0

    println("checks:")
    println(" - Strouhal close to reference (~0.2)? ", pass_strouhal)
    println(" - wake vorticity present? ", pass_vorticity)

    results[Re] = (
        fdom = fdom,
        Str = Str,
        rel_err_Str = rel_err_Str,
        max_speed = max_speed,
        mean_speed = mean_speed,
        max_vort = max_vort,
        Fx_body = body_force[1],
        Fy_body = body_force[2],
        Cd = coeffs.Cd,
        Cl = coeffs.Cl,
        pass_strouhal = pass_strouhal,
        pass_vorticity = pass_vorticity,
    )
end

println("\n=== Summary for all Re tested ===")
for Re in Re_list
    r = results[Re]
    println("Re=", Re, ": Str=", round(r.Str, sigdigits=6), ", rel_err_Str=", round(r.rel_err_Str, sigdigits=6), ", max_vort=", round(r.max_vort, sigdigits=6))
end

println("benchmark flow around circle completed.")
