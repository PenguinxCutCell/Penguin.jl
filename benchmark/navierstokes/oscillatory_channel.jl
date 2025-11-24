using Penguin
using Statistics
using LinearAlgebra
using CairoMakie
using DelimitedFiles

###########
# Geometry and base data
###########
h = 1.0
Lx = 2h
nx = 64
ny = 64
ρ = 1.0
ν = 0.05
μ = ρ * ν

y_wall_bot = -h
y_wall_top = h

mesh_p  = Penguin.Mesh((nx, ny), (Lx, 2h), (0.0, -h))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, 2h), (-0.5dx, -h))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, 2h), (0.0, -h - 0.5dy))

body = (x, y, _=0.0) -> begin
    if y < y_wall_bot
        return y_wall_bot - y
    elseif y > y_wall_top
        return y - y_wall_top
    else
        return -min(y - y_wall_bot, y_wall_top - y)
    end
end

capacity_ux = Capacity(body, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(body, mesh_uy; compute_centroids=false)
capacity_p  = Capacity(body, mesh_p;  compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
total_dofs = 2 * (nu_x + nu_y) + np

xs_ux = mesh_ux.nodes[1]
ys_ux = mesh_ux.nodes[2]
xs_uy = mesh_uy.nodes[1]
ys_uy = mesh_uy.nodes[2]
xs_p  = mesh_p.nodes[1]
ys_p  = mesh_p.nodes[2]

trim_coords(v) = length(v) > 1 ? v[1:end-1] : copy(v)
xs_ux_trim = trim_coords(xs_ux)
ys_ux_trim = trim_coords(ys_ux)
xs_uy_trim = trim_coords(xs_uy)
ys_uy_trim = trim_coords(ys_uy)

###########
# Utilities
###########
function complex_profile(F0, ω, ν_local, h_local, y)
    λ = sqrt(im * ω / ν_local)
    F0 / (im * ω) * (1 - cosh(λ * y) / cosh(λ * h_local))
end

function complex_bulk(F0, ω, ν_local, h_local)
    λ = sqrt(im * ω / ν_local)
    F0 / (im * ω) * (1 - tanh(λ * h_local) / (λ * h_local))
end

function complex_wall_shear(F0, ω, ν_local, h_local, μ_local)
    λ = sqrt(im * ω / ν_local)
    Uprime = -(F0 / (im * ω)) * λ * tanh(λ * h_local)
    μ_local * Uprime
end

function forcing_from_bulk(target_bulk, ω, ν_local, h_local)
    λ = sqrt(im * ω / ν_local)
    im * ω * target_bulk / (1 - tanh(λ * h_local) / (λ * h_local))
end

function harmonic_fit(t, s, ω)
    A = hcat(cos.(ω .* t), sin.(ω .* t))
    coeffs = A \ s
    coeffs[1] - im * coeffs[2]
end

wrap_phase_deg(θ) = mod(θ + 180, 360) - 180

function trapz(x, y)
    n = length(x)
    n == length(y) || error("trapz dimension mismatch")
    if n == 1
        return 0.0
    end
    acc = 0.0
    for i in 2:n
        acc += 0.5 * (y[i] + y[i-1]) * (x[i] - x[i-1])
    end
    acc
end

function analytic_velocity(y, t, F0, ω, ν_local, h_local)
    real(complex_profile(F0, ω, ν_local, h_local, y) * exp(im * ω * t))
end

function analyze_case(alpha; target_bulk=1.0, periods=1, steps_per_period=240)
    ω = (alpha^2 * ν) / h^2
    T = 2π / ω
    Δt = T / steps_per_period
    total_steps = periods * steps_per_period
    T_end = periods * T

    F0 = forcing_from_bulk(target_bulk, ω, ν, h)
    λ = sqrt(im * ω / ν)
    U_bulk_exact = complex_bulk(F0, ω, ν, h)
    U_center_exact = complex_profile(F0, ω, ν, h, 0.0)
    τ_exact = complex_wall_shear(F0, ω, ν, h, μ)

    inlet_velocity = (x, y, t=0.0) -> analytic_velocity(y, t, F0, ω, ν, h)

    bc_ux = BorderConditions(Dict(
        :left   => Dirichlet(inlet_velocity),
        :right  => Outflow(),
        :bottom => Dirichlet((x, y, t=0.0) -> 0.0),
        :top    => Dirichlet((x, y, t=0.0) -> 0.0)
    ))

    bc_uy = BorderConditions(Dict(
        :left   => Dirichlet((x, y, t=0.0) -> 0.0),
        :right  => Outflow(),
        :bottom => Dirichlet((x, y, t=0.0) -> 0.0),
        :top    => Dirichlet((x, y, t=0.0) -> 0.0)
    ))

    f_force = (x, y, z=0.0, t=0.0) -> real(F0 * exp(im * ω * t))
    fluid = Fluid((mesh_ux, mesh_uy),
                  (capacity_ux, capacity_uy),
                  (operator_ux, operator_uy),
                  mesh_p,
                  capacity_p,
                  operator_p,
                  μ, ρ, f_force, (x,y,z=0.0,t=0.0)->0.0)

    pressure_gauge = PinPressureGauge()
    interface_bc = Dirichlet(0.0)

    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=zeros(total_dofs))
    times, histories = solve_NavierStokesMono_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=:CN)

    xs_len = length(xs_ux_trim)
    ys_len = length(ys_ux_trim)
    center_idx = clamp(argmin(abs.(ys_ux_trim)), 1, ys_len)
    area = Lx * 2h

    total_len = length(times)
    center_series = zeros(Float64, total_len)
    bulk_series = zeros(Float64, total_len)
    shear_series = zeros(Float64, total_len)
    force_series = zeros(Float64, total_len)
    power_series = zeros(Float64, total_len)
    diss_series = zeros(Float64, total_len)

    profile_series = zeros(Float64, ys_len, steps_per_period)
    profile_counter = 0

    y_norm = ys_ux_trim ./ h
    for (step, hist) in enumerate(histories)
        ux_hist = hist[1:nu_x]
        Ux = reshape(ux_hist, (length(xs_ux), length(ys_ux)))
        Ux_trim = Ux[1:length(xs_ux_trim), 1:length(ys_ux_trim)]
        avg_profile = vec(mean(Ux_trim, dims=1))

        center_series[step] = avg_profile[center_idx]
        bulk_series[step] = trapz(ys_ux_trim, avg_profile) / (2h)

        du_dy = similar(avg_profile)
        du_dy[1] = (avg_profile[2] - avg_profile[1]) / (ys_ux_trim[2] - ys_ux_trim[1])
        du_dy[end] = (avg_profile[end] - avg_profile[end-1]) / (ys_ux_trim[end] - ys_ux_trim[end-1])
        for j in 2:length(avg_profile)-1
            du_dy[j] = (avg_profile[j+1] - avg_profile[j-1]) / (ys_ux_trim[j+1] - ys_ux_trim[j-1])
        end
        shear_series[step] = μ * du_dy[end]

        f_val = real(F0 * exp(im * ω * times[step]))
        force_series[step] = f_val
        power_series[step] = ρ * f_val * trapz(ys_ux_trim, avg_profile) * Lx
        diss_series[step] = μ * trapz(ys_ux_trim, du_dy .^ 2) * Lx

        if step > total_len - steps_per_period
            profile_counter += 1
            profile_series[:, profile_counter] = avg_profile
        end
    end

    idx_start = total_len - steps_per_period + 1
    t_window = times[idx_start:end]
    center_window = center_series[idx_start:end]
    bulk_window = bulk_series[idx_start:end]
    shear_window = shear_series[idx_start:end]
    force_window = force_series[idx_start:end]
    power_window = power_series[idx_start:end]
    diss_window = diss_series[idx_start:end]

    center_amp_num = harmonic_fit(t_window, center_window, ω)
    bulk_amp_num = harmonic_fit(t_window, bulk_window, ω)
    shear_amp_num = harmonic_fit(t_window, shear_window, ω)

    amp_err_center = abs(abs(center_amp_num) - abs(U_center_exact)) / max(abs(U_center_exact), eps())
    amp_err_bulk = abs(abs(bulk_amp_num) - abs(U_bulk_exact)) / max(abs(U_bulk_exact), eps())
    amp_err_shear = abs(abs(shear_amp_num) - abs(τ_exact)) / max(abs(τ_exact), eps())

    phase_err_center = wrap_phase_deg(rad2deg(angle(center_amp_num / U_center_exact)))
    phase_err_bulk = wrap_phase_deg(rad2deg(angle(bulk_amp_num / U_bulk_exact)))
    phase_err_shear = wrap_phase_deg(rad2deg(angle(shear_amp_num / τ_exact)))

    profile_amplitudes = Vector{ComplexF64}(undef, ys_len)
    for j in 1:ys_len
        profile_amplitudes[j] = harmonic_fit(t_window, view(profile_series, j, :), ω)
    end
    profile_exact = [complex_profile(F0, ω, ν, h, y) for y in ys_ux_trim]

    profile_tag = replace(@sprintf("alpha_%g", alpha), "." => "p")
    profile_data = hcat(ys_ux_trim ./ h,
                    real.(profile_amplitudes),
                    imag.(profile_amplitudes),
                    real.(profile_exact),
                        imag.(profile_exact))
    header_profile = ["y_over_h","Re_u_num","Im_u_num","Re_u_exact","Im_u_exact"]
    writedlm("oscillatory_channel_profile_$profile_tag.csv",
             vcat(permutedims(header_profile), profile_data))

    center_exact_t = real(U_center_exact * exp.(im .* ω .* t_window))
    bulk_exact_t = real(U_bulk_exact * exp.(im .* ω .* t_window))
    shear_exact_t = real(τ_exact * exp.(im .* ω .* t_window))

    signal_matrix = hcat(t_window,
                         center_window,
                         bulk_window,
                         shear_window,
                         center_exact_t,
                         bulk_exact_t,
                         shear_exact_t,
                         force_window)
    header_signals = ["t","u_center","u_bulk","tau_w","u_center_exact","u_bulk_exact","tau_w_exact","force"]
    writedlm("oscillatory_channel_signals_$profile_tag.csv",
             vcat(permutedims(header_signals), signal_matrix))

    fig_profile = Figure(resolution=(950, 420))
    ax_amp = Axis(fig_profile[1,1], xlabel="y / h", ylabel="Amplitude",
                  title=@sprintf("Amplitude vs y (α=%.1f)", alpha))
    lines!(ax_amp, y_norm, abs.(profile_amplitudes); color=:steelblue, label="numeric")
    lines!(ax_amp, y_norm, abs.(profile_exact); color=:black, linestyle=:dash, label="analytic")
    axislegend(ax_amp; position=:lt)
    ax_phase = Axis(fig_profile[1,2], xlabel="y / h", ylabel="Phase [deg]",
                    title="Phase vs y")
    lines!(ax_phase, y_norm, rad2deg.(angle.(profile_amplitudes)); color=:orchid, label="numeric")
    lines!(ax_phase, y_norm, rad2deg.(angle.(profile_exact)); color=:black, linestyle=:dash, label="analytic")
    axislegend(ax_phase; position=:lt)
    save("oscillatory_channel_profile_$profile_tag.png", fig_profile)
    display(fig_profile)

    fig_signals = Figure(resolution=(1100, 380))
    ax_c = Axis(fig_signals[1,1], xlabel="t / T", ylabel="u_center",
                title="Centerline velocity (last period)")
    lines!(ax_c, (t_window .- t_window[1]) ./ T, center_window; color=:blue, label="numeric")
    lines!(ax_c, (t_window .- t_window[1]) ./ T, center_exact_t; color=:black, linestyle=:dash, label="analytic")
    axislegend(ax_c; position=:rt)
    ax_b = Axis(fig_signals[1,2], xlabel="t / T", ylabel="⟨u⟩",
                title="Bulk velocity (last period)")
    lines!(ax_b, (t_window .- t_window[1]) ./ T, bulk_window; color=:green)
    lines!(ax_b, (t_window .- t_window[1]) ./ T, bulk_exact_t; color=:black, linestyle=:dash)
    ax_tau = Axis(fig_signals[1,3], xlabel="t / T", ylabel="τ_w",
                  title="Wall shear (last period)")
    lines!(ax_tau, (t_window .- t_window[1]) ./ T, shear_window; color=:crimson)
    lines!(ax_tau, (t_window .- t_window[1]) ./ T, shear_exact_t; color=:black, linestyle=:dash)
    save("oscillatory_channel_signals_$profile_tag.png", fig_signals)
    display(fig_signals)

    Pin_avg = mean(power_window)
    Phi_avg = mean(diss_window)
    energy_rel = abs(Phi_avg - Pin_avg) / max(abs(Phi_avg), eps())

    (; alpha,
       ω,
       Δt,
       target_bulk,
       amp_err_center,
       phase_err_center,
       amp_err_bulk,
       phase_err_bulk,
       amp_err_shear,
       phase_err_shear,
       energy_rel)
end

alphas = [10.0]
summary = Vector{NamedTuple}(undef, length(alphas))

for (i, α) in pairs(alphas)
    println()
    println(@sprintf("=== Running oscillatory channel case α = %.1f ===", α))
    summary[i] = analyze_case(α)
    println(@sprintf("Amplitude errors (center, bulk, shear): %.3f, %.3f, %.3f",
                     summary[i].amp_err_center,
                     summary[i].amp_err_bulk,
                     summary[i].amp_err_shear))
    println(@sprintf("Phase errors (deg): %.2f, %.2f, %.2f",
                     summary[i].phase_err_center,
                     summary[i].phase_err_bulk,
                     summary[i].phase_err_shear))
    println(@sprintf("Energy balance relative error: %.3e", summary[i].energy_rel))
end

summary_header = ["alpha","omega","dt","target_bulk",
                  "amp_err_center","phase_err_center",
                  "amp_err_bulk","phase_err_bulk",
                  "amp_err_shear","phase_err_shear",
                  "energy_rel"]
summary_rows = [Tuple(summary[i]) for i in 1:length(summary)]
summary_matrix = hcat([getindex.(summary_rows, j) for j in 1:length(summary_header)]...)
writedlm("oscillatory_channel_summary.csv",
         vcat(permutedims(summary_header), summary_matrix))

println("Saved summary to oscillatory_channel_summary.csv")
