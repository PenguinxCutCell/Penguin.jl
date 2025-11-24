using Penguin
using Statistics
using Printf
using LinearAlgebra
using CairoMakie

"""
    Kovasznay flow convergence study (steady Navier–Stokes, Re ≈ 40).

The test solves the steady incompressible Navier–Stokes equations on the unit
square for the classical Kovasznay solution. Dirichlet data on every boundary
matches the analytical velocity field, and the pressure gauge enforces zero
mean. We sample a sequence of uniform grids to measure the h-convergence of
velocity and pressure in both L₂ and L∞ norms.

Expected behaviour:
  • ‖u - uₕ‖₂, ‖v - vₕ‖₂, ‖p - pₕ‖₂ ≈ O(hʳ) with r close to the formal order
    of the staggered discretisation.
  • L∞ slopes are typically slightly lower but should follow the same trend.
"""

Re = 40.0
ρ = 1.0
ν = 1.0 / Re
μ = ρ * ν
λ = Re / 2 - sqrt(Re^2 / 4 + 4π^2)

kovasznay_u(x, y) = 1 - exp(λ * x) * cos(2π * y)
kovasznay_v(x, y) = (λ / (2π)) * exp(λ * x) * sin(2π * y)
kovasznay_p(x, y) = 0.5 * (1 - exp(2λ * x))

resolutions = [32, 64, 128, 256]
results = NamedTuple[]
best_data = nothing

trim_copy(A) = size(A, 1) > 1 && size(A, 2) > 1 ? copy(@view A[1:end-1, 1:end-1]) : copy(A)
trim_coords(v) = length(v) > 1 ? copy(v[1:end-1]) : copy(v)

println("=== Kovasznay flow steady convergence (Re ≈ 40) ===")
println(@sprintf("%6s %7s %12s %12s %12s %12s %12s %12s",
                 "N", "h", "‖u‖₂ err", "‖u‖∞ err", "‖v‖₂ err", "‖v‖∞ err",
                 "‖p‖₂ err", "‖p‖∞ err"))

body = (x, y, _=0.0) -> -1.0  # full fluid domain (no obstacle)

for N in resolutions
    nx = N
    ny = N
    Lx, Ly = 1.0, 1.0
    x0, y0 = 0.0, 0.0

    mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
    dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
    dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
    mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
    mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

    capacity_ux = Capacity(body, mesh_ux)
    capacity_uy = Capacity(body, mesh_uy)
    capacity_p  = Capacity(body, mesh_p)

    operator_ux = DiffusionOps(capacity_ux)
    operator_uy = DiffusionOps(capacity_uy)
    operator_p  = DiffusionOps(capacity_p)

    bc_ux = BorderConditions(Dict(
        :left   => Dirichlet((x, y, t=0.0) -> kovasznay_u(x, y)),
        :right  => Dirichlet((x, y, t=0.0) -> kovasznay_u(x, y)),
        :bottom => Dirichlet((x, y, t=0.0) -> kovasznay_u(x, y)),
        :top    => Dirichlet((x, y, t=0.0) -> kovasznay_u(x, y)),
    ))
    bc_uy = BorderConditions(Dict(
        :left   => Dirichlet((x, y, t=0.0) -> kovasznay_v(x, y)),
        :right  => Dirichlet((x, y, t=0.0) -> kovasznay_v(x, y)),
        :bottom => Dirichlet((x, y, t=0.0) -> kovasznay_v(x, y)),
        :top    => Dirichlet((x, y, t=0.0) -> kovasznay_v(x, y)),
    ))

    fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
    fₚ = (x, y, z=0.0, t=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (capacity_ux, capacity_uy),
                  (operator_ux, operator_uy),
                  mesh_p,
                  capacity_p,
                  operator_p,
                  μ, ρ, fᵤ, fₚ)

    nu_x = prod(operator_ux.size)
    nu_y = prod(operator_uy.size)
    np = prod(operator_p.size)
    total_dofs = 2 * (nu_x + nu_y) + np

    xs_ux = mesh_ux.nodes[1]
    ys_ux = mesh_ux.nodes[2]
    xs_uy = mesh_uy.nodes[1]
    ys_uy = mesh_uy.nodes[2]
    xs_p = mesh_p.nodes[1]
    ys_p = mesh_p.nodes[2]

    Ux_exact = [kovasznay_u(x, y) for x in xs_ux, y in ys_ux]
    Uy_exact = [kovasznay_v(x, y) for x in xs_uy, y in ys_uy]
    P_exact = [kovasznay_p(x, y) for x in xs_p, y in ys_p]

    x0_vec = zeros(total_dofs)
    x0_vec[1:nu_x] .= vec(Ux_exact)
    x0_vec[nu_x+1:2nu_x] .= vec(Ux_exact)
    x0_vec[2nu_x+1:2nu_x+nu_y] .= vec(Uy_exact)
    x0_vec[2nu_x+nu_y+1:2*(nu_x+nu_y)] .= vec(Uy_exact)
    x0_vec[2*(nu_x + nu_y) + 1:end] .= vec(P_exact)

    pressure_gauge = PinPressureGauge()
    interface_bc = Dirichlet(0.0)

    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)
    solve_NavierStokesMono_steady!(solver; tol=eps(), maxiter=8, relaxation=1.0, nlsolve_method=:picard)

    uωx = solver.x[1:nu_x]
    uωy = solver.x[2nu_x+1:2nu_x+nu_y]
    pω = solver.x[2*(nu_x + nu_y) + 1:end]

    Ux_num = reshape(uωx, Tuple(operator_ux.size))
    Uy_num = reshape(uωy, Tuple(operator_uy.size))
    P_num = reshape(pω, Tuple(operator_p.size))

    xs_ux_trim = trim_coords(xs_ux)
    ys_ux_trim = trim_coords(ys_ux)
    xs_uy_trim = trim_coords(xs_uy)
    ys_uy_trim = trim_coords(ys_uy)
    xs_p_trim = trim_coords(xs_p)
    ys_p_trim = trim_coords(ys_p)

    Ux_num_trim = trim_copy(Ux_num)
    Uy_num_trim = trim_copy(Uy_num)
    P_num_trim = trim_copy(P_num)

    Ux_exact_trim = trim_copy(Ux_exact)
    Uy_exact_trim = trim_copy(Uy_exact)
    P_exact_trim = trim_copy(P_exact)

    hx = Lx / nx
    hy = Ly / ny
    area_vel = hx * hy
    area_p = hx * hy

    ΔUx = Ux_num_trim .- Ux_exact_trim
    ΔUy = Uy_num_trim .- Uy_exact_trim
    ΔP = P_num_trim .- P_exact_trim
    ΔP .-= mean(ΔP)  # adjust constant gauge mismatch

    err_u_L2 = sqrt(sum(abs2, ΔUx) * area_vel)
    err_v_L2 = sqrt(sum(abs2, ΔUy) * area_vel)
    err_p_L2 = sqrt(sum(abs2, ΔP) * area_p)

    err_u_Linf = maximum(abs, ΔUx)
    err_v_Linf = maximum(abs, ΔUy)
    err_p_Linf = maximum(abs, ΔP)

    h = max(hx, hy)

    push!(results, (; N, h, err_u_L2, err_u_Linf, err_v_L2, err_v_Linf, err_p_L2, err_p_Linf))

    if best_data === nothing || h < best_data.h
        global best_data = (; N, h,
                             xs_ux=xs_ux_trim, ys_ux=ys_ux_trim,
                             xs_uy=xs_uy_trim, ys_uy=ys_uy_trim,
                             xs_p=xs_p_trim, ys_p=ys_p_trim,
                             Ux_num=Ux_num_trim, Uy_num=Uy_num_trim, P_num=P_num_trim,
                             Ux_exact=Ux_exact_trim, Uy_exact=Uy_exact_trim, P_exact=P_exact_trim,
                             ΔUx=ΔUx, ΔUy=ΔUy, ΔP=ΔP)
    end

    println(@sprintf("%6d %7.4f %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e",
                     N, h, err_u_L2, err_u_Linf, err_v_L2, err_v_Linf, err_p_L2, err_p_Linf))
end

function convergence_rates(errors::Vector{Float64}, hs::Vector{Float64})
    rates = Float64[]
    for i in 1:length(errors)-1
        push!(rates, log(errors[i] / errors[i+1]) / log(hs[i] / hs[i+1]))
    end
    return rates
end

hs = [r.h for r in results]
u_L2_rates = convergence_rates([r.err_u_L2 for r in results], hs)
v_L2_rates = convergence_rates([r.err_v_L2 for r in results], hs)
p_L2_rates = convergence_rates([r.err_p_L2 for r in results], hs)
u_Linf_rates = convergence_rates([r.err_u_Linf for r in results], hs)
v_Linf_rates = convergence_rates([r.err_v_Linf for r in results], hs)
p_Linf_rates = convergence_rates([r.err_p_Linf for r in results], hs)

format_rates(rates) = join([@sprintf("%.2f", r) for r in rates], ", ")

println()
println("L₂ convergence rates:")
println("  u-component: ", format_rates(u_L2_rates))
println("  v-component: ", format_rates(v_L2_rates))
println("  pressure  : ", format_rates(p_L2_rates))
println("L∞ convergence rates:")
println("  u-component: ", format_rates(u_Linf_rates))
println("  v-component: ", format_rates(v_Linf_rates))
println("  pressure  : ", format_rates(p_Linf_rates))

best = results[end]
println()
println(@sprintf("Highest resolution N=%d, h=%.4f", best.N, best.h))
println(@sprintf("Errors: ‖u‖₂=%.4e, ‖u‖∞=%.4e, ‖v‖₂=%.4e, ‖v‖∞=%.4e, ‖p‖₂=%.4e, ‖p‖∞=%.4e",
                 best.err_u_L2, best.err_u_Linf, best.err_v_L2, best.err_v_Linf, best.err_p_L2, best.err_p_Linf))

best_data === nothing && error("No solution data recorded.")

Ux_num_plot = best_data.Ux_num
Uy_num_plot = best_data.Uy_num
P_num_plot = best_data.P_num
Ux_exact_plot = best_data.Ux_exact
Uy_exact_plot = best_data.Uy_exact
P_exact_plot = best_data.P_exact
ΔUx_plot = best_data.ΔUx
ΔUy_plot = best_data.ΔUy
ΔP_plot = best_data.ΔP

xs_ux_plot = best_data.xs_ux
ys_ux_plot = best_data.ys_ux
xs_uy_plot = best_data.xs_uy
ys_uy_plot = best_data.ys_uy
xs_p_plot = best_data.xs_p
ys_p_plot = best_data.ys_p

fig = Figure(resolution=(1200, 700))

ax_u = Axis(fig[1,1], xlabel="x", ylabel="y", title="u (N=$(best_data.N))")
hm_u = heatmap!(ax_u, xs_ux_plot, ys_ux_plot, Ux_num_plot'; colormap=:plasma)
contour!(ax_u, xs_ux_plot, ys_ux_plot, Ux_exact_plot'; levels=8, color=:black, linewidth=1)
Colorbar(fig[1,2], hm_u; label="u")

ax_v = Axis(fig[1,3], xlabel="x", ylabel="y", title="v (N=$(best_data.N))")
hm_v = heatmap!(ax_v, xs_uy_plot, ys_uy_plot, Uy_num_plot'; colormap=:viridis)
contour!(ax_v, xs_uy_plot, ys_uy_plot, Uy_exact_plot'; levels=8, color=:black, linewidth=1)
Colorbar(fig[1,4], hm_v; label="v")

ax_p = Axis(fig[1,5], xlabel="x", ylabel="y", title="Pressure (N=$(best_data.N))")
hm_p = heatmap!(ax_p, xs_p_plot, ys_p_plot, P_num_plot'; colormap=:balance)
contour!(ax_p, xs_p_plot, ys_p_plot, P_exact_plot'; levels=8, color=:black, linewidth=1)
Colorbar(fig[1,6], hm_p; label="p")

ax_err_u = Axis(fig[2,1], xlabel="x", ylabel="y", title="u error")
hm_err_u = heatmap!(ax_err_u, xs_ux_plot, ys_ux_plot, ΔUx_plot'; colormap=:balance)
Colorbar(fig[2,2], hm_err_u; label="Δu")

ax_err_v = Axis(fig[2,3], xlabel="x", ylabel="y", title="v error")
hm_err_v = heatmap!(ax_err_v, xs_uy_plot, ys_uy_plot, ΔUy_plot'; colormap=:balance)
Colorbar(fig[2,4], hm_err_v; label="Δv")

ax_err_p = Axis(fig[2,5], xlabel="x", ylabel="y", title="Pressure error")
hm_err_p = heatmap!(ax_err_p, xs_p_plot, ys_p_plot, ΔP_plot'; colormap=:balance)
Colorbar(fig[2,6], hm_err_p; label="Δp")

save("navierstokes_kovasznay_highres.png", fig)
display(fig)

order = sortperm(hs; rev=true)
hs_sorted = hs[order]
u_L2_sorted = [r.err_u_L2 for r in results][order]
v_L2_sorted = [r.err_v_L2 for r in results][order]
p_L2_sorted = [r.err_p_L2 for r in results][order]
u_Linf_sorted = [r.err_u_Linf for r in results][order]
v_Linf_sorted = [r.err_v_Linf for r in results][order]
p_Linf_sorted = [r.err_p_Linf for r in results][order]

fig_conv = Figure(resolution=(800, 450))
ax_conv = Axis(fig_conv[1,1], xlabel="h", ylabel="Error", title="Kovasznay convergence", xscale=log10, yscale=log10)

lines!(ax_conv, hs_sorted, u_L2_sorted; color=:blue, label="‖u‖₂")
scatter!(ax_conv, hs_sorted, u_L2_sorted; color=:blue)
lines!(ax_conv, hs_sorted, v_L2_sorted; color=:green, label="‖v‖₂")
scatter!(ax_conv, hs_sorted, v_L2_sorted; color=:green)

lines!(ax_conv, hs_sorted, u_Linf_sorted; color=:blue, linestyle=:dash, label="‖u‖∞")
scatter!(ax_conv, hs_sorted, u_Linf_sorted; color=:blue)
lines!(ax_conv, hs_sorted, v_Linf_sorted; color=:green, linestyle=:dash, label="‖v‖∞")
scatter!(ax_conv, hs_sorted, v_Linf_sorted; color=:green)

# add reference slopes
h_ref = [hs_sorted[1], hs_sorted[end]]
slope_u = u_L2_sorted[1] / (hs_sorted[1]^2)
slope_v = v_L2_sorted[1] / (hs_sorted[1]^2)
lines!(ax_conv, h_ref, slope_u .* (h_ref .^ 2); color=:blue, linestyle=:dot, label="O(h²) ref.")
lines!(ax_conv, h_ref, slope_v .* (h_ref .^ 2); color=:green, linestyle=:dot)
axislegend(ax_conv; position=:lb)
save("navierstokes_kovasznay_convergence.png", fig_conv)
display(fig_conv)

println("--- Kovasznay convergence study complete ---")