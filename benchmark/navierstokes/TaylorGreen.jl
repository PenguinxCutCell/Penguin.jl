using Penguin
using LinearAlgebra
using Printf
using CSV
using DataFrames
using CairoMakie

# Taylor–Green vortex convergence study for the Navier–Stokes prototype.
# Domain: [0, 2π] × [0, 2π]
# Exact solution:
#   u(x,y,t) =  sin(kx) cos(ky) e^{-2νk² t}
#   v(x,y,t) = -cos(kx) sin(ky) e^{-2νk² t}
# This is a true Navier–Stokes solution with zero forcing. We enforce the exact
# velocity on the boundary so the manufactured solution is satisfied.

const Lx = 2π
const Ly = 2π
const x0 = 0.0
const y0 = 0.0

const μ = 1.0
const ρ = 1.0
const k = 1.0
const ν = μ / ρ

const t_end = 0.1
const Δt = 0.001
const scheme = :CN  # θ = 1/2 (Crank–Nicolson) for viscous terms

u_exact(x,y,t) =  sin(k * x) * cos(k * y) * exp(-2.0 * ν * k^2 * t)
v_exact(x,y,t) = -cos(k * x) * sin(k * y) * exp(-2.0 * ν * k^2 * t)

# Helper to skip boundary samples when computing errors
interior_indices(n) = 2:(n-1)

# resolutions to test
ns = [8, 16, 32, 64]

errors_u = Float64[]
errors_v = Float64[]
errors_p = Float64[]
hs = Float64[]

# cached fields for plotting (taken from finest grid)
xs_ux_plot = nothing; ys_ux_plot = nothing
xs_uy_plot = nothing; ys_uy_plot = nothing
Xp_plot = nothing;  Yp_plot = nothing
Ux_plot = nothing;  Uy_plot = nothing; P_plot = nothing

for n in ns
    nx = n; ny = n
    mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
    dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
    dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
    mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
    mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

    body = (x, y, _=0.0) -> -1.0  # whole domain is fluid

    cap_ux = Capacity(body, mesh_ux)
    cap_uy = Capacity(body, mesh_uy)
    cap_p  = Capacity(body, mesh_p)

    op_ux = DiffusionOps(cap_ux)
    op_uy = DiffusionOps(cap_uy)
    op_p  = DiffusionOps(cap_p)

    bc_ux = BorderConditions(Dict(
        :left  => Dirichlet((x,y,t)->u_exact(x,y,t)),
        :right => Dirichlet((x,y,t)->u_exact(x,y,t)),
        :bottom=> Dirichlet((x,y,t)->u_exact(x,y,t)),
        :top   => Dirichlet((x,y,t)->u_exact(x,y,t)),
    ))

    bc_uy = BorderConditions(Dict(
        :left  => Dirichlet((x,y,t)->v_exact(x,y,t)),
        :right => Dirichlet((x,y,t)->v_exact(x,y,t)),
        :bottom=> Dirichlet((x,y,t)->v_exact(x,y,t)),
        :top   => Dirichlet((x,y,t)->v_exact(x,y,t)),
    ))

    pressure_gauge = PinPressureGauge()
    bc_cut = Dirichlet(0.0)

    fᵤ = (x,y,z=0.0) -> 0.0
    fₚ = (x,y,z=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (cap_ux, cap_uy),
                  (op_ux, op_uy),
                  mesh_p,
                  cap_p,
                  op_p,
                  μ, ρ, fᵤ, fₚ)

    nu_x = prod(op_ux.size)
    nu_y = prod(op_uy.size)
    np = prod(op_p.size)
    Ntot = 2 * (nu_x + nu_y) + np
    x0_vec = zeros(Float64, Ntot)

    xs_ux = mesh_ux.nodes[1]; ys_ux = mesh_ux.nodes[2]
    xs_uy = mesh_uy.nodes[1]; ys_uy = mesh_uy.nodes[2]
    Xp = mesh_p.nodes[1];      Yp = mesh_p.nodes[2]

    u_init = zeros(Float64, nu_x)
    v_init = zeros(Float64, nu_y)
    p_init = zeros(Float64, np)

    idx = 1
    for j in 1:length(ys_ux), i in 1:length(xs_ux)
        u_init[idx] = u_exact(xs_ux[i], ys_ux[j], 0.0)
        idx += 1
    end

    idx = 1
    for j in 1:length(ys_uy), i in 1:length(xs_uy)
        v_init[idx] = v_exact(xs_uy[i], ys_uy[j], 0.0)
        idx += 1
    end


    x0_vec[1:nu_x] .= u_init
    x0_vec[nu_x+1:2nu_x] .= u_init          # tie DOFs
    x0_vec[2nu_x+1:2nu_x+nu_y] .= v_init
    x0_vec[2nu_x+nu_y+1:2*(nu_x+nu_y)] .= v_init
    x0_vec[2*(nu_x+nu_y)+1:end] .= p_init

    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut; x0=x0_vec)

    @printf("Running Navier–Stokes resolution %d×%d ...\n", nx, ny)
    times, states = solve_NavierStokesMono_unsteady!(solver; Δt=Δt, T_end=t_end, scheme=scheme, method=Base.:\)
    final = states[end]

    u_num = final[1:nu_x]
    v_num = final[2nu_x+1:2nu_x+nu_y]

    Ux = reshape(u_num, length(xs_ux), length(ys_ux))
    Uy = reshape(v_num, length(xs_uy), length(ys_uy))

    Ux_ex = [u_exact(xs_ux[i], ys_ux[j], t_end) for i in 1:length(xs_ux), j in 1:length(ys_ux)]
    Uy_ex = [v_exact(xs_uy[i], ys_uy[j], t_end) for i in 1:length(xs_uy), j in 1:length(ys_uy)]

    ix_range_u = interior_indices(length(xs_ux))
    iy_range_u = interior_indices(length(ys_ux))
    ix_range_v = interior_indices(length(xs_uy))
    iy_range_v = interior_indices(length(ys_uy))

    Vux = diag(op_ux.V)
    Vuy = diag(op_uy.V)

    function weighted_L2_grid(num, ex, mask_i, mask_j, Vdiag; remove_mean=false)
        ni, nj = size(num, 1), size(num, 2)
        total_w = 0.0
        weighted_sum = 0.0
        for j in 1:nj, i in 1:ni
            if (i in mask_i) && (j in mask_j)
                lin = (j-1)*ni + i
                w = Vdiag[lin]
                err = num[i,j] - ex[i,j]
                total_w += w
                weighted_sum += w * err
            end
        end
        mean_err = (remove_mean && total_w > 0) ? weighted_sum / total_w : 0.0

        accum = 0.0
        for j in 1:nj, i in 1:ni
            if (i in mask_i) && (j in mask_j)
                lin = (j-1)*ni + i
                w = Vdiag[lin]
                err = (num[i,j] - ex[i,j]) - mean_err
                accum += w * err^2
            end
        end
        return sqrt(accum)
    end

    err_u = weighted_L2_grid(Ux, reshape(Ux_ex, size(Ux)), ix_range_u, iy_range_u, Vux)
    err_v = weighted_L2_grid(Uy, reshape(Uy_ex, size(Uy)), ix_range_v, iy_range_v, Vuy)

    push!(errors_u, err_u)
    push!(errors_v, err_v)
    push!(hs, max(Lx / nx, Ly / ny))

    global xs_ux_plot = xs_ux; global ys_ux_plot = ys_ux
    global xs_uy_plot = xs_uy; global ys_uy_plot = ys_uy
    global Ux_plot = Ux; global Uy_plot = Uy

    @printf("  h=%.5e  ||u||_L2=%.5e  ||v||_L2=%.5e\n", hs[end], err_u, err_v)
end

function rate(h, e)
    r = Float64[]
    for i in 2:length(e)
        push!(r, log(e[i] / e[i-1]) / log(h[i] / h[i-1]))
    end
    return r
end

r_u = rate(hs, errors_u)
r_v = rate(hs, errors_v)

println("\nEstimated convergence rates (between successive resolutions):")
for i in 1:length(r_u)
    @printf("  between %d and %d: u rate=%.2f, v rate=%.2f\n",
            ns[i], ns[i+1], r_u[i], r_v[i])
end

println("\nFinal errors:")
for (i,n) in enumerate(ns)
    @printf("  %4d: h=%.3e  ||u||=%.5e  ||v||=%.5e\n",
            n, hs[i], errors_u[i], errors_v[i])
end

df = DataFrame(h=hs, error_u=errors_u, error_v=errors_v)
CSV.write("taylor_green_convergence.csv", df)

fig = Figure(resolution=(900, 500))
ax = Axis(fig[1,1], xscale=log10, yscale=log10,
          xlabel="h", ylabel="volume-integrated L2 error",
          title="Taylor–Green convergence (t = $(t_end))")

lines!(ax, hs, errors_u; label="u", color=:tomato)
scatter!(ax, hs, errors_u; color=:tomato)
lines!(ax, hs, errors_v; label="v", color=:royalblue)
scatter!(ax, hs, errors_v; color=:royalblue)

p_ref = 2.0
h_ref = [minimum(hs), maximum(hs)]
ref_line = errors_u[1] * (h_ref ./ hs[1]).^p_ref
lines!(ax, h_ref, ref_line; color=:black, linestyle=:dash, label="O(h²)")
axislegend(ax, position=:rb)

save("taylor_green_convergence.png", fig)
display(fig)

fig_snap = Figure(resolution=(1200, 400))
ax1 = Axis(fig_snap[1,1], title="u_x (n=$(last(ns)), t=$(t_end))", xlabel="x", ylabel="y")
hm1 = heatmap!(ax1, xs_ux_plot, ys_ux_plot, Ux_plot; colormap=:viridis)
Colorbar(fig_snap[1,2], hm1)

ax2 = Axis(fig_snap[1,3], title="u_y (n=$(last(ns)), t=$(t_end))", xlabel="x", ylabel="y")
hm2 = heatmap!(ax2, xs_uy_plot, ys_uy_plot, Uy_plot; colormap=:thermal)
Colorbar(fig_snap[1,4], hm2)

save("taylor_green_highest_resolution_fields.png", fig_snap)
display(fig_snap)
