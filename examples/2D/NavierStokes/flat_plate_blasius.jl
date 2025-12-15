using Penguin
using CairoMakie
using Statistics
using Interpolations
using Printf

"""
Flat-plate Blasius boundary layer solved in a half-channel with a cut-cell wall
located at y = 0. The inflow is uniform and laminar; the analytical Blasius
similarity solution is used for comparison at a downstream station.

Domain: x in [0, Lx], y in [y0, y0 + Ly] with a no-slip wall at y = 0.
Reynolds number is based on plate length Lref = 1 and freestream speed U_inf.
"""

###########
# Helpers
###########
# RK4 integration of the Blasius ODE f''' + 0.5 f f'' = 0 with f(0)=0, f'(0)=0, f''(0)=a.
function blasius_profile(; η_max=12.0, n=800)
    dη = η_max / (n - 1)
    η = range(0.0, η_max; length=n)
    f = zeros(Float64, n)
    g = zeros(Float64, n) # g = f'
    h = zeros(Float64, n) # h = f''
    h[1] = 0.469599988361 # canonical shooting value for f''(0)

    # simple explicit RK4 since the system is smooth and monotone
    for i in 1:(n-1)
        fi, gi, hi = f[i], g[i], h[i]
        k1_f, k1_g, k1_h = gi, hi, -0.5 * fi * hi
        k2_f, k2_g, k2_h = gi + 0.5 * dη * k1_g, hi + 0.5 * dη * k1_h, -0.5 * (fi + 0.5 * dη * k1_f) * (hi + 0.5 * dη * k1_h)
        k3_f, k3_g, k3_h = gi + 0.5 * dη * k2_g, hi + 0.5 * dη * k2_h, -0.5 * (fi + 0.5 * dη * k2_f) * (hi + 0.5 * dη * k2_h)
        k4_f, k4_g, k4_h = gi + dη * k3_g, hi + dη * k3_h, -0.5 * (fi + dη * k3_f) * (hi + dη * k3_h)

        f[i+1] = fi + (dη / 6.0) * (k1_f + 2k2_f + 2k3_f + k4_f)
        g[i+1] = gi + (dη / 6.0) * (k1_g + 2k2_g + 2k3_g + k4_g)
        h[i+1] = hi + (dη / 6.0) * (k1_h + 2k2_h + 2k3_h + k4_h)
    end

    return η, g
end

# Trapezoidal rule for non-uniform grids.
function trapz(x::AbstractVector, f::AbstractVector)
    @assert length(x) == length(f)
    if length(x) < 2
        return zero(eltype(f))
    end
    acc = zero(eltype(f))
    for i in 1:(length(x)-1)
        acc += 0.5 * (f[i] + f[i+1]) * (x[i+1] - x[i])
    end
    return acc
end

nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))

###########
# Geometry
###########
Lx, Ly = 1.5, 0.25
x0, y0 = 0.0, -0.02
nx, ny = 64, 64

body = (x, y, _=0.0) -> -y # wall is the zero level set at y = 0

mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

capacity_ux = Capacity(body, mesh_ux; compute_centroids=true)
capacity_uy = Capacity(body, mesh_uy; compute_centroids=true)
capacity_p  = Capacity(body, mesh_p; compute_centroids=true)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
U_inf = 1.0
Lref = 1.0
Re_L = 1.0e4
ρ = 1.0
μ = ρ * U_inf * Lref / Re_L
ν = μ / ρ

inflow_u = (x, y, t=0.0) -> U_inf
ux_left   = Dirichlet(inflow_u)
ux_right  = Outflow()
ux_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
ux_top    = Dirichlet((x, y, t=0.0) -> U_inf)

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)
uy_right = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(:left=>ux_left, :right=>ux_right, :bottom=>ux_bottom, :top=>ux_top))
bc_uy = BorderConditions(Dict(:left=>uy_zero, :right=>uy_right, :bottom=>uy_zero, :top=>uy_zero))

pressure_gauge = PinPressureGauge()
cut_bc = Dirichlet((x, y, t=0.0) -> 0.0) # no-slip on the plate

###########
# Fluid and solver
###########
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
x0_vec = zeros(2 * (nu_x + nu_y) + np)

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, cut_bc; x0=x0_vec)

println("Solving steady flat-plate boundary layer (Re_L=$(Re_L))...")
_, picard_iters, picard_res = solve_NavierStokesMono_steady!(solver;
    nlsolve_method=:picard,
    tol=1e-9,
    maxiter=40,
    relaxation=1.0)
println(@sprintf("Picard iterations = %d, residual = %.3e", picard_iters, picard_res))

###########
# Post-processing
###########
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
Ux = reshape(solver.x[1:nu_x], (length(xs), length(ys)))
Uy = reshape(solver.x[2nu_x+1:2nu_x+nu_y], (length(xs), length(ys)))

x_probe = 1.0
ix = nearest_index(xs, x_probe)
ys_fluid_mask = ys .>= 0.0
ys_fluid = ys[ys_fluid_mask]
ux_profile = Ux[ix, ys_fluid_mask]

η_tab, fprime_tab = blasius_profile()
fprime_interp = LinearInterpolation(η_tab, fprime_tab; extrapolation_bc=Flat())

η_profile = ys_fluid .* sqrt(U_inf / (2 * ν * max(x_probe, 1e-8)))
η_clamped = clamp.(η_profile, minimum(η_tab), maximum(η_tab))
ux_blasius = U_inf .* fprime_interp.(η_clamped)

profile_err = ux_profile - ux_blasius
L2_err = sqrt(mean(abs2, profile_err))
Linf_err = maximum(abs.(profile_err))

δ_star_numeric = trapz(ys_fluid, 1 .- ux_profile ./ U_inf)
δ_star_blasius = 1.7208 * sqrt(ν * x_probe / U_inf)
δ99_blasius = 4.91 * sqrt(ν * x_probe / U_inf)

println(@sprintf("Profile L2 error = %.4e, Linf error = %.4e", L2_err, Linf_err))
println(@sprintf("δ99 analytic at x=%.2f: %.4e", x_probe, δ99_blasius))

speed = sqrt.(Ux.^2 .+ Uy.^2)

fig = Figure(resolution=(1000, 450))
ax = Axis(fig[1, 1], xlabel="u / U_inf", ylabel="y", title="Blasius profile at x = $(x_probe)")
lines!(ax, ux_profile ./ U_inf, ys_fluid; label="numerical")
lines!(ax, ux_blasius ./ U_inf, ys_fluid; color=:red, linestyle=:dash, label="Blasius")
hlines!(ax, [δ99_blasius]; color=:gray, linestyle=:dot, label="δ99")
axislegend(ax, position=:rb)

ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", title="Speed magnitude")
heatmap!(ax2, xs, ys, speed; colormap=:plasma)
contour!(ax2, xs, ys, [body(x, y) for x in xs, y in ys]; levels=[0.0], color=:white)

save("flat_plate_blasius.png", fig)
println("Saved plot to flat_plate_blasius.png")
