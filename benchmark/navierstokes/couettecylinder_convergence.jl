using Penguin
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using CSV
using DataFrames
using CairoMakie

"""
Convergence study for the Stokes Couette flow between two concentric cylinders.

This script mirrors `benchmark/navierstokes/couettecylinder.jl` but sweeps a
set of uniform resolutions, solves the steady Stokes problem with cut cells,
and compares the numerical solution against the analytic Couette velocity
profile. Volume-weighted L₂ errors are computed for both velocity components,
normalised by the fluid volume estimated from the pressure capacity centroids
(`capacity_p.C_ω`). Results are written to CSV and a convergence plot is saved.
"""

###########
# Helpers reused from the single-resolution benchmark
###########
interval_min(a, b) = 0.5 * (a + b - abs(a - b))

struct CouetteCylinderParams
    R₁::Float64  # inner radius
    R₂::Float64  # outer radius
    ω::Float64   # angular velocity of inner cylinder
end

CouetteCylinderParams(; R₁=0.25, R₂=0.5, ω=1.0) = CouetteCylinderParams(R₁, R₂, ω)

function diffusion_levelset(params::CouetteCylinderParams)
    return (x, y, _=0.0) -> begin
        r = hypot(x, y)
        φ_inner = r - params.R₁          # negative inside the inner disk
        φ_outer_ext = params.R₂ - r      # negative outside the outer cylinder
        interval_min(φ_inner, φ_outer_ext)
    end
end

annulus_body(params) = (x, y, _=0.0) -> -diffusion_levelset(params)(x, y)

function tangential_velocity(r, params::CouetteCylinderParams)
    Ri, Ro = params.R₁, params.R₂
    ω = params.ω
    δ = (Ri^2 - Ro^2)
    A = ω * Ri^2 / δ
    B = -ω * Ri^2 * Ro^2 / δ
    return A * r + B / r
end

function analytic_velocity_components(x, y, params::CouetteCylinderParams)
    r = hypot(x, y)
    if r < params.R₁ || r > params.R₂
        return (0.0, 0.0)
    end
    uθ = tangential_velocity(r, params)
    if r < eps()
        return (0.0, 0.0)
    end
    ux = -uθ * y / r
    uy =  uθ * x / r
    return (ux, uy)
end

import Penguin: build_g_g

struct CouetteCutBC <: Penguin.AbstractBoundary
    value_map::Dict{UInt64, Function}
    default::Union{Float64, Function}
end

CouetteCutBC(default::Union{Float64, Function}=0.0) =
    CouetteCutBC(Dict{UInt64, Function}(), default)

function evaluate_cut_bc(bc::CouetteCutBC, mesh_id::UInt64)
    if haskey(bc.value_map, mesh_id)
        return bc.value_map[mesh_id]
    elseif bc.default isa Function
        return bc.default
    else
        return (args...) -> bc.default
    end
end

function build_g_g(op::Penguin.AbstractOperators, bc::CouetteCutBC, cap::Penguin.Capacity)
    coords = Penguin.get_all_coordinates(cap.C_γ)
    mesh_id = objectid(cap.mesh)
    f = evaluate_cut_bc(bc, mesh_id)
    return [f(coord...) for coord in coords]
end

function couette_cut_component(component::Symbol, params::CouetteCylinderParams)
    tol = 2e-3
    return (x, y, _=0.0) -> begin
        r = hypot(x, y)
        if abs(r - params.R₁) <= tol
            uθ = params.ω * params.R₁
        elseif abs(r - params.R₂) <= tol
            uθ = 0.0
        else
            return 0.0
        end
        r_safe = max(r, eps())
        if component === :ux
            return -uθ * y / r_safe
        else
            return  uθ * x / r_safe
        end
    end
end

###########
# Error utilities
###########
inside_annulus(coord, params::CouetteCylinderParams, tol::Float64) = begin
    r = hypot(coord[1], coord[2])
    return (params.R₁ - tol) <= r <= (params.R₂ + tol)
end

function fluid_volume(op_p::AbstractOperators,
                      cap_p::Penguin.Capacity,
                      params::CouetteCylinderParams,
                      tol::Float64)
    coords = cap_p.C_ω
    Vdiag = diag(op_p.V)
    vol = 0.0
    for (i, coord) in enumerate(coords)
        if inside_annulus(coord, params, tol)
            vol += Vdiag[i]
        end
    end
    return vol
end

function velocity_error(u_vec::Vector{Float64},
                        op::AbstractOperators,
                        cap::Penguin.Capacity,
                        params::CouetteCylinderParams,
                        component::Symbol,
                        fluid_vol::Float64,
                        tol::Float64)
    coords = cap.C_ω
    Vdiag = diag(op.V)
    err = 0.0
    for i in 1:length(u_vec)
        vol = Vdiag[i]
        vol <= 0 && continue
        coord = coords[i]
        inside_annulus(coord, params, tol) || continue
        exact = component === :ux ?
            analytic_velocity_components(coord[1], coord[2], params)[1] :
            analytic_velocity_components(coord[1], coord[2], params)[2]
        diff = u_vec[i] - exact
        err += vol * diff^2
    end
    return fluid_vol > 0 ? sqrt(err / fluid_vol) : 0.0
end

###########
# Problem parameters and containers
###########
params = CouetteCylinderParams()
domain_half = 0.8
Lx = Ly = 2domain_half
x0 = y0 = -domain_half
μ = 1.0
ρ = 1.0

resolutions = [8, 16, 32, 64, 128, 256, 512]
errors_ux = Float64[]
errors_uy = Float64[]
hs = Float64[]

xs_best = nothing; ys_best = nothing
xs_uy_best = nothing; ys_uy_best = nothing
Ux_best = nothing; Uy_best = nothing
body_best = nothing

for n in resolutions
    nx = n; ny = n
    mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
    dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
    dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
    mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
    mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

    body = annulus_body(params)

    capacity_ux = Capacity(body, mesh_ux; compute_centroids=true)
    capacity_uy = Capacity(body, mesh_uy; compute_centroids=true)
    capacity_p  = Capacity(body, mesh_p;  compute_centroids=true)

    operator_ux = DiffusionOps(capacity_ux)
    operator_uy = DiffusionOps(capacity_uy)
    operator_p  = DiffusionOps(capacity_p)

    zero_bc = Dirichlet((x, y, _=0.0) -> 0.0)
    bc_ux = BorderConditions(Dict(
        :left=>zero_bc, :right=>zero_bc, :bottom=>zero_bc, :top=>zero_bc
    ))
    bc_uy = BorderConditions(Dict(
        :left=>zero_bc, :right=>zero_bc, :bottom=>zero_bc, :top=>zero_bc
    ))

    pressure_gauge = MeanPressureGauge()

    cut_bc = CouetteCutBC(0.0)
    cut_bc.value_map[objectid(mesh_ux)] = couette_cut_component(:ux, params)
    cut_bc.value_map[objectid(mesh_uy)] = couette_cut_component(:uy, params)

    fᵤ = (x, y, z=0.0) -> 0.0
    fₚ = (x, y, z=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (capacity_ux, capacity_uy),
                  (operator_ux, operator_uy),
                  mesh_p, capacity_p, operator_p,
                  μ, ρ, fᵤ, fₚ)

    nu = prod(operator_ux.size)
    np = prod(operator_p.size)
    x0_vec = zeros(4*nu + np)

    solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, cut_bc; x0=x0_vec)

    @printf("Solving Couette cylinder at resolution %d × %d ...\n", nx, ny)
    solve_StokesMono!(solver; method=Base.:\)

    uωx = solver.x[1:nu]
    uωy = solver.x[2nu+1:3nu]

    tol = 0.5 * min(dx, dy)
    vol_fluid = fluid_volume(operator_p, capacity_p, params, tol)
    err_x = velocity_error(uωx, operator_ux, capacity_ux, params, :ux, vol_fluid, tol)
    err_y = velocity_error(uωy, operator_uy, capacity_uy, params, :uy, vol_fluid, tol)

    push!(errors_ux, err_x)
    push!(errors_uy, err_y)
    push!(hs, max(Lx / nx, Ly / ny))

    @printf("  h=%.5e  ||u_x||=%.5e  ||u_y||=%.5e\n", hs[end], err_x, err_y)

    if n == last(resolutions)
        xs_best = mesh_ux.nodes[1]
        ys_best = mesh_ux.nodes[2]
        xs_uy_best = mesh_uy.nodes[1]
        ys_uy_best = mesh_uy.nodes[2]
        Ux_best = reshape(uωx, (length(xs_best), length(ys_best)))
        Uy_best = reshape(uωy, (length(xs_uy_best), length(ys_uy_best)))
        body_best = [body(x, y) for x in xs_best, y in ys_best]
    end
end

function rate(h, e)
    r = Float64[]
    for i in 2:length(e)
        push!(r, log(e[i] / e[i-1]) / log(h[i] / h[i-1]))
    end
    return r
end

r_ux = rate(hs, errors_ux)
r_uy = rate(hs, errors_uy)

println("\nEstimated convergence rates (between successive resolutions):")
for i in 1:length(r_ux)
    @printf("  between %d and %d: ux rate=%.2f, uy rate=%.2f\n",
            resolutions[i], resolutions[i+1], r_ux[i], r_uy[i])
end

println("\nFinal errors:")
for (i,n) in enumerate(resolutions)
    @printf("  %4d: h=%.3e  ||u_x||=%.5e  ||u_y||=%.5e\n",
            n, hs[i], errors_ux[i], errors_uy[i])
end

df = DataFrame(h=hs, error_ux=errors_ux, error_uy=errors_uy)
CSV.write("couette_cylinder_convergence.csv", df)

fig = Figure(resolution=(900, 500))
ax = Axis(fig[1,1], xscale=log10, yscale=log10,
          xlabel="h", ylabel="volume-weighted L2 error",
          title="Couette cylinder convergence")

lines!(ax, hs, errors_ux; label="u_x", color=:seagreen)
scatter!(ax, hs, errors_ux; color=:seagreen)
lines!(ax, hs, errors_uy; label="u_y", color=:orange)
scatter!(ax, hs, errors_uy; color=:orange)

p_ref = 2.0
h_ref = [minimum(hs), maximum(hs)]
ref_line = errors_ux[1] * (h_ref ./ hs[1]).^p_ref
lines!(ax, h_ref, ref_line; color=:black, linestyle=:dash, label="O(h²)")
axislegend(ax, position=:rb)

save("stokes2d_couettecylinder_convergence.png", fig)
display(fig)

if xs_best !== nothing
    fig_fields = Figure(resolution=(1200, 400))
    ax1 = Axis(fig_fields[1,1], title="u_x (finest grid)", xlabel="x", ylabel="y")
    hm1 = heatmap!(ax1, xs_best, ys_best, Ux_best'; colormap=:viridis)
    contour!(ax1, xs_best, ys_best, body_best'; levels=[0.0], color=:white)
    Colorbar(fig_fields[1,2], hm1)

    ax2 = Axis(fig_fields[1,3], title="u_y (finest grid)", xlabel="x", ylabel="y")
    hm2 = heatmap!(ax2, xs_uy_best, ys_uy_best, Uy_best'; colormap=:plasma)
    contour!(ax2, xs_best, ys_best, body_best'; levels=[0.0], color=:white)
    Colorbar(fig_fields[1,4], hm2)

    save("stokes2d_couettecylinder_fields.png", fig_fields)
    display(fig_fields)
end
