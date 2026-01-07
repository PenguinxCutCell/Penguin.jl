using Penguin
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using CairoMakie
using DelimitedFiles
"""
Benchmark: steady Stokes Couette flow between two concentric cylinders
embedded with a level-set. The inner cylinder (radius 0.25) rotates with
unit angular velocity while the outer cylinder (radius 0.5) is stationary.

The geometry is represented via the union level-set supplied by
`diffusion_levelset` such that the fluid occupies the annulus where
`body(x,y) < 0`. The problem is solved with `StokesMono` and the
`uγ` degrees of freedom are enforced through a component-aware cut BC.

Outputs:
  * solver residual/size
  * tangential velocity profile sampled along the x-axis (y = 0)
  * L₂/L∞ errors against analytic Couette solution
"""

###########
# Level-set helpers
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

# Fluid (negative) region is the annulus => flip level-set sign
annulus_body(params) = (x, y, _=0.0) -> -diffusion_levelset(params)(x, y)

###########
# Analytic Couette solution utilities
###########
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

###########
# Component-aware cut BC
###########
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

###########
# Geometry and grids
###########
params = CouetteCylinderParams()
nx, ny = 64, 64
domain_half = 0.8
Lx = Ly = 2domain_half
x0 = y0 = -domain_half

mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

body = annulus_body(params)

capacity_ux = Capacity(body, mesh_ux; compute_centroids=true, method="VOFI", integration_method=:vofijul)
capacity_uy = Capacity(body, mesh_uy; compute_centroids=true, method="VOFI", integration_method=:vofijul)
capacity_p  = Capacity(body, mesh_p;  compute_centroids=true, method="VOFI", integration_method=:vofijul)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions on the box
###########
zero_bc = Dirichlet((x, y, _=0.0) -> 0.0)
bc_ux = BorderConditions(Dict(
    :left=>zero_bc, :right=>zero_bc, :bottom=>zero_bc, :top=>zero_bc
))
bc_uy = BorderConditions(Dict(
    :left=>zero_bc, :right=>zero_bc, :bottom=>zero_bc, :top=>zero_bc
))
pressure_gauge = MeanPressureGauge()

function couette_cut_component(component::Symbol, params::CouetteCylinderParams)
    tol = 1e-2
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

cut_bc = CouetteCutBC(0.0)
cut_bc.value_map[objectid(mesh_ux)] = couette_cut_component(:ux, params)
cut_bc.value_map[objectid(mesh_uy)] = couette_cut_component(:uy, params)

###########
# Material and solver
###########
μ = 1.0
ρ = 1.0
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
solve_StokesMono!(solver; method=Base.:\)

println("Couette cylinder benchmark solved. Unknowns = ", length(solver.x))

###########
# Post-processing: select aligned points along x=0 (upper half)
###########
x_nodes_x = mesh_ux.nodes[1]
y_nodes_x = mesh_ux.nodes[2]
LI_ux = LinearIndices((length(x_nodes_x), length(y_nodes_x)))

pω  = solver.x[4nu+1:end]
uωx = solver.x[1:nu]
uωy = solver.x[2nu+1:3nu]

i_axis = argmin(abs.(x_nodes_x .- 0.0))

profile_points = Tuple{Float64,Float64,Float64,Float64,Float64}[]
for (j, yval) in enumerate(y_nodes_x)
    if yval <= params.R₁ || yval >= params.R₂
        continue
    end
    idx = LI_ux[i_axis, j]
    Ux_val = uωx[idx]
    ux_exact = -tangential_velocity(yval, params)
    uθ_num = -Ux_val
    push!(profile_points, (yval, yval, Ux_val, ux_exact, uθ_num))
end

num_vals = [val[3] for val in profile_points]
exact_vals = [val[4] for val in profile_points]
errs = num_vals .- exact_vals
ℓ2_err = sqrt(mean(abs2, errs))
ℓinf_err = maximum(abs, errs)

println(@sprintf("Profile points: %d, L2 error = %.4e, Linf error = %.4e",
                 length(profile_points), ℓ2_err, ℓinf_err))

radial_drift = maximum(abs, uωx)
println(@sprintf("Max |u_x| (proxy for radial drift) = %.4e", radial_drift))

if isempty(profile_points)
    println("No valid profile samples captured; consider refining resolution or domain.")
else
    println("Sample (r, y, numerical u_x, exact u_x) along x=0⁺:")
    nprint = min(length(profile_points), 8)
    for k in 1:nprint
        r, yval, num_u, exact_u, _ = profile_points[k]
        println(@sprintf("  r=%.3f (y=%.3f), u_x_num=%.4f, u_x_exact=%.4f", r, yval, num_u, exact_u))
    end
end

###########
# uθ(r) profile plot
###########
if !isempty(profile_points)
    r_sorted = [pt[1] for pt in profile_points]
    uθ_num_sorted = [pt[5] for pt in profile_points]
    uθ_exact_sorted = [tangential_velocity(r, params) for r in r_sorted]

    fig_profile = Figure(resolution=(640, 360))
    axp = Axis(fig_profile[1, 1], xlabel="r", ylabel="uθ", title="Tangential velocity profile")
    scatter!(axp, r_sorted, uθ_num_sorted; color=:blue, markersize=6, label="numerical points")
    lines!(axp, r_sorted, uθ_exact_sorted; color=:red, linestyle=:dash, label="exact")
    axislegend(axp, position=:rb)
    display(fig_profile)
    save("stokes2d_couettecylinder_profile.png", fig_profile)

    # Save csv of profile data
    open("couette_cylinder_profile_$(nx).csv", "w") do io
        println(io, "r,y,u_x_numerical,u_x_exact,u_theta_numerical")
        for pt in profile_points
            @printf(io, "%.6f,%.6f,%.16e,%.16e,%.16e\n", pt[1], pt[2], pt[3], pt[4], pt[5])
        end
    end
end

###########
# Heatmap visualization
###########
Ux_field = reshape(uωx, (length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2])))
Uy_field = reshape(uωy, (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2])))
speed = sqrt.(Ux_field.^2 .+ Uy_field.^2)

# value for body < 0 regions
speed .= ifelse.(speed .== 0.0, NaN, speed)

fig = Figure(resolution=(500, 500))
ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Couette cylinder: |u| field")
heatmap!(ax, mesh_ux.nodes[1], mesh_ux.nodes[2], speed'; colormap=:viridis)

# Overlay level-set contour to highlight cylinders
body_values = [body(x, y) for x in mesh_ux.nodes[1], y in mesh_ux.nodes[2]]
contour!(ax, mesh_ux.nodes[1], mesh_ux.nodes[2], body_values'; levels=[0.0], color=:white, linewidth=2)

Colorbar(fig[1, 2], label="|u|")
display(fig)
save("stokes2d_couettecylinder_speed.png", fig)

###########
# Pressure heatmap
###########
xp = mesh_p.nodes[1]; yp = mesh_p.nodes[2]
P_field = reshape(pω, (length(xp), length(yp)))
P_field .= ifelse.(P_field .== 0.0, NaN, P_field)
fig_p = Figure(resolution=(600, 420))
axp2 = Axis(fig_p[1, 1], xlabel="x", ylabel="y", title="Couette cylinder: pressure")
hm = heatmap!(axp2, xp, yp, P_field'; colormap=:viridis)
contour!(axp2, mesh_ux.nodes[1], mesh_ux.nodes[2], body_values'; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig_p[1, 2], hm; label="pressure")
display(fig_p)
save("stokes2d_couettecylinder_pressure.png", fig_p)

# Read CSV plot for nx=[64,128,256,512] and plot profiles together
function plot_couette_profiles_from_csv(nx_values)
    fig_all = Figure(resolution=(700, 500))
    ax_all = Axis(fig_all[1, 1], xlabel="r", ylabel="uθ", title="Couette cylinder: tangential velocity profiles")
    for nx in nx_values
        csv_path = "couette_cylinder_profile_$(nx).csv"
        if !isfile(csv_path)
            println("CSV file $(csv_path) not found; skipping.")
            continue
        end
        data = readdlm(csv_path, ',', skipstart=1)
        r_vals = data[:, 1]
        uθ_numerical = data[:, 5]
        scatter!(ax_all, r_vals, uθ_numerical;label="nx=$(nx)")
    end
    # Exact solution curve
    r_fine = range(params.R₁, params.R₂, length=200)
    uθ_exact_fine = [tangential_velocity(r, params) for r in r_fine]
    lines!(ax_all, r_fine, uθ_exact_fine; color=:black, label="exact")

    # Add interface vertical lines
    vlines!(ax_all, [params.R₁, params.R₂]; color=:gray, linestyle=:dash, label="interfaces")
    axislegend(ax_all, position=:rt)
    display(fig_all)
    save("stokes2d_couettecylinder_profiles_comparison.png", fig_all)
end 

plot_couette_profiles_from_csv([8, 16, 32, 64, 128, 256, 512])