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
unit angular velocity while the outer cylinder is free slip (zero shear).

We model the free-slip condition as du_theta/dr = 0 at r = R2, yielding the
analytic angular velocity:
u_theta(r) = omega * R1^2 * r / (R2^2 + R1^2) * (R2^2 / r^2 + 1)

Cut-cell BCs in StokesMono are Dirichlet-only, so we impose the tangential
speed at r = R2 implied by this free-slip condition.

Outputs:
  * solver residual/size
  * tangential velocity profile sampled along x = 0 (y-axis)
  * L2/Linf errors against analytic free-slip Couette solution
"""

###########
# Level-set helpers
###########
interval_min(a, b) = 0.5 * (a + b - abs(a - b))

struct CouetteCylinderParams
    R1::Float64  # inner radius
    R2::Float64  # outer radius
    omega::Float64  # angular velocity of inner cylinder
end

CouetteCylinderParams(; R1=0.25, R2=0.5, omega=1.0) = CouetteCylinderParams(R1, R2, omega)

function diffusion_levelset(params::CouetteCylinderParams)
    return (x, y, _=0.0) -> begin
        r = hypot(x, y)
        phi_inner = r - params.R1          # negative inside the inner disk
        phi_outer_ext = params.R2 - r      # negative outside the outer cylinder
        interval_min(phi_inner, phi_outer_ext)
    end
end

# Fluid (negative) region is the annulus => flip level-set sign
annulus_body(params) = (x, y, _=0.0) -> -diffusion_levelset(params)(x, y)

###########
# Analytic Couette solution utilities (free slip at r = R2)
###########
function tangential_velocity(r, params::CouetteCylinderParams)
    if r < eps()
        return 0.0
    end
    R1, R2 = params.R1, params.R2
    omega = params.omega
    denom = R2^2 + R1^2
    A = omega * R1^2 / denom
    B = omega * R1^2 * R2^2 / denom
    return A * r + B / r
end

function analytic_velocity_components(x, y, params::CouetteCylinderParams)
    r = hypot(x, y)
    if r < params.R1 || r > params.R2
        return (0.0, 0.0)
    end
    u_theta = tangential_velocity(r, params)
    if r < eps()
        return (0.0, 0.0)
    end
    ux = -u_theta * y / r
    uy =  u_theta * x / r
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
    coords = Penguin.get_all_coordinates(getproperty(cap, Symbol("C_\u03b3")))
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
Lx = Ly = 2 * domain_half
x0 = y0 = -domain_half

mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

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
    u_theta_outer = tangential_velocity(params.R2, params)
    return (x, y, _=0.0) -> begin
        r = hypot(x, y)
        if abs(r - params.R1) <= tol
            u_theta = params.omega * params.R1
        elseif abs(r - params.R2) <= tol
            u_theta = u_theta_outer
        else
            return 0.0
        end
        r_safe = max(r, eps())
        if component === :ux
            return -u_theta * y / r_safe
        else
            return  u_theta * x / r_safe
        end
    end
end

cut_bc = CouetteCutBC(0.0)
cut_bc.value_map[objectid(mesh_ux)] = couette_cut_component(:ux, params)
cut_bc.value_map[objectid(mesh_uy)] = couette_cut_component(:uy, params)

###########
# Material and solver
###########
mu = 1.0
rho = 1.0
f_u = (x, y, z=0.0) -> 0.0
f_p = (x, y, z=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p, capacity_p, operator_p,
              mu, rho, f_u, f_p)

nu = prod(operator_ux.size)
np = prod(operator_p.size)
x0_vec = zeros(4 * nu + np)

solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, cut_bc; x0=x0_vec)
solve_StokesMono!(solver; method=Base.:\)

println("Couette cylinder (free slip) solved. Unknowns = ", length(solver.x))

###########
# Post-processing: select aligned points along x=0 (upper half)
###########
x_nodes_x = mesh_ux.nodes[1]
y_nodes_x = mesh_ux.nodes[2]
LI_ux = LinearIndices((length(x_nodes_x), length(y_nodes_x)))

p_omega  = solver.x[4 * nu + 1:end]
u_omega_x = solver.x[1:nu]
u_omega_y = solver.x[2 * nu + 1:3 * nu]

i_axis = argmin(abs.(x_nodes_x .- 0.0))

profile_points = Tuple{Float64, Float64, Float64, Float64, Float64}[]
for (j, yval) in enumerate(y_nodes_x)
    if yval <= params.R1 || yval >= params.R2
        continue
    end
    idx = LI_ux[i_axis, j]
    ux_val = u_omega_x[idx]
    ux_exact = -tangential_velocity(yval, params)
    u_theta_num = -ux_val
    push!(profile_points, (yval, yval, ux_val, ux_exact, u_theta_num))
end

num_vals = [val[3] for val in profile_points]
exact_vals = [val[4] for val in profile_points]
errs = num_vals .- exact_vals
l2_err = sqrt(mean(abs2, errs))
linf_err = maximum(abs, errs)

println(@sprintf("Profile points: %d, L2 error = %.4e, Linf error = %.4e",
                 length(profile_points), l2_err, linf_err))

radial_drift = maximum(abs, u_omega_x)
println(@sprintf("Max |u_x| (proxy for radial drift) = %.4e", radial_drift))

if isempty(profile_points)
    println("No valid profile samples captured; consider refining resolution or domain.")
else
    println("Sample (r, y, numerical u_x, exact u_x) along x=0+:")
    nprint = min(length(profile_points), 8)
    for k in 1:nprint
        r, yval, num_u, exact_u, _ = profile_points[k]
        println(@sprintf("  r=%.3f (y=%.3f), u_x_num=%.4f, u_x_exact=%.4f", r, yval, num_u, exact_u))
    end
end

###########
# u_theta(r) profile plot
###########
if !isempty(profile_points)
    r_sorted = [pt[1] for pt in profile_points]
    u_theta_num_sorted = [pt[5] for pt in profile_points]
    u_theta_exact_sorted = [tangential_velocity(r, params) for r in r_sorted]

    fig_profile = Figure(resolution=(640, 360))
    axp = Axis(fig_profile[1, 1], xlabel="r", ylabel="u_theta", title="Tangential velocity profile")
    scatter!(axp, r_sorted, u_theta_num_sorted; color=:blue, markersize=6, label="numerical points")
    lines!(axp, r_sorted, u_theta_exact_sorted; color=:red, linestyle=:dash, label="exact")
    axislegend(axp, position=:rb)
    display(fig_profile)
    save("stokes2d_couettecylinder_freeslip_profile.png", fig_profile)

    # Save csv of profile data
    open("couette_cylinder_free_slip_profile_$(nx).csv", "w") do io
        println(io, "r,y,u_x_numerical,u_x_exact,u_theta_numerical")
        for pt in profile_points
            @printf(io, "%.6f,%.6f,%.16e,%.16e,%.16e\n", pt[1], pt[2], pt[3], pt[4], pt[5])
        end
    end
end

###########
# Heatmap visualization
###########
Ux_field = reshape(u_omega_x, (length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2])))
Uy_field = reshape(u_omega_y, (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2])))
speed = sqrt.(Ux_field .^ 2 .+ Uy_field .^ 2)

# value for body < 0 regions
speed .= ifelse.(speed .== 0.0, NaN, speed)

fig = Figure(resolution=(500, 500))
ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Couette cylinder (free slip): |u| field")
heatmap!(ax, mesh_ux.nodes[1], mesh_ux.nodes[2], speed'; colormap=:viridis)

# Overlay level-set contour to highlight cylinders
body_values = [body(x, y) for x in mesh_ux.nodes[1], y in mesh_ux.nodes[2]]
contour!(ax, mesh_ux.nodes[1], mesh_ux.nodes[2], body_values'; levels=[0.0], color=:white, linewidth=2)

Colorbar(fig[1, 2], label="|u|")
display(fig)
save("stokes2d_couettecylinder_freeslip_speed.png", fig)

###########
# Pressure heatmap
###########
xp = mesh_p.nodes[1]
yp = mesh_p.nodes[2]
P_field = reshape(p_omega, (length(xp), length(yp)))
P_field .= ifelse.(P_field .== 0.0, NaN, P_field)
fig_p = Figure(resolution=(600, 420))
axp2 = Axis(fig_p[1, 1], xlabel="x", ylabel="y", title="Couette cylinder (free slip): pressure")
hm = heatmap!(axp2, xp, yp, P_field'; colormap=:viridis)
contour!(axp2, mesh_ux.nodes[1], mesh_ux.nodes[2], body_values'; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig_p[1, 2], hm; label="pressure")
display(fig_p)
save("stokes2d_couettecylinder_freeslip_pressure.png", fig_p)

# Read CSV plot for nx=[64,128,256,512] and plot profiles together
function plot_couette_profiles_from_csv(nx_values; prefix="couette_cylinder_free_slip_profile_")
    fig_all = Figure(resolution=(700, 500))
    ax_all = Axis(fig_all[1, 1], xlabel="r", ylabel="u_theta",
                  title="Couette cylinder (free slip): tangential velocity profiles")
    for nx in nx_values
        csv_path = "$(prefix)$(nx).csv"
        if !isfile(csv_path)
            println("CSV file $(csv_path) not found; skipping.")
            continue
        end
        data = readdlm(csv_path, ',', skipstart=1)
        r_vals = data[:, 1]
        u_theta_numerical = data[:, 5]
        scatter!(ax_all, r_vals, u_theta_numerical; label="nx=$(nx)")
    end
    # Exact solution curve
    r_fine = range(params.R1, params.R2, length=200)
    u_theta_exact_fine = [tangential_velocity(r, params) for r in r_fine]
    lines!(ax_all, r_fine, u_theta_exact_fine; color=:black, label="exact")

    # Add interface vertical lines
    vlines!(ax_all, [params.R1, params.R2]; color=:gray, linestyle=:dash, label="interfaces")
    axislegend(ax_all, position=:rt)
    display(fig_all)
    save("stokes2d_couettecylinder_freeslip_profiles_comparison.png", fig_all)
end

plot_couette_profiles_from_csv([8, 16, 32, 64, 128, 256, 512])
