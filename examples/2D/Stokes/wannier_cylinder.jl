using Penguin
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using CairoMakie
using DelimitedFiles

"""
Benchmark: Wannier (journal-bearing) flow between two eccentric cylinders in the
Stokes regime. The inner cylinder (radius R1) is centered at the origin and
rotates with unit tangential speed; the outer cylinder (radius R2) is centered
at (0, ecc) and is stationary. The exact solution from Wannier (1950) is used
for boundary conditions and error estimation.
"""

###########
# Utility helpers
###########
interval_min(a, b) = 0.5 * (a + b - abs(a - b))

###########
# Parameters and geometry
###########
struct WannierParams
    R1::Float64      # inner radius
    R2::Float64      # outer radius
    ecc::Float64     # center offset along y
    v_inner::Float64 # tangential speed on inner cylinder
    v_outer::Float64 # tangential speed on outer cylinder
end

WannierParams(; R1 = 1 / sinh(1.5),
              R2 = 1 / sinh(1.0),
              ecc = 1/tanh(1.0) - 1/tanh(1.5),
              v_inner = 1.0,
              v_outer = 0.0) = WannierParams(R1, R2, ecc, v_inner, v_outer)

inner_center(::WannierParams) = (0.0, 0.0)
outer_center(p::WannierParams) = (0.0, p.ecc)

function wannier_levelset(p::WannierParams)
    return (x, y, _ = 0.0) -> begin
        ci = inner_center(p); co = outer_center(p)
        r_inner = hypot(x - ci[1], y - ci[2])
        r_outer = hypot(x - co[1], y - co[2])
        phi_inner = r_inner - p.R1       # positive outside the inner cylinder
        phi_outer_ext = p.R2 - r_outer   # positive inside the outer cylinder
        interval_min(phi_inner, phi_outer_ext)
    end
end

# Fluid (negative) region is the annulus between the eccentric cylinders
wannier_body(p::WannierParams) = (x, y, _ = 0.0) -> -wannier_levelset(p)(x, y)

function concentric_tangential_velocity(r, p::WannierParams)
    R1, R2, v1, v2 = p.R1, p.R2, p.v_inner, p.v_outer
    if r < R1 || r > R2
        return 0.0
    end
    A = (R2 * v2 - R1 * v1) / (R2^2 - R1^2)
    B = v1 * R1 - A * R1^2
    return A * r + B / r
end

function concentric_velocity(x, y, p::WannierParams)
    r = hypot(x, y)
    uθ = concentric_tangential_velocity(r, p)
    if r < eps()
        return (0.0, 0.0)
    end
    return (-uθ * y / r, uθ * x / r)
end

###########
# Wannier analytical solution (translated from Basilisk C reference)
###########
function psiuv(x, y, r1, r2, e, v1, v2)
    d1 = (r2 * r2 - r1 * r1) / (2 * e) - e / 2
    d2 = d1 + e
    s = sqrt((r2 - r1 - e) * (r2 - r1 + e) * (r2 + r1 + e) * (r2 + r1 - e)) / (2 * e)
    l1 = log((d1 + s) / (d1 - s))
    l2 = log((d2 + s) / (d2 - s))
    den = (r2 * r2 + r1 * r1) * (l1 - l2) - 4 * s * e
    curlb = 2 * (d2 * d2 - d1 * d1) * (r1 * v1 + r2 * v2) / ((r2 * r2 + r1 * r1) * den) +
            r1 * r1 * r2 * r2 * (v1 / r1 - v2 / r2) / (s * (r1 * r1 + r2 * r2) * (d2 - d1))
    A = -0.5 * (d1 * d2 - s * s) * curlb
    B = (d1 + s) * (d2 + s) * curlb
    C = (d1 - s) * (d2 - s) * curlb
    D = (d1 * l2 - d2 * l1) * (r1 * v1 + r2 * v2) / den -
        2 * s * ((r2 * r2 - r1 * r1) / (r2 * r2 + r1 * r1)) * (r1 * v1 + r2 * v2) / den -
        r1 * r1 * r2 * r2 * (v1 / r1 - v2 / r2) / ((r1 * r1 + r2 * r2) * e)
    E = 0.5 * (l1 - l2) * (r1 * v1 + r2 * v2) / den
    F = e * (r1 * v1 + r2 * v2) / den

    y_shift = y + d2
    spy = s + y_shift
    smy = s - y_shift
    zp = x * x + spy * spy
    zm = x * x + smy * smy
    l = log(zp / zm)
    zr = 2 * (spy / zp + smy / zm)

    psi = A * l + B * y_shift * spy / zp + C * y_shift * smy / zm + D * y_shift +
          E * (x * x + y_shift * y_shift + s * s) + F * y_shift * l
    ux = -A * zr - B * ((s + 2 * y_shift) * zp - 2 * spy * spy * y_shift) / (zp * zp) -
         C * ((s - 2 * y_shift) * zm + 2 * smy * smy * y_shift) / (zm * zm) - D -
         E * 2 * y_shift - F * (l + y_shift * zr)
    uy = -A * 8 * s * x * y_shift / (zp * zm) - B * 2 * x * y_shift * spy / (zp * zp) -
         C * 2 * x * y_shift * smy / (zm * zm) + E * 2 * x - F * 8 * s * x * y_shift * y_shift / (zp * zm)

    return ux, uy, psi
end

function wannier_velocity(x, y, p::WannierParams)
    # Use Couette closed-form when concentric (ecc ≈ 0) to avoid division by zero in Wannier formula
    if abs(p.ecc) < 1e-12
        return concentric_velocity(x, y, p)
    end
    ux, uy, _ = psiuv(x, y - p.ecc, p.R1, p.R2, p.ecc, p.v_inner, p.v_outer)
    return (ux, uy)
end

###########
# Component-aware cut boundary condition
###########
import Penguin: build_g_g

struct WannierCutBC <: Penguin.AbstractBoundary
    value_map::Dict{UInt64, Function}
    default::Union{Float64, Function}
end

WannierCutBC(default::Union{Float64, Function} = 0.0) =
    WannierCutBC(Dict{UInt64, Function}(), default)

function evaluate_cut_bc(bc::WannierCutBC, mesh_id::UInt64)
    if haskey(bc.value_map, mesh_id)
        return bc.value_map[mesh_id]
    elseif bc.default isa Function
        return bc.default
    else
        return (args...) -> bc.default
    end
end

function build_g_g(op::Penguin.AbstractOperators, bc::WannierCutBC, cap::Penguin.Capacity)
    coords = Penguin.get_all_coordinates(cap.C_γ)
    mesh_id = objectid(cap.mesh)
    f = evaluate_cut_bc(bc, mesh_id)
    return [f(coord...) for coord in coords]
end

function wannier_cut_component(component::Symbol, p::WannierParams, tol::Float64)
    ci = inner_center(p)
    co = outer_center(p)
    return (x, y, _ = 0.0) -> begin
        r_inner = hypot(x - ci[1], y - ci[2])
        r_outer = hypot(x - co[1], y - co[2])
        if abs(r_inner - p.R1) <= tol
            uθ = p.v_inner
            r_safe = max(r_inner, eps())
            return component === :ux ? -uθ * (y - ci[2]) / r_safe : uθ * (x - ci[1]) / r_safe
        elseif abs(r_outer - p.R2) <= tol
            uθ = p.v_outer
            r_safe = max(r_outer, eps())
            return component === :ux ? -uθ * (y - co[2]) / r_safe : uθ * (x - co[1]) / r_safe
        else
            return 0.0
        end
    end
end

###########
# Geometry and grids
###########
params = WannierParams()

nx, ny = 128, 128
L_half = 1.25
Lx = Ly = 2 * L_half
x0 = y0 = -L_half

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

body = wannier_body(params)

capacity_ux = Capacity(body, mesh_ux; compute_centroids = true, method = "VOFI", integration_method = :vofijul)
capacity_uy = Capacity(body, mesh_uy; compute_centroids = true, method = "VOFI", integration_method = :vofijul)
capacity_p = Capacity(body, mesh_p; compute_centroids = true, method = "VOFI", integration_method = :vofijul)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p = DiffusionOps(capacity_p)

###########
# Boundary conditions on the box
###########
zero_bc = Dirichlet((x, y, _ = 0.0) -> 0.0)
bc_ux = BorderConditions(Dict(:left => zero_bc, :right => zero_bc, :bottom => zero_bc, :top => zero_bc))
bc_uy = BorderConditions(Dict(:left => zero_bc, :right => zero_bc, :bottom => zero_bc, :top => zero_bc))
pressure_gauge = MeanPressureGauge()

cut_bc = WannierCutBC(0.0)
tol_cut = 1.0 * min(dx, dy)  # wide enough to capture the embedded interface cells
cut_bc.value_map[objectid(mesh_ux)] = wannier_cut_component(:ux, params, tol_cut)
cut_bc.value_map[objectid(mesh_uy)] = wannier_cut_component(:uy, params, tol_cut)

###########
# Material and solver
###########
mu = 1.0
rho = 1.0
f_u = (x, y, z = 0.0) -> 0.0
f_p = (x, y, z = 0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p, capacity_p, operator_p,
              mu, rho, f_u, f_p)

nu = prod(operator_ux.size)
np = prod(operator_p.size)
x0_vec = zeros(4 * nu + np)

solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, cut_bc; x0 = x0_vec)
solve_StokesMono!(solver; method = Base.:\)

println("Wannier cylinder benchmark solved. Unknowns = ", length(solver.x))

###########
# Post-processing: velocity fields and norms
###########
x_nodes_x = mesh_ux.nodes[1]
y_nodes_x = mesh_ux.nodes[2]
LI_ux = LinearIndices((length(x_nodes_x), length(y_nodes_x)))

p_vec = solver.x[4 * nu + 1:end]
u_vec_x = solver.x[1:nu]
u_vec_y = solver.x[2 * nu + 1:3 * nu]

global sample_count = 0
global sum_sq_speed = 0.0
global sum_sq_ux = 0.0
global sum_sq_uy = 0.0
global max_err_speed = 0.0
global max_err_ux = 0.0
global max_err_uy = 0.0

for (j, yval) in enumerate(y_nodes_x), (i, xval) in enumerate(x_nodes_x)
    if body(xval, yval) >= 0
        continue
    end
    idx = LI_ux[i, j]
    ux_num = u_vec_x[idx]
    uy_num = u_vec_y[idx]
    ux_exact, uy_exact = wannier_velocity(xval, yval, params)
    dux = ux_num - ux_exact
    duy = uy_num - uy_exact
    ds = sqrt(dux * dux + duy * duy)

    global sample_count += 1
    global sum_sq_speed += ds * ds
    global sum_sq_ux += dux * dux
    global sum_sq_uy += duy * duy
    global max_err_speed = max(max_err_speed, abs(ds))
    global max_err_ux = max(max_err_ux, abs(dux))
    global max_err_uy = max(max_err_uy, abs(duy))
end

if sample_count == 0
    println("No fluid samples found; check geometry or domain size.")
else
    l2_speed = sqrt(sum_sq_speed / sample_count)
    l2_ux = sqrt(sum_sq_ux / sample_count)
    l2_uy = sqrt(sum_sq_uy / sample_count)
    println(@sprintf("Samples=%d, L2(|u| error)=%.4e, Linf(|u| error)=%.4e", sample_count, l2_speed, max_err_speed))
    println(@sprintf("L2(ux)=%.4e, Linf(ux)=%.4e", l2_ux, max_err_ux))
    println(@sprintf("L2(uy)=%.4e, Linf(uy)=%.4e", l2_uy, max_err_uy))
end

###########
# Profile along x = 0 (through cylinder centers)
###########
i_axis = argmin(abs.(x_nodes_x .- 0.0))
profile_points = Vector{NTuple{8, Float64}}()

for (j, yval) in enumerate(y_nodes_x)
    xval = x_nodes_x[i_axis]
    if body(xval, yval) >= 0
        continue
    end
    idx = LI_ux[i_axis, j]
    ux_num = u_vec_x[idx]
    uy_num = u_vec_y[idx]
    ux_exact, uy_exact = wannier_velocity(xval, yval, params)
    push!(profile_points, (xval, yval, ux_num, uy_num, ux_exact, uy_exact,
                           sqrt(ux_num^2 + uy_num^2), sqrt(ux_exact^2 + uy_exact^2)))
end

if isempty(profile_points)
    println("No profile samples along x=0 captured.")
else
    open("wannier_cylinder_profile_$(nx).csv", "w") do io
        println(io, "x,y,ux_num,uy_num,ux_exact,uy_exact,speed_num,speed_exact")
        for pt in profile_points
            @printf(io, "%.6f,%.6f,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n", pt...)
        end
    end
    println(@sprintf("Saved %d profile samples to wannier_cylinder_profile_%d.csv", length(profile_points), nx))
end

###########
# Visualization
###########
Ux_field = reshape(u_vec_x, (length(mesh_ux.nodes[1]), length(mesh_ux.nodes[2])))
Uy_field = reshape(u_vec_y, (length(mesh_uy.nodes[1]), length(mesh_uy.nodes[2])))
speed_num = sqrt.(Ux_field .^ 2 .+ Uy_field .^ 2)

mask = [body(x, y) < 0 ? 1.0 : NaN for x in mesh_ux.nodes[1], y in mesh_ux.nodes[2]]

speed_num_masked = speed_num .* mask
speed_exact = [begin ux_e, uy_e = wannier_velocity(x, y, params); sqrt(ux_e^2 + uy_e^2) end
               for x in mesh_ux.nodes[1], y in mesh_ux.nodes[2]]
speed_exact_masked = speed_exact .* mask
speed_err = (speed_num .- speed_exact) .* mask

body_values = [body(x, y) for x in mesh_ux.nodes[1], y in mesh_ux.nodes[2]]

fig = Figure(resolution = (520, 520))
ax = Axis(fig[1, 1], xlabel = "x", ylabel = "y", title = "Wannier flow: |u| field")
heatmap!(ax, mesh_ux.nodes[1], mesh_ux.nodes[2], speed_num_masked'; colormap = :viridis)
contour!(ax, mesh_ux.nodes[1], mesh_ux.nodes[2], body_values'; levels = [0.0], color = :white, linewidth = 2)
Colorbar(fig[1, 2], label = "|u|")
display(fig)
save("stokes2d_wannier_speed.png", fig)

fig_err = Figure(resolution = (520, 520))
ax_err = Axis(fig_err[1, 1], xlabel = "x", ylabel = "y", title = "Wannier flow: |u| error")
heatmap!(ax_err, mesh_ux.nodes[1], mesh_ux.nodes[2], speed_err'; colormap = :viridis)
contour!(ax_err, mesh_ux.nodes[1], mesh_ux.nodes[2], body_values'; levels = [0.0], color = :black, linewidth = 1.5)
Colorbar(fig_err[1, 2], label = "|u| error")
display(fig_err)
save("stokes2d_wannier_speed_error.png", fig_err)

fig_exact = Figure(resolution = (520, 520))
ax_ex = Axis(fig_exact[1, 1], xlabel = "x", ylabel = "y", title = "Wannier flow: |u| exact")
heatmap!(ax_ex, mesh_ux.nodes[1], mesh_ux.nodes[2], speed_exact_masked'; colormap = :viridis)
contour!(ax_ex, mesh_ux.nodes[1], mesh_ux.nodes[2], body_values'; levels = [0.0], color = :white, linewidth = 2)
Colorbar(fig_exact[1, 2], label = "|u| exact")
display(fig_exact)
save("stokes2d_wannier_speed_exact.png", fig_exact)
