using Penguin
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using DelimitedFiles
using IterativeSolvers
using CairoMakie

"""
Refinement study for 2D parallel plates in the Stokes regime.

Tracks:
  - Mass conservation defect (net boundary flux)
  - QoIs: flux Q, wall shear tau_w (Couette), dissipation vs power input
  - Robustness: min cut-cell volume V_min, min aperture A_min, solver iterations/residuals
  - One rotated geometry run (grid-rotation sensitivity)

Notes:
  - The channel is embedded in a larger box with cut-cell walls.
  - Poiseuille is driven by prescribed inflow/outflow profile (dp/dx equivalent).
  - Couette is driven by moving the upper plate via cut-cell interface BCs.

Theory (lubrication scaling, h ≪ L):
  - Poiseuille: Q ∝ (h^3/μ) (Δp/L)
  - Couette: Q ∝ U h, and dissipation ∝ μ U^2 / h
  - Wall shear scales as τ_w ∼ μ U / h
"""

###########
# Geometry helpers
###########
plate_normal(angle_rad) = (-sin(angle_rad), cos(angle_rad))
plate_tangent(n) = (n[2], -n[1])

function plate_body(center, normal, gap)
    return (x, y, _=0.0) -> begin
        dist = (x - center[1]) * normal[1] + (y - center[2]) * normal[2]
        abs(dist) - 0.5 * gap
    end
end

###########
# Cut-cell BC that can vary per component/mesh
###########
import Penguin: build_g_g

struct PlateCutBC <: Penguin.AbstractBoundary
    value_map::Dict{UInt64, Function}
    default::Union{Float64, Function}
end

PlateCutBC(default::Union{Float64, Function} = 0.0) =
    PlateCutBC(Dict{UInt64, Function}(), default)

function evaluate_cut_bc(bc::PlateCutBC, mesh_id::UInt64)
    if haskey(bc.value_map, mesh_id)
        return bc.value_map[mesh_id]
    elseif bc.default isa Function
        return bc.default
    else
        return (args...) -> bc.default
    end
end

function build_g_g(op::Penguin.AbstractOperators, bc::PlateCutBC, cap::Penguin.Capacity)
    coords = Penguin.get_all_coordinates(cap.C_γ)
    mesh_id = objectid(cap.mesh)
    f = evaluate_cut_bc(bc, mesh_id)
    return [f(coord...) for coord in coords]
end

function plate_cut_component(component::Symbol, center, normal, gap, U_wall)
    t = plate_tangent(normal)
    return (x, y, _=0.0) -> begin
        dist = (x - center[1]) * normal[1] + (y - center[2]) * normal[2]
        if dist >= 0.0 && abs(dist) <= 0.5 * gap + 1e-12
            return component === :ux ? U_wall * t[1] : U_wall * t[2]
        else
            return 0.0
        end
    end
end

###########
# Metrics helpers
###########
function min_positive(values)
    pos = values[values .> 0.0]
    return isempty(pos) ? 0.0 : minimum(pos)
end

function capacity_minima(cap_ux, cap_uy, cap_p)
    v_vals = vcat(diag(cap_ux.V), diag(cap_uy.V), diag(cap_p.V))
    a_vals = vcat(diag(cap_ux.A[1]), diag(cap_ux.A[2]), diag(cap_uy.A[1]), diag(cap_uy.A[2]))
    return min_positive(v_vals), min_positive(a_vals)
end

function solver_history_metrics(history)
    if history === nothing
        return (0, NaN, false)
    end
    resnorms = if hasproperty(history, :resnorms)
        history.resnorms
    elseif hasproperty(history, :data) && haskey(history.data, :resnorm)
        history.data[:resnorm]
    else
        Float64[]
    end
    if isempty(resnorms)
        return (0, NaN, false)
    end
    niter = max(length(resnorms) - 1, 0)
    final_res = resnorms[end]
    stagnation = length(resnorms) >= 5 && (resnorms[end] / resnorms[end-4] > 0.9)
    return (niter, final_res, stagnation)
end

function boundary_net_flux(Ux, Uy, A_ux, A_uy, x_nodes_x, y_nodes_x, x_nodes_y, y_nodes_y, body)
    i_left = 1
    i_right = length(x_nodes_x)
    j_bottom = 1
    j_top = length(y_nodes_y)
    flux = 0.0

    for j in 1:length(y_nodes_x)
        x = x_nodes_x[i_right]
        y = y_nodes_x[j]
        if body(x, y) < 0 && A_ux[i_right, j] > 0
            flux += Ux[i_right, j] * A_ux[i_right, j]
        end
        x = x_nodes_x[i_left]
        y = y_nodes_x[j]
        if body(x, y) < 0 && A_ux[i_left, j] > 0
            flux -= Ux[i_left, j] * A_ux[i_left, j]
        end
    end

    for i in 1:length(x_nodes_y)
        x = x_nodes_y[i]
        y = y_nodes_y[j_top]
        if body(x, y) < 0 && A_uy[i, j_top] > 0
            flux += Uy[i, j_top] * A_uy[i, j_top]
        end
        y = y_nodes_y[j_bottom]
        if body(x, y) < 0 && A_uy[i, j_bottom] > 0
            flux -= Uy[i, j_bottom] * A_uy[i, j_bottom]
        end
    end
    return flux
end

function profile_on_midline(Ux, x_nodes_x, y_nodes_x, body, x_mid)
    i_mid = argmin(abs.(x_nodes_x .- x_mid))
    ys = Float64[]
    ux_vals = Float64[]
    for j in 1:length(y_nodes_x)
        x = x_nodes_x[i_mid]
        y = y_nodes_x[j]
        if body(x, y) < 0
            push!(ys, y)
            push!(ux_vals, Ux[i_mid, j])
        end
    end
    return ys, ux_vals
end

function integrate_flux_from_profile(ys, ux_vals)
    if length(ys) < 2
        return 0.0
    end
    dy = diff(ys)
    return sum(0.5 .* (ux_vals[1:end-1] .+ ux_vals[2:end]) .* dy)
end

function wall_shear_from_profile(ys, ux_vals, y_bot, y_top, mu, U_wall)
    if length(ys) < 2
        return (NaN, NaN)
    end
    y1 = ys[1]
    yN = ys[end]
    tau_bottom = mu * ux_vals[1] / (y1 - y_bot)
    tau_top = mu * (U_wall - ux_vals[end]) / (y_top - yN)
    return (tau_bottom, tau_top)
end

function dissipation_from_profile(ys, ux_vals, mu, Lx)
    if length(ys) < 2
        return 0.0
    end
    dudy = diff(ux_vals) ./ diff(ys)
    return Lx * sum(mu .* dudy .^ 2 .* diff(ys))
end

###########
# Case runner
###########
function run_parallel_plate_case(; nx, ny, Lx, Ly, gap, angle_deg, mode, dpdx, U_wall, mu, rho)
    x0 = 0.0
    y0 = 0.0
    angle = deg2rad(angle_deg)
    center = (x0 + 0.5 * Lx, y0 + 0.5 * Ly)
    normal = plate_normal(angle)

    body = plate_body(center, normal, gap)

    mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
    dx = Lx / nx
    dy = Ly / ny
    mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
    mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

    capacity_ux = Capacity(body, mesh_ux; compute_centroids=true, method="VOFI", integration_method=:vofijul)
    capacity_uy = Capacity(body, mesh_uy; compute_centroids=true, method="VOFI", integration_method=:vofijul)
    capacity_p = Capacity(body, mesh_p; compute_centroids=true, method="VOFI", integration_method=:vofijul)

    operator_ux = DiffusionOps(capacity_ux)
    operator_uy = DiffusionOps(capacity_uy)
    operator_p = DiffusionOps(capacity_p)

    zero_bc = Dirichlet((x, y, _=0.0) -> 0.0)
    bc_ux = BorderConditions(Dict(:left => zero_bc, :right => zero_bc, :bottom => zero_bc, :top => zero_bc))
    bc_uy = BorderConditions(Dict(:left => zero_bc, :right => zero_bc, :bottom => zero_bc, :top => zero_bc))

    if mode == :poiseuille
        y_bot = center[2] - 0.5 * gap
        y_top = center[2] + 0.5 * gap
        poiseuille_profile = (x, y, _=0.0) -> begin
            if y < y_bot || y > y_top
                return 0.0
            end
            y_local = y - y_bot
            return (dpdx / (2 * mu)) * y_local * (gap - y_local)
        end
        bc_ux = BorderConditions(Dict(:left => Dirichlet(poiseuille_profile),
                                      :right => Dirichlet(poiseuille_profile),
                                      :bottom => zero_bc,
                                      :top => zero_bc))
        bc_uy = BorderConditions(Dict(:left => zero_bc, :right => zero_bc, :bottom => zero_bc, :top => zero_bc))
        cut_bc = Dirichlet(0.0)
    elseif mode == :couette
        cut_bc = PlateCutBC(0.0)
        cut_bc.value_map[objectid(mesh_ux)] = plate_cut_component(:ux, center, normal, gap, U_wall)
        cut_bc.value_map[objectid(mesh_uy)] = plate_cut_component(:uy, center, normal, gap, U_wall)
    else
        error("Unknown mode: $mode")
    end

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

    solver = StokesMono(fluid, (bc_ux, bc_uy), PinPressureGauge(), cut_bc; x0 = x0_vec)
    solve_StokesMono!(solver; method=Base.:\)

    u_vec_x = solver.x[1:nu]
    u_vec_y = solver.x[2 * nu + 1:3 * nu]

    x_nodes_x = mesh_ux.nodes[1]
    y_nodes_x = mesh_ux.nodes[2]
    x_nodes_y = mesh_uy.nodes[1]
    y_nodes_y = mesh_uy.nodes[2]

    Ux = reshape(u_vec_x, (length(x_nodes_x), length(y_nodes_x)))
    Uy = reshape(u_vec_y, (length(x_nodes_y), length(y_nodes_y)))
    A_ux = reshape(diag(capacity_ux.A[1]), size(Ux))
    A_uy = reshape(diag(capacity_uy.A[2]), size(Uy))

    net_flux = boundary_net_flux(Ux, Uy, A_ux, A_uy, x_nodes_x, y_nodes_x, x_nodes_y, y_nodes_y, body)

    U_ref = mode == :poiseuille ? abs(dpdx) * gap^2 / (8 * mu) : abs(U_wall)
    mass_defect = abs(net_flux) / (U_ref * Lx)

    x_mid = x0 + 0.5 * Lx
    ys, ux_profile = profile_on_midline(Ux, x_nodes_x, y_nodes_x, body, x_mid)
    Q = integrate_flux_from_profile(ys, ux_profile)

    y_bot = center[2] - 0.5 * gap
    y_top = center[2] + 0.5 * gap
    tau_bottom, tau_top = wall_shear_from_profile(ys, ux_profile, y_bot, y_top, mu, U_wall)
    dissipation = dissipation_from_profile(ys, ux_profile, mu, Lx)

    Qin = mode == :poiseuille ? dpdx * Lx * Q : tau_top * U_wall * Lx

    V_min, A_min = capacity_minima(capacity_ux, capacity_uy, capacity_p)
    history = isempty(solver.ch) ? nothing : solver.ch[end]
    niter, res_final, stagnation = solver_history_metrics(history)

    return (; mode, nx, ny, dx, dy, gap, angle_deg,
            Q, mass_defect, tau_bottom, tau_top, dissipation, P_in=Qin,
            V_min, A_min, niter, res_final, stagnation)
end

###########
# Study configuration
###########
Lx = 1.0
Ly = 1.0
mu = 1.0
rho = 1.0

gaps = [1/10, 1/20, 1/40, 1/80]
cells_per_gap = [1, 2, 4, 8]

results = Vector{NamedTuple}()

for mode in (:poiseuille, :couette)
    for gap in gaps
        for cpg in cells_per_gap
            nx = Int(round(Lx * cpg / gap))
            ny = nx
            println(@sprintf("Case=%s gap=%.4f cells_per_gap=%d nx=%d", String(mode), gap, cpg, nx))
            dpdx = mode == :poiseuille ? 1.0 : 0.0
            U_wall = mode == :couette ? 1.0 : 0.0
            #res = run_parallel_plate_case(nx=nx, ny=ny, Lx=Lx, Ly=Ly, gap=gap,
            #                              angle_deg=0.0, mode=mode, dpdx=dpdx,
            #                              U_wall=U_wall, mu=mu, rho=rho)
            #push!(results, res)
        end
    end
end

# Rotated robustness run (Couette only)
rot_gap = 1/40
rot_cells_per_gap = 4
rot_angle_deg = 10.0
rot_nx = Int(round(Lx * rot_cells_per_gap / rot_gap))
println(@sprintf("Rotated Couette gap=%.4f cells_per_gap=%d nx=%d angle=%.1f deg",
                 rot_gap, rot_cells_per_gap, rot_nx, rot_angle_deg))
#rot_res = run_parallel_plate_case(nx=rot_nx, ny=rot_nx, Lx=Lx, Ly=Ly, gap=rot_gap,
#                                  angle_deg=rot_angle_deg, mode=:couette, dpdx=0.0,
#                                  U_wall=1.0, mu=mu, rho=rho)
#push!(results, rot_res)

###########
# Output
###########
header = ["mode", "nx", "ny", "dx", "dy", "gap", "angle_deg",
          "Q", "mass_defect", "tau_bottom", "tau_top", "dissipation", "P_in",
          "V_min", "A_min", "niter", "res_final", "stagnation"]

data = [map(h -> getfield(r, Symbol(h)), header) for r in results]
#open("parallel_plates_refinement.csv", "w") do io
#    println(io, join(header, ","))
#    for row in data
#        println(io, join(row, ","))
#    end
#end

println("Saved results to parallel_plates_refinement.csv")

###########
# Read CSV and plot summary
###########
csv_path = "parallel_plates_refinement.csv"
if isfile(csv_path)
    raw = readdlm(csv_path, ',', Any, '\n')
    if size(raw, 1) > 1
        header = String.(raw[1, :])
        rows = raw[2:end, :]

        col_index(name) = findfirst(==(name), header)
        function col_data(name)
            idx = col_index(name)
            idx === nothing && error("Missing column: $name")
            return rows[:, idx]
        end

        to_float(x) = x isa Number ? Float64(x) : parse(Float64, String(x))
        to_str(x) = x isa AbstractString ? String(x) : string(x)

        mode = map(to_str, col_data("mode"))
        gap = map(to_float, col_data("gap"))
        dy = map(to_float, col_data("dy"))
        angle = map(to_float, col_data("angle_deg"))
        Q = map(to_float, col_data("Q"))
        dissipation = map(to_float, col_data("dissipation"))
        V_min = map(to_float, col_data("V_min"))
        mass_defect = map(to_float, col_data("mass_defect"))

        gap_over_dy = gap ./ dy
        resolved = gap_over_dy .>= 2.0
        angle0 = abs.(angle) .< 1e-12

        function finest_per_gap(mask)
            gap_key = round.(gap, digits=8)
            keys = unique(gap_key[mask])
            idxs = Int[]
            for g in keys
                ids = findall(mask .& (gap_key .== g))
                if !isempty(ids)
                    best = ids[argmin(dy[ids])]
                    push!(idxs, best)
                end
            end
            return idxs
        end

        fig = Figure(resolution = (1200, 400))

        ax_q = Axis(fig[1, 1], xlabel = "gap h", ylabel = "Q", title = "Poiseuille Q vs gap",
                    xscale = log10, yscale = log10)
        mask_p = (mode .== "poiseuille") .& angle0 .& isfinite.(Q) .& (Q .> 0.0)
        mask_p_res = mask_p .& resolved
        mask_p_under = mask_p .& .!resolved
        scatter!(ax_q, gap[mask_p_res], Q[mask_p_res], color = :steelblue, markersize = 8, label = "resolved")
        scatter!(ax_q, gap[mask_p_under], Q[mask_p_under], color = :gray, markersize = 8, label = "under-resolved")
        finest_idx = finest_per_gap(mask_p_res)
        if !isempty(finest_idx)
            order = sortperm(gap[finest_idx])
            h = gap[finest_idx][order]
            q = Q[finest_idx][order]
            lines!(ax_q, h, q, color = :black, label = "finest per gap")
            anchor_h = h[end]
            anchor_q = q[end]
            ref = anchor_q .* (h ./ anchor_h) .^ 3
            lines!(ax_q, h, ref, linestyle = :dash, linewidth = 2, color = :black, label = "slope 3")
        end
        text!(ax_q, 0.97, 0.03, text = "Theory: Q ∝ h^3",
              align = (:right, :bottom), space = :relative)
        text!(ax_q, 0.97, 0.10, text = "gap h = plate spacing; cell size Δy = dy",
              align = (:right, :bottom), space = :relative)
        axislegend(ax_q, position = :lt)

        ax_d = Axis(fig[1, 2], xlabel = "gap h", ylabel = "dissipation", title = "Couette dissipation",
                    xscale = log10, yscale = log10)
        mask_c = (mode .== "couette") .& angle0 .& isfinite.(dissipation) .& (dissipation .> 0.0)
        mask_c_res = mask_c .& resolved
        mask_c_under = mask_c .& .!resolved
        scatter!(ax_d, gap[mask_c_res], dissipation[mask_c_res], color = :darkorange, markersize = 8, label = "resolved")
        scatter!(ax_d, gap[mask_c_under], dissipation[mask_c_under], color = :gray, markersize = 8, label = "under-resolved")
        finest_idx_c = finest_per_gap(mask_c_res)
        if !isempty(finest_idx_c)
            order = sortperm(gap[finest_idx_c])
            h = gap[finest_idx_c][order]
            d = dissipation[finest_idx_c][order]
            lines!(ax_d, h, d, color = :black, label = "finest per gap")
            anchor_h = h[end]
            anchor_d = d[end]
            ref = anchor_d .* (h ./ anchor_h) .^ (-1)
            lines!(ax_d, h, ref, linestyle = :dash, linewidth = 2, color = :black, label = "slope -1")
        end
        text!(ax_d, 0.03, 0.1, text = "Theory: dissipation ∝ h^{-1}",
              align = (:left, :top), space = :relative)
        text!(ax_d, 0.03, 0.18, text = "resolved when h/Δy ≥ 2",
              align = (:left, :top), space = :relative)
        axislegend(ax_d, position = :rt)

        ax_it = Axis(fig[1, 3], xlabel = "gap h", ylabel = "mass defect", title = "Mass defect vs gap",
                     xscale = log10)
        mask_md = isfinite.(mass_defect) .& (mass_defect .>= 0.0)
        scatter!(ax_it, gap[mask_md .& (mode .== "poiseuille") .& angle0],
                 mass_defect[mask_md .& (mode .== "poiseuille") .& angle0],
                 color = :steelblue, markersize = 8, label = "poiseuille")
        scatter!(ax_it, gap[mask_md .& (mode .== "couette") .& angle0],
                 mass_defect[mask_md .& (mode .== "couette") .& angle0],
                 color = :darkorange, markersize = 8, label = "couette")
        scatter!(ax_it, gap[mask_md .& (mode .== "couette") .& .!angle0],
                 mass_defect[mask_md .& (mode .== "couette") .& .!angle0],
                 color = :black, marker = :diamond, markersize = 9, label = "couette rotated")
        text!(ax_it, 0.97,0.03, text = "Mass defect = |Q_in - Q_out| / (U_ref Lx)",
              align = (:right, :bottom), space = :relative)
        axislegend(ax_it, position = :rt)

        display(fig)
        save("parallel_plates_refinement_summary.png", fig)
        println("Saved plot to parallel_plates_refinement_summary.png")
    end
end
