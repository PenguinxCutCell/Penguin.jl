using Penguin
using LinearAlgebra
using Printf
using DelimitedFiles
using IterativeSolvers

"""
Cylinder translating with the surrounding uniform flow (no relative motion).
The body moves at the same speed as the imposed far-field velocity; in the
inviscid limit the net force should be zero. This provides a sanity check
that the cut-cell moving-body setup yields near-zero forces when the fluid
and body are co-moving.
"""

# --- Geometry and motion ---
nx, ny = 32, 32
Lx, Ly = 6.0, 6.0
x0, y0 = -Lx / 2, -Ly / 2

radius = 0.5
center_x0 = 0.0
center_y0 = 0.0
U∞ = 1.0                      # uniform flow speed (x-direction)

center_x = t -> center_x0 + U∞ * t 

body = (x, y, t) -> begin
    cx = center_x(t)
    radius - sqrt((x - cx)^2 + (y - center_y0)^2)
end

# --- Physics ---
μ = 1.0
ρ = 1.0
fᵤ = (x, y, z=0.0) -> 0.0
fₚ = (x, y, z=0.0) -> 0.0

# --- Time integration ---
Δt = 0.001
T_end = 0.1                   # short run; body stays inside the domain
scheme = :BE

geometry_method = "VOFI"
capacity_kwargs = (; method=geometry_method,
                    integration_method=:vofijul,
                    compute_centroids=true)

# --- Meshes ---
mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

# --- Initial capacities/operators (t=0) ---
cap_ux0 = Capacity((x, y, _=0) -> body(x, y, 0.0), mesh_ux)
cap_uy0 = Capacity((x, y, _=0) -> body(x, y, 0.0), mesh_uy)
cap_p0  = Capacity((x, y, _=0) -> body(x, y, 0.0), mesh_p)
op_ux0 = DiffusionOps(cap_ux0)
op_uy0 = DiffusionOps(cap_uy0)
op_p0  = DiffusionOps(cap_p0)

# --- Boundary conditions ---
ux_uni = Dirichlet(U∞)
uy_zero = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(:left => ux_uni, :right => ux_uni,
                              :bottom => ux_uni, :top => ux_uni))
bc_uy = BorderConditions(Dict(:left => uy_zero, :right => uy_zero,
                              :bottom => uy_zero, :top => uy_zero))
pressure_gauge = PinPressureGauge()

# Body surface follows the uniform flow (no slip relative to surrounding flow)
bc_cut = (
    Dirichlet((x, y, t) -> U∞),                        # u_x on body
    Dirichlet((x, y, t) -> 0.0)                        # u_y on body
)

# --- Fluid object and solver ---
fluid = Fluid((mesh_ux, mesh_uy),
              (cap_ux0, cap_uy0),
              (op_ux0, op_uy0),
              mesh_p, cap_p0, op_p0,
              μ, ρ, fᵤ, fₚ)

nu_x = prod(op_ux0.size)
nu_y = prod(op_uy0.size)
np   = prod(op_p0.size)
Ntot = 2 * (nu_x + nu_y) + np
x0_vec = zeros(Ntot)

solver = MovingStokesUnsteadyMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut;
                                  scheme=scheme, x0=x0_vec)

# --- Utility: dump uωx to CSV (filter zeros and compare to 1.0) ---
function write_uomega_x_csv(path, state_vec, nu_x)
    uωx = state_vec[1:nu_x]
    nz_idx = findall(!iszero, uωx)
    vals = uωx[nz_idx]
    diff = vals .- 1.0

    open(path, "w") do io
        println(io, "idx,uomega_x,uomega_x_minus_1")
        for (i, v, d) in zip(nz_idx, vals, diff)
            @printf(io, "%d,%.16e,%.16e\n", i, v, d)
        end
    end
    println("Saved uωx CSV to $(path) with $(length(vals)) nonzero entries.")
    return nothing
end

function l2_error_uomega_x(path)
    # if file has only header (no data rows), skip
    lines = readlines(path)
    if length(lines) ≤ 1
        println("No nonzero uωx entries in $(path); L2 error = 0.0")
        return 0.0
    end
    data = readdlm(path, ',', skipstart=1)
    if isempty(data)
        println("No nonzero uωx entries in $(path); L2 error = 0.0")
        return 0.0
    end
    diffs = data[:, 3]
    l2 = sqrt(sum(abs2, diffs) / length(diffs))
    println("L2 error of uωx-1.0 = $(l2)")
    return l2
end
# ...existing code...
function write_uomega_x_all(prefix, states, times, nu_x)
    errors = Float64[]
    for (k, state) in pairs(states)
        t = times[k]
        fname = @sprintf("%s_step_%03d_t%.4f.csv", prefix, k-1, t)
        write_uomega_x_csv(fname, state, nu_x)
        push!(errors, l2_error_uomega_x(fname))
    end
    if !isempty(errors)
        maxerr, idx = findmax(errors)
        println(@sprintf("Max L2 error over %d states: %.6e (at step %d, t=%.4f)",
                         length(errors), maxerr, idx-1, times[idx]))
    end
    return errors
end

# --- Run simulation ---
println("Running co-moving cylinder in uniform flow (expected near-zero force)")
times, states = solve_MovingStokesUnsteadyMono!(solver, body, mesh_p,
                        Δt, 0.0, T_end, (bc_ux, bc_uy), bc_cut;
                        scheme=scheme,
                        method=IterativeSolvers.gmres,
                        geometry_method=geometry_method,
                        integration_method=capacity_kwargs.integration_method,
                        compute_centroids=capacity_kwargs.compute_centroids)

# --- Velocity dump (final state, bulk x-velocity uωx) ---
if !isempty(states)
    final_state = states[end]
    write_uomega_x_csv("uomega_x.csv", final_state, nu_x)
    l2_error_uomega_x("uomega_x.csv")
    write_uomega_x_all("uomega_x", states, times, nu_x)
end