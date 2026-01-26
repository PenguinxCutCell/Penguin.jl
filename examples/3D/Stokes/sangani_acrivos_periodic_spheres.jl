using Penguin
using LinearAlgebra
using Printf
using IterativeSolvers
using DelimitedFiles

"""
Sangani & Acrivos (1982): steady Stokes flow past a periodic array of spheres.

We mirror the periodic body-force setup: a fully periodic unit cube with a
sphere at the center, driven by a uniform body force in the x-direction.
We report the volume-averaged streamwise velocity in the fluid region and
compare it against the tabulated values in `sangani.acrivos`.

Note: The table is interpreted as (void_fraction, mean_velocity).
Increase `nx, ny, nz` if you need tighter agreement at high solid fractions.
"""

const SANGANI_REFERENCE = let
    table_path = joinpath(@__DIR__, "sangani.acrivos")
    data = readdlm(table_path, ';')
    data = Float64.(data)
    [(data[i, 1], data[i, 2]) for i in 1:size(data, 1)]
end

const TARGET_VOID_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 0.9]

# Geometry and mesh (periodic cell with centered sphere)
Lx, Ly, Lz = 1.0, 1.0, 1.0
domain_volume = Lx * Ly * Lz
origin = (-0.5, -0.5, -0.5)  # cell spans [-0.5,0.5]^3
sphere_center = (0.0, 0.0, 0.0)

nx, ny, nz = 24, 24, 24  # number of pressure cells in each direction

mesh_p = Penguin.Mesh((nx, ny, nz), (Lx, Ly, Lz), origin)
Δx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
Δy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
Δz = mesh_p.nodes[3][2] - mesh_p.nodes[3][1]
mesh_ux = Penguin.Mesh((nx, ny, nz), (Lx, Ly, Lz), (origin[1] - 0.5 * Δx, origin[2], origin[3]))
mesh_uy = Penguin.Mesh((nx, ny, nz), (Lx, Ly, Lz), (origin[1], origin[2] - 0.5 * Δy, origin[3]))
mesh_uz = Penguin.Mesh((nx, ny, nz), (Lx, Ly, Lz), (origin[1], origin[2], origin[3] - 0.5 * Δz))

# Physics and boundary conditions
mu = 1.0
rho = 1.0
forcing_u = ((x, y, z) -> 1.0, (x, y, z) -> 0.0, (x, y, z) -> 0.0)
forcing_p = (x, y, z) -> 0.0

# Fully periodic velocity field
bc_ux = BorderConditions(Dict(
    :left=>Outflow(), :right=>Outflow(),
    :bottom=>Outflow(), :top=>Outflow(),
    :front=>Outflow(), :back=>Outflow()
))
bc_uy = BorderConditions(Dict(
    :left=>Outflow(), :right=>Outflow(),
    :bottom=>Outflow(), :top=>Outflow(),
    :front=>Outflow(), :back=>Outflow()
))
bc_uz = BorderConditions(Dict(
    :left=>Outflow(), :right=>Outflow(),
    :bottom=>Outflow(), :top=>Outflow(),
    :front=>Outflow(), :back=>Outflow()
))
pressure_gauge = PinPressureGauge()
cut_bc = Dirichlet(0.0)

# Utilities
sphere_levelset(radius; center=(0.0, 0.0, 0.0)) = (x, y, z, _=0.0) -> radius -
    sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2)

function fluid_mean(field::Vector{Float64}, ::Capacity)
    return isempty(field) ? 0.0 : sum(field) / length(field)
end

function solve_sangani_case(void_target::Float64, ref_mean_u::Float64)
    println("Solving Sangani & Acrivos case with target void fraction = $void_target")

    solid_fraction = 1.0 - void_target
    radius = (solid_fraction * domain_volume * 3.0 / (4.0 * pi))^(1.0 / 3.0)
    body = sphere_levelset(radius; center=sphere_center)

    cap_ux = Capacity(body, mesh_ux; compute_centroids=false, method="VOFI", integration_method=:vofijul)
    cap_uy = Capacity(body, mesh_uy; compute_centroids=false, method="VOFI", integration_method=:vofijul)
    cap_uz = Capacity(body, mesh_uz; compute_centroids=false, method="VOFI", integration_method=:vofijul)
    cap_p  = Capacity(body, mesh_p;  compute_centroids=false, method="VOFI", integration_method=:vofijul)

    op_ux = DiffusionOps(cap_ux)
    op_uy = DiffusionOps(cap_uy)
    op_uz = DiffusionOps(cap_uz)
    op_p  = DiffusionOps(cap_p)

    nu_x = prod(op_ux.size)
    nu_y = prod(op_uy.size)
    nu_z = prod(op_uz.size)
    np   = prod(op_p.size)
    x0   = zeros(2 * (nu_x + nu_y + nu_z) + np)

    fluid = Fluid((mesh_ux, mesh_uy, mesh_uz),
                  (cap_ux, cap_uy, cap_uz),
                  (op_ux, op_uy, op_uz),
                  mesh_p, cap_p, op_p,
                  mu, rho, forcing_u, forcing_p)

    solver = StokesMono(fluid, (bc_ux, bc_uy, bc_uz), pressure_gauge, cut_bc; x0=x0)
    solve_StokesMono!(solver; method=IterativeSolvers.gmres)

    uomega_x = solver.x[1:nu_x]
    mean_u = fluid_mean(uomega_x, cap_ux)

    fluid_volume = sum(diag(cap_p.V))
    void_effective = fluid_volume / domain_volume

    rel_err = abs(mean_u - ref_mean_u) / ref_mean_u
    println("  Measured void fraction = $void_effective, Mean U = $mean_u (Ref: $ref_mean_u, RelErr: $rel_err)")

    return (; void_target, void_effective, radius, mean_u, ref_mean_u, rel_err)
end

results = NamedTuple[]

function interpolate_reference(void_target::Float64, reference::Vector{Tuple{Float64, Float64}})
    sorted_ref = sort(reference, by=first)
    voids = first.(sorted_ref)
    means = last.(sorted_ref)

    void_target <= voids[1] && return means[1]
    void_target >= voids[end] && return means[end]

    idx = searchsortedlast(voids, void_target)
    if voids[idx] == void_target
        return means[idx]
    end
    v0, v1 = voids[idx], voids[idx + 1]
    m0, m1 = means[idx], means[idx + 1]
    t = (void_target - v0) / (v1 - v0)
    return m0 + t * (m1 - m0)
end

for void_fraction in TARGET_VOID_FRACTIONS
    ref_mean_u = interpolate_reference(void_fraction, SANGANI_REFERENCE)
    res = solve_sangani_case(void_fraction, ref_mean_u)
    push!(results, res)
end

@printf("%8s %12s %12s %12s %10s\n",
        "Void", "Void_eff", "U_avg", "Ref", "RelErr")
for res in results
    @printf("%8.3f %12.4f %12.4e %12.4e %9.2f%%\n",
            res.void_target, res.void_effective, res.mean_u,
            res.ref_mean_u, 100 * res.rel_err)
end
