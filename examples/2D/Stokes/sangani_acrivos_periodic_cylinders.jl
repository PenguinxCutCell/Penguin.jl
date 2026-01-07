using Penguin
using LinearAlgebra
using CairoMakie
using Printf
using IterativeSolvers

"""
Sangani & Acrivos (1982): steady Stokes flow past a periodic array of cylinders.

We mirror the Basilisk setup: a fully periodic unit square with a cylinder at
the center, driven by a uniform body force in the streamwise direction. The
drag per unit length is reported as F/(mu*Ubar), where Ubar is the
volume-averaged streamwise velocity in the fluid region. Results are compared
against Table 1 of Sangani & Acrivos (1982). Increase `nx, ny` if you need
tighter agreement at high volume fractions.
"""

const SANGANI_REFERENCE = [
    (0.05, 15.56),
    (0.10, 24.83),
    (0.20, 51.53),
    (0.30, 102.90),
    (0.40, 217.89),
    (0.50, 532.55),
    (0.60, 1.763e3),
    (0.70, 1.352e4),
    (0.75, 1.263e5),
]

# Geometry and mesh (periodic cell with centered cylinder)
Lx = 1.0
Ly = 1.0
domain_area = Lx * Ly
origin = (-0.5, -0.5)  # cell spans [-0.5,0.5]×[-0.5,0.5]
cylinder_center = (0.0, 0.0)
nx, ny = 128, 128  # number of pressure cells in each direction

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), origin)   # pressure (cell-centered)
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (origin[1] - 0.5 * dx, origin[2]))  # x-velocity faces
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (origin[1], origin[2] - 0.5 * dy))  # y-velocity faces

# Physics and boundary conditions
mu = 1.0
rho = 1.0
forcing_u = ((x, y, z=0.0) -> 1.0,  # uniform body force (Basilisk: a = 1)
             (x, y, z=0.0) -> 0.0)
forcing_p = (x, y, z=0.0) -> 0.0

# Fully periodic velocity field
bc_ux = BorderConditions(Dict(
    :left=>Outflow(), :right=>Outflow(),
    :bottom=>Outflow(), :top=>Outflow()
))
bc_uy = BorderConditions(Dict(
    :left=>Outflow(), :right=>Outflow(),
    :bottom=>Outflow(), :top=>Outflow()
))
pressure_gauge = PinPressureGauge()
cut_bc = Dirichlet(0.0)

# Utilities
cylinder_levelset(radius; center=(0.0, 0.0)) = (x, y, _=0.0) -> radius - hypot(x - center[1], y - center[2])

function fluid_mean(field::Vector{Float64}, cap::Capacity)
    weights = diag(cap.V)
    total = sum(weights)
    total == 0 && return 0.0
    return sum(field .* weights) / total
end

function solve_sangani_case(phi_target::Float64, ref_drag::Float64; return_fields::Bool=false)
    println("Solving Sangani & Acrivos case with target volume fraction = $phi_target")
    radius = sqrt(phi_target * domain_area / pi)
    center = cylinder_center
    body = cylinder_levelset(radius; center=center)

    cap_ux = Capacity(body, mesh_ux; compute_centroids=false, method="VOFI", integration_method=:vofijul)
    cap_uy = Capacity(body, mesh_uy; compute_centroids=false, method="VOFI", integration_method=:vofijul)
    cap_p  = Capacity(body, mesh_p;  compute_centroids=false, method="VOFI", integration_method=:vofijul)

    op_ux = DiffusionOps(cap_ux)
    op_uy = DiffusionOps(cap_uy)
    op_p  = DiffusionOps(cap_p)

    nu_x = prod(op_ux.size)
    nu_y = prod(op_uy.size)
    np   = prod(op_p.size)
    x0   = zeros(2 * (nu_x + nu_y) + np)

    fluid = Fluid((mesh_ux, mesh_uy),
                  (cap_ux, cap_uy),
                  (op_ux, op_uy),
                  mesh_p, cap_p, op_p,
                  mu, rho, forcing_u, forcing_p)

    solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, cut_bc; x0=x0)
    solve_StokesMono!(solver; method=IterativeSolvers.gmres)

    uomega_x = solver.x[1:nu_x]
    uomega_y = solver.x[2nu_x + 1:2nu_x + nu_y]
    mean_u = fluid_mean(uomega_x, cap_ux)

    fluid_area = sum(diag(cap_p.V))
    phi_effective = 1 - fluid_area / domain_area

    force_diag = compute_navierstokes_force_diagnostics(solver)
    body_force = navierstokes_reaction_force_components(force_diag; acting_on=:body)
    drag = body_force[1]
    println("  Computed drag force = $drag")
    drag_nd = abs(drag) / (mu * max(abs(mean_u), eps()))
    rel_err = abs(drag_nd - ref_drag) / ref_drag
    println("  Measured phi = $phi_effective, Mean U = $mean_u, Drag/(mu*U) = $drag_nd (Ref: $ref_drag, RelErr: $rel_err)")

    base = (; phi_target, phi_effective, radius, mean_u, drag, drag_nd, ref_drag, rel_err, center)

    if return_fields
        Ux = reshape(uomega_x, op_ux.size)
        Uy = reshape(uomega_y, op_uy.size)
        xs = mesh_ux.nodes[1]
        ys = mesh_ux.nodes[2]
        speed = sqrt.(Ux.^2 .+ Uy.^2)
        return (; base..., Ux, Uy, xs, ys, speed)
    end

    return base
end

results = NamedTuple[]
global plot_data = nothing
for (idx, (phi, ref)) in enumerate(SANGANI_REFERENCE)
    res = solve_sangani_case(phi, ref; return_fields=idx == length(SANGANI_REFERENCE))
    push!(results, res)
    if hasproperty(res, :speed)
        global plot_data = res
    end
end

@printf("%6s %10s %12s %12s %12s %10s\n",
        "Phi", "Phi_eff", "U_avg", "F/(mu*U)", "Ref", "RelErr")
for res in results
    @printf("%6.2f %10.4f %12.4e %12.4f %12.2f %9.2f%%\n",
            res.phi_target, res.phi_effective, res.mean_u,
            res.drag_nd, res.ref_drag, 100 * res.rel_err)
end

if plot_data !== nothing
    fig = Figure(resolution=(900, 450))
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y",
              title=@sprintf("|u|, Φ=%.2f", plot_data.phi_target))
    hm = heatmap!(ax, plot_data.xs, plot_data.ys, plot_data.speed; colormap=:plasma)
    levelset = cylinder_levelset(plot_data.radius; center=plot_data.center)
    contour!(ax, plot_data.xs, plot_data.ys,
             [levelset(x, y) for x in plot_data.xs, y in plot_data.ys];
             levels=[0.0], color=:white, linewidth=2)
    Colorbar(fig[1, 2], hm, label="|u|")
    save("sangani_velocity_magnitude.png", fig)
    display(fig)
    println("Saved figure: sangani_velocity_magnitude.png")
end
