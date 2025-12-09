# Moving - Stokes - Unsteady - Monophasic - Rigid FSI (2D)
"""
    MovingStokesFSI2D(stokes, body_shape, mass, center0, velocity0; external_force=(t,c,v)->SVector(0.0, 0.0))

Explicit fluid-structure coupling for a rigid 2D body driven by hydrodynamic
forces from `MovingStokesUnsteadyMono`. The body is translated (no deformation)
by Newton's law; the fluid is advanced on a space-time cut-cell grid using the
current rigid velocity.

Arguments
- `stokes::MovingStokesUnsteadyMono{2}`: underlying moving Stokes solver (set up
  with the usual boundary conditions and pressure gauge; `bc_cut` is overwritten
  each step to match the body velocity).
- `body_shape::Function`: level-set evaluator taking `(x, y, center::SVector)`
  and returning `phi(x, y)` at the given center position.
- `mass::Real`: rigid-body mass.
- `center0, velocity0`: initial center position and velocity (2 entries each).
- `external_force`: optional `(t, center, velocity) -> SVector(2)` body force.
"""
mutable struct MovingStokesFSI2D
    stokes::MovingStokesUnsteadyMono{2}
    body_shape::Function
    mass::Float64
    center::SVector{2,Float64}
    velocity::SVector{2,Float64}
    external_force::Function
    centers::Vector{SVector{2,Float64}}
    velocities::Vector{SVector{2,Float64}}
    forces::Vector{SVector{2,Float64}}
end

function MovingStokesFSI2D(stokes::MovingStokesUnsteadyMono{2},
                           body_shape::Function,
                           mass::Real,
                           center0::AbstractVector{<:Real},
                           velocity0::AbstractVector{<:Real};
                           external_force::Function=(t, c, v) -> SVector(0.0, 0.0))
    length(center0) == 2 || error("center0 must have length 2")
    length(velocity0) == 2 || error("velocity0 must have length 2")
    mass_val = Float64(mass)
    c0 = SVector{2,Float64}(center0...)
    v0 = SVector{2,Float64}(velocity0...)
    return MovingStokesFSI2D(stokes, body_shape, mass_val, c0, v0, external_force,
                             SVector{2,Float64}[], SVector{2,Float64}[], SVector{2,Float64}[])
end

# Linearly interpolate the rigid-body center between t0 and t1 (used to build the
# space-time body indicator for a single time slab).
function rigid_body_path(body_shape::Function,
                         c0::SVector{2,Float64},
                         c1::SVector{2,Float64},
                         t0::Float64,
                         t1::Float64)
    return (x::Float64, y::Float64, t::Float64) -> begin
        alpha = t1 == t0 ? 1.0 : clamp((t - t0) / (t1 - t0), 0.0, 1.0)
        c = (1 - alpha) * c0 + alpha * c1
        return body_shape(x, y, c)
    end
end

# Compute hydrodynamic force components (volume-integrated divergence of stress)
# for the current moving Stokes solution on a single time slab.
function compute_moving_stokes_force_2D(s::MovingStokesUnsteadyMono{2}, data)
    nu_x = data.nu_x
    nu_y = data.nu_y
    np = data.np
    total_velocity_dofs = 2 * (nu_x + nu_y)
    length(s.x) == total_velocity_dofs + np || error("State length mismatch for force computation.")

    pω = Vector{Float64}(view(s.x, total_velocity_dofs + 1:total_velocity_dofs + np))

    pressure_vec_x = -Vector{Float64}(data.grad_x * pω)
    pressure_vec_y = -Vector{Float64}(data.grad_y * pω)

    # Helper for viscous contribution of one component at t^{n+1}
    function viscous_part(op::DiffusionOps, cap::Capacity, μ, uω, uγ)
        G = op.G[1:end÷2, 1:end÷2]
        H = op.H[1:end÷2, 1:end÷2]
        W = op.Wꜝ[1:end÷2, 1:end÷2]
        Iμ = build_I_D(op, μ, cap)
        Iμ = Iμ[1:end÷2, 1:end÷2]
        grad_vec = G * uω
        if size(H, 2) != 0
            grad_vec .+= H * uγ
        end
        mixed = W * grad_vec
        return Vector{Float64}(Iμ * (G' * mixed))
    end

    uωx = Vector{Float64}(view(s.x, 1:nu_x))
    uγx = Vector{Float64}(view(s.x, nu_x + 1:2nu_x))
    uωy = Vector{Float64}(view(s.x, 2nu_x + 1:2nu_x + nu_y))
    uγy = Vector{Float64}(view(s.x, 2nu_x + nu_y + 1:2nu_x + 2nu_y))

    visc_x = viscous_part(data.op_ux, data.cap_ux, s.fluid.μ, uωx, uγx)
    visc_y = viscous_part(data.op_uy, data.cap_uy, s.fluid.μ, uωy, uγy)

    force_density_x = pressure_vec_x .+ visc_x
    force_density_y = pressure_vec_y .+ visc_y
    integrated_force = SVector(sum(force_density_x), sum(force_density_y))

    return (; pressure=(pressure_vec_x, pressure_vec_y),
            viscous=(visc_x, visc_y),
            force_density=(force_density_x, force_density_y),
            integrated_force=integrated_force)
end

"""
    solve_MovingStokesFSI2D!(fsi, mesh, Δt, Tₛ, Tₑ, bc_b_u; scheme=:BE, kwargs...)

Advance the coupled fluid-rigid system explicitly:
1. Predict body path over the step with current velocity (linear trajectory).
2. Solve the moving Stokes system on that slab.
3. Integrate hydrodynamic force, update rigid velocity by Newton's law, and
   advance the center position along the predicted path.

Returns `(times, fluid_states, centers, velocities, forces)`.
"""
function solve_MovingStokesFSI2D!(fsi::MovingStokesFSI2D,
                                   mesh::AbstractMesh,
                                   Δt::Float64, Tₛ::Float64, Tₑ::Float64,
                                   bc_b_u::Tuple{BorderConditions, BorderConditions};
                                   scheme::Symbol=:BE,
                                   method=Base.:\,
                                   algorithm=nothing,
                                   geometry_method::String="VOFI",
                                   external_force::Function=fsi.external_force,
                                   kwargs...)
    println("Solving the problem:")
    println("- Moving problem (rigid FSI)")
    println("- Monophasic problem")
    println("- Unsteady problem")
    println("- Stokes problem")

    θ = scheme == :CN ? 0.5 : 1.0

    s = fsi.stokes
    s.bc_u = bc_b_u

    # Staggered meshes (match prescribed-motion solver)
    dx = mesh.nodes[1][2] - mesh.nodes[1][1]
    dy = mesh.nodes[2][2] - mesh.nodes[2][1]
    mesh_ux = Penguin.Mesh((length(mesh.nodes[1])-1, length(mesh.nodes[2])-1),
                            (mesh.nodes[1][end] - mesh.nodes[1][1], mesh.nodes[2][end] - mesh.nodes[2][1]),
                            (mesh.nodes[1][1] - 0.5*dx, mesh.nodes[2][1]))
    mesh_uy = Penguin.Mesh((length(mesh.nodes[1])-1, length(mesh.nodes[2])-1),
                            (mesh.nodes[1][end] - mesh.nodes[1][1], mesh.nodes[2][end] - mesh.nodes[2][1]),
                            (mesh.nodes[1][1], mesh.nodes[2][1] - 0.5*dy))

    # Reset histories
    empty!(s.times); empty!(s.states)
    empty!(fsi.centers); empty!(fsi.velocities); empty!(fsi.forces)

    t = Tₛ
    push!(s.times, t)
    push!(s.states, copy(s.x))
    push!(fsi.centers, fsi.center)
    push!(fsi.velocities, fsi.velocity)

    println("Time : $(t)")

    while t < Tₑ - 1e-12 * max(1.0, Tₑ)
        dt_step = min(Δt, Tₑ - t)
        t_next = t + dt_step
        println("Time : $(t_next)")

        c0 = fsi.center
        c1 = c0 + dt_step * fsi.velocity  # explicit trajectory for this slab

        body_fn = rigid_body_path(fsi.body_shape, c0, c1, t, t_next)

        # Cut-cell velocity follows current rigid velocity
        v_body = fsi.velocity
        bc_cut = (Dirichlet((x, y, t_local) -> v_body[1]),
                  Dirichlet((x, y, t_local) -> v_body[2]))
        s.bc_cut = normalize_cut_bc(bc_cut, 2)

        # Space-time meshes and capacities
        STmesh_ux = Penguin.SpaceTimeMesh(mesh_ux, [t, t_next], tag=mesh.tag)
        STmesh_uy = Penguin.SpaceTimeMesh(mesh_uy, [t, t_next], tag=mesh.tag)
        STmesh_p = Penguin.SpaceTimeMesh(mesh, [t, t_next], tag=mesh.tag)

        capacity_ux = Capacity(body_fn, STmesh_ux; method=geometry_method, kwargs...)
        capacity_uy = Capacity(body_fn, STmesh_uy; method=geometry_method, kwargs...)
        capacity_p = Capacity(body_fn, STmesh_p; method=geometry_method, kwargs...)

        operator_ux = DiffusionOps(capacity_ux)
        operator_uy = DiffusionOps(capacity_uy)
        operator_p = DiffusionOps(capacity_p)

        data = stokes2D_moving_blocks(s.fluid,
                                       (operator_ux, operator_uy),
                                       (capacity_ux, capacity_uy),
                                       operator_p, capacity_p,
                                       scheme)

        x_prev = s.x
        assemble_stokes2D_moving!(s, data, dt_step, x_prev, t, t_next, θ, mesh)

        solve_moving_stokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        # Hydrodynamic force on body is the reaction of the fluid force
        force_data = compute_moving_stokes_force_2D(s, data)
        F_fluid = force_data.integrated_force
        F_body = -F_fluid
        F_ext = SVector{2,Float64}(external_force(t_next, c0, v_body))
        a = (F_body + F_ext) / fsi.mass

        v_new = v_body + dt_step * a
        c_new = c1

        push!(s.times, t_next)
        push!(s.states, copy(s.x))
        push!(fsi.centers, c_new)
        push!(fsi.velocities, v_new)
        push!(fsi.forces, F_body)
        println("Solver Extremum : ", maximum(abs.(s.x)))
        println("Rigid center    : ", c_new, " | velocity: ", v_new, " | force: ", F_body)

        fsi.center = c_new
        fsi.velocity = v_new
        t = t_next
    end

    return s.times, s.states, fsi.centers, fsi.velocities, fsi.forces
end
