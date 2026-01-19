# Moving Rayleigh-Benard instability with a melting boundary.

struct RayleighBenardMeltingSetup
    mesh_ux::Mesh{2}
    mesh_uy::Mesh{2}
    mesh_p::Mesh{2}
    mesh_T::Mesh{2}
    bc_ux::BorderConditions
    bc_uy::BorderConditions
    bc_T::BorderConditions
    bc_cut_u::AbstractBoundary
    bc_cut_T::AbstractBoundary
    pressure_gauge::AbstractPressureGauge
    ν::Float64
    ρ::Float64
    κ::Union{Float64, Function}
    scalar_source::Function
    strategy::CouplingStrategy
    β::Float64
    gravity::NTuple{2, Float64}
    T_ref::Float64
    fᵤ::Function
    fₚ::Function
    store_states::Bool
    initial_velocity::Union{Nothing, Vector{Float64}}
    initial_temperature::Union{Nothing, Vector{Float64}}
end

function RayleighBenardMeltingSetup(mesh_ux::Mesh{2},
                                    mesh_uy::Mesh{2},
                                    mesh_p::Mesh{2},
                                    mesh_T::Mesh{2},
                                    bc_ux::BorderConditions,
                                    bc_uy::BorderConditions,
                                    bc_T::BorderConditions,
                                    bc_cut_u::AbstractBoundary,
                                    bc_cut_T::AbstractBoundary,
                                    pressure_gauge::AbstractPressureGauge;
                                    ν::Real,
                                    ρ::Real=1.0,
                                    κ::Union{Real, Function}=1.0,
                                    scalar_source::Function=(x, y, z=0.0, t=0.0) -> 0.0,
                                    strategy::CouplingStrategy=PicardCoupling(),
                                    β::Real=1.0,
                                    gravity::NTuple{2, Real}=(0.0, -1.0),
                                    T_ref::Real=0.0,
                                    fᵤ::Function=(x, y, z=0.0, t=0.0) -> 0.0,
                                    fₚ::Function=(x, y, z=0.0, t=0.0) -> 0.0,
                                    store_states::Bool=false,
                                    initial_velocity::Union{Nothing, Vector{Float64}}=nothing,
                                    initial_temperature::Union{Nothing, Vector{Float64}}=nothing)
    g_tuple = (float(gravity[1]), float(gravity[2]))
    return RayleighBenardMeltingSetup(
        mesh_ux, mesh_uy, mesh_p, mesh_T, bc_ux, bc_uy, bc_T, bc_cut_u, bc_cut_T,
        pressure_gauge, float(ν), float(ρ), κ, scalar_source, strategy, float(β),
        g_tuple, float(T_ref), fᵤ, fₚ, store_states, initial_velocity, initial_temperature
    )
end

function _build_coupler_for_body(setup::RayleighBenardMeltingSetup, body::Function;
                                 T0::Union{Nothing, Vector{Float64}}=setup.initial_temperature,
                                 U0::Union{Nothing, Vector{Float64}}=setup.initial_velocity)
    capacity_ux = Capacity(body, setup.mesh_ux; method="VOFI", integration_method=:vofijul)
    capacity_uy = Capacity(body, setup.mesh_uy; method="VOFI", integration_method=:vofijul)
    capacity_p  = Capacity(body, setup.mesh_p; method="VOFI", integration_method=:vofijul)
    capacity_T  = Capacity(body, setup.mesh_T; method="VOFI", integration_method=:vofijul)

    operator_ux = DiffusionOps(capacity_ux)
    operator_uy = DiffusionOps(capacity_uy)
    operator_p  = DiffusionOps(capacity_p)

    fluid = Fluid(
        (setup.mesh_ux, setup.mesh_uy),
        (capacity_ux, capacity_uy),
        (operator_ux, operator_uy),
        setup.mesh_p,
        capacity_p,
        operator_p,
        setup.ν,
        setup.ρ,
        setup.fᵤ,
        setup.fₚ
    )

    nu_x = prod(operator_ux.size)
    nu_y = prod(operator_uy.size)
    np = prod(operator_p.size)
    vel_len = 2 * (nu_x + nu_y) + np
    x0 = if U0 === nothing
        zeros(Float64, vel_len)
    else
        length(U0) == vel_len || error("Initial velocity vector length mismatch (got $(length(U0)), expected $vel_len).")
        copy(U0)
    end

    ns_solver = NavierStokesMono(fluid, (setup.bc_ux, setup.bc_uy), setup.pressure_gauge, setup.bc_cut_u; x0=x0)

    Nx_T = length(setup.mesh_T.nodes[1])
    Ny_T = length(setup.mesh_T.nodes[2])
    N_scalar = Nx_T * Ny_T
    T_init = if T0 === nothing
        zeros(Float64, 2 * N_scalar)
    else
        length(T0) == 2 * N_scalar || error("Initial temperature vector length mismatch (got $(length(T0)), expected $(2 * N_scalar)).")
        copy(T0)
    end

    coupler = NavierStokesScalarCoupler(
        ns_solver,
        capacity_T,
        setup.κ,
        setup.scalar_source,
        setup.bc_T,
        setup.bc_cut_T;
        strategy=setup.strategy,
        β=setup.β,
        gravity=setup.gravity,
        T_ref=setup.T_ref,
        T0=T_init,
        store_states=setup.store_states
    )
    return coupler, capacity_T
end

function _interp_from_positions(interpo::String, mesh::AbstractMesh, xf)
    centroids = range(mesh.nodes[2][1], mesh.nodes[2][end], length=length(mesh.nodes[2]))
    if interpo == "quad"
        return quad_interpol(centroids, xf)
    elseif interpo == "cubic"
        return cubic_interpol(centroids, xf)
    else
        return lin_interpol(centroids, xf)
    end
end

function solve_MovingRayleighBenardMelting2D!(
    setup::RayleighBenardMeltingSetup,
    Interface_position,
    Hₙ⁰,
    Δt::Float64,
    Tₑ::Float64,
    ic::InterfaceConditions,
    mesh::AbstractMesh,
    scheme::String;
    sₙ=nothing,
    interpo::String="linear",
    Newton_params=(200, 1e-8, 1e-8, 1.0),
    cfl_target=0.5,
    Δt_min=1e-4,
    Δt_max=1.0,
    adaptive_timestep::Bool=false,
    method=IterativeSolvers.gmres,
    algorithm=nothing,
    kwargs...
)
    if length(mesh.nodes) != 2
        error("solve_MovingRayleighBenardMelting2D! supports 2D meshes only.")
    end

    s_current = isnothing(sₙ) ? _interp_from_positions(interpo, mesh, Interface_position) : sₙ
    current_xf = Interface_position
    current_Hₙ = Hₙ⁰
    new_xf = current_xf
    new_Hₙ = current_Hₙ

    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Vector{Any}()
    reconstruct = Vector{Any}()
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (0.0, Δt))
    temperature_history = Vector{Vector{Float64}}()
    velocity_history = Vector{Vector{Float64}}()

    prev_T_state = setup.initial_temperature
    prev_U_state = setup.initial_velocity
    ρL = ic.flux.value

    scheme_sym = lowercase(scheme) in ("cn", "crank_nicolson", "cranknicolson") ? :CN : :BE

    t = 0.0
    k = 1
    while t < Tₑ
        err = Inf
        err_rel = Inf
        iter = 0
        Interface_term = zeros(length(current_Hₙ))
        s_next = _interp_from_positions(interpo, mesh, new_xf)

        while (iter < Newton_params[1]) && (err > Newton_params[2]) && (err_rel > Newton_params[3])
            iter += 1
            body = (xx, yy, tt, _=0) -> begin
                t_norm = (tt - t) / Δt
                a = 2.0 * t_norm^2 - 3.0 * t_norm + 1.0
                b = -4.0 * t_norm^2 + 4.0 * t_norm
                c = 2.0 * t_norm^2 - t_norm
                pos_start = s_current(yy)
                pos_mid = 0.5 * (s_current(yy) + s_next(yy))
                pos_end = s_next(yy)
                x_interp = a * pos_start + b * pos_mid + c * pos_end
                return xx - x_interp
            end

            STmesh = SpaceTimeMesh(mesh, [t, t + Δt], tag=mesh.tag)
            capacity = Capacity(body, STmesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase = Phase(capacity, operator, setup.scalar_source, setup.κ)

            body_liquid = (xx, yy, _=0) -> xx - s_next(yy)
            coupler, _ = _build_coupler_for_body(setup, body_liquid; T0=prev_T_state, U0=prev_U_state)
            step!(coupler; Δt=Δt, t_prev=t, scheme=scheme_sym, method=method, algorithm=algorithm, kwargs...)
            Tᵢ = copy(coupler.scalar_state)
            prev_T_state = Tᵢ
            prev_U_state = copy(coupler.velocity_state)

            dims = operator.size
            spatial_shape = spatial_shape_from_dims(dims)
            nx = spatial_shape[1]
            ny = length(spatial_shape) >= 2 ? spatial_shape[2] : 1

            Hₙ, Hₙ₊₁ = extract_height_profiles(phase.capacity, dims)

            W! = operator.Wꜝ[1:end÷2, 1:end÷2]
            G  = operator.G[1:end÷2, 1:end÷2]
            H  = operator.H[1:end÷2, 1:end÷2]
            Id = build_I_D(operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:end÷2, 1:end÷2]
            Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
            Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            Interface_term = reshape(Interface_term, (nx, ny))
            Interface_term = (1 / ρL) * vec(sum(Interface_term, dims=1))

            res = Hₙ₊₁ - Hₙ - Interface_term
            new_Hₙ = current_Hₙ .+ Newton_params[4] .* res
            err = maximum(abs.(new_Hₙ .- current_Hₙ))
            denom = max(eps(), maximum(abs.(current_xf)))
            err_rel = err / denom

            if !haskey(residuals, k)
                residuals[k] = Float64[]
            end
            push!(residuals[k], err)

            new_xf = interface_positions_from_heights(new_Hₙ, mesh)
            ensure_periodic!(new_xf)
            s_next = _interp_from_positions(interpo, mesh, new_xf)

            if (err <= Newton_params[2]) || (err_rel <= Newton_params[3])
                push!(xf_log, new_xf)
                break
            end

            current_Hₙ = new_Hₙ
            current_xf = new_xf
        end

        push!(temperature_history, copy(prev_T_state))
        push!(velocity_history, copy(prev_U_state))
        push!(reconstruct, s_next)

        if adaptive_timestep
            velocity_field = abs.(Interface_term)
            time_left = Tₑ - t
            Δt_max_current = min(Δt_max, time_left)
            Δt, cfl = adapt_timestep(velocity_field, mesh, cfl_target, Δt, Δt_min, Δt_max_current;
                                     growth_factor=1.1, shrink_factor=0.8, safety_factor=0.9)
            push!(timestep_history, (t, Δt))
            println("Adaptive timestep: Δt = $(round(Δt, digits=6)), CFL = $(round(cfl, digits=3))")
        end

        t += Δt
        current_Hₙ = new_Hₙ
        current_xf = new_xf
        s_current = s_next
        k += 1
        push!(timestep_history, (t, Δt))
    end

    return temperature_history, velocity_history, residuals, xf_log, reconstruct, timestep_history
end
