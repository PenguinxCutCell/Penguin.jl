"""
    NavierStokesProjectionMono

Incremental projection (pressure-correction) Navier–Stokes solver for a
single phase on staggered velocity / cell-centered pressure grids. The
algorithm advances velocities with an implicit viscous term and explicit
convection, predicts an intermediate divergence-full velocity, solves a
pressure Poisson equation, then corrects both velocity and pressure.

Unknown ordering matches `NavierStokesMono`: `[uω₁, uγ₁, ..., uωₙ, uγₙ, pω]`.
"""
mutable struct NavierStokesProjectionMono{N}
    fluid::Fluid{N}
    bc_u::NTuple{N, BorderConditions}
    pressure_gauge::AbstractPressureGauge
    bc_cut::AbstractBoundary
    convection::Union{Nothing,NavierStokesConvection{N}}
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    x::Vector{Float64}
    prev_conv::Union{Nothing,NTuple{N,Vector{Float64}}}
end

function NavierStokesProjectionMono(fluid::Fluid{N},
                                    bc_u::NTuple{N,BorderConditions},
                                    pressure_gauge::AbstractPressureGauge,
                                    bc_cut::AbstractBoundary;
                                    x0=zeros(0)) where {N}
    nu_components = ntuple(i -> prod(fluid.operator_u[i].size), N)
    np = prod(fluid.operator_p.size)
    Nvel = 2 * sum(nu_components)
    Ntot = Nvel + np
    x_init = length(x0) == Ntot ? copy(x0) : zeros(Ntot)

    A = spzeros(Float64, max(1, Nvel), max(1, Nvel))
    b = zeros(Float64, max(1, Nvel))

    convection = build_convection_data(fluid)

    return NavierStokesProjectionMono{N}(fluid, bc_u, pressure_gauge, bc_cut,
                                         convection, A, b, x_init, nothing)
end

NavierStokesProjectionMono(fluid::Fluid{1},
                           bc_u::BorderConditions,
                           pressure_gauge::AbstractPressureGauge,
                           bc_cut::AbstractBoundary;
                           x0=zeros(0)) = NavierStokesProjectionMono(fluid, (bc_u,), pressure_gauge, bc_cut; x0=x0)

function NavierStokesProjectionMono(fluid::Fluid{N},
                                    bc_u_args::Vararg{BorderConditions,N};
                                    pressure_gauge::AbstractPressureGauge=DEFAULT_PRESSURE_GAUGE,
                                    bc_cut::AbstractBoundary,
                                    bc_p::Union{Nothing,BorderConditions}=nothing,
                                    x0=zeros(0)) where {N}
    gauge = bc_p === nothing ? pressure_gauge : normalize_pressure_gauge(bc_p)
    return NavierStokesProjectionMono(fluid, Tuple(bc_u_args), gauge, bc_cut; x0=x0)
end

# Block helpers --------------------------------------------------------------

function navierstokesproj1D_blocks(s::NavierStokesProjectionMono)
    op_u = s.fluid.operator_u[1]
    cap_u = s.fluid.capacity_u[1]
    op_p = s.fluid.operator_p
    cap_p = s.fluid.capacity_p

    nu = prod(op_u.size)
    np = prod(op_p.size)

    μ = s.fluid.μ
    Iμ = build_I_D(op_u, μ, cap_u)

    WG_G = op_u.Wꜝ * op_u.G
    WG_H = op_u.Wꜝ * op_u.H
    visc_u_ω = (Iμ * op_u.G' * WG_G)
    visc_u_γ = (Iμ * op_u.G' * WG_H)

    grad = -(op_p.G + op_p.H)
    @assert size(grad, 1) == nu "Pressure gradient rows must match velocity DOFs for 1D Navier–Stokes"

    Gp = op_p.G
    Hp = op_p.H
    Gp_u = Gp[1:nu, :]
    Hp_u = Hp[1:nu, :]
    div_u_ω = -(Gp_u' + Hp_u')
    div_u_γ =  (Hp_u')

    ρ = s.fluid.ρ
    mass = build_I_D(op_u, ρ, cap_u) * op_u.V

    return (; nu_components=(nu,),
            nu, np,
            op_u, op_p,
            cap_u, cap_p,
            visc_u_ω, visc_u_γ,
            grad,
            div_u_ω, div_u_γ,
            tie = I(nu),
            mass,
            V = op_u.V)
end

function navierstokesproj2D_blocks(s::NavierStokesProjectionMono)
    ops_u = s.fluid.operator_u
    caps_u = s.fluid.capacity_u
    op_p = s.fluid.operator_p
    cap_p = s.fluid.capacity_p

    nu_x = prod(ops_u[1].size)
    nu_y = prod(ops_u[2].size)
    np = prod(op_p.size)

    μ = s.fluid.μ
    Iμ_x = build_I_D(ops_u[1], μ, caps_u[1])
    Iμ_y = build_I_D(ops_u[2], μ, caps_u[2])

    WGx_Gx = ops_u[1].Wꜝ * ops_u[1].G
    WGx_Hx = ops_u[1].Wꜝ * ops_u[1].H
    visc_x_ω = (Iμ_x * ops_u[1].G' * WGx_Gx)
    visc_x_γ = (Iμ_x * ops_u[1].G' * WGx_Hx)

    WGy_Gy = ops_u[2].Wꜝ * ops_u[2].G
    WGy_Hy = ops_u[2].Wꜝ * ops_u[2].H
    visc_y_ω = (Iμ_y * ops_u[2].G' * WGy_Gy)
    visc_y_γ = (Iμ_y * ops_u[2].G' * WGy_Hy)

    grad_full = (op_p.G + op_p.H)
    total_grad_rows = size(grad_full, 1)
    @assert total_grad_rows == nu_x + nu_y "Pressure gradient rows ($(total_grad_rows)) must match velocity DOFs ($(nu_x + nu_y))."

    x_rows = 1:nu_x
    y_rows = nu_x+1:nu_x+nu_y
    grad_x = -grad_full[x_rows, :]
    grad_y = -grad_full[y_rows, :]

    Gp = op_p.G
    Hp = op_p.H
    Gp_x = Gp[x_rows, :]
    Hp_x = Hp[x_rows, :]
    Gp_y = Gp[y_rows, :]
    Hp_y = Hp[y_rows, :]
    div_x_ω = - (Gp_x' + Hp_x')
    div_x_γ =   (Hp_x')
    div_y_ω = - (Gp_y' + Hp_y')
    div_y_γ =   (Hp_y')

    ρ = s.fluid.ρ
    mass_x = build_I_D(ops_u[1], ρ, caps_u[1]) * ops_u[1].V
    mass_y = build_I_D(ops_u[2], ρ, caps_u[2]) * ops_u[2].V

    return (; nu_components=(nu_x, nu_y),
            nu_x, nu_y, np,
            op_ux = ops_u[1], op_uy = ops_u[2], op_p, cap_px = caps_u[1], cap_py = caps_u[2], cap_p,
            visc_x_ω, visc_x_γ, visc_y_ω, visc_y_γ,
            grad_x, grad_y,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ,
            tie_x = I(nu_x), tie_y = I(nu_y),
            mass_x, mass_y,
            Vx = ops_u[1].V, Vy = ops_u[2].V)
end

# Convection helper (reuses stencils from `navierstokes.jl`) ------------------

function compute_convection_vectors!(s::NavierStokesProjectionMono,
                                     data,
                                     advecting_state::AbstractVector{<:Real},
                                     advected_state::AbstractVector{<:Real}=advecting_state)
    convection = s.convection
    convection === nothing && error("Convection data not available for the current Navier–Stokes configuration.")

    nu_components = data.nu_components
    N = length(nu_components)

    uω_adv = Vector{Vector{Float64}}(undef, N)
    uγ_adv = Vector{Vector{Float64}}(undef, N)
    offset = 0
    for i in 1:N
        n = nu_components[i]
        uω_adv[i] = Vector{Float64}(view(advecting_state, offset+1:offset+n))
        offset += n
        uγ_adv[i] = Vector{Float64}(view(advecting_state, offset+1:offset+n))
        offset += n
    end
    uω_adv_tuple = Tuple(uω_adv)
    uγ_adv_tuple = Tuple(uγ_adv)

    same_state = advected_state === advecting_state
    qω_tuple = nothing
    qγ_tuple = nothing
    if same_state
        qω_tuple = uω_adv_tuple
        qγ_tuple = uγ_adv_tuple
    else
        uω_advected = Vector{Vector{Float64}}(undef, N)
        uγ_advected = Vector{Vector{Float64}}(undef, N)
        offset = 0
        for i in 1:N
            n = nu_components[i]
            uω_advected[i] = Vector{Float64}(view(advected_state, offset+1:offset+n))
            offset += n
            uγ_advected[i] = Vector{Float64}(view(advected_state, offset+1:offset+n))
            offset += n
        end
        qω_tuple = Tuple(uω_advected)
        qγ_tuple = Tuple(uγ_advected)
    end

    qω_tuple = qω_tuple::NTuple{N,Vector{Float64}}
    qγ_tuple = qγ_tuple::NTuple{N,Vector{Float64}}

    bulk = ntuple(Val(N)) do i
        idx = Int(i)
        build_convection_matrix(convection.stencils[idx], uω_adv_tuple)
    end

    K_adv = ntuple(Val(N)) do i
        idx = Int(i)
        build_K_matrix(convection.stencils[idx], rotated_interfaces(uγ_adv_tuple, idx))
    end

    K_advected = same_state ? K_adv : ntuple(Val(N)) do i
        idx = Int(i)
        build_K_matrix(convection.stencils[idx], rotated_interfaces(qγ_tuple, idx))
    end

    conv_vectors = ntuple(Val(N)) do i
        idx = Int(i)
        bulk[idx] * qω_tuple[idx] - 0.5 * (K_adv[idx] * qω_tuple[idx] + K_advected[idx] * uω_adv_tuple[idx])
    end

    return conv_vectors
end

# Linear solve wrapper --------------------------------------------------------

function _solve_linear(A::SparseMatrixCSC{Float64,Int},
                       b::Vector{Float64};
                       method=Base.:\, algorithm=nothing, kwargs...)
    Ared, bred, keep_rows, keep_cols = remove_zero_rows_cols!(A, b)

    kwargs_nt = (; kwargs...)
    precond_builder = haskey(kwargs_nt, :precond_builder) ? kwargs_nt.precond_builder : nothing
    if precond_builder !== nothing
        kwargs_nt = Base.structdiff(kwargs_nt, (precond_builder=precond_builder,))
    end

    precond_kwargs = (;)
    if precond_builder !== nothing
        precond_result = try
            precond_builder(Ared)
        catch err
            if err isa MethodError
                precond_builder(Ared, bred)
            else
                rethrow(err)
            end
        end
        precond_kwargs = _preconditioner_kwargs(precond_result)
    end

    solve_kwargs = merge(kwargs_nt, precond_kwargs)

    xred = nothing
    if algorithm !== nothing
        prob = LinearSolve.LinearProblem(Ared, bred)
        sol = LinearSolve.solve(prob, algorithm; solve_kwargs...)
        xred = sol.u
    elseif method === Base.:\
        try
            xred = Ared \ bred
        catch e
            if e isa SingularException
                @warn "Direct solver hit SingularException; falling back to bicgstabl" sizeA=size(Ared)
                xred = IterativeSolvers.bicgstabl(Ared, bred)
            else
                rethrow(e)
            end
        end
    else
        xred = method(Ared, bred; solve_kwargs...)
    end

    xfull = zeros(Float64, size(A, 2))
    xfull[keep_cols] = xred
    return xfull
end

# Boundary projection on velocity vectors ------------------------------------

function _impose_velocity_dirichlet_values!(uωx::Vector{Float64}, uγx::Vector{Float64},
                                            uωy::Vector{Float64}, uγy::Vector{Float64},
                                            bc_ux::BorderConditions, bc_uy::BorderConditions,
                                            mesh_u::NTuple{2,AbstractMesh},
                                            t::Float64)
    mesh_ux, mesh_uy = mesh_u
    nx = length(mesh_ux.nodes[1]); ny = length(mesh_ux.nodes[2])
    nx_y = length(mesh_uy.nodes[1]); ny_y = length(mesh_uy.nodes[2])
    @assert nx == nx_y && ny == ny_y "Velocity component meshes must share grid dimensions"

    LIx = LinearIndices((nx, ny))
    LIy = LinearIndices((nx_y, ny_y))

    iright = max(nx - 1, 1)
    jtop   = max(ny - 1, 1)

    xs_x = mesh_ux.nodes[1]; ys_x = mesh_ux.nodes[2]
    xs_y = mesh_uy.nodes[1]; ys_y = mesh_uy.nodes[2]

    eval_boundary_func(f, x, y, t) = try
        f(x, y, t)
    catch err
        err isa MethodError || rethrow(err)
        try
            f(x, y)
        catch err2
            err2 isa MethodError || rethrow(err2)
            f(x, y, 0.0)
        end
    end
    eval_val(bc, x, y, t) = (bc isa Dirichlet) ? (bc.value isa Function ? eval_boundary_func(bc.value, x, y, t) : bc.value) : nothing

    bcx_bottom = get(bc_ux.borders, :bottom, nothing)
    bcy_bottom = get(bc_uy.borders, :bottom, nothing)
    bcx_top    = get(bc_ux.borders, :top, nothing)
    bcy_top    = get(bc_uy.borders, :top, nothing)
    bcx_left   = get(bc_ux.borders, :left, nothing)
    bcy_left   = get(bc_uy.borders, :left, nothing)
    bcx_right  = get(bc_ux.borders, :right, nothing)
    bcy_right  = get(bc_uy.borders, :right, nothing)

    # Bottom/top
    for (jx, bcx, bcy) in ((1, bcx_bottom, bcy_bottom), (jtop, bcx_top, bcy_top))
        jy = jx
        for i in 1:nx
            if bcx isa Dirichlet
                valx = eval_val(bcx, xs_x[i], ys_x[jx], t)
                if valx !== nothing
                    lix = LIx[i, jx]
                    uωx[lix] = Float64(valx)
                    uγx[lix] = Float64(valx)
                end
            end
            if bcy isa Dirichlet
                valy = eval_val(bcy, xs_y[i], ys_y[jy], t)
                if valy !== nothing
                    liy = LIy[i, jy]
                    uωy[liy] = Float64(valy)
                    uγy[liy] = Float64(valy)
                end
            end
        end
    end

    # Left/right
    for (i, bcx, bcy) in ((1, bcx_left, bcy_left), (iright, bcx_right, bcy_right))
        for j in 1:ny
            if bcx isa Dirichlet
                valx = eval_val(bcx, xs_x[i], ys_x[j], t)
                if valx !== nothing
                    lix = LIx[i, j]
                    uωx[lix] = Float64(valx)
                    uγx[lix] = Float64(valx)
                end
            end
            if bcy isa Dirichlet
                valy = eval_val(bcy, xs_y[i], ys_y[j], t)
                if valy !== nothing
                    liy = LIy[i, j]
                    uωy[liy] = Float64(valy)
                    uγy[liy] = Float64(valy)
                end
            end
        end
    end
end

# Projection step assembly ----------------------------------------------------

function _assemble_predictor_2D(s::NavierStokesProjectionMono, data, Δt::Float64,
                                x_prev::AbstractVector{<:Real}, p_prev::AbstractVector{<:Real},
                                t_prev::Float64, t_next::Float64, θ::Float64,
                                conv_prev::Union{Nothing,NTuple{2,Vector{Float64}}})
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y

    rows = 2 * sum_nu
    cols = 2 * sum_nu
    A = spzeros(Float64, rows, cols)

    mass_x_dt = (1.0 / Δt) * data.mass_x
    mass_y_dt = (1.0 / Δt) * data.mass_y
    θc = 1.0 - θ

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y

    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = mass_x_dt + θ * data.visc_x_ω
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = θ * data.visc_x_γ

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = mass_y_dt + θ * data.visc_y_ω
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = θ * data.visc_y_γ

    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    uωx_prev = view(x_prev, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev, off_uγy+1:off_uγy+nu_y)

    f_prev_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_prev)
    f_next_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_next)
    load_x = data.Vx * (θ .* f_next_x .+ θc .* f_prev_x)

    f_prev_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_prev)
    f_next_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_next)
    load_y = data.Vy * (θ .* f_next_y .+ θc .* f_prev_y)

    conv_curr = compute_convection_vectors!(s, data, x_prev)
    ρ = s.fluid.ρ
    ρ isa Function && error("Spatially varying density is not supported yet for projection solver.")
    ρ_val = float(ρ)

    rhs_mom_x = mass_x_dt * Vector{Float64}(uωx_prev)
    rhs_mom_x .-= θc * (data.visc_x_ω * Vector{Float64}(uωx_prev) + data.visc_x_γ * Vector{Float64}(uγx_prev))
    # Pressure gradient is treated explicitly: add (-∇pⁿ)
    rhs_mom_x .+= data.grad_x * Vector{Float64}(p_prev)
    rhs_mom_x .-= ρ_val .* (conv_prev === nothing ? conv_curr[1] : 1.5 .* conv_curr[1] .- 0.5 .* conv_prev[1])
    rhs_mom_x .+= load_x

    rhs_mom_y = mass_y_dt * Vector{Float64}(uωy_prev)
    rhs_mom_y .-= θc * (data.visc_y_ω * Vector{Float64}(uωy_prev) + data.visc_y_γ * Vector{Float64}(uγy_prev))
    rhs_mom_y .+= data.grad_y * Vector{Float64}(p_prev)
    rhs_mom_y .-= ρ_val .* (conv_prev === nothing ? conv_curr[2] : 1.5 .* conv_curr[2] .- 0.5 .* conv_prev[2])
    rhs_mom_y .+= load_y

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, t_next)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, t_next)

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y)

    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 t=t_next)

    return A, b, conv_curr
end

function _assemble_pressure_poisson(data, Δt::Float64, div_u::Vector{Float64}, ρ_val::Float64)
    nu_x = data.nu_x
    nu_y = data.nu_y
    np = data.np

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y

    uωx = view(div_u, off_uωx+1:off_uωx+nu_x)
    uγx = view(div_u, off_uγx+1:off_uγx+nu_x)
    uωy = view(div_u, off_uωy+1:off_uωy+nu_y)
    uγy = view(div_u, off_uγy+1:off_uγy+nu_y)

    div_star = data.div_x_ω * Vector{Float64}(uωx) + data.div_x_γ * Vector{Float64}(uγx) +
               data.div_y_ω * Vector{Float64}(uωy) + data.div_y_γ * Vector{Float64}(uγy)

    # Poisson operator Lp = ∇·∇ (only ω velocities receive the pressure correction)
    Lp = data.div_x_ω * data.grad_x + data.div_y_ω * data.grad_y

    # Solve Lp * δp = (ρ/Δt) * div(u*)
    A = Lp
    b = (ρ_val / Δt) .* div_star

    return A, b
end

# Time integration ------------------------------------------------------------

function solve_NavierStokesProjection_unsteady!(s::NavierStokesProjectionMono; Δt::Float64, T_end::Float64,
                                                scheme::Symbol=:CN, method=Base.:\,
                                                algorithm=nothing, store_states::Bool=true, kwargs...)
    θ = scheme_to_theta(scheme)
    N = length(s.fluid.operator_u)
    N == 2 || error("Projection solver currently implemented for 2D (got N=$(N)).")

    data = navierstokesproj2D_blocks(s)

    p_offset = 2 * (data.nu_x + data.nu_y)
    np = data.np
    Ntot = p_offset + np

    x_prev = length(s.x) == Ntot ? copy(s.x) : zeros(Float64, Ntot)
    p_prev = view(x_prev, p_offset+1:p_offset+np)

    histories = store_states ? Vector{Vector{Float64}}() : Vector{Vector{Float64}}()
    if store_states
        push!(histories, copy(x_prev))
    end
    times = Float64[0.0]

    conv_prev = s.prev_conv
    if conv_prev !== nothing && length(conv_prev) != length(data.nu_components)
        conv_prev = nothing
    end

    t = 0.0
    println("[NavierStokesProjection] Starting unsteady solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
    while t < T_end - 1e-12 * max(1.0, T_end)
        dt_step = min(Δt, T_end - t)
        t_next = t + dt_step

        A_mom, b_mom, conv_curr = _assemble_predictor_2D(s, data, dt_step, x_prev, p_prev,
                                                         t, t_next, θ, conv_prev)
        u_star = _solve_linear(A_mom, b_mom; method=method, algorithm=algorithm, kwargs...)

        ρ_val = s.fluid.ρ
        ρ_val isa Function && error("Spatially varying density is not supported yet for projection solver.")
        ρ_val = float(ρ_val)

        A_p, b_p = _assemble_pressure_poisson(data, dt_step, u_star, ρ_val)
        apply_pressure_gauge!(A_p, b_p, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                              p_offset=0, np=np, row_start=1)
        δp = _solve_linear(A_p, b_p; method=method, algorithm=algorithm, kwargs...)

        grad_corr_x = (Δt / ρ_val) .* (data.grad_x * δp)
        grad_corr_y = (Δt / ρ_val) .* (data.grad_y * δp)

        # grad_x/grad_y already carry a leading minus sign; we must ADD the correction
        # to realize u^{n+1} = u* - (Δt/ρ)∇δp.
        uωx_corr = Vector{Float64}(view(u_star, 1:data.nu_x)) .+ grad_corr_x
        uγx_corr = Vector{Float64}(view(u_star, data.nu_x+1:2*data.nu_x))
        uωy_corr = Vector{Float64}(view(u_star, 2*data.nu_x+1:2*data.nu_x+data.nu_y)) .+ grad_corr_y
        uγy_corr = Vector{Float64}(view(u_star, 2*data.nu_x+data.nu_y+1:2*(data.nu_x+data.nu_y)))

        _impose_velocity_dirichlet_values!(uωx_corr, uγx_corr, uωy_corr, uγy_corr,
                                           s.bc_u[1], s.bc_u[2], s.fluid.mesh_u, t_next)

        p_new = Vector{Float64}(p_prev) .+ δp

        x_next = zeros(Float64, Ntot)
        x_next[1:data.nu_x] = uωx_corr
        x_next[data.nu_x+1:2*data.nu_x] = uγx_corr
        x_next[2*data.nu_x+1:2*data.nu_x+data.nu_y] = uωy_corr
        x_next[2*data.nu_x+data.nu_y+1:2*(data.nu_x+data.nu_y)] = uγy_corr
        x_next[p_offset+1:end] = p_new

        x_prev = x_next
        p_prev = view(x_prev, p_offset+1:p_offset+np)
        conv_prev = ntuple(Val(length(data.nu_components))) do i
            copy(conv_curr[Int(i)])
        end

        push!(times, t_next)
        if store_states
            push!(histories, copy(x_prev))
        end
        max_state = maximum(abs, x_prev)
        println("[NavierStokesProjection] t=$(round(t_next; digits=6)) max|state|=$(max_state)")

        t = t_next
    end

    s.prev_conv = conv_prev
    s.x = x_prev
    return times, histories
end
