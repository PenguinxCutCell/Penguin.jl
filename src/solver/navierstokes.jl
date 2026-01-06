struct ConvectiveStencil{N}
    primary_dim::Int
    D_plus::NTuple{N,SparseMatrixCSC{Float64,Int}}
    S_minus::NTuple{N,SparseMatrixCSC{Float64,Int}}
    S_plus_primary::SparseMatrixCSC{Float64,Int}
    A::NTuple{N,SparseMatrixCSC{Float64,Int}}
    Ht::SparseMatrixCSC{Float64,Int}
end

struct NavierStokesConvection{N}
    stencils::NTuple{N,ConvectiveStencil{N}}
end

"""
    NavierStokesMono

Prototype incompressible Navier–Stokes solver that reuses the staggered Stokes
layout but augments the momentum equation with an explicit, skew-symmetric
convection operator. The convection matrices (`Cx`, `Cy`) and their interface
counterparts (`Kx`, `Ky`) follow the discrete flux-form described in the
project notes. Time integration is implicit for viscous/pressure terms while
convection is advanced with an Adams–Bashforth extrapolation.
"""
mutable struct NavierStokesMono{N}
    fluid::Fluid{N}
    bc_u::NTuple{N, BorderConditions}
    pressure_gauge::AbstractPressureGauge
    bc_cut::AbstractBoundary
    convection::Union{Nothing,NavierStokesConvection{N}}  # Convection data when available (N ≥ 1)
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    x::Vector{Float64}
    prev_conv::Union{Nothing,NTuple{N,Vector{Float64}}}
    last_conv_ops::Union{Nothing,NamedTuple}
    ch::Vector{Any}
    residual_history::Vector{Float64}  # Store Picard iteration residuals
end

"""
    NavierStokesMono(fluid, bc_u, bc_p, bc_cut; x0=zeros(0))

Construct a Navier–Stokes solver scaffold. Unknown ordering matches the Stokes
setup: `[uω₁, uγ₁, ..., uωₙ, uγₙ, pω]`.
"""
function NavierStokesMono(fluid::Fluid{N},
                          bc_u::NTuple{N,BorderConditions},
                          pressure_gauge::AbstractPressureGauge,
                          bc_cut::AbstractBoundary;
                          x0=zeros(0)) where {N}
    nu_components = ntuple(i -> prod(fluid.operator_u[i].size), N)
    np = prod(fluid.operator_p.size)
    Ntot = 2 * sum(nu_components) + np
    x_init = length(x0) == Ntot ? copy(x0) : zeros(Ntot)

    A = spzeros(Float64, Ntot, Ntot)
    b = zeros(Ntot)

    convection = build_convection_data(fluid)

    return NavierStokesMono{N}(fluid, bc_u, pressure_gauge, bc_cut,
                               convection, A, b, x_init,
                               nothing, nothing, Any[], Float64[])
end

NavierStokesMono(fluid::Fluid{1},
                 bc_u::BorderConditions,
                 pressure_gauge::AbstractPressureGauge,
                 bc_cut::AbstractBoundary;
                 x0=zeros(0)) = NavierStokesMono(fluid, (bc_u,), pressure_gauge, bc_cut; x0=x0)

function NavierStokesMono(fluid::Fluid{N},
                          bc_u_args::Vararg{BorderConditions,N};
                          pressure_gauge::AbstractPressureGauge=DEFAULT_PRESSURE_GAUGE,
                          bc_cut::AbstractBoundary,
                          bc_p::Union{Nothing,BorderConditions}=nothing,
                          x0=zeros(0)) where {N}
    gauge = bc_p === nothing ? pressure_gauge : normalize_pressure_gauge(bc_p)
    return NavierStokesMono(fluid, Tuple(bc_u_args), gauge, bc_cut; x0=x0)
end

function build_convection_data(fluid::Fluid{N}) where {N}
    stencils = ntuple(Val(N)) do i
        build_convective_stencil(fluid.capacity_u[Int(i)], fluid.operator_u[Int(i)], Int(i))
    end
    return NavierStokesConvection{N}(stencils)
end

function build_convective_stencil(capacity::AbstractCapacity,
                                  op::AbstractOperators,
                                  primary_dim::Int)
    _, _, _, _, _, D_p, S_m, S_p = compute_base_operators(capacity)
    N = length(D_p)
    @assert 1 ≤ primary_dim ≤ N "Primary dimension $(primary_dim) out of bounds for N=$(N)"

    D_plus = ntuple(Val(N)) do i
        D_p[Int(i)]
    end

    S_minus = ntuple(Val(N)) do i
        S_m[Int(i)]
    end

    return ConvectiveStencil{N}(primary_dim,
                                 D_plus,
                                 S_minus,
                                 S_p[primary_dim],
                                 capacity.A,
                                 op.H')
end

# Safe sparse products -------------------------------------------------------

@inline function safe_mul(A::SparseMatrixCSC{Float64,Int}, v::AbstractVector{<:Real})
    size(A, 2) == 0 && return zeros(Float64, size(A, 1))
    @assert size(A, 2) == length(v) "Dimension mismatch: size(A,2)=$(size(A,2)) length(v)=$(length(v))"
    return A * Vector{Float64}(v)
end

@inline function build_convection_matrix(stencil::ConvectiveStencil{N},
                                         u_components::NTuple{N,AbstractVector{<:Real}}) where {N}
    primary = stencil.primary_dim

    flux_primary = stencil.S_minus[primary] * safe_mul(stencil.A[primary], u_components[primary])
    term = stencil.D_plus[primary] * spdiagm(0 => flux_primary) * stencil.S_minus[primary]

    for j in 1:N
        j == primary && continue
        A_cross = stencil.A[j]
        if size(A_cross, 2) == 0 || length(u_components[j]) == 0
            continue
        end
        flux_cross = stencil.S_minus[primary] * safe_mul(A_cross, u_components[j])
        term += stencil.D_plus[j] * spdiagm(0 => flux_cross) * stencil.S_minus[j]
    end

    return term
end

@inline function build_K_matrix(stencil::ConvectiveStencil,
                                uγ::AbstractVector{<:Real})
    size(stencil.Ht, 2) == 0 && return spzeros(Float64, size(stencil.S_plus_primary, 1), size(stencil.S_plus_primary, 1))
    @assert size(stencil.Ht, 2) == length(uγ) "Dimension mismatch for interface velocities"
    weights = stencil.S_plus_primary * (stencil.Ht * Vector{Float64}(uγ))
    return spdiagm(0 => weights)
end

@inline function rotated_interfaces(uγ_tuple::NTuple{N,Vector{Float64}}, idx::Int) where {N}
    total = 0
    for j in 1:N
        total += length(uγ_tuple[j])
    end
    result = Vector{Float64}(undef, total)
    pos = 1
    for shift in 0:N-1
        comp = mod1(idx + shift, N)
        vals = uγ_tuple[comp]
        len = length(vals)
        if len > 0
            copyto!(result, pos, vals, 1, len)
        end
        pos += len
    end
    return result
end

# Block builders -------------------------------------------------------------

function navierstokes1D_blocks(s::NavierStokesMono)
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

function navierstokes2D_blocks(s::NavierStokesMono)
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

function navierstokes3D_blocks(s::NavierStokesMono)
    ops_u = s.fluid.operator_u
    caps_u = s.fluid.capacity_u
    op_p = s.fluid.operator_p
    cap_p = s.fluid.capacity_p

    nu_x = prod(ops_u[1].size)
    nu_y = prod(ops_u[2].size)
    nu_z = prod(ops_u[3].size)
    np = prod(op_p.size)

    μ = s.fluid.μ
    Iμ_x = build_I_D(ops_u[1], μ, caps_u[1])
    Iμ_y = build_I_D(ops_u[2], μ, caps_u[2])
    Iμ_z = build_I_D(ops_u[3], μ, caps_u[3])

    WGx_Gx = ops_u[1].Wꜝ * ops_u[1].G
    WGx_Hx = ops_u[1].Wꜝ * ops_u[1].H
    visc_x_ω = (Iμ_x * ops_u[1].G' * WGx_Gx)
    visc_x_γ = (Iμ_x * ops_u[1].G' * WGx_Hx)

    WGy_Gy = ops_u[2].Wꜝ * ops_u[2].G
    WGy_Hy = ops_u[2].Wꜝ * ops_u[2].H
    visc_y_ω = (Iμ_y * ops_u[2].G' * WGy_Gy)
    visc_y_γ = (Iμ_y * ops_u[2].G' * WGy_Hy)

    WGz_Gz = ops_u[3].Wꜝ * ops_u[3].G
    WGz_Hz = ops_u[3].Wꜝ * ops_u[3].H
    visc_z_ω = (Iμ_z * ops_u[3].G' * WGz_Gz)
    visc_z_γ = (Iμ_z * ops_u[3].G' * WGz_Hz)

    grad_full = (op_p.G + op_p.H)
    total_grad_rows = size(grad_full, 1)
    @assert total_grad_rows == nu_x + nu_y + nu_z "Pressure gradient rows ($(total_grad_rows)) must match velocity DOFs ($(nu_x + nu_y + nu_z))."

    x_rows = 1:nu_x
    y_rows = nu_x+1:nu_x+nu_y
    z_rows = nu_x+nu_y+1:nu_x+nu_y+nu_z

    grad_x = -grad_full[x_rows, :]
    grad_y = -grad_full[y_rows, :]
    grad_z = -grad_full[z_rows, :]

    Gp = op_p.G
    Hp = op_p.H
    Gp_x = Gp[x_rows, :]
    Hp_x = Hp[x_rows, :]
    Gp_y = Gp[y_rows, :]
    Hp_y = Hp[y_rows, :]
    Gp_z = Gp[z_rows, :]
    Hp_z = Hp[z_rows, :]
    div_x_ω = - (Gp_x' + Hp_x')
    div_x_γ =   (Hp_x')
    div_y_ω = - (Gp_y' + Hp_y')
    div_y_γ =   (Hp_y')
    div_z_ω = - (Gp_z' + Hp_z')
    div_z_γ =   (Hp_z')

    ρ = s.fluid.ρ
    mass_x = build_I_D(ops_u[1], ρ, caps_u[1]) * ops_u[1].V
    mass_y = build_I_D(ops_u[2], ρ, caps_u[2]) * ops_u[2].V
    mass_z = build_I_D(ops_u[3], ρ, caps_u[3]) * ops_u[3].V

    return (; nu_components=(nu_x, nu_y, nu_z),
            nu_x, nu_y, nu_z, np,
            op_ux = ops_u[1], op_uy = ops_u[2], op_uz = ops_u[3], op_p,
            cap_px = caps_u[1], cap_py = caps_u[2], cap_pz = caps_u[3], cap_p,
            visc_x_ω, visc_x_γ, visc_y_ω, visc_y_γ, visc_z_ω, visc_z_γ,
            grad_x, grad_y, grad_z,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ, div_z_ω, div_z_γ,
            tie_x = I(nu_x), tie_y = I(nu_y), tie_z = I(nu_z),
            mass_x, mass_y, mass_z,
            Vx = ops_u[1].V, Vy = ops_u[2].V, Vz = ops_u[3].V)
end

# Convection helpers ---------------------------------------------------------

function compute_convection_vectors!(s::NavierStokesMono,
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

    SplusHt = ntuple(Val(N)) do i
        idx = Int(i)
        convection.stencils[idx].S_plus_primary * convection.stencils[idx].Ht
    end

    conv_vectors = ntuple(Val(N)) do i
        idx = Int(i)
        bulk[idx] * qω_tuple[idx] - 0.5 * (K_adv[idx] * qω_tuple[idx] + K_advected[idx] * uω_adv_tuple[idx])
    end

    s.last_conv_ops = (bulk=bulk,
                       K_adv=K_adv,
                       K_advected=K_advected,
                       K_mean=ntuple(Val(N)) do i
                           idx = Int(i)
                           0.5 * (K_adv[idx] + K_advected[idx])
                       end,
                       SplusHt=SplusHt,
                       uω_adv=uω_adv_tuple,
                       uγ_adv=uγ_adv_tuple)

    return conv_vectors
end

# Assembly ------------------------------------------------------------------


function assemble_navierstokes1D_unsteady!(s::NavierStokesMono, data, Δt::Float64,
                                           x_prev::AbstractVector{<:Real},
                                           p_half_prev::AbstractVector{<:Real},
                                           t_prev::Float64, t_next::Float64,
                                           θ::Float64,
                                           conv_prev::Union{Nothing,NTuple{1,Vector{Float64}}})
    nu = data.nu
    np = data.np

    rows = 2 * nu + np
    cols = 2 * nu + np
    A = spzeros(Float64, rows, cols)

    mass_dt = (1.0 / Δt) * data.mass
    θc = 1.0 - θ

    off_uω = 0
    off_uγ = nu
    off_p  = 2 * nu

    row_uω = 0
    row_uγ = nu
    row_con = 2 * nu

    # Momentum block
    A[row_uω+1:row_uω+nu, off_uω+1:off_uω+nu] = mass_dt + θ * data.visc_u_ω
    A[row_uω+1:row_uω+nu, off_uγ+1:off_uγ+nu] = θ * data.visc_u_γ
    A[row_uω+1:row_uω+nu, off_p+1:off_p+np]   = data.grad

    # Tie row
    A[row_uγ+1:row_uγ+nu, off_uγ+1:off_uγ+nu] = data.tie

    # Continuity row
    con_rows = row_con + 1:row_con + np
    A[con_rows, off_uω+1:off_uω+nu] = data.div_u_ω
    A[con_rows, off_uγ+1:off_uγ+nu] = data.div_u_γ

    uω_prev = view(x_prev, off_uω+1:off_uω+nu)
    uγ_prev = view(x_prev, off_uγ+1:off_uγ+nu)

    f_prev = safe_build_source(data.op_u, s.fluid.fᵤ, data.cap_u, t_prev, 1)
    f_next = safe_build_source(data.op_u, s.fluid.fᵤ, data.cap_u, t_next, 1)
    load = data.V * (θ .* f_next .+ θc .* f_prev)

    rhs_mom = mass_dt * Vector{Float64}(uω_prev)
    rhs_mom .-= θc * (data.visc_u_ω * Vector{Float64}(uω_prev) + data.visc_u_γ * Vector{Float64}(uγ_prev))

    rhs_mom .+= load

    conv_curr = compute_convection_vectors!(s, data, x_prev)
    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    if conv_prev === nothing
        rhs_mom .-= ρ_val .* conv_curr[1]
    else
        rhs_mom .-= ρ_val .* (1.5 .* conv_curr[1] .- 0.5 .* conv_prev[1])
    end

    g_cut_next = safe_build_g(data.op_u, s.bc_cut, data.cap_u, t_next)
    b = vcat(rhs_mom, g_cut_next, zeros(np))

    apply_velocity_dirichlet!(A, b, s.bc_u[1], s.fluid.mesh_u[1];
                              nu=nu, uω_offset=off_uω, uγ_offset=off_uγ, t=t_next)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return conv_curr
end

function assemble_navierstokes2D_unsteady!(s::NavierStokesMono, data, Δt::Float64,
                                           x_prev::AbstractVector{<:Real},
                                           p_half_prev::AbstractVector{<:Real},
                                           t_prev::Float64, t_next::Float64,
                                           θ::Float64,
                                           conv_prev::Union{Nothing,NTuple{2,Vector{Float64}}})
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    mass_x_dt = (1.0 / Δt) * data.mass_x
    mass_y_dt = (1.0 / Δt) * data.mass_y
    θc = 1.0 - θ

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_con = 2 * sum_nu

    # Momentum blocks
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = mass_x_dt + θ * data.visc_x_ω
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = θ * data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = mass_y_dt + θ * data.visc_y_ω
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = θ * data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    # Tie rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    # Continuity rows
    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ

    uωx_prev = view(x_prev, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev, off_uγy+1:off_uγy+nu_y)

    f_prev_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_prev, 1)
    f_next_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_next, 1)
    load_x = data.Vx * (θ .* f_next_x .+ θc .* f_prev_x)

    f_prev_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_prev, 2)
    f_next_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_next, 2)
    load_y = data.Vy * (θ .* f_next_y .+ θc .* f_prev_y)

    rhs_mom_x = mass_x_dt * Vector{Float64}(uωx_prev)
    rhs_mom_x .-= θc * (data.visc_x_ω * Vector{Float64}(uωx_prev) + data.visc_x_γ * Vector{Float64}(uγx_prev))

    rhs_mom_y = mass_y_dt * Vector{Float64}(uωy_prev)
    rhs_mom_y .-= θc * (data.visc_y_ω * Vector{Float64}(uωy_prev) + data.visc_y_γ * Vector{Float64}(uγy_prev))

    rhs_mom_x .+= load_x
    rhs_mom_y .+= load_y

    conv_curr = compute_convection_vectors!(s, data, x_prev)
    
    # Get density for convection terms
    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ  # Use constant ρ for now
    
    if conv_prev === nothing
        rhs_mom_x .-= ρ_val .* conv_curr[1]
        rhs_mom_y .-= ρ_val .* conv_curr[2]
    else
        rhs_mom_x .-= ρ_val .* (1.5 .* conv_curr[1] .- 0.5 .* conv_prev[1])
        rhs_mom_y .-= ρ_val .* (1.5 .* conv_curr[2] .- 0.5 .* conv_prev[2])
    end

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, t_next)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, t_next)

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y, zeros(np))

    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 t=t_next)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return conv_curr
end

function assemble_navierstokes3D_unsteady!(s::NavierStokesMono, data, Δt::Float64,
                                           x_prev::AbstractVector{<:Real},
                                           p_half_prev::AbstractVector{<:Real},
                                           t_prev::Float64, t_next::Float64,
                                           θ::Float64,
                                           conv_prev::Union{Nothing,NTuple{3,Vector{Float64}}})
    nu_x = data.nu_x
    nu_y = data.nu_y
    nu_z = data.nu_z
    sum_nu = nu_x + nu_y + nu_z
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    mass_x_dt = (1.0 / Δt) * data.mass_x
    mass_y_dt = (1.0 / Δt) * data.mass_y
    mass_z_dt = (1.0 / Δt) * data.mass_z
    θc = 1.0 - θ

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_uωz = 2 * nu_x + 2 * nu_y
    off_uγz = 2 * nu_x + 2 * nu_y + nu_z
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_uωz = 2 * nu_x + 2 * nu_y
    row_uγz = 2 * nu_x + 2 * nu_y + nu_z
    row_con = 2 * sum_nu

    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = mass_x_dt + θ * data.visc_x_ω
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = θ * data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = mass_y_dt + θ * data.visc_y_ω
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = θ * data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    A[row_uωz+1:row_uωz+nu_z, off_uωz+1:off_uωz+nu_z] = mass_z_dt + θ * data.visc_z_ω
    A[row_uωz+1:row_uωz+nu_z, off_uγz+1:off_uγz+nu_z] = θ * data.visc_z_γ
    A[row_uωz+1:row_uωz+nu_z, off_p+1:off_p+np]       = data.grad_z

    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y
    A[row_uγz+1:row_uγz+nu_z, off_uγz+1:off_uγz+nu_z] = data.tie_z

    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ
    A[con_rows, off_uωz+1:off_uωz+nu_z] = data.div_z_ω
    A[con_rows, off_uγz+1:off_uγz+nu_z] = data.div_z_γ

    uωx_prev = view(x_prev, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev, off_uγy+1:off_uγy+nu_y)
    uωz_prev = view(x_prev, off_uωz+1:off_uωz+nu_z)
    uγz_prev = view(x_prev, off_uγz+1:off_uγz+nu_z)

    f_prev_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_prev, 1)
    f_next_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_next, 1)
    load_x = data.Vx * (θ .* f_next_x .+ θc .* f_prev_x)

    f_prev_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_prev, 2)
    f_next_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_next, 2)
    load_y = data.Vy * (θ .* f_next_y .+ θc .* f_prev_y)

    f_prev_z = safe_build_source(data.op_uz, s.fluid.fᵤ, data.cap_pz, t_prev, 3)
    f_next_z = safe_build_source(data.op_uz, s.fluid.fᵤ, data.cap_pz, t_next, 3)
    load_z = data.Vz * (θ .* f_next_z .+ θc .* f_prev_z)

    rhs_mom_x = mass_x_dt * Vector{Float64}(uωx_prev)
    rhs_mom_x .-= θc * (data.visc_x_ω * Vector{Float64}(uωx_prev) + data.visc_x_γ * Vector{Float64}(uγx_prev))

    rhs_mom_y = mass_y_dt * Vector{Float64}(uωy_prev)
    rhs_mom_y .-= θc * (data.visc_y_ω * Vector{Float64}(uωy_prev) + data.visc_y_γ * Vector{Float64}(uγy_prev))

    rhs_mom_z = mass_z_dt * Vector{Float64}(uωz_prev)
    rhs_mom_z .-= θc * (data.visc_z_ω * Vector{Float64}(uωz_prev) + data.visc_z_γ * Vector{Float64}(uγz_prev))

    rhs_mom_x .+= load_x
    rhs_mom_y .+= load_y
    rhs_mom_z .+= load_z

    conv_curr = compute_convection_vectors!(s, data, x_prev)

    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    if conv_prev === nothing
        rhs_mom_x .-= ρ_val .* conv_curr[1]
        rhs_mom_y .-= ρ_val .* conv_curr[2]
        rhs_mom_z .-= ρ_val .* conv_curr[3]
    else
        rhs_mom_x .-= ρ_val .* (1.5 .* conv_curr[1] .- 0.5 .* conv_prev[1])
        rhs_mom_y .-= ρ_val .* (1.5 .* conv_curr[2] .- 0.5 .* conv_prev[2])
        rhs_mom_z .-= ρ_val .* (1.5 .* conv_curr[3] .- 0.5 .* conv_prev[3])
    end

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, t_next)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, t_next)
    g_cut_z = safe_build_g(data.op_uz, s.bc_cut, data.cap_pz, t_next)

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y, rhs_mom_z, g_cut_z, zeros(np))

    apply_velocity_dirichlet_3D!(A, b, s.bc_u[1], s.bc_u[2], s.bc_u[3], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y, nu_z=nu_z,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 uωz_off=off_uωz, uγz_off=off_uγz,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 row_uωz_off=row_uωz, row_uγz_off=row_uγz,
                                 t=t_next)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return conv_curr
end

function assemble_navierstokes1D_unsteady_picard!(s::NavierStokesMono, data, Δt::Float64,
                                                  x_prev::AbstractVector{<:Real},
                                                  x_iter::AbstractVector{<:Real},
                                                  p_half_prev::AbstractVector{<:Real},
                                                  t_prev::Float64, t_next::Float64,
                                                  θ::Float64,
                                                  conv_prev::Union{Nothing,NTuple{1,Vector{Float64}}})
    nu = data.nu
    np = data.np

    rows = 2 * nu + np
    cols = 2 * nu + np
    A = spzeros(Float64, rows, cols)

    mass_dt = (1.0 / Δt) * data.mass
    θc = 1.0 - θ

    off_uω = 0
    off_uγ = nu
    off_p  = 2 * nu

    row_uω = 0
    row_uγ = nu
    row_con = 2 * nu

    compute_convection_vectors!(s, data, x_iter)
    ops = s.last_conv_ops
    @assert ops !== nothing
    bulk = ops.bulk
    K_adv = ops.K_adv

    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    conv_prev_vec = conv_prev === nothing ? zeros(Float64, nu) : conv_prev[1]

    # Momentum block with Picard linearized convection
    A[row_uω+1:row_uω+nu, off_uω+1:off_uω+nu] = mass_dt + θ * (data.visc_u_ω + ρ_val * bulk[1] - 0.5 * ρ_val * K_adv[1])
    A[row_uω+1:row_uω+nu, off_uγ+1:off_uγ+nu] = θ * data.visc_u_γ
    A[row_uω+1:row_uω+nu, off_p+1:off_p+np]   = data.grad

    # Tie row
    A[row_uγ+1:row_uγ+nu, off_uγ+1:off_uγ+nu] = data.tie

    # Continuity row
    con_rows = row_con + 1:row_con + np
    A[con_rows, off_uω+1:off_uω+nu] = data.div_u_ω
    A[con_rows, off_uγ+1:off_uγ+nu] = data.div_u_γ

    uω_prev = view(x_prev, off_uω+1:off_uω+nu)
    uγ_prev = view(x_prev, off_uγ+1:off_uγ+nu)

    f_prev = safe_build_source(data.op_u, s.fluid.fᵤ, data.cap_u, t_prev, 1)
    f_next = safe_build_source(data.op_u, s.fluid.fᵤ, data.cap_u, t_next, 1)
    load = data.V * (θ .* f_next .+ θc .* f_prev)

    rhs_mom = mass_dt * Vector{Float64}(uω_prev)
    rhs_mom .-= θc * (data.visc_u_ω * Vector{Float64}(uω_prev) + data.visc_u_γ * Vector{Float64}(uγ_prev))

    rhs_mom .+= load
    rhs_mom .-= (1.0 - θ) * ρ_val .* conv_prev_vec

    g_cut_next = safe_build_g(data.op_u, s.bc_cut, data.cap_u, t_next)
    b = vcat(rhs_mom, g_cut_next, zeros(np))

    apply_velocity_dirichlet!(A, b, s.bc_u[1], s.fluid.mesh_u[1];
                              nu=nu, uω_offset=off_uω, uγ_offset=off_uγ, t=t_next)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end

function assemble_navierstokes2D_unsteady_picard!(s::NavierStokesMono, data, Δt::Float64,
                                                  x_prev::AbstractVector{<:Real},
                                                  x_iter::AbstractVector{<:Real},
                                                  p_half_prev::AbstractVector{<:Real},
                                                  t_prev::Float64, t_next::Float64,
                                                  θ::Float64,
                                                  conv_prev::Union{Nothing,NTuple{2,Vector{Float64}}})
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    mass_x_dt = (1.0 / Δt) * data.mass_x
    mass_y_dt = (1.0 / Δt) * data.mass_y
    θc = 1.0 - θ

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_con = 2 * sum_nu

    compute_convection_vectors!(s, data, x_iter)
    ops = s.last_conv_ops
    @assert ops !== nothing
    bulk = ops.bulk
    K_adv = ops.K_adv

    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    conv_prev_x = conv_prev === nothing ? zeros(Float64, nu_x) : conv_prev[1]
    conv_prev_y = conv_prev === nothing ? zeros(Float64, nu_y) : conv_prev[2]

    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = mass_x_dt + θ * (data.visc_x_ω + ρ_val * bulk[1] - 0.5 * ρ_val * K_adv[1])
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = θ * data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = mass_y_dt + θ * (data.visc_y_ω + ρ_val * bulk[2] - 0.5 * ρ_val * K_adv[2])
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = θ * data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ

    uωx_prev = view(x_prev, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev, off_uγy+1:off_uγy+nu_y)

    f_prev_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_prev, 1)
    f_next_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_next, 1)
    load_x = data.Vx * (θ .* f_next_x .+ θc .* f_prev_x)

    f_prev_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_prev, 2)
    f_next_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_next, 2)
    load_y = data.Vy * (θ .* f_next_y .+ θc .* f_prev_y)

    rhs_mom_x = mass_x_dt * Vector{Float64}(uωx_prev)
    rhs_mom_x .-= θc * (data.visc_x_ω * Vector{Float64}(uωx_prev) + data.visc_x_γ * Vector{Float64}(uγx_prev))

    rhs_mom_y = mass_y_dt * Vector{Float64}(uωy_prev)
    rhs_mom_y .-= θc * (data.visc_y_ω * Vector{Float64}(uωy_prev) + data.visc_y_γ * Vector{Float64}(uγy_prev))

    rhs_mom_x .+= load_x
    rhs_mom_y .+= load_y

    rhs_mom_x .-= (1.0 - θ) * ρ_val .* conv_prev_x
    rhs_mom_y .-= (1.0 - θ) * ρ_val .* conv_prev_y

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, t_next)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, t_next)

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y, zeros(np))

    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 t=t_next)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end

function assemble_navierstokes3D_unsteady_picard!(s::NavierStokesMono, data, Δt::Float64,
                                                  x_prev::AbstractVector{<:Real},
                                                  x_iter::AbstractVector{<:Real},
                                                  p_half_prev::AbstractVector{<:Real},
                                                  t_prev::Float64, t_next::Float64,
                                                  θ::Float64,
                                                  conv_prev::Union{Nothing,NTuple{3,Vector{Float64}}})
    nu_x = data.nu_x
    nu_y = data.nu_y
    nu_z = data.nu_z
    sum_nu = nu_x + nu_y + nu_z
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    mass_x_dt = (1.0 / Δt) * data.mass_x
    mass_y_dt = (1.0 / Δt) * data.mass_y
    mass_z_dt = (1.0 / Δt) * data.mass_z
    θc = 1.0 - θ

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_uωz = 2 * nu_x + 2 * nu_y
    off_uγz = 2 * nu_x + 2 * nu_y + nu_z
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_uωz = 2 * nu_x + 2 * nu_y
    row_uγz = 2 * nu_x + 2 * nu_y + nu_z
    row_con = 2 * sum_nu

    compute_convection_vectors!(s, data, x_iter)
    ops = s.last_conv_ops
    @assert ops !== nothing
    bulk = ops.bulk
    K_adv = ops.K_adv

    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    conv_prev_x = conv_prev === nothing ? zeros(Float64, nu_x) : conv_prev[1]
    conv_prev_y = conv_prev === nothing ? zeros(Float64, nu_y) : conv_prev[2]
    conv_prev_z = conv_prev === nothing ? zeros(Float64, nu_z) : conv_prev[3]

    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = mass_x_dt + θ * (data.visc_x_ω + ρ_val * bulk[1] - 0.5 * ρ_val * K_adv[1])
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = θ * data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = mass_y_dt + θ * (data.visc_y_ω + ρ_val * bulk[2] - 0.5 * ρ_val * K_adv[2])
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = θ * data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    A[row_uωz+1:row_uωz+nu_z, off_uωz+1:off_uωz+nu_z] = mass_z_dt + θ * (data.visc_z_ω + ρ_val * bulk[3] - 0.5 * ρ_val * K_adv[3])
    A[row_uωz+1:row_uωz+nu_z, off_uγz+1:off_uγz+nu_z] = θ * data.visc_z_γ
    A[row_uωz+1:row_uωz+nu_z, off_p+1:off_p+np]       = data.grad_z

    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y
    A[row_uγz+1:row_uγz+nu_z, off_uγz+1:off_uγz+nu_z] = data.tie_z

    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ
    A[con_rows, off_uωz+1:off_uωz+nu_z] = data.div_z_ω
    A[con_rows, off_uγz+1:off_uγz+nu_z] = data.div_z_γ

    uωx_prev = view(x_prev, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev, off_uγy+1:off_uγy+nu_y)
    uωz_prev = view(x_prev, off_uωz+1:off_uωz+nu_z)
    uγz_prev = view(x_prev, off_uγz+1:off_uγz+nu_z)

    f_prev_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_prev, 1)
    f_next_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_next, 1)
    load_x = data.Vx * (θ .* f_next_x .+ θc .* f_prev_x)

    f_prev_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_prev, 2)
    f_next_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_next, 2)
    load_y = data.Vy * (θ .* f_next_y .+ θc .* f_prev_y)

    f_prev_z = safe_build_source(data.op_uz, s.fluid.fᵤ, data.cap_pz, t_prev, 3)
    f_next_z = safe_build_source(data.op_uz, s.fluid.fᵤ, data.cap_pz, t_next, 3)
    load_z = data.Vz * (θ .* f_next_z .+ θc .* f_prev_z)

    rhs_mom_x = mass_x_dt * Vector{Float64}(uωx_prev)
    rhs_mom_x .-= θc * (data.visc_x_ω * Vector{Float64}(uωx_prev) + data.visc_x_γ * Vector{Float64}(uγx_prev))

    rhs_mom_y = mass_y_dt * Vector{Float64}(uωy_prev)
    rhs_mom_y .-= θc * (data.visc_y_ω * Vector{Float64}(uωy_prev) + data.visc_y_γ * Vector{Float64}(uγy_prev))

    rhs_mom_z = mass_z_dt * Vector{Float64}(uωz_prev)
    rhs_mom_z .-= θc * (data.visc_z_ω * Vector{Float64}(uωz_prev) + data.visc_z_γ * Vector{Float64}(uγz_prev))

    rhs_mom_x .+= load_x
    rhs_mom_y .+= load_y
    rhs_mom_z .+= load_z

    rhs_mom_x .-= (1.0 - θ) * ρ_val .* conv_prev_x
    rhs_mom_y .-= (1.0 - θ) * ρ_val .* conv_prev_y
    rhs_mom_z .-= (1.0 - θ) * ρ_val .* conv_prev_z

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, t_next)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, t_next)
    g_cut_z = safe_build_g(data.op_uz, s.bc_cut, data.cap_pz, t_next)

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y, rhs_mom_z, g_cut_z, zeros(np))

    apply_velocity_dirichlet_3D!(A, b, s.bc_u[1], s.bc_u[2], s.bc_u[3], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y, nu_z=nu_z,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 uωz_off=off_uωz, uγz_off=off_uγz,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 row_uωz_off=row_uωz, row_uγz_off=row_uγz,
                                 t=t_next)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end


function assemble_navierstokes1D_steady_picard!(s::NavierStokesMono,
                                                data,
                                                advecting_state::AbstractVector{<:Real})
    nu = data.nu
    np = data.np

    rows = 2 * nu + np
    cols = 2 * nu + np
    A = spzeros(Float64, rows, cols)

    off_uω = 0
    off_uγ = nu
    off_p  = 2 * nu

    row_uω = 0
    row_uγ = nu
    row_con = 2 * nu

    compute_convection_vectors!(s, data, advecting_state)
    ops = s.last_conv_ops
    @assert ops !== nothing
    bulk = ops.bulk
    K_adv = ops.K_adv

    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    A[row_uω+1:row_uω+nu, off_uω+1:off_uω+nu] = data.visc_u_ω + ρ_val * bulk[1] - 0.5 * ρ_val * K_adv[1]
    A[row_uω+1:row_uω+nu, off_uγ+1:off_uγ+nu] = data.visc_u_γ
    A[row_uω+1:row_uω+nu, off_p+1:off_p+np]   = data.grad

    A[row_uγ+1:row_uγ+nu, off_uγ+1:off_uγ+nu] = data.tie

    con_rows = row_con+1:row_con+np
    A[con_rows, off_uω+1:off_uω+nu] = data.div_u_ω
    A[con_rows, off_uγ+1:off_uγ+nu] = data.div_u_γ

    f = safe_build_source(data.op_u, s.fluid.fᵤ, data.cap_u, nothing, 1)
    load = data.V * f

    g_cut = safe_build_g(data.op_u, s.bc_cut, data.cap_u, nothing)

    b = vcat(load, g_cut, zeros(np))

    apply_velocity_dirichlet!(A, b, s.bc_u[1], s.fluid.mesh_u[1];
                              nu=nu, uω_offset=off_uω, uγ_offset=off_uγ, t=nothing)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end

function assemble_navierstokes2D_steady_picard!(s::NavierStokesMono,
                                                data,
                                                advecting_state::AbstractVector{<:Real})
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_con = 2 * sum_nu

    # Build convection linearization for the current iterate
    compute_convection_vectors!(s, data, advecting_state)
    ops = s.last_conv_ops
    @assert ops !== nothing
    bulk = ops.bulk
    K_diag = ops.K_adv

    # Get density for convection terms
    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ  # Use constant ρ for now, spatial variation needs more work
    
    # Momentum rows with Picard linearized convection
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = data.visc_x_ω + ρ_val * bulk[1] - 0.5 * ρ_val * K_diag[1]
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = data.visc_y_ω + ρ_val * bulk[2] - 0.5 * ρ_val * K_diag[2]
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    # Tie rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    # Continuity
    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ

    # Forcing (steady)
    f_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, nothing, 1)
    f_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, nothing, 2)
    load_x = data.Vx * f_x
    load_y = data.Vy * f_y

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, nothing)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, nothing)

    b = vcat(load_x, g_cut_x, load_y, g_cut_y, zeros(np))

    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 t=nothing)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end

function assemble_navierstokes3D_steady_picard!(s::NavierStokesMono,
                                                data,
                                                advecting_state::AbstractVector{<:Real})
    nu_x = data.nu_x
    nu_y = data.nu_y
    nu_z = data.nu_z
    sum_nu = nu_x + nu_y + nu_z
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_uωz = 2 * nu_x + 2 * nu_y
    off_uγz = 2 * nu_x + 2 * nu_y + nu_z
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_uωz = 2 * nu_x + 2 * nu_y
    row_uγz = 2 * nu_x + 2 * nu_y + nu_z
    row_con = 2 * sum_nu

    compute_convection_vectors!(s, data, advecting_state)
    ops = s.last_conv_ops
    @assert ops !== nothing
    bulk = ops.bulk
    K_diag = ops.K_adv

    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = data.visc_x_ω + ρ_val * bulk[1] - 0.5 * ρ_val * K_diag[1]
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = data.visc_y_ω + ρ_val * bulk[2] - 0.5 * ρ_val * K_diag[2]
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    A[row_uωz+1:row_uωz+nu_z, off_uωz+1:off_uωz+nu_z] = data.visc_z_ω + ρ_val * bulk[3] - 0.5 * ρ_val * K_diag[3]
    A[row_uωz+1:row_uωz+nu_z, off_uγz+1:off_uγz+nu_z] = data.visc_z_γ
    A[row_uωz+1:row_uωz+nu_z, off_p+1:off_p+np]       = data.grad_z

    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y
    A[row_uγz+1:row_uγz+nu_z, off_uγz+1:off_uγz+nu_z] = data.tie_z

    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ
    A[con_rows, off_uωz+1:off_uωz+nu_z] = data.div_z_ω
    A[con_rows, off_uγz+1:off_uγz+nu_z] = data.div_z_γ

    f_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, nothing, 1)
    f_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, nothing, 2)
    f_z = safe_build_source(data.op_uz, s.fluid.fᵤ, data.cap_pz, nothing, 3)
    load_x = data.Vx * f_x
    load_y = data.Vy * f_y
    load_z = data.Vz * f_z

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, nothing)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, nothing)
    g_cut_z = safe_build_g(data.op_uz, s.bc_cut, data.cap_pz, nothing)

    b = vcat(load_x, g_cut_x, load_y, g_cut_y, load_z, g_cut_z, zeros(np))

    apply_velocity_dirichlet_3D!(A, b, s.bc_u[1], s.bc_u[2], s.bc_u[3], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y, nu_z=nu_z,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 uωz_off=off_uωz, uγz_off=off_uγz,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 row_uωz_off=row_uωz, row_uγz_off=row_uγz,
                                 t=nothing)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end

# Linear solve ---------------------------------------------------------------

function solve_navierstokes_linear_system!(s::NavierStokesMono; method=Base.:\, algorithm=nothing, kwargs...)
    Ared, bred, keep_idx_rows, keep_idx_cols = remove_zero_rows_cols!(s.A, s.b)

    kwargs_nt = (; kwargs...)
    precond_builder = haskey(kwargs_nt, :precond_builder) ? kwargs_nt.precond_builder : nothing
    if precond_builder !== nothing
        kwargs_nt = Base.structdiff(kwargs_nt, (precond_builder=precond_builder,))
    end

    precond_kwargs = (;)
    if precond_builder !== nothing
        precond_result = try
            precond_builder(Ared, s)
        catch err
            if err isa MethodError
                precond_builder(Ared)
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
        log = get(solve_kwargs, :log, false)
        if log
            xred, ch = method(Ared, bred; solve_kwargs...)
            push!(s.ch, ch)
        else
            xred = method(Ared, bred; solve_kwargs...)
        end
    end

    N = size(s.A, 2)
    s.x = zeros(N)
    s.x[keep_idx_cols] = xred
    return s
end

# Time integration ----------------------------------------------------------

function solve_NavierStokesMono_unsteady!(s::NavierStokesMono; Δt::Float64, T_end::Float64,
                                          scheme::Symbol=:CN, method=Base.:\,
                                          algorithm=nothing, store_states::Bool=true,
                                          kwargs...)
    θ = scheme_to_theta(scheme)
    N = length(s.fluid.operator_u)

    if N == 1
        data = navierstokes1D_blocks(s)

        p_offset = 2 * data.nu
        np = data.np
        Ntot = p_offset + np

        x_prev = length(s.x) == Ntot ? copy(s.x) : zeros(Ntot)

        p_half_prev = zeros(np)
        if length(s.x) == Ntot && !isempty(s.x)
            p_half_prev .= s.x[p_offset+1:p_offset+np]
        end

        histories = store_states ? Vector{Vector{Float64}}() : Vector{Vector{Float64}}()
        if store_states
            push!(histories, copy(x_prev))
        end
        times = Float64[0.0]

        conv_prev = s.prev_conv
        if conv_prev !== nothing && length(conv_prev) != 1
            conv_prev = nothing
        end

        t = 0.0
        println("[NavierStokesMono] Starting unsteady 1D solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
        while t < T_end - 1e-12 * max(1.0, T_end)
            dt_step = min(Δt, T_end - t)
            t_next = t + dt_step

            conv_curr = assemble_navierstokes1D_unsteady!(s, data, dt_step, x_prev, p_half_prev, t, t_next, θ, conv_prev)
            solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

            x_prev = copy(s.x)
            p_half_prev .= s.x[p_offset+1:p_offset+np]
            conv_prev = ntuple(Val(1)) do i
                copy(conv_curr[Int(i)])
            end

            push!(times, t_next)
            if store_states
                push!(histories, x_prev)
            end
            max_state = maximum(abs, x_prev)
            println("[NavierStokesMono] t=$(round(t_next; digits=6)) max|state|=$(max_state)")

            t = t_next
        end

        s.prev_conv = conv_prev
        return times, histories
    elseif N == 2
        data = navierstokes2D_blocks(s)

        p_offset = 2 * (data.nu_x + data.nu_y)
        np = data.np
        Ntot = p_offset + np

        x_prev = length(s.x) == Ntot ? copy(s.x) : zeros(Ntot)

        p_half_prev = zeros(np)
        if length(s.x) == Ntot && !isempty(s.x)
            p_half_prev .= s.x[p_offset+1:p_offset+np]
        end

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
        println("[NavierStokesMono] Starting unsteady solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
        while t < T_end - 1e-12 * max(1.0, T_end)
            dt_step = min(Δt, T_end - t)
            t_next = t + dt_step

            conv_curr = assemble_navierstokes2D_unsteady!(s, data, dt_step, x_prev, p_half_prev, t, t_next, θ, conv_prev)
            solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

            x_prev = copy(s.x)
            p_half_prev .= s.x[p_offset+1:p_offset+np]
            N_comp = length(data.nu_components)
            conv_prev = ntuple(Val(N_comp)) do i
                copy(conv_curr[Int(i)])
            end

            push!(times, t_next)
            if store_states
                push!(histories, x_prev)
            end
            max_state = maximum(abs, x_prev)
            println("[NavierStokesMono] t=$(round(t_next; digits=6)) max|state|=$(max_state)")

            t = t_next
        end

        s.prev_conv = conv_prev
        return times, histories
    elseif N == 3
        data = navierstokes3D_blocks(s)

        sum_nu = data.nu_x + data.nu_y + data.nu_z
        p_offset = 2 * sum_nu
        np = data.np
        Ntot = p_offset + np

        x_prev = length(s.x) == Ntot ? copy(s.x) : zeros(Ntot)

        p_half_prev = zeros(np)
        if length(s.x) == Ntot && !isempty(s.x)
            p_half_prev .= s.x[p_offset+1:p_offset+np]
        end

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
        println("[NavierStokesMono] Starting unsteady solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
        while t < T_end - 1e-12 * max(1.0, T_end)
            dt_step = min(Δt, T_end - t)
            t_next = t + dt_step

            conv_curr = assemble_navierstokes3D_unsteady!(s, data, dt_step, x_prev, p_half_prev, t, t_next, θ, conv_prev)
            solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

            x_prev = copy(s.x)
            p_half_prev .= s.x[p_offset+1:p_offset+np]
            N_comp = length(data.nu_components)
            conv_prev = ntuple(Val(N_comp)) do i
                copy(conv_curr[Int(i)])
            end

            push!(times, t_next)
            if store_states
                push!(histories, x_prev)
            end
            max_state = maximum(abs, x_prev)
            println("[NavierStokesMono] t=$(round(t_next; digits=6)) max|state|=$(max_state)")

            t = t_next
        end

        s.prev_conv = conv_prev
        return times, histories
    else
        error("Navier–Stokes unsteady solver not implemented for N=$(N)")
    end
end

function solve_NavierStokesMono_unsteady_picard!(s::NavierStokesMono; Δt::Float64, T_end::Float64,
                                                 scheme::Symbol=:CN, inner_tol::Float64=1e-6,
                                                 inner_maxiter::Int=10, relaxation::Float64=1.0,
                                                 method=Base.:\, algorithm=nothing,
                                                 store_states::Bool=true, kwargs...)
    θ = scheme_to_theta(scheme)
    θ_relax = clamp(relaxation, 0.0, 1.0)
    N = length(s.fluid.operator_u)
    empty!(s.residual_history)
    s.prev_conv = nothing

    if N == 1
        data = navierstokes1D_blocks(s)

        p_offset = 2 * data.nu
        np = data.np
        Ntot = p_offset + np

        x_prev = length(s.x) == Ntot ? copy(s.x) : zeros(Ntot)

        p_half_prev = zeros(np)
        if length(s.x) == Ntot && !isempty(s.x)
            p_half_prev .= s.x[p_offset+1:p_offset+np]
        end

        histories = store_states ? Vector{Vector{Float64}}() : Vector{Vector{Float64}}()
        if store_states
            push!(histories, copy(x_prev))
        end
        times = Float64[0.0]

        t = 0.0
        println("[NavierStokesMono] Starting unsteady 1D Picard solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
        while t < T_end - 1e-12 * max(1.0, T_end)
            dt_step = min(Δt, T_end - t)
            t_next = t + dt_step

            conv_prev_tuple = (1.0 - θ) == 0.0 ? nothing : compute_convection_vectors!(s, data, x_prev)

            x_iter = copy(x_prev)
            residual = Inf
            iter = 0

            while iter < inner_maxiter && residual > inner_tol
                assemble_navierstokes1D_unsteady_picard!(s, data, dt_step, x_prev, x_iter, p_half_prev,
                                                         t, t_next, θ, conv_prev_tuple)
                solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

                x_raw = s.x
                x_new = θ_relax .* x_raw .+ (1.0 - θ_relax) .* x_iter

                vel_residual = maximum(abs, (x_new .- x_iter)[1:p_offset])
                residual = vel_residual
                push!(s.residual_history, residual)

                x_iter .= x_new
                s.x .= x_new

                iter += 1
                println("[NavierStokesMono] t=$(round(t_next; digits=6)) Picard iter=$(iter) max|Δu|=$(residual)")
            end

            if residual > inner_tol
                @warn "Navier–Stokes unsteady Picard (1D) did not reach tolerance" time=t_next final_residual=residual iterations=iter tol=inner_tol
            end

            x_prev = copy(x_iter)
            p_half_prev .= x_prev[p_offset+1:p_offset+np]

            push!(times, t_next)
            if store_states
                push!(histories, copy(x_prev))
            end
            max_state = maximum(abs, x_prev)
            println("[NavierStokesMono] t=$(round(t_next; digits=6)) max|state|=$(max_state)")

            t = t_next
        end

        s.prev_conv = nothing
        return times, histories
    elseif N == 2
        data = navierstokes2D_blocks(s)

        p_offset = 2 * (data.nu_x + data.nu_y)
        np = data.np
        Ntot = p_offset + np

        x_prev = length(s.x) == Ntot ? copy(s.x) : zeros(Ntot)

        p_half_prev = zeros(np)
        if length(s.x) == Ntot && !isempty(s.x)
            p_half_prev .= s.x[p_offset+1:p_offset+np]
        end

        histories = store_states ? Vector{Vector{Float64}}() : Vector{Vector{Float64}}()
        if store_states
            push!(histories, copy(x_prev))
        end
        times = Float64[0.0]

        t = 0.0
        println("[NavierStokesMono] Starting unsteady Picard solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
        while t < T_end - 1e-12 * max(1.0, T_end)
            dt_step = min(Δt, T_end - t)
            t_next = t + dt_step

            conv_prev_tuple = (1.0 - θ) == 0.0 ? nothing : compute_convection_vectors!(s, data, x_prev)

            x_iter = copy(x_prev)
            residual = Inf
            iter = 0

            while iter < inner_maxiter && residual > inner_tol
                assemble_navierstokes2D_unsteady_picard!(s, data, dt_step, x_prev, x_iter, p_half_prev,
                                                         t, t_next, θ, conv_prev_tuple)
                solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

                x_raw = s.x
                x_new = θ_relax .* x_raw .+ (1.0 - θ_relax) .* x_iter

                vel_residual = maximum(abs, (x_new .- x_iter)[1:p_offset])
                residual = vel_residual
                push!(s.residual_history, residual)

                x_iter .= x_new
                s.x .= x_new

                iter += 1
                println("[NavierStokesMono] t=$(round(t_next; digits=6)) Picard iter=$(iter) max|Δu|=$(residual)")
            end

            if residual > inner_tol
                @warn "Navier–Stokes unsteady Picard (2D) did not reach tolerance" time=t_next final_residual=residual iterations=iter tol=inner_tol
            end

            x_prev = copy(x_iter)
            p_half_prev .= x_prev[p_offset+1:p_offset+np]

            push!(times, t_next)
            if store_states
                push!(histories, copy(x_prev))
            end
            max_state = maximum(abs, x_prev)
            println("[NavierStokesMono] t=$(round(t_next; digits=6)) max|state|=$(max_state)")

            t = t_next
        end

        s.prev_conv = nothing
        return times, histories
    elseif N == 3
        data = navierstokes3D_blocks(s)

        p_offset = 2 * (data.nu_x + data.nu_y + data.nu_z)
        np = data.np
        Ntot = p_offset + np

        x_prev = length(s.x) == Ntot ? copy(s.x) : zeros(Ntot)

        p_half_prev = zeros(np)
        if length(s.x) == Ntot && !isempty(s.x)
            p_half_prev .= s.x[p_offset+1:p_offset+np]
        end

        histories = store_states ? Vector{Vector{Float64}}() : Vector{Vector{Float64}}()
        if store_states
            push!(histories, copy(x_prev))
        end
        times = Float64[0.0]

        t = 0.0
        println("[NavierStokesMono] Starting unsteady Picard solve (3D) up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
        while t < T_end - 1e-12 * max(1.0, T_end)
            dt_step = min(Δt, T_end - t)
            t_next = t + dt_step

            conv_prev_tuple = (1.0 - θ) == 0.0 ? nothing : compute_convection_vectors!(s, data, x_prev)

            x_iter = copy(x_prev)
            residual = Inf
            iter = 0

            while iter < inner_maxiter && residual > inner_tol
                assemble_navierstokes3D_unsteady_picard!(s, data, dt_step, x_prev, x_iter, p_half_prev,
                                                         t, t_next, θ, conv_prev_tuple)
                solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

                x_raw = s.x
                x_new = θ_relax .* x_raw .+ (1.0 - θ_relax) .* x_iter

                vel_residual = maximum(abs, (x_new .- x_iter)[1:p_offset])
                residual = vel_residual
                push!(s.residual_history, residual)

                x_iter .= x_new
                s.x .= x_new

                iter += 1
                println("[NavierStokesMono] t=$(round(t_next; digits=6)) Picard iter=$(iter) max|Δu|=$(residual)")
            end

            if residual > inner_tol
                @warn "Navier–Stokes unsteady Picard (3D) did not reach tolerance" time=t_next final_residual=residual iterations=iter tol=inner_tol
            end

            x_prev = copy(x_iter)
            p_half_prev .= x_prev[p_offset+1:p_offset+np]

            push!(times, t_next)
            if store_states
                push!(histories, copy(x_prev))
            end
            max_state = maximum(abs, x_prev)
            println("[NavierStokesMono] t=$(round(t_next; digits=6)) max|state|=$(max_state)")

            t = t_next
        end

        s.prev_conv = nothing
        return times, histories
    else
        error("Navier–Stokes unsteady Picard solver not implemented for N=$(N)")
    end
end

function build_convection_operators(s::NavierStokesMono, state::AbstractVector{<:Real})
    N = length(s.fluid.operator_u)
    data = if N == 1
        navierstokes1D_blocks(s)
    elseif N == 2
        navierstokes2D_blocks(s)
    else
        navierstokes3D_blocks(s)
    end
    conv_vectors = compute_convection_vectors!(s, data, state)
    return s.last_conv_ops, conv_vectors
end

function solve_NavierStokesMono_steady!(s::NavierStokesMono; tol=1e-8, maxiter::Int=25,
                                        relaxation::Float64=1.0, method=Base.:\,
                                        algorithm=nothing, nlsolve_method::Symbol=:picard,
                                        kwargs...)
    N = length(s.fluid.operator_u)

    if N == 1
        if nlsolve_method == :picard
            return solve_NavierStokesMono_steady_picard_1D!(s; tol=tol, maxiter=maxiter,
                                                           relaxation=relaxation, method=method,
                                                           algorithm=algorithm, kwargs...)
        elseif nlsolve_method == :newton
            return solve_NavierStokesMono_steady_newton_1D!(s; tol=tol, maxiter=maxiter,
                                                           method=method, algorithm=algorithm, kwargs...)
        else
            error("Unknown nlsolve_method: $(nlsolve_method). Use :picard or :newton")
        end
    elseif N == 2
        if nlsolve_method == :picard
            return solve_NavierStokesMono_steady_picard!(s; tol=tol, maxiter=maxiter,
                                                       relaxation=relaxation, method=method,
                                                       algorithm=algorithm, kwargs...)
        elseif nlsolve_method == :newton
            return solve_NavierStokesMono_steady_newton!(s; tol=tol, maxiter=maxiter,
                                                       method=method, algorithm=algorithm, kwargs...)
        else
            error("Unknown nlsolve_method: $(nlsolve_method). Use :picard or :newton")
        end
    elseif N == 3
        if nlsolve_method == :picard
            return solve_NavierStokesMono_steady_picard_3D!(s; tol=tol, maxiter=maxiter,
                                                           relaxation=relaxation, method=method,
                                                           algorithm=algorithm, kwargs...)
        elseif nlsolve_method == :newton
            return solve_NavierStokesMono_steady_newton!(s; tol=tol, maxiter=maxiter,
                                                         method=method, algorithm=algorithm, kwargs...)
        else
            error("Unknown nlsolve_method: $(nlsolve_method). Use :picard or :newton")
        end
    else
        error("Steady Navier–Stokes solver not implemented for N=$(N)")
    end
end


function solve_NavierStokesMono_steady_picard_1D!(s::NavierStokesMono; tol=1e-8, maxiter::Int=25,
                                                 relaxation::Float64=1.0, method=Base.:\,
                                                 algorithm=nothing, kwargs...)
    θ_relax = clamp(relaxation, 0.0, 1.0)
    data = navierstokes1D_blocks(s)
    x_iter = copy(s.x)
    residual = Inf
    iter = 0

    empty!(s.residual_history)

    println("[NavierStokesMono] Starting steady 1D Picard iterations (tol=$(tol), maxiter=$(maxiter), relaxation=$(θ_relax))")

    while iter < maxiter && residual > tol
        assemble_navierstokes1D_steady_picard!(s, data, x_iter)
        solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        x_new = θ_relax .* s.x .+ (1.0 - θ_relax) .* x_iter

        p_offset = 2 * data.nu
        velocity_residual = maximum(abs, (x_new .- x_iter)[1:p_offset])
        residual = velocity_residual
        push!(s.residual_history, residual)

        x_iter .= x_new
        s.x .= x_new

        iter += 1
        println("[NavierStokesMono] Picard iter=$(iter) max|Δu|=$(residual)")
    end

    if residual > tol
        @warn "Navier–Stokes steady 1D Picard did not reach tolerance" final_residual=residual iterations=iter tol=tol
    end

    s.prev_conv = nothing
    return s.x, iter, residual
end

function solve_NavierStokesMono_steady_newton_1D!(s::NavierStokesMono; tol=1e-8, maxiter::Int=25,
                                                 method=Base.:\, algorithm=nothing, kwargs...)
    data = navierstokes1D_blocks(s)
    x_iter = copy(s.x)
    residual = Inf
    iter = 0

    empty!(s.residual_history)

    println("[NavierStokesMono] Starting steady 1D Newton iterations (tol=$(tol), maxiter=$(maxiter))")

    while iter < maxiter && residual > tol
        F_val = compute_navierstokes1D_residual!(s, data, x_iter)
        J_val = compute_navierstokes1D_jacobian!(s, data, x_iter)

        rhs = -F_val

        nu = data.nu
        off_uω = 0
        off_uγ = nu
        off_p  = 2 * nu

        apply_velocity_dirichlet_1D_newton!(J_val, rhs, x_iter, s.bc_u[1], s.fluid.mesh_u[1];
                                             nu=nu,
                                             uω_off=off_uω, uγ_off=off_uγ,
                                             row_uω_off=0, row_uγ_off=nu,
                                             t=nothing)

        apply_pressure_gauge_newton!(J_val, rhs, x_iter, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                                     p_offset=off_p, np=data.np,
                                     row_start=2 * nu + 1,
                                     t=nothing)

        s.A = J_val
        s.b = rhs
        solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        x_new = x_iter .+ s.x

        p_offset = 2 * nu
        velocity_residual = maximum(abs, (x_new .- x_iter)[1:p_offset])
        residual = velocity_residual
        push!(s.residual_history, residual)

        x_iter .= x_new
        s.x .= x_new

        iter += 1
        println("[NavierStokesMono] Newton iter=$(iter) max|Δu|=$(residual)")
    end

    if residual > tol
        @warn "Navier–Stokes steady 1D Newton did not reach tolerance" final_residual=residual iterations=iter tol=tol
    end

    s.prev_conv = nothing
    return s.x, iter, residual
end

function solve_NavierStokesMono_steady_picard!(s::NavierStokesMono; tol=1e-8, maxiter::Int=25,
                                              relaxation::Float64=1.0, method=Base.:\,
                                              algorithm=nothing, kwargs...)
    θ_relax = clamp(relaxation, 0.0, 1.0)
    N = length(s.fluid.operator_u)
    N == 2 || error("Steady Navier–Stokes Picard solver currently implemented for 2D (N=$(N)).")

    data = navierstokes2D_blocks(s)
    x_iter = copy(s.x)
    residual = Inf
    iter = 0
    
    # Clear and initialize residual history
    empty!(s.residual_history)

    println("[NavierStokesMono] Starting steady Picard iterations (tol=$(tol), maxiter=$(maxiter), relaxation=$(θ_relax))")

    while iter < maxiter && residual > tol
        assemble_navierstokes2D_steady_picard!(s, data, x_iter)
        solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        x_new = θ_relax .* s.x .+ (1.0 - θ_relax) .* x_iter
        
        # Calculate residual only on velocity components (exclude pressure)
        p_offset = 2 * (data.nu_x + data.nu_y)
        velocity_residual = maximum(abs, (x_new .- x_iter)[1:p_offset])
        residual = velocity_residual
        
        # Store residual in history
        push!(s.residual_history, residual)

        x_iter .= x_new
        s.x .= x_new

        iter += 1
        println("[NavierStokesMono] Picard iter=$(iter) max|Δu|=$(residual)")
    end

    if residual > tol
        @warn "Navier–Stokes steady Picard did not reach tolerance" final_residual=residual iterations=iter tol=tol
    end

    s.prev_conv = nothing
    return s.x, iter, residual
end

function solve_NavierStokesMono_steady_picard_3D!(s::NavierStokesMono; tol=1e-8, maxiter::Int=25,
                                                 relaxation::Float64=1.0, method=Base.:\,
                                                 algorithm=nothing, kwargs...)
    θ_relax = clamp(relaxation, 0.0, 1.0)
    N = length(s.fluid.operator_u)
    N == 3 || error("Steady Navier–Stokes Picard (3D) requires N=3 (got $(N))")

    data = navierstokes3D_blocks(s)
    x_iter = copy(s.x)
    residual = Inf
    iter = 0

    empty!(s.residual_history)

    println("[NavierStokesMono] Starting steady 3D Picard iterations (tol=$(tol), maxiter=$(maxiter), relaxation=$(θ_relax))")

    while iter < maxiter && residual > tol
        assemble_navierstokes3D_steady_picard!(s, data, x_iter)
        solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        x_new = θ_relax .* s.x .+ (1.0 - θ_relax) .* x_iter

        p_offset = 2 * (data.nu_x + data.nu_y + data.nu_z)
        velocity_residual = maximum(abs, (x_new .- x_iter)[1:p_offset])
        residual = velocity_residual
        push!(s.residual_history, residual)

        x_iter .= x_new
        s.x .= x_new

        iter += 1
        println("[NavierStokesMono] Picard iter=$(iter) max|Δu|=$(residual)")
    end

    if residual > tol
        @warn "Navier–Stokes steady 3D Picard did not reach tolerance" final_residual=residual iterations=iter tol=tol
    end

    s.prev_conv = nothing
    return s.x, iter, residual
end

function solve_NavierStokesMono_steady_newton!(s::NavierStokesMono; tol=1e-8, maxiter::Int=25,
                                              method=Base.:\, algorithm=nothing, kwargs...)
    N = length(s.fluid.operator_u)
    if N == 2
        data = navierstokes2D_blocks(s)
    elseif N == 3
        data = navierstokes3D_blocks(s)
    else
        error("Steady Navier–Stokes Newton solver not implemented for N=$(N).")
    end
    x_iter = copy(s.x)
    residual = Inf
    iter = 0
    
    # Clear and initialize residual history
    empty!(s.residual_history)

    println("[NavierStokesMono] Starting steady Newton iterations (tol=$(tol), maxiter=$(maxiter))")

    while iter < maxiter && residual > tol
        # Compute residual and Jacobian
        F_val = N == 2 ? compute_navierstokes2D_residual!(s, data, x_iter) :
                         compute_navierstokes3D_residual!(s, data, x_iter)
        J_val = N == 2 ? compute_navierstokes2D_jacobian!(s, data, x_iter) :
                         compute_navierstokes3D_jacobian!(s, data, x_iter)

        rhs = -F_val

        if N == 2
            nu_x = data.nu_x
            nu_y = data.nu_y
            sum_nu = nu_x + nu_y
            off_uωx = 0
            off_uγx = nu_x
            off_uωy = 2 * nu_x
            off_uγy = 2 * nu_x + nu_y
            off_p   = 2 * sum_nu

            apply_velocity_dirichlet_2D_newton!(J_val, rhs, x_iter, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                                nu_x=nu_x, nu_y=nu_y,
                                                uωx_off=off_uωx, uγx_off=off_uγx,
                                                uωy_off=off_uωy, uγy_off=off_uγy,
                                                row_uωx_off=0, row_uγx_off=nu_x,
                                                row_uωy_off=2*nu_x, row_uγy_off=2*nu_x+nu_y,
                                                t=nothing)

            apply_pressure_gauge_newton!(J_val, rhs, x_iter, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                                         p_offset=off_p, np=data.np,
                                         row_start=2*sum_nu+1,
                                         t=nothing)
        else
            nu_x = data.nu_x
            nu_y = data.nu_y
            nu_z = data.nu_z
            sum_nu = nu_x + nu_y + nu_z
            off_uωx = 0
            off_uγx = nu_x
            off_uωy = 2 * nu_x
            off_uγy = 2 * nu_x + nu_y
            off_uωz = 2 * nu_x + 2 * nu_y
            off_uγz = 2 * nu_x + 2 * nu_y + nu_z
            off_p   = 2 * sum_nu

            apply_velocity_dirichlet_3D_newton!(J_val, rhs, x_iter, s.bc_u[1], s.bc_u[2], s.bc_u[3], s.fluid.mesh_u;
                                                nu_x=nu_x, nu_y=nu_y, nu_z=nu_z,
                                                uωx_off=off_uωx, uγx_off=off_uγx,
                                                uωy_off=off_uωy, uγy_off=off_uγy,
                                                uωz_off=off_uωz, uγz_off=off_uγz,
                                                row_uωx_off=0, row_uγx_off=nu_x,
                                                row_uωy_off=2*nu_x, row_uγy_off=2*nu_x+nu_y,
                                                row_uωz_off=2*nu_x+2*nu_y, row_uγz_off=2*nu_x+2*nu_y+nu_z,
                                                t=nothing)

            apply_pressure_gauge_newton!(J_val, rhs, x_iter, s.pressure_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                                         p_offset=off_p, np=data.np,
                                         row_start=2*sum_nu+1,
                                         t=nothing)
        end

        # Solve Newton step: J * Δx = rhs
        s.A = J_val
        s.b = rhs
        solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)
        
        # Update solution: x_new = x_iter + Δx
        x_new = x_iter .+ s.x
        
        # Calculate residual only on velocity components (exclude pressure)
        p_offset = N == 2 ? 2 * (data.nu_x + data.nu_y) : 2 * (data.nu_x + data.nu_y + data.nu_z)
        velocity_residual = maximum(abs, (x_new .- x_iter)[1:p_offset])
        residual = velocity_residual
        
        # Store residual in history
        push!(s.residual_history, residual)

        x_iter .= x_new
        s.x .= x_new

        iter += 1
        println("[NavierStokesMono] Newton iter=$(iter) max|Δu|=$(residual)")
    end

    if residual > tol
        @warn "Navier–Stokes steady Newton did not reach tolerance" final_residual=residual iterations=iter tol=tol
    end

    s.prev_conv = nothing
    return s.x, iter, residual
end

# Newton method helper functions

function compute_navierstokes1D_residual!(s::NavierStokesMono, data, x_state::AbstractVector{<:Real})
    nu = data.nu
    np = data.np

    off_uω = 0
    off_uγ = nu
    off_p  = 2 * nu

    uω = view(x_state, off_uω+1:off_uω+nu)
    uγ = view(x_state, off_uγ+1:off_uγ+nu)
    pω = view(x_state, off_p+1:off_p+np)

    conv_vectors = compute_convection_vectors!(s, data, x_state)

    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    f = safe_build_source(data.op_u, s.fluid.fᵤ, data.cap_u, nothing, 1)
    load = data.V * f

    uω_vec = Vector{Float64}(uω)
    uγ_vec = Vector{Float64}(uγ)
    p_vec = Vector{Float64}(pω)

    F_mom = data.visc_u_ω * uω_vec + data.visc_u_γ * uγ_vec + ρ_val * conv_vectors[1] + data.grad * p_vec - load

    g_cut = safe_build_g(data.op_u, s.bc_cut, data.cap_u, nothing)
    F_tie = uγ_vec - g_cut

    F_cont = data.div_u_ω * uω_vec + data.div_u_γ * uγ_vec

    return vcat(F_mom, F_tie, F_cont)
end

function compute_navierstokes1D_jacobian!(s::NavierStokesMono, data, x_state::AbstractVector{<:Real})
    nu = data.nu
    np = data.np

    rows = 2 * nu + np
    cols = 2 * nu + np
    J = spzeros(Float64, rows, cols)

    off_uω = 0
    off_uγ = nu
    off_p  = 2 * nu

    row_uω = 0
    row_uγ = nu
    row_con = 2 * nu

    compute_convection_vectors!(s, data, x_state)
    ops = s.last_conv_ops
    @assert ops !== nothing
    bulk = ops.bulk[1]
    K_adv = ops.K_adv[1]

    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    J[row_uω+1:row_uω+nu, off_uω+1:off_uω+nu] = data.visc_u_ω + ρ_val * bulk - 0.5 * ρ_val * K_adv
    J[row_uω+1:row_uω+nu, off_uγ+1:off_uγ+nu] = data.visc_u_γ
    J[row_uω+1:row_uω+nu, off_p+1:off_p+np]   = data.grad

    J[row_uγ+1:row_uγ+nu, off_uγ+1:off_uγ+nu] = data.tie

    con_rows = row_con+1:row_con+np
    J[con_rows, off_uω+1:off_uω+nu] = data.div_u_ω
    J[con_rows, off_uγ+1:off_uγ+nu] = data.div_u_γ

    return J
end

function compute_navierstokes2D_residual!(s::NavierStokesMono, data, x_state::AbstractVector{<:Real})
    """
    Compute the nonlinear residual F(x) = 0 for steady Navier-Stokes:
    F = [momentum_x_residual; tie_x_residual; momentum_y_residual; tie_y_residual; continuity_residual]
    """
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y
    np = data.np
    
    # Extract state variables
    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_p   = 2 * sum_nu
    
    uωx = view(x_state, off_uωx+1:off_uωx+nu_x)
    uγx = view(x_state, off_uγx+1:off_uγx+nu_x)
    uωy = view(x_state, off_uωy+1:off_uωy+nu_y)
    uγy = view(x_state, off_uγy+1:off_uγy+nu_y)
    pω  = view(x_state, off_p+1:off_p+np)
    
    # Compute convection terms
    conv_vectors = compute_convection_vectors!(s, data, x_state)
    
    # Get density for convection terms
    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ
    
    # Forcing terms
    f_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, nothing, 1)
    f_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, nothing, 2)
    load_x = data.Vx * f_x
    load_y = data.Vy * f_y
    
    # Compute residuals
    # Momentum x: -μ∇²u + ρ(u·∇)u + ∇p - f = 0
    F_mom_x = data.visc_x_ω * Vector{Float64}(uωx) + data.visc_x_γ * Vector{Float64}(uγx) + 
              ρ_val * conv_vectors[1] + data.grad_x * Vector{Float64}(pω) - load_x
    
    # Tie x: uγ - g_cut = 0
    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, nothing)
    F_tie_x = Vector{Float64}(uγx) - g_cut_x
    
    # Momentum y: -μ∇²v + ρ(u·∇)v + ∇p - f = 0
    F_mom_y = data.visc_y_ω * Vector{Float64}(uωy) + data.visc_y_γ * Vector{Float64}(uγy) + 
              ρ_val * conv_vectors[2] + data.grad_y * Vector{Float64}(pω) - load_y
    
    # Tie y: vγ - g_cut = 0
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, nothing)
    F_tie_y = Vector{Float64}(uγy) - g_cut_y
    
    # Continuity: ∇·u = 0
    F_cont = data.div_x_ω * Vector{Float64}(uωx) + data.div_x_γ * Vector{Float64}(uγx) + 
             data.div_y_ω * Vector{Float64}(uωy) + data.div_y_γ * Vector{Float64}(uγy)
    
    # Combine all residuals
    F = vcat(F_mom_x, F_tie_x, F_mom_y, F_tie_y, F_cont)
    
    return F
end

function compute_navierstokes2D_jacobian!(s::NavierStokesMono, data, x_state::AbstractVector{<:Real})
    """
    Compute the Jacobian matrix J = ∂F/∂x for Newton method
    """
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y
    np = data.np
    
    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    J = spzeros(Float64, rows, cols)
    
    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_p   = 2 * sum_nu
    
    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_con = 2 * sum_nu
    
    # Compute convection Jacobian terms
    compute_convection_vectors!(s, data, x_state)
    ops = s.last_conv_ops
    @assert ops !== nothing
    bulk = ops.bulk
    K_adv = ops.K_adv
    
    # Get density for convection terms
    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ
    
    # Momentum x Jacobian: ∂F_mom_x/∂x
    J[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = data.visc_x_ω + ρ_val * bulk[1] - 0.5 * ρ_val * K_adv[1]
    J[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = data.visc_x_γ
    J[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x
    
    # Tie x Jacobian: ∂F_tie_x/∂x
    J[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    
    # Momentum y Jacobian: ∂F_mom_y/∂x
    J[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = data.visc_y_ω + ρ_val * bulk[2] - 0.5 * ρ_val * K_adv[2]
    J[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = data.visc_y_γ
    J[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y
    
    # Tie y Jacobian: ∂F_tie_y/∂x
    J[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y
    
    # Continuity Jacobian: ∂F_cont/∂x
    con_rows = row_con+1:row_con+np
    J[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    J[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    J[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    J[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ
    
    return J
end

function compute_navierstokes3D_residual!(s::NavierStokesMono, data, x_state::AbstractVector{<:Real})
    nu_x = data.nu_x
    nu_y = data.nu_y
    nu_z = data.nu_z
    np = data.np

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_uωz = 2 * nu_x + 2 * nu_y
    off_uγz = 2 * nu_x + 2 * nu_y + nu_z
    off_p   = 2 * (nu_x + nu_y + nu_z)

    uωx = view(x_state, off_uωx+1:off_uωx+nu_x)
    uγx = view(x_state, off_uγx+1:off_uγx+nu_x)
    uωy = view(x_state, off_uωy+1:off_uωy+nu_y)
    uγy = view(x_state, off_uγy+1:off_uγy+nu_y)
    uωz = view(x_state, off_uωz+1:off_uωz+nu_z)
    uγz = view(x_state, off_uγz+1:off_uγz+nu_z)
    pω = view(x_state, off_p+1:off_p+np)

    conv_vectors = compute_convection_vectors!(s, data, x_state)
    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    f_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, nothing, 1)
    f_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, nothing, 2)
    f_z = safe_build_source(data.op_uz, s.fluid.fᵤ, data.cap_pz, nothing, 3)
    load_x = data.Vx * f_x
    load_y = data.Vy * f_y
    load_z = data.Vz * f_z

    uωx_vec = Vector{Float64}(uωx)
    uγx_vec = Vector{Float64}(uγx)
    uωy_vec = Vector{Float64}(uωy)
    uγy_vec = Vector{Float64}(uγy)
    uωz_vec = Vector{Float64}(uωz)
    uγz_vec = Vector{Float64}(uγz)
    p_vec = Vector{Float64}(pω)

    F_mom_x = data.visc_x_ω * uωx_vec + data.visc_x_γ * uγx_vec + ρ_val * conv_vectors[1] + data.grad_x * p_vec - load_x
    F_mom_y = data.visc_y_ω * uωy_vec + data.visc_y_γ * uγy_vec + ρ_val * conv_vectors[2] + data.grad_y * p_vec - load_y
    F_mom_z = data.visc_z_ω * uωz_vec + data.visc_z_γ * uγz_vec + ρ_val * conv_vectors[3] + data.grad_z * p_vec - load_z

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, nothing)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, nothing)
    g_cut_z = safe_build_g(data.op_uz, s.bc_cut, data.cap_pz, nothing)

    F_tie_x = uγx_vec - g_cut_x
    F_tie_y = uγy_vec - g_cut_y
    F_tie_z = uγz_vec - g_cut_z

    F_cont = data.div_x_ω * uωx_vec + data.div_x_γ * uγx_vec +
             data.div_y_ω * uωy_vec + data.div_y_γ * uγy_vec +
             data.div_z_ω * uωz_vec + data.div_z_γ * uγz_vec

    return vcat(F_mom_x, F_tie_x, F_mom_y, F_tie_y, F_mom_z, F_tie_z, F_cont)
end

function compute_navierstokes3D_jacobian!(s::NavierStokesMono, data, x_state::AbstractVector{<:Real})
    nu_x = data.nu_x
    nu_y = data.nu_y
    nu_z = data.nu_z
    sum_nu = nu_x + nu_y + nu_z
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    J = spzeros(Float64, rows, cols)

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_uωz = 2 * nu_x + 2 * nu_y
    off_uγz = 2 * nu_x + 2 * nu_y + nu_z
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_uωz = 2 * nu_x + 2 * nu_y
    row_uγz = 2 * nu_x + 2 * nu_y + nu_z
    row_con = 2 * sum_nu

    compute_convection_vectors!(s, data, x_state)
    ops = s.last_conv_ops
    @assert ops !== nothing
    bulk = ops.bulk
    K_adv = ops.K_adv

    ρ = s.fluid.ρ
    ρ_val = ρ isa Function ? 1.0 : ρ

    J[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = data.visc_x_ω + ρ_val * bulk[1] - 0.5 * ρ_val * K_adv[1]
    J[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = data.visc_x_γ
    J[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    J[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x

    J[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = data.visc_y_ω + ρ_val * bulk[2] - 0.5 * ρ_val * K_adv[2]
    J[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = data.visc_y_γ
    J[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    J[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    J[row_uωz+1:row_uωz+nu_z, off_uωz+1:off_uωz+nu_z] = data.visc_z_ω + ρ_val * bulk[3] - 0.5 * ρ_val * K_adv[3]
    J[row_uωz+1:row_uωz+nu_z, off_uγz+1:off_uγz+nu_z] = data.visc_z_γ
    J[row_uωz+1:row_uωz+nu_z, off_p+1:off_p+np]       = data.grad_z

    J[row_uγz+1:row_uγz+nu_z, off_uγz+1:off_uγz+nu_z] = data.tie_z

    con_rows = row_con+1:row_con+np
    J[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    J[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    J[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    J[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ
    J[con_rows, off_uωz+1:off_uωz+nu_z] = data.div_z_ω
    J[con_rows, off_uγz+1:off_uγz+nu_z] = data.div_z_γ

    return J
end

# Boundary condition application for Newton method residuals

function apply_velocity_dirichlet_1D_newton!(A::SparseMatrixCSC{Float64, Int}, rhs::Vector{Float64},
                                             x_state::AbstractVector{<:Real},
                                             bc_u::BorderConditions,
                                             mesh_u::AbstractMesh;
                                             nu::Int,
                                             uω_off::Int, uγ_off::Int,
                                             row_uω_off::Int, row_uγ_off::Int,
                                             t::Union{Nothing,Float64}=nothing)
    xnodes = mesh_u.nodes[1]
    iL = 1
    iR = max(length(xnodes) - 1, 1)

    function eval_value(bc, x)
        isnothing(bc) && return nothing
        bc isa Dirichlet || return nothing
        v = bc.value
        if v isa Function
            return t === nothing ? v(x) : v(x, 0.0, t)
        else
            return v
        end
    end

    left_bc = get(bc_u.borders, :bottom, get(bc_u.borders, :left, nothing))
    right_bc = get(bc_u.borders, :top,    get(bc_u.borders, :right, nothing))

    if left_bc isa Dirichlet
        vL = eval_value(left_bc, xnodes[1])
        if vL !== nothing
            col = uω_off + iL
            delta = Float64(vL) - Float64(x_state[col])
            enforce_dirichlet!(A, rhs, row_uω_off + iL, col, delta)
            colγ = uγ_off + iL
            deltaγ = Float64(vL) - Float64(x_state[colγ])
            enforce_dirichlet!(A, rhs, row_uγ_off + iL, colγ, deltaγ)
        end
    elseif left_bc isa Symmetry
        col = uω_off + iL
        delta = -Float64(x_state[col])
        enforce_dirichlet!(A, rhs, row_uω_off + iL, col, delta)
        colγ = uγ_off + iL
        deltaγ = -Float64(x_state[colγ])
        enforce_dirichlet!(A, rhs, row_uγ_off + iL, colγ, deltaγ)
    elseif left_bc isa Outflow
        neighbor = min(iL + 1, nu)
        col = uω_off + iL
        col_adj = uω_off + neighbor
        rhs_val = -(Float64(x_state[col]) - Float64(x_state[col_adj]))
        enforce_zero_gradient!(A, rhs, row_uω_off + iL, col, col_adj, rhs_val)
        colγ = uγ_off + iL
        colγ_adj = uγ_off + neighbor
        rhs_gamma = -(Float64(x_state[colγ]) - Float64(x_state[colγ_adj]))
        enforce_zero_gradient!(A, rhs, row_uγ_off + iL, colγ, colγ_adj, rhs_gamma)
    end

    if right_bc isa Dirichlet
        vR = eval_value(right_bc, xnodes[end])
        if vR !== nothing
            col = uω_off + iR
            delta = Float64(vR) - Float64(x_state[col])
            enforce_dirichlet!(A, rhs, row_uω_off + iR, col, delta)
            colγ = uγ_off + iR
            deltaγ = Float64(vR) - Float64(x_state[colγ])
            enforce_dirichlet!(A, rhs, row_uγ_off + iR, colγ, deltaγ)
        end
    elseif right_bc isa Symmetry
        col = uω_off + iR
        delta = -Float64(x_state[col])
        enforce_dirichlet!(A, rhs, row_uω_off + iR, col, delta)
        colγ = uγ_off + iR
        deltaγ = -Float64(x_state[colγ])
        enforce_dirichlet!(A, rhs, row_uγ_off + iR, colγ, deltaγ)
    elseif right_bc isa Outflow
        neighbor = max(iR - 1, iL)
        col = uω_off + iR
        col_adj = uω_off + neighbor
        rhs_val = -(Float64(x_state[col]) - Float64(x_state[col_adj]))
        enforce_zero_gradient!(A, rhs, row_uω_off + iR, col, col_adj, rhs_val)
        colγ = uγ_off + iR
        colγ_adj = uγ_off + neighbor
        rhs_gamma = -(Float64(x_state[colγ]) - Float64(x_state[colγ_adj]))
        enforce_zero_gradient!(A, rhs, row_uγ_off + iR, colγ, colγ_adj, rhs_gamma)
    end

    return nothing
end

function apply_velocity_dirichlet_2D_newton!(A::SparseMatrixCSC{Float64, Int}, rhs::Vector{Float64},
                                             x_state::AbstractVector{<:Real},
                                             bc_ux::BorderConditions,
                                             bc_uy::BorderConditions,
                                             mesh_u::NTuple{2,AbstractMesh};
                                             nu_x::Int, nu_y::Int,
                                             uωx_off::Int, uγx_off::Int,
                                             uωy_off::Int, uγy_off::Int,
                                             row_uωx_off::Int, row_uγx_off::Int,
                                             row_uωy_off::Int, row_uγy_off::Int,
                                             t::Union{Nothing,Float64}=nothing)
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

    eval_val(bc, x, y) = (bc isa Dirichlet) ? (bc.value isa Function ? eval_boundary_func(bc.value, x, y) : bc.value) : nothing
    eval_val(bc, x, y, t) = (bc isa Dirichlet) ? (bc.value isa Function ? bc.value(x, y, t) : bc.value) : nothing

    function eval_boundary_func(f, x, y)
        try
            return f(x, y)
        catch MethodError
            return f(x, y, 0.0)
        end
    end

    bcx_bottom = get(bc_ux.borders, :bottom, nothing)
    bcy_bottom = get(bc_uy.borders, :bottom, nothing)
    bcx_top    = get(bc_ux.borders, :top, nothing)
    bcy_top    = get(bc_uy.borders, :top, nothing)
    bcx_left   = get(bc_ux.borders, :left, nothing)
    bcy_left   = get(bc_uy.borders, :left, nothing)
    bcx_right  = get(bc_ux.borders, :right, nothing)
    bcy_right  = get(bc_uy.borders, :right, nothing)

    # Bottom/top boundaries
    for (jx, bcx, bcy) in ((1, bcx_bottom, bcy_bottom), (jtop, bcx_top, bcy_top))
        for i in 1:nx
            if bcx isa Dirichlet
                vx = t === nothing ? eval_val(bcx, xs_x[i], ys_x[jx]) : eval_val(bcx, xs_x[i], ys_x[jx], t)
                if vx !== nothing
                    lix = LIx[i, jx]
                    col = uωx_off + lix
                    delta = Float64(vx) - Float64(x_state[col])
                    enforce_dirichlet!(A, rhs, row_uωx_off + lix, col, delta)
                    colγ = uγx_off + lix
                    deltaγ = Float64(vx) - Float64(x_state[colγ])
                    enforce_dirichlet!(A, rhs, row_uγx_off + lix, colγ, deltaγ)
                end
            elseif bcx isa Symmetry
                lix = LIx[i, jx]
                neighbor = jx == 1 ? LIx[i, min(jx + 1, ny)] : LIx[i, max(jx - 1, 1)]
                col = uωx_off + lix
                col_adj = uωx_off + neighbor
                rhs_val = -(Float64(x_state[col]) - Float64(x_state[col_adj]))
                enforce_zero_gradient!(A, rhs, row_uωx_off + lix, col, col_adj, rhs_val)
                colγ = uγx_off + lix
                colγ_adj = uγx_off + neighbor
                rhs_gamma = -(Float64(x_state[colγ]) - Float64(x_state[colγ_adj]))
                enforce_zero_gradient!(A, rhs, row_uγx_off + lix, colγ, colγ_adj, rhs_gamma)
            elseif bcx isa Outflow
                lix = LIx[i, jx]
                neighbor = jx == 1 ? LIx[i, min(jx + 1, ny)] : LIx[i, max(jx - 1, 1)]
                col = uωx_off + lix
                col_adj = uωx_off + neighbor
                rhs_val = -(Float64(x_state[col]) - Float64(x_state[col_adj]))
                enforce_zero_gradient!(A, rhs, row_uωx_off + lix, col, col_adj, rhs_val)
                colγ = uγx_off + lix
                colγ_adj = uγx_off + neighbor
                rhs_gamma = -(Float64(x_state[colγ]) - Float64(x_state[colγ_adj]))
                enforce_zero_gradient!(A, rhs, row_uγx_off + lix, colγ, colγ_adj, rhs_gamma)
            end
        end
        for i in 1:nx_y
            if bcy isa Dirichlet
                vy = t === nothing ? eval_val(bcy, xs_y[i], ys_y[jx]) : eval_val(bcy, xs_y[i], ys_y[jx], t)
                if vy !== nothing
                    liy = LIy[i, jx]
                    col = uωy_off + liy
                    delta = Float64(vy) - Float64(x_state[col])
                    enforce_dirichlet!(A, rhs, row_uωy_off + liy, col, delta)
                    colγ = uγy_off + liy
                    deltaγ = Float64(vy) - Float64(x_state[colγ])
                    enforce_dirichlet!(A, rhs, row_uγy_off + liy, colγ, deltaγ)
                end
            elseif bcy isa Symmetry
                liy = LIy[i, jx]
                col = uωy_off + liy
                delta = -Float64(x_state[col])
                enforce_dirichlet!(A, rhs, row_uωy_off + liy, col, delta)
                colγ = uγy_off + liy
                deltaγ = -Float64(x_state[colγ])
                enforce_dirichlet!(A, rhs, row_uγy_off + liy, colγ, deltaγ)
            elseif bcy isa Outflow
                liy = LIy[i, jx]
                neighbor = jx == 1 ? LIy[i, min(jx + 1, ny_y)] : LIy[i, max(jx - 1, 1)]
                col = uωy_off + liy
                col_adj = uωy_off + neighbor
                rhs_val = -(Float64(x_state[col]) - Float64(x_state[col_adj]))
                enforce_zero_gradient!(A, rhs, row_uωy_off + liy, col, col_adj, rhs_val)
                colγ = uγy_off + liy
                colγ_adj = uγy_off + neighbor
                rhs_gamma = -(Float64(x_state[colγ]) - Float64(x_state[colγ_adj]))
                enforce_zero_gradient!(A, rhs, row_uγy_off + liy, colγ, colγ_adj, rhs_gamma)
            end
        end
    end

    # Left/right boundaries
    for (ix, bcx, bcy) in ((1, bcx_left, bcy_left), (iright, bcx_right, bcy_right))
        for j in 1:ny
            if bcx isa Dirichlet
                vx = t === nothing ? eval_val(bcx, xs_x[ix], ys_x[j]) : eval_val(bcx, xs_x[ix], ys_x[j], t)
                if vx !== nothing
                    lix = LIx[ix, j]
                    col = uωx_off + lix
                    delta = Float64(vx) - Float64(x_state[col])
                    enforce_dirichlet!(A, rhs, row_uωx_off + lix, col, delta)
                    colγ = uγx_off + lix
                    deltaγ = Float64(vx) - Float64(x_state[colγ])
                    enforce_dirichlet!(A, rhs, row_uγx_off + lix, colγ, deltaγ)
                end
            elseif bcx isa Symmetry
                lix = LIx[ix, j]
                col = uωx_off + lix
                delta = -Float64(x_state[col])
                enforce_dirichlet!(A, rhs, row_uωx_off + lix, col, delta)
                colγ = uγx_off + lix
                deltaγ = -Float64(x_state[colγ])
                enforce_dirichlet!(A, rhs, row_uγx_off + lix, colγ, deltaγ)
            elseif bcx isa Outflow
                lix = LIx[ix, j]
                neighbor = ix == 1 ? LIx[min(ix + 1, nx), j] : LIx[max(ix - 1, 1), j]
                col = uωx_off + lix
                col_adj = uωx_off + neighbor
                rhs_val = -(Float64(x_state[col]) - Float64(x_state[col_adj]))
                enforce_zero_gradient!(A, rhs, row_uωx_off + lix, col, col_adj, rhs_val)
                colγ = uγx_off + lix
                colγ_adj = uγx_off + neighbor
                rhs_gamma = -(Float64(x_state[colγ]) - Float64(x_state[colγ_adj]))
                enforce_zero_gradient!(A, rhs, row_uγx_off + lix, colγ, colγ_adj, rhs_gamma)
            end
        end
        for j in 1:ny_y
            if bcy isa Dirichlet
                vy = t === nothing ? eval_val(bcy, xs_y[ix], ys_y[j]) : eval_val(bcy, xs_y[ix], ys_y[j], t)
                if vy !== nothing
                    liy = LIy[ix, j]
                    col = uωy_off + liy
                    delta = Float64(vy) - Float64(x_state[col])
                    enforce_dirichlet!(A, rhs, row_uωy_off + liy, col, delta)
                    colγ = uγy_off + liy
                    deltaγ = Float64(vy) - Float64(x_state[colγ])
                    enforce_dirichlet!(A, rhs, row_uγy_off + liy, colγ, deltaγ)
                end
            elseif bcy isa Symmetry
                liy = LIy[ix, j]
                neighbor = ix == 1 ? LIy[min(ix + 1, nx_y), j] : LIy[max(ix - 1, 1), j]
                col = uωy_off + liy
                col_adj = uωy_off + neighbor
                rhs_val = -(Float64(x_state[col]) - Float64(x_state[col_adj]))
                enforce_zero_gradient!(A, rhs, row_uωy_off + liy, col, col_adj, rhs_val)
                colγ = uγy_off + liy
                colγ_adj = uγy_off + neighbor
                rhs_gamma = -(Float64(x_state[colγ]) - Float64(x_state[colγ_adj]))
                enforce_zero_gradient!(A, rhs, row_uγy_off + liy, colγ, colγ_adj, rhs_gamma)
            elseif bcy isa Outflow
                liy = LIy[ix, j]
                neighbor = ix == 1 ? LIy[min(ix + 1, nx_y), j] : LIy[max(ix - 1, 1), j]
                col = uωy_off + liy
                col_adj = uωy_off + neighbor
                rhs_val = -(Float64(x_state[col]) - Float64(x_state[col_adj]))
                enforce_zero_gradient!(A, rhs, row_uωy_off + liy, col, col_adj, rhs_val)
                colγ = uγy_off + liy
                colγ_adj = uγy_off + neighbor
                rhs_gamma = -(Float64(x_state[colγ]) - Float64(x_state[colγ_adj]))
                enforce_zero_gradient!(A, rhs, row_uγy_off + liy, colγ, colγ_adj, rhs_gamma)
            end
        end
    end

    return nothing
end

function apply_velocity_dirichlet_3D_newton!(A::SparseMatrixCSC{Float64, Int}, rhs::Vector{Float64},
                                             x_state::AbstractVector{<:Real},
                                             bc_ux::BorderConditions,
                                             bc_uy::BorderConditions,
                                             bc_uz::BorderConditions,
                                             mesh_u::NTuple{3,AbstractMesh};
                                             nu_x::Int, nu_y::Int, nu_z::Int,
                                             uωx_off::Int, uγx_off::Int,
                                             uωy_off::Int, uγy_off::Int,
                                             uωz_off::Int, uγz_off::Int,
                                             row_uωx_off::Int, row_uγx_off::Int,
                                             row_uωy_off::Int, row_uγy_off::Int,
                                             row_uωz_off::Int, row_uγz_off::Int,
                                             t::Union{Nothing,Float64}=nothing)
    mesh_ux, mesh_uy, mesh_uz = mesh_u
    nx = length(mesh_ux.nodes[1]); ny = length(mesh_ux.nodes[2]); nz = length(mesh_ux.nodes[3])
    nx_y = length(mesh_uy.nodes[1]); ny_y = length(mesh_uy.nodes[2]); nz_y = length(mesh_uy.nodes[3])
    nx_z = length(mesh_uz.nodes[1]); ny_z = length(mesh_uz.nodes[2]); nz_z = length(mesh_uz.nodes[3])
    @assert nx == nx_y == nx_z && ny == ny_y == ny_z && nz == nz_y == nz_z "Velocity meshes must share grid dimensions"

    LIx = LinearIndices((nx, ny, nz))
    LIy = LinearIndices((nx, ny, nz))
    LIz = LinearIndices((nx, ny, nz))

    iright = max(nx - 1, 1)
    jtop   = max(ny - 1, 1)
    kfront = max(nz - 1, 1)

    xs_x, ys_x, zs_x = mesh_ux.nodes
    xs_y, ys_y, zs_y = mesh_uy.nodes
    xs_z, ys_z, zs_z = mesh_uz.nodes

    eval_val(bc, x, y, z) = (bc isa Dirichlet) ? (bc.value isa Function ? eval_val_fn(bc.value, x, y, z) : bc.value) : nothing
    eval_val(bc, x, y, z, t) = (bc isa Dirichlet) ? (bc.value isa Function ? bc.value(x, y, z, t) : bc.value) : nothing

    function eval_val_fn(f, x, y, z)
        try
            return f(x, y, z)
        catch MethodError
            return f(x, y, z, 0.0)
        end
    end

    bcx_bottom = get(bc_ux.borders, :bottom, nothing)
    bcy_bottom = get(bc_uy.borders, :bottom, nothing)
    bcz_bottom = get(bc_uz.borders, :bottom, nothing)
    bcx_top    = get(bc_ux.borders, :top, nothing)
    bcy_top    = get(bc_uy.borders, :top, nothing)
    bcz_top    = get(bc_uz.borders, :top, nothing)

    bcx_left   = get(bc_ux.borders, :left, nothing)
    bcy_left   = get(bc_uy.borders, :left, nothing)
    bcz_left   = get(bc_uz.borders, :left, nothing)
    bcx_right  = get(bc_ux.borders, :right, nothing)
    bcy_right  = get(bc_uy.borders, :right, nothing)
    bcz_right  = get(bc_uz.borders, :right, nothing)

    bcx_back  = get(bc_ux.borders, :back, get(bc_ux.borders, :backward, nothing))
    bcy_back  = get(bc_uy.borders, :back, get(bc_uy.borders, :backward, nothing))
    bcz_back  = get(bc_uz.borders, :back, get(bc_uz.borders, :backward, nothing))
    bcx_front = get(bc_ux.borders, :front, get(bc_ux.borders, :forward, nothing))
    bcy_front = get(bc_uy.borders, :front, get(bc_uy.borders, :forward, nothing))
    bcz_front = get(bc_uz.borders, :front, get(bc_uz.borders, :forward, nothing))

    function enforce_dirichlet_delta(row, col, target)
        delta = Float64(target) - Float64(x_state[col])
        enforce_dirichlet!(A, rhs, row, col, delta)
    end

    function zero_gradient_delta(row, col_a, col_b)
        rhs_val = -(Float64(x_state[col_a]) - Float64(x_state[col_b]))
        enforce_zero_gradient!(A, rhs, row, col_a, col_b, rhs_val)
    end

    # Bottom/top (vary along x and z)
    for (jy, bcx, bcy, bcz) in ((1, bcx_bottom, bcy_bottom, bcz_bottom), (jtop, bcx_top, bcy_top, bcz_top))
        for i in 1:nx, k in 1:nz
            lix = LIx[i, jy, k]
            liy = LIy[i, jy, k]
            liz = LIz[i, jy, k]
            if bcx isa Dirichlet
                vx = t === nothing ? eval_val(bcx, xs_x[i], ys_x[jy], zs_x[k]) : eval_val(bcx, xs_x[i], ys_x[jy], zs_x[k], t)
                if vx !== nothing
                    enforce_dirichlet_delta(row_uωx_off + lix, uωx_off + lix, vx)
                    enforce_dirichlet_delta(row_uγx_off + lix, uγx_off + lix, vx)
                end
            elseif bcx isa Outflow || bcx isa Symmetry
                neighbor = jy == 1 ? LIx[i, min(jy+1, ny), k] : LIx[i, max(jy-1, 1), k]
                zero_gradient_delta(row_uωx_off + lix, uωx_off + lix, uωx_off + neighbor)
                zero_gradient_delta(row_uγx_off + lix, uγx_off + lix, uγx_off + neighbor)
            end
            if bcy isa Dirichlet
                vy = t === nothing ? eval_val(bcy, xs_y[i], ys_y[jy], zs_y[k]) : eval_val(bcy, xs_y[i], ys_y[jy], zs_y[k], t)
                if vy !== nothing
                    enforce_dirichlet_delta(row_uωy_off + liy, uωy_off + liy, vy)
                    enforce_dirichlet_delta(row_uγy_off + liy, uγy_off + liy, vy)
                end
            elseif bcy isa Symmetry
                enforce_dirichlet_delta(row_uωy_off + liy, uωy_off + liy, 0.0)
                enforce_dirichlet_delta(row_uγy_off + liy, uγy_off + liy, 0.0)
            elseif bcy isa Outflow
                neighbor = jy == 1 ? LIy[i, min(jy+1, ny), k] : LIy[i, max(jy-1, 1), k]
                zero_gradient_delta(row_uωy_off + liy, uωy_off + liy, uωy_off + neighbor)
                zero_gradient_delta(row_uγy_off + liy, uγy_off + liy, uγy_off + neighbor)
            end
            if bcz isa Dirichlet
                vz = t === nothing ? eval_val(bcz, xs_z[i], ys_z[jy], zs_z[k]) : eval_val(bcz, xs_z[i], ys_z[jy], zs_z[k], t)
                if vz !== nothing
                    enforce_dirichlet_delta(row_uωz_off + liz, uωz_off + liz, vz)
                    enforce_dirichlet_delta(row_uγz_off + liz, uγz_off + liz, vz)
                end
            elseif bcz isa Outflow || bcz isa Symmetry
                neighbor = jy == 1 ? LIz[i, min(jy+1, ny), k] : LIz[i, max(jy-1, 1), k]
                zero_gradient_delta(row_uωz_off + liz, uωz_off + liz, uωz_off + neighbor)
                zero_gradient_delta(row_uγz_off + liz, uγz_off + liz, uγz_off + neighbor)
            end
        end
    end

    # Left/right (vary along y,z)
    for (ix, bcx, bcy, bcz) in ((1, bcx_left, bcy_left, bcz_left), (iright, bcx_right, bcy_right, bcz_right))
        for j in 1:ny, k in 1:nz
            lix = LIx[ix, j, k]
            liy = LIy[ix, j, k]
            liz = LIz[ix, j, k]
            if bcx isa Dirichlet
                vx = t === nothing ? eval_val(bcx, xs_x[ix], ys_x[j], zs_x[k]) : eval_val(bcx, xs_x[ix], ys_x[j], zs_x[k], t)
                if vx !== nothing
                    enforce_dirichlet_delta(row_uωx_off + lix, uωx_off + lix, vx)
                    enforce_dirichlet_delta(row_uγx_off + lix, uγx_off + lix, vx)
                end
            elseif bcx isa Symmetry
                enforce_dirichlet_delta(row_uωx_off + lix, uωx_off + lix, 0.0)
                enforce_dirichlet_delta(row_uγx_off + lix, uγx_off + lix, 0.0)
            elseif bcx isa Outflow
                neighbor = ix == 1 ? LIx[min(ix+1, nx), j, k] : LIx[max(ix-1, 1), j, k]
                zero_gradient_delta(row_uωx_off + lix, uωx_off + lix, uωx_off + neighbor)
                zero_gradient_delta(row_uγx_off + lix, uγx_off + lix, uγx_off + neighbor)
            end
            if bcy isa Dirichlet
                vy = t === nothing ? eval_val(bcy, xs_y[ix], ys_y[j], zs_y[k]) : eval_val(bcy, xs_y[ix], ys_y[j], zs_y[k], t)
                if vy !== nothing
                    enforce_dirichlet_delta(row_uωy_off + liy, uωy_off + liy, vy)
                    enforce_dirichlet_delta(row_uγy_off + liy, uγy_off + liy, vy)
                end
            elseif bcy isa Symmetry
                neighbor = ix == 1 ? LIy[min(ix+1, nx), j, k] : LIy[max(ix-1, 1), j, k]
                zero_gradient_delta(row_uωy_off + liy, uωy_off + liy, uωy_off + neighbor)
                zero_gradient_delta(row_uγy_off + liy, uγy_off + liy, uγy_off + neighbor)
            elseif bcy isa Outflow
                neighbor = ix == 1 ? LIy[min(ix+1, nx), j, k] : LIy[max(ix-1, 1), j, k]
                zero_gradient_delta(row_uωy_off + liy, uωy_off + liy, uωy_off + neighbor)
                zero_gradient_delta(row_uγy_off + liy, uγy_off + liy, uγy_off + neighbor)
            end
            if bcz isa Dirichlet
                vz = t === nothing ? eval_val(bcz, xs_z[ix], ys_z[j], zs_z[k]) : eval_val(bcz, xs_z[ix], ys_z[j], zs_z[k], t)
                if vz !== nothing
                    enforce_dirichlet_delta(row_uωz_off + liz, uωz_off + liz, vz)
                    enforce_dirichlet_delta(row_uγz_off + liz, uγz_off + liz, vz)
                end
            elseif bcz isa Outflow || bcz isa Symmetry
                neighbor = ix == 1 ? LIz[min(ix+1, nx), j, k] : LIz[max(ix-1, 1), j, k]
                zero_gradient_delta(row_uωz_off + liz, uωz_off + liz, uωz_off + neighbor)
                zero_gradient_delta(row_uγz_off + liz, uγz_off + liz, uγz_off + neighbor)
            end
        end
    end

    # Back/front (vary along x,y)
    for (kidx, bcx, bcy, bcz) in ((1, bcx_back, bcy_back, bcz_back), (kfront, bcx_front, bcy_front, bcz_front))
        for i in 1:nx, j in 1:ny
            lix = LIx[i, j, kidx]
            liy = LIy[i, j, kidx]
            liz = LIz[i, j, kidx]
            if bcx isa Dirichlet
                vx = t === nothing ? eval_val(bcx, xs_x[i], ys_x[j], zs_x[kidx]) : eval_val(bcx, xs_x[i], ys_x[j], zs_x[kidx], t)
                if vx !== nothing
                    enforce_dirichlet_delta(row_uωx_off + lix, uωx_off + lix, vx)
                    enforce_dirichlet_delta(row_uγx_off + lix, uγx_off + lix, vx)
                end
            elseif bcx isa Outflow || bcx isa Symmetry
                neighbor = kidx == 1 ? LIx[i, j, min(kidx+1, nz)] : LIx[i, j, max(kidx-1, 1)]
                zero_gradient_delta(row_uωx_off + lix, uωx_off + lix, uωx_off + neighbor)
                zero_gradient_delta(row_uγx_off + lix, uγx_off + lix, uγx_off + neighbor)
            end
            if bcy isa Dirichlet
                vy = t === nothing ? eval_val(bcy, xs_y[i], ys_y[j], zs_y[kidx]) : eval_val(bcy, xs_y[i], ys_y[j], zs_y[kidx], t)
                if vy !== nothing
                    enforce_dirichlet_delta(row_uωy_off + liy, uωy_off + liy, vy)
                    enforce_dirichlet_delta(row_uγy_off + liy, uγy_off + liy, vy)
                end
            elseif bcy isa Outflow || bcy isa Symmetry
                neighbor = kidx == 1 ? LIy[i, j, min(kidx+1, nz)] : LIy[i, j, max(kidx-1, 1)]
                zero_gradient_delta(row_uωy_off + liy, uωy_off + liy, uωy_off + neighbor)
                zero_gradient_delta(row_uγy_off + liy, uγy_off + liy, uγy_off + neighbor)
            end
            if bcz isa Dirichlet
                vz = t === nothing ? eval_val(bcz, xs_z[i], ys_z[j], zs_z[kidx]) : eval_val(bcz, xs_z[i], ys_z[j], zs_z[kidx], t)
                if vz !== nothing
                    enforce_dirichlet_delta(row_uωz_off + liz, uωz_off + liz, vz)
                    enforce_dirichlet_delta(row_uγz_off + liz, uγz_off + liz, vz)
                end
            elseif bcz isa Symmetry
                enforce_dirichlet_delta(row_uωz_off + liz, uωz_off + liz, 0.0)
                enforce_dirichlet_delta(row_uγz_off + liz, uγz_off + liz, 0.0)
            elseif bcz isa Outflow
                neighbor = kidx == 1 ? LIz[i, j, min(kidx+1, nz)] : LIz[i, j, max(kidx-1, 1)]
                zero_gradient_delta(row_uωz_off + liz, uωz_off + liz, uωz_off + neighbor)
                zero_gradient_delta(row_uγz_off + liz, uγz_off + liz, uγz_off + neighbor)
            end
        end
    end

    return nothing
end

function apply_pressure_gauge_newton!(A::SparseMatrixCSC{Float64,Int}, rhs::Vector{Float64},
                                      x_state::AbstractVector{<:Real},
                                      gauge::AbstractPressureGauge,
                                      _mesh_p::AbstractMesh,
                                      capacity_p::AbstractCapacity;
                                      p_offset::Int, np::Int,
                                      row_start::Int,
                                      t::Union{Nothing,Float64}=nothing)
    diagV = diag(capacity_p.V)
    if gauge isa PinPressureGauge
        idx = gauge.index
        if idx === nothing
            tol = 1e-12
            idx = findfirst(x -> x > tol, diagV)
            idx === nothing && (idx = 1)
        end
        1 ≤ idx ≤ np || error("PinPressureGauge index $(idx) outside valid range 1:$(np)")
        row = row_start + idx - 1
        col = p_offset + idx
        delta = -Float64(x_state[col])
        enforce_dirichlet!(A, rhs, row, col, delta)
    elseif gauge isa MeanPressureGauge
        weights = copy(diagV)
        if isempty(weights)
            error("MeanPressureGauge requires at least one pressure DOF")
        end
        if all(isapprox.(weights, 0.0; atol=1e-12))
            weights .= 1.0
        end
        total = sum(weights)
        total == 0.0 && (weights .= 1.0; total = sum(weights))
        weights ./= total
        row = row_start
        A[row, :] .= 0.0
        for j in 1:np
            A[row, p_offset + j] = weights[j]
        end
        p_segment = view(x_state, p_offset+1:p_offset+np)
        rhs[row] = -dot(weights, p_segment)
    else
        error("Unknown pressure gauge type $(typeof(gauge))")
    end
    return nothing
end

function compute_navierstokes_force_diagnostics(s::NavierStokesMono)
    N = length(s.fluid.operator_u)
    data = N == 1 ? navierstokes1D_blocks(s) :
           N == 2 ? navierstokes2D_blocks(s) :
           error("Force diagnostics currently implemented for 1D or 2D Navier–Stokes (got N=$(N)).")

    nu_components = data.nu_components
    total_velocity_dofs = 2 * sum(nu_components)
    np = data.np
    length(s.x) == total_velocity_dofs + np || error("State vector length mismatch: expected $(total_velocity_dofs + np), got $(length(s.x)).")
    pω = Vector{Float64}(view(s.x, total_velocity_dofs + 1:total_velocity_dofs + np))

    grads = Vector{SparseMatrixCSC{Float64,Int}}(undef, N)
    Vmats = Vector{SparseMatrixCSC{Float64,Int}}(undef, N)
    if N == 1
        grads[1] = data.grad
        Vmats[1] = data.V
    elseif N == 2
        grads[1] = data.grad_x
        grads[2] = data.grad_y
        Vmats[1] = data.Vx
        Vmats[2] = data.Vy
    end

    g_p = Vector{Vector{Float64}}(undef, N)
    L_u = Vector{Vector{Float64}}(undef, N)
    pressure_part = Vector{Vector{Float64}}(undef, N)
    viscous_part = Vector{Vector{Float64}}(undef, N)
    force_density = Vector{Vector{Float64}}(undef, N)

    integrated_pressure = zeros(Float64, N)
    integrated_viscous = zeros(Float64, N)
    integrated_force = zeros(Float64, N)

    offset = 0
    for α in 1:N
        nu = nu_components[α]
        uω = Vector{Float64}(view(s.x, offset + 1:offset + nu))
        uγ = Vector{Float64}(view(s.x, offset + nu + 1:offset + 2nu))
        offset += 2nu

        grad = grads[α]
        gp_vec = -Vector{Float64}(grad * pω)
        pressure_vec = -gp_vec

        op = s.fluid.operator_u[α]
        G_u = Vector{Float64}(op.G * uω)
        if size(op.H, 2) == 0
            H_u = zeros(Float64, size(op.G, 1))
        else
            H_u = Vector{Float64}(op.H * uγ)
        end
        W_dagger = op.Wꜝ
        mixed = Vector{Float64}(W_dagger * (G_u + H_u))
        Lu_vec = Vector{Float64}(op.G' * mixed)
        Iμ = build_I_D(op, s.fluid.μ, s.fluid.capacity_u[α])
        visc_vec = Vector{Float64}(Iμ * Lu_vec)

        force_vec = pressure_vec .+ visc_vec

        g_p[α] = gp_vec
        L_u[α] = Lu_vec
        pressure_part[α] = pressure_vec
        viscous_part[α] = visc_vec
        force_density[α] = force_vec

        V = Vmats[α]
        integrated_pressure[α] = sum(Vector{Float64}(pressure_vec))
        integrated_viscous[α] = sum(Vector{Float64}(visc_vec))
        integrated_force[α] = sum(Vector{Float64}(force_vec))
    end

    return (; g_p=Tuple(g_p),
            L_u=Tuple(L_u),
            pressure=Tuple(pressure_part),
            viscous=Tuple(viscous_part),
            force_density=Tuple(force_density),
            integrated_pressure=integrated_pressure,
            integrated_viscous=integrated_viscous,
            integrated_force=integrated_force)
end

function navierstokes_reaction_force_components(force_data::NamedTuple; acting_on::Symbol=:body)
    allowed = (:body, :fluid)
    acting_on ∈ allowed || error("acting_on must be one of $(allowed), got $(acting_on).")
    sign = acting_on === :body ? -1.0 : 1.0
    forces = force_data.integrated_force
    return sign .* forces
end

function drag_lift_coefficients(force_data::NamedTuple; ρ::Real, U_ref::Real, length_ref::Real, acting_on::Symbol=:body)
    forces = navierstokes_reaction_force_components(force_data; acting_on=acting_on)
    length(forces) ≥ 2 || error("Drag/Lift coefficients require at least 2 components (found $(length(forces))).")
    denom = ρ * U_ref^2 * length_ref
    denom == 0 && error("Reference denominator for drag/lift coefficients is zero.")
    drag = forces[1]
    lift = forces[2]
    Cd = 2 * drag / denom
    Cl = 2 * lift / denom
    return (; drag=drag,
            lift=lift,
            Cd=Cd,
            Cl=Cl,
            reference=(ρ=ρ, U_ref=U_ref, length=length_ref))
end

function pressure_trace_on_cut(s::NavierStokesMono; center::NTuple{2,Float64}, tol::Float64=1e-10, sort_by_angle::Bool=true)
    length(s.fluid.operator_u) == 2 || error("pressure_trace_on_cut currently implemented for 2D configurations.")
    cap = s.fluid.capacity_p
    np = prod(s.fluid.operator_p.size)
    nu_components = ntuple(i -> prod(s.fluid.operator_u[i].size), 2)
    total_velocity_dofs = 2 * sum(nu_components)
    pω = Vector{Float64}(view(s.x, total_velocity_dofs + 1:total_velocity_dofs + np))

    ops_u = s.fluid.operator_u
    caps_u = s.fluid.capacity_u

    nu_x, nu_y = nu_components
    uωx = Vector{Float64}(view(s.x, 1:nu_x))
    uωy = Vector{Float64}(view(s.x, 2nu_x + 1:2nu_x + nu_y))

    mesh_ux = caps_u[1].mesh
    mesh_uy = caps_u[2].mesh
    xs_ux, ys_ux = mesh_ux.nodes
    xs_uy, ys_uy = mesh_uy.nodes

    Ux = reshape(uωx, (length(xs_ux), length(ys_ux)))
    Uy = reshape(uωy, (length(xs_uy), length(ys_uy)))

    dx_ux = length(xs_ux) > 1 ? minimum(diff(xs_ux)) : 1.0
    dy_ux = length(ys_ux) > 1 ? minimum(diff(ys_ux)) : 1.0
    dx_uy = length(xs_uy) > 1 ? minimum(diff(xs_uy)) : 1.0
    dy_uy = length(ys_uy) > 1 ? minimum(diff(ys_uy)) : 1.0
    δx = max(eps(), minimum((dx_ux, dx_uy)))
    δy = max(eps(), minimum((dy_ux, dy_uy)))

    μ = s.fluid.μ
    body = cap.body
    body_eval = if applicable(body, 0.0, 0.0)
        (x::Float64, y::Float64) -> body(x, y)
    elseif applicable(body, 0.0, 0.0, 0.0)
        (x::Float64, y::Float64) -> body(x, y, 0.0)
    else
        error("capacity body does not provide a 2D evaluation signature.")
    end
    identity2 = SMatrix{2,2,Float64}(1.0, 0.0,
                                     0.0, 1.0)

    function bilinear_interpolate(xs::Vector{Float64}, ys::Vector{Float64}, field::AbstractMatrix, x::Float64, y::Float64)
        x_clamped = clamp(x, xs[1], xs[end])
        y_clamped = clamp(y, ys[1], ys[end])

        ix_hi = searchsortedfirst(xs, x_clamped)
        if ix_hi <= 1
            ix_low, ix_hi = 1, min(length(xs), 2)
        elseif ix_hi > length(xs)
            ix_low, ix_hi = length(xs)-1, length(xs)
        elseif xs[ix_hi] == x_clamped
            ix_low = max(ix_hi - 1, 1)
        else
            ix_low = max(ix_hi - 1, 1)
        end
        iy_hi = searchsortedfirst(ys, y_clamped)
        if iy_hi <= 1
            iy_low, iy_hi = 1, min(length(ys), 2)
        elseif iy_hi > length(ys)
            iy_low, iy_hi = length(ys)-1, length(ys)
        elseif ys[iy_hi] == y_clamped
            iy_low = max(iy_hi - 1, 1)
        else
            iy_low = max(iy_hi - 1, 1)
        end

        x1, x2 = xs[ix_low], xs[ix_hi]
        y1, y2 = ys[iy_low], ys[iy_hi]
        tx = x2 ≈ x1 ? 0.0 : (x_clamped - x1) / (x2 - x1)
        ty = y2 ≈ y1 ? 0.0 : (y_clamped - y1) / (y2 - y1)

        f11 = field[ix_low, iy_low]
        f21 = field[ix_hi, iy_low]
        f12 = field[ix_low, iy_hi]
        f22 = field[ix_hi, iy_hi]

        return (1 - tx) * (1 - ty) * f11 +
               tx * (1 - ty) * f21 +
               (1 - tx) * ty * f12 +
               tx * ty * f22
    end

    function velocity_gradient(x::Float64, y::Float64)
        ux_x_plus = bilinear_interpolate(xs_ux, ys_ux, Ux, x + δx, y)
        ux_x_minus = bilinear_interpolate(xs_ux, ys_ux, Ux, x - δx, y)
        dux_dx = (ux_x_plus - ux_x_minus) / (2δx)

        ux_y_plus = bilinear_interpolate(xs_ux, ys_ux, Ux, x, y + δy)
        ux_y_minus = bilinear_interpolate(xs_ux, ys_ux, Ux, x, y - δy)
        dux_dy = (ux_y_plus - ux_y_minus) / (2δy)

        uy_x_plus = bilinear_interpolate(xs_uy, ys_uy, Uy, x + δx, y)
        uy_x_minus = bilinear_interpolate(xs_uy, ys_uy, Uy, x - δx, y)
        duy_dx = (uy_x_plus - uy_x_minus) / (2δx)

        uy_y_plus = bilinear_interpolate(xs_uy, ys_uy, Uy, x, y + δy)
        uy_y_minus = bilinear_interpolate(xs_uy, ys_uy, Uy, x, y - δy)
        duy_dy = (uy_y_plus - uy_y_minus) / (2δy)

        return SMatrix{2,2,Float64}(dux_dx, duy_dx,
                                     dux_dy, duy_dy)
    end

    function interface_normal(x::Float64, y::Float64)
        δn = 0.5 * min(δx, δy)
        δn = max(δn, sqrt(eps()))
        fx_plus = body_eval(x + δn, y)
        fx_minus = body_eval(x - δn, y)
        fy_plus = body_eval(x, y + δn)
        fy_minus = body_eval(x, y - δn)
        grad = SVector((fx_plus - fx_minus) / (2δn),
                       (fy_plus - fy_minus) / (2δn))
        norm_grad = norm(grad)
        norm_grad == 0.0 && return nothing
        n_candidate = grad / norm_grad
        probe = body_eval(x + 1e-4 * n_candidate[1], y + 1e-4 * n_candidate[2])
        if probe > 0
            n_candidate = -n_candidate
        end
        return n_candidate
    end

    Γ_diag = diag(cap.Γ)
    Cγ = cap.C_γ
    θ = Float64[]
    p_vals = Float64[]
    weights = Float64[]
    coords = Vector{Tuple{Float64,Float64}}()
    normals = Vector{SVector{2,Float64}}()
    traction_vectors = Vector{SVector{2,Float64}}()
    integrated_forces = Vector{SVector{2,Float64}}()
    pressure_from_stress = Float64[]

    for (i, γ) in enumerate(Γ_diag)
        if γ ≤ tol || i > length(pω)
            continue
        end
        centroid = i ≤ length(Cγ) ? Cγ[i] : nothing
        if centroid === nothing || all(iszero, centroid)
            continue
        end
        angle = atan(centroid[2] - center[2], centroid[1] - center[1])
        push!(θ, angle)
        push!(p_vals, pω[i])
        push!(weights, γ)
        push!(coords, (centroid[1], centroid[2]))

        normal_vec = interface_normal(centroid[1], centroid[2])
        if normal_vec === nothing
            push!(normals, SVector(0.0, 0.0))
            push!(traction_vectors, SVector(0.0, 0.0))
            push!(integrated_forces, SVector(0.0, 0.0))
            push!(pressure_from_stress, pω[i])
            continue
        end

        grad_u = velocity_gradient(centroid[1], centroid[2])
        sym_grad = grad_u + grad_u'
        σ = μ * sym_grad - pω[i] * identity2
        traction = σ * normal_vec
        force_vec = traction * γ
        push!(normals, normal_vec)
        push!(traction_vectors, traction)
        push!(integrated_forces, force_vec)
        push!(pressure_from_stress, -dot(traction, normal_vec))
    end

    if sort_by_angle
        order = sortperm(θ)
        θ = θ[order]
        p_vals = p_vals[order]
        weights = weights[order]
        coords = coords[order]
        normals = normals[order]
        traction_vectors = traction_vectors[order]
        integrated_forces = integrated_forces[order]
        pressure_from_stress = pressure_from_stress[order]
    end

    total_force = SVector(0.0, 0.0)
    for f in integrated_forces
        total_force = total_force + f
    end

    return (; θ=θ,
            p=p_vals,
            weights=weights,
            coords=coords,
            normals=normals,
            traction=traction_vectors,
            integrated_force=integrated_forces,
            p_from_stress=pressure_from_stress,
            total_force=total_force)
end
