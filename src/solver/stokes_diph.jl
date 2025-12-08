"""
    StokesDiph

Prototype two-phase Stokes solver with per-phase fluids on staggered grids.
Unknown ordering (2D): `[u₁xᵠ, u₁xᵞ, u₁yᵠ, u₁yᵞ, p₁, u₂xᵠ, u₂xᵞ, u₂yᵠ, u₂yᵞ, p₂]`
Unknown ordering (3D) follows the same pattern, adding z components; 1D is the
degenerate single-component variant.
Rows enforce momentum, velocity continuity at the interface, traction continuity,
and incompressibility for each phase.
"""
mutable struct StokesDiph{N}
    fluid_a::Fluid{N}
    fluid_b::Fluid{N}
    bc_u_a::NTuple{N, BorderConditions}
    bc_u_b::NTuple{N, BorderConditions}
    ic::NTuple{N, InterfaceConditions}
    pressure_gauge::NTuple{2, AbstractPressureGauge}
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    x::Vector{Float64}
    ch::Vector{Any}
end

@inline function normalize_pressure_gauges(g::AbstractPressureGauge)
    return (normalize_pressure_gauge(g), normalize_pressure_gauge(g))
end
@inline function normalize_pressure_gauges(gt::NTuple{2,AbstractPressureGauge})
    return (normalize_pressure_gauge(gt[1]), normalize_pressure_gauge(gt[2]))
end

function StokesDiph(fluid_a::Fluid{N},
                    fluid_b::Fluid{N},
                    bc_u_a::NTuple{N,BorderConditions},
                    bc_u_b::NTuple{N,BorderConditions},
                    ic::NTuple{N,InterfaceConditions};
                    pressure_gauge::Union{AbstractPressureGauge,NTuple{2,AbstractPressureGauge}}=DEFAULT_PRESSURE_GAUGE,
                    x0=zeros(0)) where {N}
    (1 <= N <= 3) || error("StokesDiph currently implemented for 1D/2D/3D (received N=$(N)).")

    gauge_tuple = pressure_gauge isa AbstractPressureGauge ? normalize_pressure_gauges(pressure_gauge) :
                  normalize_pressure_gauges(pressure_gauge)

    nu_a = ntuple(i -> prod(fluid_a.operator_u[i].size), N)
    nu_b = ntuple(i -> prod(fluid_b.operator_u[i].size), N)
    np_a = prod(fluid_a.operator_p.size)
    np_b = prod(fluid_b.operator_p.size)

    Ntot = 2 * (sum(nu_a) + sum(nu_b)) + np_a + np_b
    x_init = length(x0) == Ntot ? copy(x0) : zeros(Ntot)

    A = spzeros(Float64, Ntot, Ntot)
    b = zeros(Ntot)

    s = StokesDiph{N}(fluid_a, fluid_b, bc_u_a, bc_u_b, ic, gauge_tuple,
                      A, b, x_init, Any[])
    assemble_stokes_diph!(s)
    return s
end

# Legacy signature accepting an unused cut/interface argument
function StokesDiph(fluid_a::Fluid{N},
                    fluid_b::Fluid{N},
                    bc_u_a::NTuple{N,BorderConditions},
                    bc_u_b::NTuple{N,BorderConditions},
                    ic::NTuple{N,InterfaceConditions},
                    pressure_gauge::Union{AbstractPressureGauge,NTuple{2,AbstractPressureGauge}},
                    _bc_cut;
                    x0=zeros(0)) where {N}
    return StokesDiph(fluid_a, fluid_b, bc_u_a, bc_u_b, ic; pressure_gauge=pressure_gauge, x0=x0)
end

# Convenience: positional pressure gauge
function StokesDiph(fluid_a::Fluid{N},
                    fluid_b::Fluid{N},
                    bc_u_a::NTuple{N,BorderConditions},
                    bc_u_b::NTuple{N,BorderConditions},
                    ic::NTuple{N,InterfaceConditions},
                    pressure_gauge::Union{AbstractPressureGauge,NTuple{2,AbstractPressureGauge}};
                    x0=zeros(0)) where {N}
    return StokesDiph(fluid_a, fluid_b, bc_u_a, bc_u_b, ic; pressure_gauge=pressure_gauge, x0=x0)
end

# Phase-wise operators -------------------------------------------------------

function stokes2D_phase_blocks(fluid::Fluid{2})
    ops_u = fluid.operator_u
    caps_u = fluid.capacity_u
    op_p = fluid.operator_p
    cap_p = fluid.capacity_p

    nu_x = prod(ops_u[1].size)
    nu_y = prod(ops_u[2].size)
    np = prod(op_p.size)

    μ = fluid.μ
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

    traction_x_ω = (Iμ_x * ops_u[1].H' * (ops_u[1].Wꜝ * ops_u[1].G))
    traction_x_γ = (Iμ_x * ops_u[1].H' * (ops_u[1].Wꜝ * ops_u[1].H))
    traction_y_ω = (Iμ_y * ops_u[2].H' * (ops_u[2].Wꜝ * ops_u[2].G))
    traction_y_γ = (Iμ_y * ops_u[2].H' * (ops_u[2].Wꜝ * ops_u[2].H))

    return (; nu_x, nu_y, np,
            op_ux = ops_u[1], op_uy = ops_u[2], op_p, cap_px = caps_u[1], cap_py = caps_u[2], cap_p,
            visc_x_ω, visc_x_γ, visc_y_ω, visc_y_γ,
            grad_x, grad_y,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ,
            traction_x_ω, traction_x_γ, traction_y_ω, traction_y_γ,
            Vx = ops_u[1].V, Vy = ops_u[2].V)
end

function stokes1D_phase_blocks(fluid::Fluid{1})
    op_u = fluid.operator_u[1]
    cap_u = fluid.capacity_u[1]
    op_p = fluid.operator_p
    cap_p = fluid.capacity_p

    nu = prod(op_u.size)
    np = prod(op_p.size)

    μ = fluid.μ
    Iμ = build_I_D(op_u, μ, cap_u)

    WG_G = op_u.Wꜝ * op_u.G
    WG_H = op_u.Wꜝ * op_u.H
    visc_x_ω = (Iμ * op_u.G' * WG_G)
    visc_x_γ = (Iμ * op_u.G' * WG_H)

    grad_x = -(op_p.G + op_p.H)
    @assert size(grad_x, 1) == nu

    Gp = op_p.G; Hp = op_p.H
    Gp_x = Gp[1:nu, :]; Hp_x = Hp[1:nu, :]
    div_x_ω = -(Gp_x' + Hp_x')
    div_x_γ =  (Hp_x')

    traction_x_ω = (Iμ * op_u.H' * WG_G)
    traction_x_γ = (Iμ * op_u.H' * WG_H)

    return (; nu, np,
            op_ux = op_u, op_p, cap_px = cap_u, cap_p,
            visc_x_ω, visc_x_γ,
            grad_x,
            div_x_ω, div_x_γ,
            traction_x_ω, traction_x_γ,
            Vx = op_u.V)
end

function stokes3D_phase_blocks(fluid::Fluid{3})
    ops_u = fluid.operator_u
    caps_u = fluid.capacity_u
    op_p = fluid.operator_p
    cap_p = fluid.capacity_p

    nu_x = prod(ops_u[1].size)
    nu_y = prod(ops_u[2].size)
    nu_z = prod(ops_u[3].size)
    np = prod(op_p.size)

    μ = fluid.μ
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
    @assert total_grad_rows == nu_x + nu_y + nu_z

    x_rows = 1:nu_x
    y_rows = nu_x+1:nu_x+nu_y
    z_rows = nu_x+nu_y+1:nu_x+nu_y+nu_z
    grad_x = -grad_full[x_rows, :]
    grad_y = -grad_full[y_rows, :]
    grad_z = -grad_full[z_rows, :]

    Gp = op_p.G; Hp = op_p.H
    Gp_x = Gp[x_rows, :]; Hp_x = Hp[x_rows, :]
    Gp_y = Gp[y_rows, :]; Hp_y = Hp[y_rows, :]
    Gp_z = Gp[z_rows, :]; Hp_z = Hp[z_rows, :]
    div_x_ω = -(Gp_x' + Hp_x'); div_x_γ = Hp_x'
    div_y_ω = -(Gp_y' + Hp_y'); div_y_γ = Hp_y'
    div_z_ω = -(Gp_z' + Hp_z'); div_z_γ = Hp_z'

    traction_x_ω = (Iμ_x * ops_u[1].H' * WGx_Gx)
    traction_x_γ = (Iμ_x * ops_u[1].H' * WGx_Hx)
    traction_y_ω = (Iμ_y * ops_u[2].H' * WGy_Gy)
    traction_y_γ = (Iμ_y * ops_u[2].H' * WGy_Hy)
    traction_z_ω = (Iμ_z * ops_u[3].H' * WGz_Gz)
    traction_z_γ = (Iμ_z * ops_u[3].H' * WGz_Hz)

    return (; nu_x, nu_y, nu_z, np,
            op_ux = ops_u[1], op_uy = ops_u[2], op_uz = ops_u[3], op_p,
            cap_px = caps_u[1], cap_py = caps_u[2], cap_pz = caps_u[3], cap_p,
            visc_x_ω, visc_x_γ, visc_y_ω, visc_y_γ, visc_z_ω, visc_z_γ,
            grad_x, grad_y, grad_z,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ, div_z_ω, div_z_γ,
            traction_x_ω, traction_x_γ, traction_y_ω, traction_y_γ, traction_z_ω, traction_z_γ,
            Vx = ops_u[1].V, Vy = ops_u[2].V, Vz = ops_u[3].V)
end

# Assembly -------------------------------------------------------------------

function assemble_stokes_diph!(s::StokesDiph)
    N = length(s.fluid_a.operator_u)
    N == length(s.fluid_b.operator_u) || error("Phase operator dimensionality mismatch.")
    if N == 1
        return assemble_stokes1D_diph!(s)
    elseif N == 2
        return assemble_stokes2D_diph!(s)
    elseif N == 3
        return assemble_stokes3D_diph!(s)
    else
        error("assemble_stokes_diph! is implemented for 1D/2D/3D (N=$(N)).")
    end
end

function assemble_stokes2D_diph!(s::StokesDiph)
    data_a = stokes2D_phase_blocks(s.fluid_a)
    data_b = stokes2D_phase_blocks(s.fluid_b)

    nu_ax = data_a.nu_x; nu_ay = data_a.nu_y
    nu_bx = data_b.nu_x; nu_by = data_b.nu_y
    @assert nu_ax == nu_bx "Velocity x-DOF counts must match between phases"
    @assert nu_ay == nu_by "Velocity y-DOF counts must match between phases"
    np_a = data_a.np; np_b = data_b.np

    # Column offsets (uωx,uγx,uωy,uγy,p) for each phase
    off_a_uωx = 0
    off_a_uγx = nu_ax
    off_a_uωy = 2 * nu_ax
    off_a_uγy = 2 * nu_ax + nu_ay
    off_a_p   = 2 * (nu_ax + nu_ay)

    off_b_uωx = off_a_p + np_a
    off_b_uγx = off_b_uωx + nu_bx
    off_b_uωy = off_b_uγx + nu_bx
    off_b_uγy = off_b_uωy + nu_by
    off_b_p   = off_b_uγy + nu_by

    # Row offsets
    row_a_momx   = 0
    row_cont_x   = nu_ax
    row_a_momy   = 2 * nu_ax
    row_cont_y   = 2 * nu_ax + nu_ay
    row_a_div    = 2 * (nu_ax + nu_ay)
    row_b_momx   = row_a_div + np_a
    n_tx         = size(data_a.traction_x_ω, 1)
    n_ty         = size(data_a.traction_y_ω, 1)
    row_trac_x   = row_b_momx + nu_bx
    row_b_momy   = row_trac_x + n_tx
    row_trac_y   = row_b_momy + nu_by
    row_b_div    = row_trac_y + n_ty

    rows = row_b_div + np_b
    cols = off_b_p + np_b
    A = spzeros(Float64, rows, cols)

    # Phase 1 momentum
    A[row_a_momx+1:row_a_momx+nu_ax, off_a_uωx+1:off_a_uωx+nu_ax] = data_a.visc_x_ω
    A[row_a_momx+1:row_a_momx+nu_ax, off_a_uγx+1:off_a_uγx+nu_ax] = data_a.visc_x_γ
    A[row_a_momx+1:row_a_momx+nu_ax, off_a_p+1:off_a_p+np_a]       = data_a.grad_x

    A[row_a_momy+1:row_a_momy+nu_ay, off_a_uωy+1:off_a_uωy+nu_ay] = data_a.visc_y_ω
    A[row_a_momy+1:row_a_momy+nu_ay, off_a_uγy+1:off_a_uγy+nu_ay] = data_a.visc_y_γ
    A[row_a_momy+1:row_a_momy+nu_ay, off_a_p+1:off_a_p+np_a]      = data_a.grad_y

    # Interface scalar jump (defaults to continuity) for uγ
    jump_x = s.ic[1].scalar
    jump_y = s.ic[2].scalar
    Iα1x = jump_x.α₁ * I(nu_ax)
    Iα2x = jump_x.α₂ * I(nu_ax)
    Iα1y = jump_y.α₁ * I(nu_ay)
    Iα2y = jump_y.α₂ * I(nu_ay)

    A[row_cont_x+1:row_cont_x+nu_ax, off_a_uγx+1:off_a_uγx+nu_ax] = Iα1x
    A[row_cont_x+1:row_cont_x+nu_ax, off_b_uγx+1:off_b_uγx+nu_bx] .= -Iα2x

    A[row_cont_y+1:row_cont_y+nu_ay, off_a_uγy+1:off_a_uγy+nu_ay] = Iα1y
    A[row_cont_y+1:row_cont_y+nu_ay, off_b_uγy+1:off_b_uγy+nu_by] .= -Iα2y

    # Phase 1 continuity
    con1_rows = row_a_div+1:row_a_div+np_a
    A[con1_rows, off_a_uωx+1:off_a_uωx+nu_ax] = data_a.div_x_ω
    A[con1_rows, off_a_uγx+1:off_a_uγx+nu_ax] = data_a.div_x_γ
    A[con1_rows, off_a_uωy+1:off_a_uωy+nu_ay] = data_a.div_y_ω
    A[con1_rows, off_a_uγy+1:off_a_uγy+nu_ay] = data_a.div_y_γ

    # Phase 2 momentum
    A[row_b_momx+1:row_b_momx+nu_bx, off_b_uωx+1:off_b_uωx+nu_bx] = data_b.visc_x_ω
    A[row_b_momx+1:row_b_momx+nu_bx, off_b_uγx+1:off_b_uγx+nu_bx] = data_b.visc_x_γ
    A[row_b_momx+1:row_b_momx+nu_bx, off_b_p+1:off_b_p+np_b]      = data_b.grad_x

    A[row_b_momy+1:row_b_momy+nu_by, off_b_uωy+1:off_b_uωy+nu_by] = data_b.visc_y_ω
    A[row_b_momy+1:row_b_momy+nu_by, off_b_uγy+1:off_b_uγy+nu_by] = data_b.visc_y_γ
    A[row_b_momy+1:row_b_momy+nu_by, off_b_p+1:off_b_p+np_b]      = data_b.grad_y

    # Traction jump (flux) β₁ T¹ + β₂ T² = h
    flux_x = s.ic[1].flux
    flux_y = s.ic[2].flux
    Iβ1x = flux_x.β₁ * I(n_tx)
    Iβ2x = flux_x.β₂ * I(n_tx)
    Iβ1y = flux_y.β₁ * I(n_ty)
    Iβ2y = flux_y.β₂ * I(n_ty)

    A[row_trac_x+1:row_trac_x+n_tx, off_a_uωx+1:off_a_uωx+nu_ax] = -Iβ1x * data_a.traction_x_ω
    A[row_trac_x+1:row_trac_x+n_tx, off_a_uγx+1:off_a_uγx+nu_ax] = -Iβ1x * data_a.traction_x_γ
    A[row_trac_x+1:row_trac_x+n_tx, off_a_p+1:off_a_p+np_a]      .-= Iβ1x * data_a.cap_px.Γ
    A[row_trac_x+1:row_trac_x+n_tx, off_b_uωx+1:off_b_uωx+nu_bx] .+= Iβ2x * data_b.traction_x_ω
    A[row_trac_x+1:row_trac_x+n_tx, off_b_uγx+1:off_b_uγx+nu_bx] .+= Iβ2x * data_b.traction_x_γ
    A[row_trac_x+1:row_trac_x+n_tx, off_b_p+1:off_b_p+np_b]      .+= Iβ2x * data_b.cap_px.Γ

    A[row_trac_y+1:row_trac_y+n_ty, off_a_uωy+1:off_a_uωy+nu_ay] = -Iβ1y * data_a.traction_y_ω
    A[row_trac_y+1:row_trac_y+n_ty, off_a_uγy+1:off_a_uγy+nu_ay] = -Iβ1y * data_a.traction_y_γ
    A[row_trac_y+1:row_trac_y+n_ty, off_a_p+1:off_a_p+np_a]      .-= Iβ1y * data_a.cap_py.Γ
    A[row_trac_y+1:row_trac_y+n_ty, off_b_uωy+1:off_b_uωy+nu_by] .+= Iβ2y * data_b.traction_y_ω
    A[row_trac_y+1:row_trac_y+n_ty, off_b_uγy+1:off_b_uγy+nu_by] .+= Iβ2y * data_b.traction_y_γ
    A[row_trac_y+1:row_trac_y+n_ty, off_b_p+1:off_b_p+np_b]      .+= Iβ2y * data_b.cap_py.Γ

    # Phase 2 continuity
    con2_rows = row_b_div+1:row_b_div+np_b
    A[con2_rows, off_b_uωx+1:off_b_uωx+nu_bx] = data_b.div_x_ω
    A[con2_rows, off_b_uγx+1:off_b_uγx+nu_bx] = data_b.div_x_γ
    A[con2_rows, off_b_uωy+1:off_b_uωy+nu_by] = data_b.div_y_ω
    A[con2_rows, off_b_uγy+1:off_b_uγy+nu_by] = data_b.div_y_γ

    jump_x = s.ic[1].scalar
    jump_y = s.ic[2].scalar
    flux_x = s.ic[1].flux
    flux_y = s.ic[2].flux

    g_jump_x = build_g_g(data_a.op_ux, jump_x, data_a.cap_px)
    g_jump_y = build_g_g(data_a.op_uy, jump_y, data_a.cap_py)
    h_flux_x = build_g_g(data_a.op_ux, flux_x, data_a.cap_px)
    h_flux_y = build_g_g(data_a.op_uy, flux_y, data_a.cap_py)

    f₁x = safe_build_source(data_a.op_ux, s.fluid_a.fᵤ, data_a.cap_px, nothing)
    f₁y = safe_build_source(data_a.op_uy, s.fluid_a.fᵤ, data_a.cap_py, nothing)
    f₂x = safe_build_source(data_b.op_ux, s.fluid_b.fᵤ, data_b.cap_px, nothing)
    f₂y = safe_build_source(data_b.op_uy, s.fluid_b.fᵤ, data_b.cap_py, nothing)

    b_mom_ax = data_a.Vx * f₁x
    b_mom_ay = data_a.Vy * f₁y
    b_mom_bx = data_b.Vx * f₂x
    b_mom_by = data_b.Vy * f₂y

    b = vcat(b_mom_ax,
             g_jump_x,
             b_mom_ay,
             g_jump_y,
             zeros(np_a),
             b_mom_bx,
             h_flux_x,
             b_mom_by,
             h_flux_y,
             zeros(np_b))

    apply_velocity_dirichlet_2D!(A, b, s.bc_u_a[1], s.bc_u_a[2], s.fluid_a.mesh_u;
                                 nu_x=nu_ax, nu_y=nu_ay,
                                 uωx_off=off_a_uωx, uγx_off=off_a_uγx,
                                 uωy_off=off_a_uωy, uγy_off=off_a_uγy,
                                 row_uωx_off=row_a_momx, row_uγx_off=row_cont_x,
                                 row_uωy_off=row_a_momy, row_uγy_off=row_cont_y)

    apply_velocity_dirichlet_2D!(A, b, s.bc_u_b[1], s.bc_u_b[2], s.fluid_b.mesh_u;
                                 nu_x=nu_bx, nu_y=nu_by,
                                 uωx_off=off_b_uωx, uγx_off=off_b_uγx,
                                 uωy_off=off_b_uωy, uγy_off=off_b_uγy,
                                 row_uωx_off=row_b_momx, row_uγx_off=row_trac_x,
                                 row_uωy_off=row_b_momy, row_uγy_off=row_trac_y)

    apply_pressure_gauge!(A, b, s.pressure_gauge[1], s.fluid_a.mesh_p, s.fluid_a.capacity_p;
                          p_offset=off_a_p, np=np_a, row_start=row_a_div+1)
    apply_pressure_gauge!(A, b, s.pressure_gauge[2], s.fluid_b.mesh_p, s.fluid_b.capacity_p;
                          p_offset=off_b_p, np=np_b, row_start=row_b_div+1)

    s.A = A
    s.b = b
    return nothing
end

function assemble_stokes3D_diph!(s::StokesDiph)
    data_a = stokes3D_phase_blocks(s.fluid_a)
    data_b = stokes3D_phase_blocks(s.fluid_b)

    nu_ax = data_a.nu_x; nu_ay = data_a.nu_y; nu_az = data_a.nu_z
    nu_bx = data_b.nu_x; nu_by = data_b.nu_y; nu_bz = data_b.nu_z
    @assert nu_ax == nu_bx && nu_ay == nu_by && nu_az == nu_bz "Velocity DOFs must match between phases in 3D"
    np_a = data_a.np; np_b = data_b.np

    off_a_uωx = 0
    off_a_uγx = nu_ax
    off_a_uωy = 2 * nu_ax
    off_a_uγy = 2 * nu_ax + nu_ay
    off_a_uωz = 2 * nu_ax + 2 * nu_ay
    off_a_uγz = 2 * nu_ax + 2 * nu_ay + nu_az
    off_a_p   = 2 * (nu_ax + nu_ay + nu_az)

    off_b_uωx = off_a_p + np_a
    off_b_uγx = off_b_uωx + nu_bx
    off_b_uωy = off_b_uγx + nu_bx
    off_b_uγy = off_b_uωy + nu_by
    off_b_uωz = off_b_uγy + nu_by
    off_b_uγz = off_b_uωz + nu_bz
    off_b_p   = off_b_uγz + nu_bz

    row_a_momx = 0
    row_cont_x = nu_ax
    row_a_momy = 2 * nu_ax
    row_cont_y = 2 * nu_ax + nu_ay
    row_a_momz = 2 * nu_ax + 2 * nu_ay
    row_cont_z = 2 * nu_ax + 2 * nu_ay + nu_az
    row_a_div  = 2 * (nu_ax + nu_ay + nu_az)

    n_tx = size(data_a.traction_x_ω, 1)
    n_ty = size(data_a.traction_y_ω, 1)
    n_tz = size(data_a.traction_z_ω, 1)

    row_b_momx = row_a_div + np_a
    row_trac_x = row_b_momx + nu_bx
    row_b_momy = row_trac_x + n_tx
    row_trac_y = row_b_momy + nu_by
    row_b_momz = row_trac_y + n_ty
    row_trac_z = row_b_momz + nu_bz
    row_b_div  = row_trac_z + n_tz

    rows = row_b_div + np_b
    cols = off_b_p + np_b
    A = spzeros(Float64, rows, cols)

    # Phase 1 momentum
    A[row_a_momx+1:row_a_momx+nu_ax, off_a_uωx+1:off_a_uωx+nu_ax] = data_a.visc_x_ω
    A[row_a_momx+1:row_a_momx+nu_ax, off_a_uγx+1:off_a_uγx+nu_ax] = data_a.visc_x_γ
    A[row_a_momx+1:row_a_momx+nu_ax, off_a_p+1:off_a_p+np_a]      = data_a.grad_x

    A[row_a_momy+1:row_a_momy+nu_ay, off_a_uωy+1:off_a_uωy+nu_ay] = data_a.visc_y_ω
    A[row_a_momy+1:row_a_momy+nu_ay, off_a_uγy+1:off_a_uγy+nu_ay] = data_a.visc_y_γ
    A[row_a_momy+1:row_a_momy+nu_ay, off_a_p+1:off_a_p+np_a]      = data_a.grad_y

    A[row_a_momz+1:row_a_momz+nu_az, off_a_uωz+1:off_a_uωz+nu_az] = data_a.visc_z_ω
    A[row_a_momz+1:row_a_momz+nu_az, off_a_uγz+1:off_a_uγz+nu_az] = data_a.visc_z_γ
    A[row_a_momz+1:row_a_momz+nu_az, off_a_p+1:off_a_p+np_a]      = data_a.grad_z

    # Interface velocity jump/continuity
    jump_x = s.ic[1].scalar
    jump_y = s.ic[2].scalar
    jump_z = s.ic[3].scalar
    Iα1x = jump_x === nothing ? I(nu_ax) : jump_x.α₁ * I(nu_ax)
    Iα2x = jump_x === nothing ? I(nu_ax) : jump_x.α₂ * I(nu_ax)
    Iα1y = jump_y === nothing ? I(nu_ay) : jump_y.α₁ * I(nu_ay)
    Iα2y = jump_y === nothing ? I(nu_ay) : jump_y.α₂ * I(nu_ay)
    Iα1z = jump_z === nothing ? I(nu_az) : jump_z.α₁ * I(nu_az)
    Iα2z = jump_z === nothing ? I(nu_az) : jump_z.α₂ * I(nu_az)

    A[row_cont_x+1:row_cont_x+nu_ax, off_a_uγx+1:off_a_uγx+nu_ax] = Iα1x
    A[row_cont_x+1:row_cont_x+nu_ax, off_b_uγx+1:off_b_uγx+nu_bx] .= -Iα2x
    A[row_cont_y+1:row_cont_y+nu_ay, off_a_uγy+1:off_a_uγy+nu_ay] = Iα1y
    A[row_cont_y+1:row_cont_y+nu_ay, off_b_uγy+1:off_b_uγy+nu_by] .= -Iα2y
    A[row_cont_z+1:row_cont_z+nu_az, off_a_uγz+1:off_a_uγz+nu_az] = Iα1z
    A[row_cont_z+1:row_cont_z+nu_az, off_b_uγz+1:off_b_uγz+nu_bz] .= -Iα2z

    # Phase 1 continuity
    con1_rows = row_a_div+1:row_a_div+np_a
    A[con1_rows, off_a_uωx+1:off_a_uωx+nu_ax] = data_a.div_x_ω
    A[con1_rows, off_a_uγx+1:off_a_uγx+nu_ax] = data_a.div_x_γ
    A[con1_rows, off_a_uωy+1:off_a_uωy+nu_ay] = data_a.div_y_ω
    A[con1_rows, off_a_uγy+1:off_a_uγy+nu_ay] = data_a.div_y_γ
    A[con1_rows, off_a_uωz+1:off_a_uωz+nu_az] = data_a.div_z_ω
    A[con1_rows, off_a_uγz+1:off_a_uγz+nu_az] = data_a.div_z_γ

    # Phase 2 momentum
    A[row_b_momx+1:row_b_momx+nu_bx, off_b_uωx+1:off_b_uωx+nu_bx] = data_b.visc_x_ω
    A[row_b_momx+1:row_b_momx+nu_bx, off_b_uγx+1:off_b_uγx+nu_bx] = data_b.visc_x_γ
    A[row_b_momx+1:row_b_momx+nu_bx, off_b_p+1:off_b_p+np_b]      = data_b.grad_x

    A[row_b_momy+1:row_b_momy+nu_by, off_b_uωy+1:off_b_uωy+nu_by] = data_b.visc_y_ω
    A[row_b_momy+1:row_b_momy+nu_by, off_b_uγy+1:off_b_uγy+nu_by] = data_b.visc_y_γ
    A[row_b_momy+1:row_b_momy+nu_by, off_b_p+1:off_b_p+np_b]      = data_b.grad_y

    A[row_b_momz+1:row_b_momz+nu_bz, off_b_uωz+1:off_b_uωz+nu_bz] = data_b.visc_z_ω
    A[row_b_momz+1:row_b_momz+nu_bz, off_b_uγz+1:off_b_uγz+nu_bz] = data_b.visc_z_γ
    A[row_b_momz+1:row_b_momz+nu_bz, off_b_p+1:off_b_p+np_b]      = data_b.grad_z

    # Traction jump
    flux_x = s.ic[1].flux
    flux_y = s.ic[2].flux
    flux_z = s.ic[3].flux
    Iβ1x = flux_x === nothing ? I(n_tx) : flux_x.β₁ * I(n_tx)
    Iβ2x = flux_x === nothing ? I(n_tx) : flux_x.β₂ * I(n_tx)
    Iβ1y = flux_y === nothing ? I(n_ty) : flux_y.β₁ * I(n_ty)
    Iβ2y = flux_y === nothing ? I(n_ty) : flux_y.β₂ * I(n_ty)
    Iβ1z = flux_z === nothing ? I(n_tz) : flux_z.β₁ * I(n_tz)
    Iβ2z = flux_z === nothing ? I(n_tz) : flux_z.β₂ * I(n_tz)

    A[row_trac_x+1:row_trac_x+n_tx, off_a_uωx+1:off_a_uωx+nu_ax] = -Iβ1x * data_a.traction_x_ω
    A[row_trac_x+1:row_trac_x+n_tx, off_a_uγx+1:off_a_uγx+nu_ax] = -Iβ1x * data_a.traction_x_γ
    #A[row_trac_x+1:row_trac_x+n_tx, off_a_p+1:off_a_p+np_a]      .-= Iβ1x * data_a.cap_px.Γ
    A[row_trac_x+1:row_trac_x+n_tx, off_b_uωx+1:off_b_uωx+nu_bx] .+= Iβ2x * data_b.traction_x_ω
    A[row_trac_x+1:row_trac_x+n_tx, off_b_uγx+1:off_b_uγx+nu_bx] .+= Iβ2x * data_b.traction_x_γ
    #A[row_trac_x+1:row_trac_x+n_tx, off_b_p+1:off_b_p+np_b]      .+= Iβ2x * data_b.cap_px.Γ

    A[row_trac_y+1:row_trac_y+n_ty, off_a_uωy+1:off_a_uωy+nu_ay] = -Iβ1y * data_a.traction_y_ω
    A[row_trac_y+1:row_trac_y+n_ty, off_a_uγy+1:off_a_uγy+nu_ay] = -Iβ1y * data_a.traction_y_γ
    #A[row_trac_y+1:row_trac_y+n_ty, off_a_p+1:off_a_p+np_a]      .-= Iβ1y * data_a.cap_py.Γ
    A[row_trac_y+1:row_trac_y+n_ty, off_b_uωy+1:off_b_uωy+nu_by] .+= Iβ2y * data_b.traction_y_ω
    A[row_trac_y+1:row_trac_y+n_ty, off_b_uγy+1:off_b_uγy+nu_by] .+= Iβ2y * data_b.traction_y_γ
    #A[row_trac_y+1:row_trac_y+n_ty, off_b_p+1:off_b_p+np_b]      .+= Iβ2y * data_b.cap_py.Γ

    A[row_trac_z+1:row_trac_z+n_tz, off_a_uωz+1:off_a_uωz+nu_az] = -Iβ1z * data_a.traction_z_ω
    A[row_trac_z+1:row_trac_z+n_tz, off_a_uγz+1:off_a_uγz+nu_az] = -Iβ1z * data_a.traction_z_γ
    #A[row_trac_z+1:row_trac_z+n_tz, off_a_p+1:off_a_p+np_a]      .-= Iβ1z * data_a.cap_pz.Γ
    A[row_trac_z+1:row_trac_z+n_tz, off_b_uωz+1:off_b_uωz+nu_bz] .+= Iβ2z * data_b.traction_z_ω
    A[row_trac_z+1:row_trac_z+n_tz, off_b_uγz+1:off_b_uγz+nu_bz] .+= Iβ2z * data_b.traction_z_γ
    #A[row_trac_z+1:row_trac_z+n_tz, off_b_p+1:off_b_p+np_b]      .+= Iβ2z * data_b.cap_pz.Γ

    # Phase 2 continuity
    con2_rows = row_b_div+1:row_b_div+np_b
    A[con2_rows, off_b_uωx+1:off_b_uωx+nu_bx] = data_b.div_x_ω
    A[con2_rows, off_b_uγx+1:off_b_uγx+nu_bx] = data_b.div_x_γ
    A[con2_rows, off_b_uωy+1:off_b_uωy+nu_by] = data_b.div_y_ω
    A[con2_rows, off_b_uγy+1:off_b_uγy+nu_by] = data_b.div_y_γ
    A[con2_rows, off_b_uωz+1:off_b_uωz+nu_bz] = data_b.div_z_ω
    A[con2_rows, off_b_uγz+1:off_b_uγz+nu_bz] = data_b.div_z_γ

    # RHS
    g_jump_x = jump_x === nothing ? zeros(nu_ax) : build_g_g(data_a.op_ux, jump_x, data_a.cap_px)
    g_jump_y = jump_y === nothing ? zeros(nu_ay) : build_g_g(data_a.op_uy, jump_y, data_a.cap_py)
    g_jump_z = jump_z === nothing ? zeros(nu_az) : build_g_g(data_a.op_uz, jump_z, data_a.cap_pz)
    h_flux_x = flux_x === nothing ? zeros(n_tx) : build_g_g(data_a.op_ux, flux_x, data_a.cap_px)
    h_flux_y = flux_y === nothing ? zeros(n_ty) : build_g_g(data_a.op_uy, flux_y, data_a.cap_py)
    h_flux_z = flux_z === nothing ? zeros(n_tz) : build_g_g(data_a.op_uz, flux_z, data_a.cap_pz)

    f₁x = safe_build_source(data_a.op_ux, s.fluid_a.fᵤ, data_a.cap_px, nothing)
    f₁y = safe_build_source(data_a.op_uy, s.fluid_a.fᵤ, data_a.cap_py, nothing)
    f₁z = safe_build_source(data_a.op_uz, s.fluid_a.fᵤ, data_a.cap_pz, nothing)
    f₂x = safe_build_source(data_b.op_ux, s.fluid_b.fᵤ, data_b.cap_px, nothing)
    f₂y = safe_build_source(data_b.op_uy, s.fluid_b.fᵤ, data_b.cap_py, nothing)
    f₂z = safe_build_source(data_b.op_uz, s.fluid_b.fᵤ, data_b.cap_pz, nothing)

    b_mom_ax = data_a.Vx * f₁x
    b_mom_ay = data_a.Vy * f₁y
    b_mom_az = data_a.Vz * f₁z
    b_mom_bx = data_b.Vx * f₂x
    b_mom_by = data_b.Vy * f₂y
    b_mom_bz = data_b.Vz * f₂z

    b = vcat(b_mom_ax,
             g_jump_x,
             b_mom_ay,
             g_jump_y,
             b_mom_az,
             g_jump_z,
             zeros(np_a),
             b_mom_bx,
             h_flux_x,
             b_mom_by,
             h_flux_y,
             b_mom_bz,
             h_flux_z,
             zeros(np_b))

    apply_velocity_dirichlet_3D!(A, b, s.bc_u_a[1], s.bc_u_a[2], s.bc_u_a[3], s.fluid_a.mesh_u;
                                 nu_x=nu_ax, nu_y=nu_ay, nu_z=nu_az,
                                 uωx_off=off_a_uωx, uγx_off=off_a_uγx,
                                 uωy_off=off_a_uωy, uγy_off=off_a_uγy,
                                 uωz_off=off_a_uωz, uγz_off=off_a_uγz,
                                 row_uωx_off=row_a_momx, row_uγx_off=row_cont_x,
                                 row_uωy_off=row_a_momy, row_uγy_off=row_cont_y,
                                 row_uωz_off=row_a_momz, row_uγz_off=row_cont_z)

    apply_velocity_dirichlet_3D!(A, b, s.bc_u_b[1], s.bc_u_b[2], s.bc_u_b[3], s.fluid_b.mesh_u;
                                 nu_x=nu_bx, nu_y=nu_by, nu_z=nu_bz,
                                 uωx_off=off_b_uωx, uγx_off=off_b_uγx,
                                 uωy_off=off_b_uωy, uγy_off=off_b_uγy,
                                 uωz_off=off_b_uωz, uγz_off=off_b_uγz,
                                 row_uωx_off=row_b_momx, row_uγx_off=row_trac_x,
                                 row_uωy_off=row_b_momy, row_uγy_off=row_trac_y,
                                 row_uωz_off=row_b_momz, row_uγz_off=row_trac_z)

    apply_pressure_gauge!(A, b, s.pressure_gauge[1], s.fluid_a.mesh_p, s.fluid_a.capacity_p;
                          p_offset=off_a_p, np=np_a, row_start=row_a_div+1)
    apply_pressure_gauge!(A, b, s.pressure_gauge[2], s.fluid_b.mesh_p, s.fluid_b.capacity_p;
                          p_offset=off_b_p, np=np_b, row_start=row_b_div+1)

    s.A = A
    s.b = b
    return nothing
end

function assemble_stokes1D_diph!(s::StokesDiph)
    data_a = stokes1D_phase_blocks(s.fluid_a)
    data_b = stokes1D_phase_blocks(s.fluid_b)

    nu = data_a.nu
    @assert nu == data_b.nu "Velocity DOFs must match between phases in 1D"
    np_a = data_a.np
    np_b = data_b.np

    off_a_uωx = 0
    off_a_uγx = nu
    off_a_p   = 2 * nu
    off_b_uωx = off_a_p + np_a
    off_b_uγx = off_b_uωx + nu
    off_b_p   = off_b_uγx + nu

    row_a_momx = 0
    row_cont_x = nu
    row_a_div  = 2 * nu
    row_b_momx = row_a_div + np_a
    n_tx       = size(data_a.traction_x_ω, 1)
    row_trac_x = row_b_momx + nu
    row_b_div  = row_trac_x + n_tx

    rows = row_b_div + np_b
    cols = off_b_p + np_b
    A = spzeros(Float64, rows, cols)

    # Phase 1 momentum
    A[row_a_momx+1:row_a_momx+nu, off_a_uωx+1:off_a_uωx+nu] = data_a.visc_x_ω
    A[row_a_momx+1:row_a_momx+nu, off_a_uγx+1:off_a_uγx+nu] = data_a.visc_x_γ
    A[row_a_momx+1:row_a_momx+nu, off_a_p+1:off_a_p+np_a]   = data_a.grad_x

    # Interface velocity jump/continuity
    jump_x = s.ic[1].scalar
    Iα1x = jump_x === nothing ? I(nu) : jump_x.α₁ * I(nu)
    Iα2x = jump_x === nothing ? I(nu) : jump_x.α₂ * I(nu)
    A[row_cont_x+1:row_cont_x+nu, off_a_uγx+1:off_a_uγx+nu] = Iα1x
    A[row_cont_x+1:row_cont_x+nu, off_b_uγx+1:off_b_uγx+nu] .= -Iα2x

    # Phase 1 continuity
    con1_rows = row_a_div+1:row_a_div+np_a
    A[con1_rows, off_a_uωx+1:off_a_uωx+nu] = data_a.div_x_ω
    A[con1_rows, off_a_uγx+1:off_a_uγx+nu] = data_a.div_x_γ

    # Phase 2 momentum
    A[row_b_momx+1:row_b_momx+nu, off_b_uωx+1:off_b_uωx+nu] = data_b.visc_x_ω
    A[row_b_momx+1:row_b_momx+nu, off_b_uγx+1:off_b_uγx+nu] = data_b.visc_x_γ
    A[row_b_momx+1:row_b_momx+nu, off_b_p+1:off_b_p+np_b]   = data_b.grad_x

    # Traction jump β₁T¹ + β₂T² = h
    flux_x = s.ic[1].flux
    Iβ1x = flux_x === nothing ? I(n_tx) : flux_x.β₁ * I(n_tx)
    Iβ2x = flux_x === nothing ? I(n_tx) : flux_x.β₂ * I(n_tx)

    A[row_trac_x+1:row_trac_x+n_tx, off_a_uωx+1:off_a_uωx+nu] = -Iβ1x * data_a.traction_x_ω
    A[row_trac_x+1:row_trac_x+n_tx, off_a_uγx+1:off_a_uγx+nu] = -Iβ1x * data_a.traction_x_γ
    #A[row_trac_x+1:row_trac_x+n_tx, off_a_p+1:off_a_p+np_a]   .-= Iβ1x * data_a.cap_px.Γ
    A[row_trac_x+1:row_trac_x+n_tx, off_b_uωx+1:off_b_uωx+nu] .+= Iβ2x * data_b.traction_x_ω
    A[row_trac_x+1:row_trac_x+n_tx, off_b_uγx+1:off_b_uγx+nu] .+= Iβ2x * data_b.traction_x_γ
    #A[row_trac_x+1:row_trac_x+n_tx, off_b_p+1:off_b_p+np_b]   .+= Iβ2x * data_b.cap_px.Γ

    # Phase 2 continuity
    con2_rows = row_b_div+1:row_b_div+np_b
    A[con2_rows, off_b_uωx+1:off_b_uωx+nu] = data_b.div_x_ω
    A[con2_rows, off_b_uγx+1:off_b_uγx+nu] = data_b.div_x_γ

    # RHS
    g_jump_x = jump_x === nothing ? zeros(nu) : build_g_g(data_a.op_ux, jump_x, data_a.cap_px)
    h_flux_x = flux_x === nothing ? zeros(n_tx) : build_g_g(data_a.op_ux, flux_x, data_a.cap_px)

    f₁x = safe_build_source(data_a.op_ux, s.fluid_a.fᵤ, data_a.cap_px, nothing)
    f₂x = safe_build_source(data_b.op_ux, s.fluid_b.fᵤ, data_b.cap_px, nothing)
    b_mom_ax = data_a.Vx * f₁x
    b_mom_bx = data_b.Vx * f₂x

    b = vcat(b_mom_ax,
             g_jump_x,
             zeros(np_a),
             b_mom_bx,
             h_flux_x,
             zeros(np_b))

    apply_velocity_dirichlet!(A, b, s.bc_u_a[1], s.fluid_a.mesh_u[1];
                              nu=nu, uω_offset=off_a_uωx, uγ_offset=off_a_uγx)
    apply_velocity_dirichlet!(A, b, s.bc_u_b[1], s.fluid_b.mesh_u[1];
                              nu=nu, uω_offset=off_b_uωx, uγ_offset=off_b_uγx)

    apply_pressure_gauge!(A, b, s.pressure_gauge[1], s.fluid_a.mesh_p, s.fluid_a.capacity_p;
                          p_offset=off_a_p, np=np_a, row_start=row_a_div+1)
    apply_pressure_gauge!(A, b, s.pressure_gauge[2], s.fluid_b.mesh_p, s.fluid_b.capacity_p;
                          p_offset=off_b_p, np=np_b, row_start=row_b_div+1)

    s.A = A
    s.b = b
    return nothing
end

# Linear solve ---------------------------------------------------------------

function solve_stokes_linear_system!(s::StokesDiph; method=Base.:\, algorithm=nothing, kwargs...)
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

function solve_StokesDiph!(s::StokesDiph; method=Base.:\, algorithm=nothing, kwargs...)
    println("[StokesDiph] Assembling steady diphasic Stokes system and solving")
    assemble_stokes_diph!(s)
    solve_stokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)
    return s
end

# Unsteady (2D) --------------------------------------------------------------

function stokes2D_phase_unsteady_blocks(fluid::Fluid{2})
    data = stokes2D_phase_blocks(fluid)
    ρ = fluid.ρ
    mass_x = build_I_D(data.op_ux, ρ, data.cap_px) * data.op_ux.V
    mass_y = build_I_D(data.op_uy, ρ, data.cap_py) * data.op_uy.V
    return (; data..., mass_x, mass_y)
end

function assemble_unsteady_stokes2D_diph!(s::StokesDiph,
                                          data_a,
                                          data_b,
                                          Δt::Float64,
                                          x_prev::AbstractVector{<:Real},
                                          t_prev::Float64,
                                          t_next::Float64,
                                          θ::Float64)
    nu_ax = data_a.nu_x; nu_ay = data_a.nu_y
    nu_bx = data_b.nu_x; nu_by = data_b.nu_y
    @assert nu_ax == nu_bx "Velocity x-DOF counts must match between phases"
    @assert nu_ay == nu_by "Velocity y-DOF counts must match between phases"
    np_a = data_a.np; np_b = data_b.np

    mass_ax_dt = (1.0 / Δt) * data_a.mass_x
    mass_ay_dt = (1.0 / Δt) * data_a.mass_y
    mass_bx_dt = (1.0 / Δt) * data_b.mass_x
    mass_by_dt = (1.0 / Δt) * data_b.mass_y
    θc = 1.0 - θ

    # Column offsets (uωx,uγx,uωy,uγy,p) for each phase
    off_a_uωx = 0
    off_a_uγx = nu_ax
    off_a_uωy = 2 * nu_ax
    off_a_uγy = 2 * nu_ax + nu_ay
    off_a_p   = 2 * (nu_ax + nu_ay)

    off_b_uωx = off_a_p + np_a
    off_b_uγx = off_b_uωx + nu_bx
    off_b_uωy = off_b_uγx + nu_bx
    off_b_uγy = off_b_uωy + nu_by
    off_b_p   = off_b_uγy + nu_by

    # Row offsets
    row_a_momx   = 0
    row_cont_x   = nu_ax
    row_a_momy   = 2 * nu_ax
    row_cont_y   = 2 * nu_ax + nu_ay
    row_a_div    = 2 * (nu_ax + nu_ay)
    row_b_momx   = row_a_div + np_a
    n_tx         = size(data_a.traction_x_ω, 1)
    n_ty         = size(data_a.traction_y_ω, 1)
    row_trac_x   = row_b_momx + nu_bx
    row_b_momy   = row_trac_x + n_tx
    row_trac_y   = row_b_momy + nu_by
    row_b_div    = row_trac_y + n_ty

    rows = row_b_div + np_b
    cols = off_b_p + np_b
    A = spzeros(Float64, rows, cols)

    # Phase 1 momentum
    A[row_a_momx+1:row_a_momx+nu_ax, off_a_uωx+1:off_a_uωx+nu_ax] = mass_ax_dt - θ * data_a.visc_x_ω
    A[row_a_momx+1:row_a_momx+nu_ax, off_a_uγx+1:off_a_uγx+nu_ax] = -θ * data_a.visc_x_γ
    A[row_a_momx+1:row_a_momx+nu_ax, off_a_p+1:off_a_p+np_a]       = data_a.grad_x

    A[row_a_momy+1:row_a_momy+nu_ay, off_a_uωy+1:off_a_uωy+nu_ay] = mass_ay_dt - θ * data_a.visc_y_ω
    A[row_a_momy+1:row_a_momy+nu_ay, off_a_uγy+1:off_a_uγy+nu_ay] = -θ * data_a.visc_y_γ
    A[row_a_momy+1:row_a_momy+nu_ay, off_a_p+1:off_a_p+np_a]      = data_a.grad_y

    # Interface scalar jump (defaults to continuity) for uγ
    jump_x = s.ic[1].scalar
    jump_y = s.ic[2].scalar
    Iα1x = jump_x.α₁ * I(nu_ax)
    Iα2x = jump_x.α₂ * I(nu_ax)
    Iα1y = jump_y.α₁ * I(nu_ay)
    Iα2y = jump_y.α₂ * I(nu_ay)

    A[row_cont_x+1:row_cont_x+nu_ax, off_a_uγx+1:off_a_uγx+nu_ax] = Iα1x
    A[row_cont_x+1:row_cont_x+nu_ax, off_b_uγx+1:off_b_uγx+nu_bx] .= -Iα2x

    A[row_cont_y+1:row_cont_y+nu_ay, off_a_uγy+1:off_a_uγy+nu_ay] = Iα1y
    A[row_cont_y+1:row_cont_y+nu_ay, off_b_uγy+1:off_b_uγy+nu_by] .= -Iα2y

    # Phase 1 continuity
    con1_rows = row_a_div+1:row_a_div+np_a
    A[con1_rows, off_a_uωx+1:off_a_uωx+nu_ax] = data_a.div_x_ω
    A[con1_rows, off_a_uγx+1:off_a_uγx+nu_ax] = data_a.div_x_γ
    A[con1_rows, off_a_uωy+1:off_a_uωy+nu_ay] = data_a.div_y_ω
    A[con1_rows, off_a_uγy+1:off_a_uγy+nu_ay] = data_a.div_y_γ

    # Phase 2 momentum
    A[row_b_momx+1:row_b_momx+nu_bx, off_b_uωx+1:off_b_uωx+nu_bx] = mass_bx_dt - θ * data_b.visc_x_ω
    A[row_b_momx+1:row_b_momx+nu_bx, off_b_uγx+1:off_b_uγx+nu_bx] = -θ * data_b.visc_x_γ
    A[row_b_momx+1:row_b_momx+nu_bx, off_b_p+1:off_b_p+np_b]      = data_b.grad_x

    A[row_b_momy+1:row_b_momy+nu_by, off_b_uωy+1:off_b_uωy+nu_by] = mass_by_dt - θ * data_b.visc_y_ω
    A[row_b_momy+1:row_b_momy+nu_by, off_b_uγy+1:off_b_uγy+nu_by] = -θ * data_b.visc_y_γ
    A[row_b_momy+1:row_b_momy+nu_by, off_b_p+1:off_b_p+np_b]      = data_b.grad_y

    # Traction jump (flux) β₁ T¹ + β₂ T² = h
    flux_x = s.ic[1].flux
    flux_y = s.ic[2].flux
    Iβ1x = flux_x.β₁ * I(n_tx)
    Iβ2x = flux_x.β₂ * I(n_tx)
    Iβ1y = flux_y.β₁ * I(n_ty)
    Iβ2y = flux_y.β₂ * I(n_ty)

    A[row_trac_x+1:row_trac_x+n_tx, off_a_uωx+1:off_a_uωx+nu_ax] = -Iβ1x * data_a.traction_x_ω
    A[row_trac_x+1:row_trac_x+n_tx, off_a_uγx+1:off_a_uγx+nu_ax] = -Iβ1x * data_a.traction_x_γ
    A[row_trac_x+1:row_trac_x+n_tx, off_b_uωx+1:off_b_uωx+nu_bx] .+= Iβ2x * data_b.traction_x_ω
    A[row_trac_x+1:row_trac_x+n_tx, off_b_uγx+1:off_b_uγx+nu_bx] .+= Iβ2x * data_b.traction_x_γ

    A[row_trac_y+1:row_trac_y+n_ty, off_a_uωy+1:off_a_uωy+nu_ay] = -Iβ1y * data_a.traction_y_ω
    A[row_trac_y+1:row_trac_y+n_ty, off_a_uγy+1:off_a_uγy+nu_ay] = -Iβ1y * data_a.traction_y_γ
    A[row_trac_y+1:row_trac_y+n_ty, off_b_uωy+1:off_b_uωy+nu_by] .+= Iβ2y * data_b.traction_y_ω
    A[row_trac_y+1:row_trac_y+n_ty, off_b_uγy+1:off_b_uγy+nu_by] .+= Iβ2y * data_b.traction_y_γ

    # Phase 2 continuity
    con2_rows = row_b_div+1:row_b_div+np_b
    A[con2_rows, off_b_uωx+1:off_b_uωx+nu_bx] = data_b.div_x_ω
    A[con2_rows, off_b_uγx+1:off_b_uγx+nu_bx] = data_b.div_x_γ
    A[con2_rows, off_b_uωy+1:off_b_uωy+nu_by] = data_b.div_y_ω
    A[con2_rows, off_b_uγy+1:off_b_uγy+nu_by] = data_b.div_y_γ

    # RHS
    g_jump_x = jump_x === nothing ? zeros(nu_ax) : build_g_g(data_a.op_ux, jump_x, data_a.cap_px)
    g_jump_y = jump_y === nothing ? zeros(nu_ay) : build_g_g(data_a.op_uy, jump_y, data_a.cap_py)
    h_flux_x = flux_x === nothing ? zeros(n_tx) : build_g_g(data_a.op_ux, flux_x, data_a.cap_px)
    h_flux_y = flux_y === nothing ? zeros(n_ty) : build_g_g(data_a.op_uy, flux_y, data_a.cap_py)

    u1ωx_prev = view(x_prev, off_a_uωx+1:off_a_uωx+nu_ax)
    u1γx_prev = view(x_prev, off_a_uγx+1:off_a_uγx+nu_ax)
    u1ωy_prev = view(x_prev, off_a_uωy+1:off_a_uωy+nu_ay)
    u1γy_prev = view(x_prev, off_a_uγy+1:off_a_uγy+nu_ay)

    u2ωx_prev = view(x_prev, off_b_uωx+1:off_b_uωx+nu_bx)
    u2γx_prev = view(x_prev, off_b_uγx+1:off_b_uγx+nu_bx)
    u2ωy_prev = view(x_prev, off_b_uωy+1:off_b_uωy+nu_by)
    u2γy_prev = view(x_prev, off_b_uγy+1:off_b_uγy+nu_by)

    f₁x_prev = safe_build_source(data_a.op_ux, s.fluid_a.fᵤ, data_a.cap_px, t_prev)
    f₁x_next = safe_build_source(data_a.op_ux, s.fluid_a.fᵤ, data_a.cap_px, t_next)
    load_ax = data_a.Vx * (θ .* f₁x_next .+ θc .* f₁x_prev)

    f₁y_prev = safe_build_source(data_a.op_uy, s.fluid_a.fᵤ, data_a.cap_py, t_prev)
    f₁y_next = safe_build_source(data_a.op_uy, s.fluid_a.fᵤ, data_a.cap_py, t_next)
    load_ay = data_a.Vy * (θ .* f₁y_next .+ θc .* f₁y_prev)

    f₂x_prev = safe_build_source(data_b.op_ux, s.fluid_b.fᵤ, data_b.cap_px, t_prev)
    f₂x_next = safe_build_source(data_b.op_ux, s.fluid_b.fᵤ, data_b.cap_px, t_next)
    load_bx = data_b.Vx * (θ .* f₂x_next .+ θc .* f₂x_prev)

    f₂y_prev = safe_build_source(data_b.op_uy, s.fluid_b.fᵤ, data_b.cap_py, t_prev)
    f₂y_next = safe_build_source(data_b.op_uy, s.fluid_b.fᵤ, data_b.cap_py, t_next)
    load_by = data_b.Vy * (θ .* f₂y_next .+ θc .* f₂y_prev)

    rhs_mom_ax = mass_ax_dt * u1ωx_prev
    rhs_mom_ax .-= θc * (data_a.visc_x_ω * u1ωx_prev + data_a.visc_x_γ * u1γx_prev)
    rhs_mom_ax .+= load_ax

    rhs_mom_ay = mass_ay_dt * u1ωy_prev
    rhs_mom_ay .-= θc * (data_a.visc_y_ω * u1ωy_prev + data_a.visc_y_γ * u1γy_prev)
    rhs_mom_ay .+= load_ay

    rhs_mom_bx = mass_bx_dt * u2ωx_prev
    rhs_mom_bx .-= θc * (data_b.visc_x_ω * u2ωx_prev + data_b.visc_x_γ * u2γx_prev)
    rhs_mom_bx .+= load_bx

    rhs_mom_by = mass_by_dt * u2ωy_prev
    rhs_mom_by .-= θc * (data_b.visc_y_ω * u2ωy_prev + data_b.visc_y_γ * u2γy_prev)
    rhs_mom_by .+= load_by

    b = vcat(rhs_mom_ax,
             g_jump_x,
             rhs_mom_ay,
             g_jump_y,
             zeros(np_a),
             rhs_mom_bx,
             h_flux_x,
             rhs_mom_by,
             h_flux_y,
             zeros(np_b))

    apply_velocity_dirichlet_2D!(A, b, s.bc_u_a[1], s.bc_u_a[2], s.fluid_a.mesh_u;
                                 nu_x=nu_ax, nu_y=nu_ay,
                                 uωx_off=off_a_uωx, uγx_off=off_a_uγx,
                                 uωy_off=off_a_uωy, uγy_off=off_a_uγy,
                                 row_uωx_off=row_a_momx, row_uγx_off=row_cont_x,
                                 row_uωy_off=row_a_momy, row_uγy_off=row_cont_y,
                                 t=t_next)

    apply_velocity_dirichlet_2D!(A, b, s.bc_u_b[1], s.bc_u_b[2], s.fluid_b.mesh_u;
                                 nu_x=nu_bx, nu_y=nu_by,
                                 uωx_off=off_b_uωx, uγx_off=off_b_uγx,
                                 uωy_off=off_b_uωy, uγy_off=off_b_uγy,
                                 row_uωx_off=row_b_momx, row_uγx_off=row_trac_x,
                                 row_uωy_off=row_b_momy, row_uγy_off=row_trac_y,
                                 t=t_next)

    apply_pressure_gauge!(A, b, s.pressure_gauge[1], s.fluid_a.mesh_p, s.fluid_a.capacity_p;
                          p_offset=off_a_p, np=np_a, row_start=row_a_div+1)
    apply_pressure_gauge!(A, b, s.pressure_gauge[2], s.fluid_b.mesh_p, s.fluid_b.capacity_p;
                          p_offset=off_b_p, np=np_b, row_start=row_b_div+1)

    s.A = A
    s.b = b
    return nothing
end

function solve_StokesDiph_unsteady!(s::StokesDiph; Δt::Float64, T_end::Float64,
                                    scheme::Symbol=:CN, method=Base.:\,
                                    algorithm=nothing, store_states::Bool=true,
                                    kwargs...)
    θ = scheme_to_theta(scheme)
    N = length(s.fluid_a.operator_u)
    N == 2 || error("StokesDiph unsteady solver implemented only for 2D (N=$(N))")

    data_a = stokes2D_phase_unsteady_blocks(s.fluid_a)
    data_b = stokes2D_phase_unsteady_blocks(s.fluid_b)

    nu_ax = data_a.nu_x
    nu_ay = data_a.nu_y
    np_a = data_a.np
    np_b = data_b.np

    Ntot = 4 * (nu_ax + nu_ay) + np_a + np_b
    x_prev = length(s.x) == Ntot ? copy(s.x) : zeros(Ntot)

    histories = store_states ? Vector{Vector{Float64}}() : Vector{Vector{Float64}}()
    if store_states
        push!(histories, copy(x_prev))
    end
    times = Float64[0.0]

    t = 0.0
    println("[StokesDiph] Starting unsteady diphasic solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
    while t < T_end - 1e-12 * max(1.0, T_end)
        dt_step = min(Δt, T_end - t)
        t_next = t + dt_step

        assemble_unsteady_stokes2D_diph!(s, data_a, data_b, dt_step, x_prev, t, t_next, θ)
        solve_stokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        x_prev = copy(s.x)
        push!(times, t_next)
        if store_states
            push!(histories, x_prev)
        end
        println("[StokesDiph] t=$(round(t_next; digits=6)) max|state|=$(maximum(abs, x_prev))")

        t = t_next
    end

    return times, histories
end
