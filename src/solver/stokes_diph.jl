"""
    StokesDiph

Prototype two-phase Stokes solver with per-phase fluids on staggered grids.
Unknown ordering (2D): `[u₁xᵠ, u₁xᵞ, u₁yᵠ, u₁yᵞ, p₁, u₂xᵠ, u₂xᵞ, u₂yᵠ, u₂yᵞ, p₂]`
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
    N == 2 || error("StokesDiph currently implemented for 2D (received N=$(N)).")

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

# Assembly -------------------------------------------------------------------

function assemble_stokes_diph!(s::StokesDiph)
    N = length(s.fluid_a.operator_u)
    N == length(s.fluid_b.operator_u) || error("Phase operator dimensionality mismatch.")
    if N != 2
        error("assemble_stokes_diph! is implemented for 2D only (N=$(N)).")
    end
    return assemble_stokes2D_diph!(s)
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

    g_jump_x = jump_x === nothing ? zeros(nu_ax) : build_g_g(data_a.op_ux, jump_x, data_a.cap_px)
    g_jump_y = jump_y === nothing ? zeros(nu_ay) : build_g_g(data_a.op_uy, jump_y, data_a.cap_py)
    h_flux_x = flux_x === nothing ? zeros(n_tx) : build_g_g(data_a.op_ux, flux_x, data_a.cap_px)
    h_flux_y = flux_y === nothing ? zeros(n_ty) : build_g_g(data_a.op_uy, flux_y, data_a.cap_py)

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
