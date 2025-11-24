mutable struct StokesDiph
    fluid_a::Fluid{2}
    fluid_b::Fluid{2}
    bc_u_a::NTuple{2,BorderConditions}
    bc_u_b::NTuple{2,BorderConditions}
    interface::InterfaceConditions
    pressure_gauge_a::AbstractPressureGauge
    pressure_gauge_b::AbstractPressureGauge
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    x::Vector{Float64}
    ch::Vector{Any}
end

function StokesDiph(fluid_a::Fluid{2}, fluid_b::Fluid{2},
                    bc_u_a::NTuple{2,BorderConditions},
                    bc_u_b::NTuple{2,BorderConditions},
                    interface::InterfaceConditions;
                    pressure_gauge_a::AbstractPressureGauge=DEFAULT_PRESSURE_GAUGE,
                    pressure_gauge_b::AbstractPressureGauge=DEFAULT_PRESSURE_GAUGE,
                    x0=zeros(0))
    nu_components = ntuple(i -> prod(fluid_a.operator_u[i].size), 2)
    np = prod(fluid_a.operator_p.size)
    Ntot = 2 * sum(nu_components) + np
    Ntot = 2 * Ntot  # two phases
    x_init = length(x0) == Ntot ? copy(x0) : zeros(Ntot)
    A = spzeros(Float64, Ntot, Ntot)
    b = zeros(Ntot)
    return StokesDiph(fluid_a, fluid_b, bc_u_a, bc_u_b, interface,
                      pressure_gauge_a, pressure_gauge_b,
                      A, b, x_init, Any[])
end

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

    ρ = fluid.ρ
    mass_x = build_I_D(ops_u[1], ρ, caps_u[1]) * ops_u[1].V
    mass_y = build_I_D(ops_u[2], ρ, caps_u[2]) * ops_u[2].V

    return (; nu_x, nu_y, np,
            op_ux = ops_u[1], op_uy = ops_u[2], op_p, cap_px = caps_u[1], cap_py = caps_u[2], cap_p,
            visc_x_ω, visc_x_γ, visc_y_ω, visc_y_γ,
            grad_x, grad_y,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ,
            tie_x = I(nu_x), tie_y = I(nu_y),
            mass_x, mass_y,
            Vx = ops_u[1].V, Vy = ops_u[2].V)
end

@inline function velocity_coords(mesh::AbstractMesh)
    xs = mesh.nodes[1]
    ys = mesh.nodes[2]
    coords = Vector{Tuple{Float64,Float64}}(undef, length(xs) * length(ys))
    idx = 1
    for j in eachindex(ys)
        for i in eachindex(xs)
            coords[idx] = (xs[i], ys[j])
            idx += 1
        end
    end
    return coords
end

@inline function coeff_vector(coeff, coords)
    if coeff isa Function
        return Float64[coeff(c...) for c in coords]
    else
        return fill(Float64(coeff), length(coords))
    end
end

function assemble_stokes_diph!(s::StokesDiph)
    data_a = stokes2D_phase_blocks(s.fluid_a)
    data_b = stokes2D_phase_blocks(s.fluid_b)

    @assert data_a.nu_x == data_b.nu_x "Velocity x-grid mismatch"
    @assert data_a.nu_y == data_b.nu_y "Velocity y-grid mismatch"
    @assert data_a.np == data_b.np "Pressure grid mismatch"
    mesh_ux = s.fluid_a.mesh_u[1]
    mesh_uy = s.fluid_a.mesh_u[2]
    ux_coords = velocity_coords(mesh_ux)
    uy_coords = velocity_coords(mesh_uy)

    nu_x = data_a.nu_x
    nu_y = data_a.nu_y
    sum_nu = nu_x + nu_y
    np = data_a.np

    rows_phase = 2 * sum_nu + np
    jump_rows = nu_x + nu_y
    flux_rows = nu_x + nu_y
    total_rows = 2 * rows_phase + jump_rows + flux_rows
    total_cols = 2 * (2 * sum_nu + np)

    A = spzeros(Float64, total_rows, total_cols)
    b = zeros(total_rows)

    # Phase offsets
    off_u1ωx = 0
    off_u1γx = nu_x
    off_u1ωy = 2 * nu_x
    off_u1γy = 2 * nu_x + nu_y
    off_p1   = 2 * sum_nu

    base2 = off_p1 + np
    off_u2ωx = base2
    off_u2γx = base2 + nu_x
    off_u2ωy = base2 + 2 * nu_x
    off_u2γy = base2 + 2 * nu_x + nu_y
    off_p2   = base2 + 2 * sum_nu

    # Row offsets
    row_phase1 = 0
    row_phase2 = rows_phase
    row_jump = 2 * rows_phase
    row_flux = row_jump + jump_rows

    # --- Phase 1 blocks (identical to StokesMono 2D) ---
    row_u1ωx = row_phase1
    row_u1γx = row_phase1 + nu_x
    row_u1ωy = row_phase1 + 2 * nu_x
    row_u1γy = row_phase1 + 2 * nu_x + nu_y
    row_con1 = row_phase1 + 2 * sum_nu

    A[row_u1ωx+1:row_u1ωx+nu_x, off_u1ωx+1:off_u1ωx+nu_x] = data_a.visc_x_ω
    A[row_u1ωx+1:row_u1ωx+nu_x, off_u1γx+1:off_u1γx+nu_x] = data_a.visc_x_γ
    A[row_u1ωx+1:row_u1ωx+nu_x, off_p1+1:off_p1+np]       = data_a.grad_x

    A[row_u1ωy+1:row_u1ωy+nu_y, off_u1ωy+1:off_u1ωy+nu_y] = data_a.visc_y_ω
    A[row_u1ωy+1:row_u1ωy+nu_y, off_u1γy+1:off_u1γy+nu_y] = data_a.visc_y_γ
    A[row_u1ωy+1:row_u1ωy+nu_y, off_p1+1:off_p1+np]       = data_a.grad_y

    con_rows = row_con1+1:row_con1+np
    A[con_rows, off_u1ωx+1:off_u1ωx+nu_x] = data_a.div_x_ω
    A[con_rows, off_u1γx+1:off_u1γx+nu_x] = data_a.div_x_γ
    A[con_rows, off_u1ωy+1:off_u1ωy+nu_y] = data_a.div_y_ω
    A[con_rows, off_u1γy+1:off_u1γy+nu_y] = data_a.div_y_γ

    f₁x = safe_build_source(data_a.op_ux, s.fluid_a.fᵤ, data_a.cap_px, nothing)
    f₁y = safe_build_source(data_a.op_uy, s.fluid_a.fᵤ, data_a.cap_py, nothing)
    b[row_u1ωx+1:row_u1ωx+nu_x] = data_a.Vx * f₁x
    b[row_u1ωy+1:row_u1ωy+nu_y] = data_a.Vy * f₁y
    b[row_u1γx+1:row_u1γx+nu_x] .= 0.0
    b[row_u1γy+1:row_u1γy+nu_y] .= 0.0

    # --- Phase 2 blocks ---
    row_u2ωx = row_phase2
    row_u2γx = row_phase2 + nu_x
    row_u2ωy = row_phase2 + 2 * nu_x
    row_u2γy = row_phase2 + 2 * nu_x + nu_y
    row_con2 = row_phase2 + 2 * sum_nu

    A[row_u2ωx+1:row_u2ωx+nu_x, off_u2ωx+1:off_u2ωx+nu_x] = data_b.visc_x_ω
    A[row_u2ωx+1:row_u2ωx+nu_x, off_u2γx+1:off_u2γx+nu_x] = data_b.visc_x_γ
    A[row_u2ωx+1:row_u2ωx+nu_x, off_p2+1:off_p2+np]       = data_b.grad_x

    A[row_u2ωy+1:row_u2ωy+nu_y, off_u2ωy+1:off_u2ωy+nu_y] = data_b.visc_y_ω
    A[row_u2ωy+1:row_u2ωy+nu_y, off_u2γy+1:off_u2γy+nu_y] = data_b.visc_y_γ
    A[row_u2ωy+1:row_u2ωy+nu_y, off_p2+1:off_p2+np]       = data_b.grad_y

    con_rows2 = row_con2+1:row_con2+np
    A[con_rows2, off_u2ωx+1:off_u2ωx+nu_x] = data_b.div_x_ω
    A[con_rows2, off_u2γx+1:off_u2γx+nu_x] = data_b.div_x_γ
    A[con_rows2, off_u2ωy+1:off_u2ωy+nu_y] = data_b.div_y_ω
    A[con_rows2, off_u2γy+1:off_u2γy+nu_y] = data_b.div_y_γ

    f₂x = safe_build_source(data_b.op_ux, s.fluid_b.fᵤ, data_b.cap_px, nothing)
    f₂y = safe_build_source(data_b.op_uy, s.fluid_b.fᵤ, data_b.cap_py, nothing)
    b[row_u2ωx+1:row_u2ωx+nu_x] = data_b.Vx * f₂x
    b[row_u2ωy+1:row_u2ωy+nu_y] = data_b.Vy * f₂y
    b[row_u2γx+1:row_u2γx+nu_x] .= 0.0
    b[row_u2γy+1:row_u2γy+nu_y] .= 0.0

    # --- Interface scalar jump (uγ continuity) ---
    jump = s.interface.scalar
    jump_vec_x1 = coeff_vector(jump.α₁, ux_coords)
    jump_vec_x2 = coeff_vector(jump.α₂, ux_coords)
    jump_vec_y1 = coeff_vector(jump.α₁, uy_coords)
    jump_vec_y2 = coeff_vector(jump.α₂, uy_coords)

    row_sx = row_jump + 1
    row_sy = row_jump + nu_x + 1
    # [[αu]] = α₂u₂ - α₁u₁ = g  (signs matter for continuity)
    A[row_sx-1+1:row_sx-1+nu_x, off_u1γx+1:off_u1γx+nu_x] = -spdiagm(0 => jump_vec_x1)
    A[row_sx-1+1:row_sx-1+nu_x, off_u2γx+1:off_u2γx+nu_x] =  spdiagm(0 => jump_vec_x2)
    b[row_sx:row_sx+nu_x-1] = safe_build_g(data_a.op_ux, jump, data_a.cap_px, nothing)

    A[row_sy-1+1:row_sy-1+nu_y, off_u1γy+1:off_u1γy+nu_y] = -spdiagm(0 => jump_vec_y1)
    A[row_sy-1+1:row_sy-1+nu_y, off_u2γy+1:off_u2γy+nu_y] =  spdiagm(0 => jump_vec_y2)
    b[row_sy:row_sy+nu_y-1] = safe_build_g(data_a.op_uy, jump, data_a.cap_py, nothing)

    # --- Interface flux jump (traction continuity) ---
    flux = s.interface.flux
    flux_vec_x1 = coeff_vector(flux.β₁, ux_coords)
    flux_vec_x2 = coeff_vector(flux.β₂, ux_coords)
    flux_vec_y1 = coeff_vector(flux.β₁, uy_coords)
    flux_vec_y2 = coeff_vector(flux.β₂, uy_coords)

    Iμx_a = build_I_D(data_a.op_ux, s.fluid_a.μ, s.fluid_a.capacity_u[1])
    Iμx_b = build_I_D(data_b.op_ux, s.fluid_b.μ, s.fluid_b.capacity_u[1])
    Iμy_a = build_I_D(data_a.op_uy, s.fluid_a.μ, s.fluid_a.capacity_u[2])
    Iμy_b = build_I_D(data_b.op_uy, s.fluid_b.μ, s.fluid_b.capacity_u[2])

    Txω_a = Iμx_a * (data_a.op_ux.H' * (data_a.op_ux.Wꜝ * data_a.op_ux.G))
    Txγ_a = Iμx_a * (data_a.op_ux.H' * (data_a.op_ux.Wꜝ * data_a.op_ux.H))
    Txω_b = Iμx_b * (data_b.op_ux.H' * (data_b.op_ux.Wꜝ * data_b.op_ux.G))
    Txγ_b = Iμx_b * (data_b.op_ux.H' * (data_b.op_ux.Wꜝ * data_b.op_ux.H))

    Tyω_a = Iμy_a * (data_a.op_uy.H' * (data_a.op_uy.Wꜝ * data_a.op_uy.G))
    Tyγ_a = Iμy_a * (data_a.op_uy.H' * (data_a.op_uy.Wꜝ * data_a.op_uy.H))
    Tyω_b = Iμy_b * (data_b.op_uy.H' * (data_b.op_uy.Wꜝ * data_b.op_uy.G))
    Tyγ_b = Iμy_b * (data_b.op_uy.H' * (data_b.op_uy.Wꜝ * data_b.op_uy.H))

    row_fx = row_flux + 1
    row_fy = row_flux + nu_x + 1

    # [[βσ·n]] = β₂σ₂·n - β₁σ₁·n = g (traction continuity)
    A[row_fx-1+1:row_fx-1+nu_x, off_u1ωx+1:off_u1ωx+nu_x] = -spdiagm(0 => flux_vec_x1) * Txω_a
    A[row_fx-1+1:row_fx-1+nu_x, off_u1γx+1:off_u1γx+nu_x] = -spdiagm(0 => flux_vec_x1) * Txγ_a
    A[row_fx-1+1:row_fx-1+nu_x, off_u2ωx+1:off_u2ωx+nu_x] =  spdiagm(0 => flux_vec_x2) * Txω_b
    A[row_fx-1+1:row_fx-1+nu_x, off_u2γx+1:off_u2γx+nu_x] =  spdiagm(0 => flux_vec_x2) * Txγ_b
    b[row_fx:row_fx+nu_x-1] = safe_build_g(data_a.op_ux, flux, data_a.cap_px, nothing)

    A[row_fy-1+1:row_fy-1+nu_y, off_u1ωy+1:off_u1ωy+nu_y] = -spdiagm(0 => flux_vec_y1) * Tyω_a
    A[row_fy-1+1:row_fy-1+nu_y, off_u1γy+1:off_u1γy+nu_y] = -spdiagm(0 => flux_vec_y1) * Tyγ_a
    A[row_fy-1+1:row_fy-1+nu_y, off_u2ωy+1:off_u2ωy+nu_y] =  spdiagm(0 => flux_vec_y2) * Tyω_b
    A[row_fy-1+1:row_fy-1+nu_y, off_u2γy+1:off_u2γy+nu_y] =  spdiagm(0 => flux_vec_y2) * Tyγ_b
    b[row_fy:row_fy+nu_y-1] = safe_build_g(data_a.op_uy, flux, data_a.cap_py, nothing)

    # Boundary conditions
    apply_velocity_dirichlet_2D!(A, b, s.bc_u_a[1], s.bc_u_a[2], s.fluid_a.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_u1ωx, uγx_off=off_u1γx,
                                 uωy_off=off_u1ωy, uγy_off=off_u1γy,
                                 row_uωx_off=row_u1ωx, row_uγx_off=row_u1γx,
                                 row_uωy_off=row_u1ωy, row_uγy_off=row_u1γy)
    apply_velocity_dirichlet_2D!(A, b, s.bc_u_b[1], s.bc_u_b[2], s.fluid_b.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_u2ωx, uγx_off=off_u2γx,
                                 uωy_off=off_u2ωy, uγy_off=off_u2γy,
                                 row_uωx_off=row_u2ωx, row_uγx_off=row_u2γx,
                                 row_uωy_off=row_u2ωy, row_uγy_off=row_u2γy)

    apply_pressure_gauge!(A, b, s.pressure_gauge_a, s.fluid_a.mesh_p, s.fluid_a.capacity_p;
                                 p_offset=off_p1, np=np, row_start=row_con1+1)

    s.A = A
    s.b = b
    return nothing
end

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
    println("[StokesDiph] Assembling steady diphasic Stokes system")
    assemble_stokes_diph!(s)
    solve_stokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)
    return s
end
