# Navier-Stokes Projection Method Solvers (2D Only)
# Implements three projection methods for incompressible Navier-Stokes equations
# with explicit Adams-Bashforth 2 for convection

using LinearAlgebra
using SparseArrays

"""
Projection method types for Navier-Stokes solvers
"""
@enum ProjectionMethod begin
    ChorinTemam       # Original non-incremental projection
    IncrementalPC     # Incremental Pressure-Correction (Godunov/Van Kan)
    RotationalPC      # Rotational Incremental Pressure-Correction (best accuracy)
end

"""
    NavierStokesProj2D

2D Navier-Stokes solver using projection methods.
State vector: [uωx; uγx; uωy; uγy] (velocity components only, pressure separate)
Pressure is stored separately and updated via projection.

# Fields
- `fluid::Fluid{2}`: Fluid properties and operators
- `bc_u::NTuple{2,BorderConditions}`: Velocity boundary conditions for x and y
- `pressure_gauge::AbstractPressureGauge`: Pressure gauge constraint
- `bc_cut::AbstractBoundary`: Cut-cell/interface BC
- `method::ProjectionMethod`: Which projection method to use
- `u::Vector{Float64}`: Current velocity field
- `p::Vector{Float64}`: Current pressure field
- `conv_prev::Union{Nothing,NTuple{2,Vector{Float64}}}`: Previous convection terms (for AB2)
- `convection::Union{Nothing,NavierStokesConvection{2}}`: Convection data structures
- `dt::Float64`: Time step size
"""
mutable struct NavierStokesProj2D
    fluid::Fluid{2}
    bc_u::NTuple{2,BorderConditions}
    pressure_gauge::AbstractPressureGauge
    bc_cut::AbstractBoundary
    method::ProjectionMethod
    u::Vector{Float64}          # Velocity state [uωx; uγx; uωy; uγy]
    p::Vector{Float64}          # Pressure field
    conv_prev::Union{Nothing,NTuple{2,Vector{Float64}}}  # Previous convection for AB2
    convection::Union{Nothing,NavierStokesConvection{2}}  # Convection operators
    dt::Float64
end

"""
    NavierStokesProj2D(fluid, bc_u, pressure_gauge, bc_cut, method; u0=zeros(0), p0=zeros(0), dt=0.01)

Construct a 2D Navier-Stokes projection method solver.
"""
function NavierStokesProj2D(fluid::Fluid{2},
                             bc_u::NTuple{2,BorderConditions},
                             pressure_gauge::AbstractPressureGauge,
                             bc_cut::AbstractBoundary,
                             method::ProjectionMethod=RotationalPC;
                             u0=zeros(0),
                             p0=zeros(0),
                             dt::Float64=0.01)
    
    nu_x = prod(fluid.operator_u[1].size)
    nu_y = prod(fluid.operator_u[2].size)
    np = prod(fluid.operator_p.size)
    
    # Total velocity DOFs: 2 components × 2 fields (ω, γ) per component
    nu_total = 2 * (nu_x + nu_y)
    
    u = length(u0) == nu_total ? copy(u0) : zeros(nu_total)
    p = length(p0) == np ? copy(p0) : zeros(np)
    
    # Build convection operators (reuse from existing NavierStokes implementation)
    # This requires build_convection_data from navierstokes.jl to be available
    convection = nothing
    try
        convection = build_convection_data(fluid)
    catch
        @warn "Could not build convection operators; projection will use simplified convection"
    end
    
    return NavierStokesProj2D(fluid, bc_u, pressure_gauge, bc_cut, method,
                               u, p, nothing, convection, dt)
end

"""Helper to build operator blocks for 2D projection methods"""
function build_projection_blocks(s::NavierStokesProj2D)
    ops_u = s.fluid.operator_u
    caps_u = s.fluid.capacity_u
    op_p = s.fluid.operator_p
    cap_p = s.fluid.capacity_p
    
    nu_x = prod(ops_u[1].size)
    nu_y = prod(ops_u[2].size)
    np = prod(op_p.size)
    
    # Viscosity operators
    μ = s.fluid.μ
    Iμ_x = build_I_D(ops_u[1], μ, caps_u[1])
    Iμ_y = build_I_D(ops_u[2], μ, caps_u[2])
    
    WGx_Gx = ops_u[1].Wꜝ * ops_u[1].G
    WGx_Hx = ops_u[1].Wꜝ * ops_u[1].H
    visc_x_ω = Iμ_x * ops_u[1].G' * WGx_Gx
    visc_x_γ = Iμ_x * ops_u[1].G' * WGx_Hx
    
    WGy_Gy = ops_u[2].Wꜝ * ops_u[2].G
    WGy_Hy = ops_u[2].Wꜝ * ops_u[2].H
    visc_y_ω = Iμ_y * ops_u[2].G' * WGy_Gy
    visc_y_γ = Iμ_y * ops_u[2].G' * WGy_Hy
    
    # Pressure gradient operators
    grad_full = (op_p.G + op_p.H)
    x_rows = 1:nu_x
    y_rows = nu_x+1:nu_x+nu_y
    grad_x = -grad_full[x_rows, :]
    grad_y = -grad_full[y_rows, :]
    
    # Divergence operators
    Gp = op_p.G
    Hp = op_p.H
    Gp_x = Gp[x_rows, :]
    Hp_x = Hp[x_rows, :]
    Gp_y = Gp[y_rows, :]
    Hp_y = Hp[y_rows, :]
    div_x_ω = -(Gp_x' + Hp_x')
    div_x_γ = (Hp_x')
    div_y_ω = -(Gp_y' + Hp_y')
    div_y_γ = (Hp_y')
    
    # Mass matrices
    ρ = s.fluid.ρ
    mass_x = build_I_D(ops_u[1], ρ, caps_u[1]) * ops_u[1].V
    mass_y = build_I_D(ops_u[2], ρ, caps_u[2]) * ops_u[2].V
    
    return (; nu_x, nu_y, np,
            op_ux=ops_u[1], op_uy=ops_u[2], op_p,
            cap_px=caps_u[1], cap_py=caps_u[2], cap_p,
            visc_x_ω, visc_x_γ, visc_y_ω, visc_y_γ,
            grad_x, grad_y,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ,
            tie_x=I(nu_x), tie_y=I(nu_y),
            mass_x, mass_y,
            Vx=ops_u[1].V, Vy=ops_u[2].V)
end

"""Compute convection terms (explicit) using current velocity"""
function compute_convection_2d!(s::NavierStokesProj2D, data, u_state::AbstractVector{<:Real})
    nu_x = data.nu_x
    nu_y = data.nu_y
    
    # Extract velocity components
    uωx = view(u_state, 1:nu_x)
    uγx = view(u_state, nu_x+1:2*nu_x)
    uωy = view(u_state, 2*nu_x+1:2*nu_x+nu_y)
    uγy = view(u_state, 2*nu_x+nu_y+1:2*(nu_x+nu_y))
    
    # If convection operators are available, use them
    if s.convection !== nothing
        try
            # Use the same convection computation as NavierStokesMono
            uω_tuple = (Vector{Float64}(uωx), Vector{Float64}(uωy))
            uγ_tuple = (Vector{Float64}(uγx), Vector{Float64}(uγy))
            
            # Build convection matrices
            bulk_x = build_convection_matrix(s.convection.stencils[1], uω_tuple)
            bulk_y = build_convection_matrix(s.convection.stencils[2], uω_tuple)
            
            # Build K matrices for interface contributions
            K_x = build_K_matrix(s.convection.stencils[1], rotated_interfaces(uγ_tuple, 1))
            K_y = build_K_matrix(s.convection.stencils[2], rotated_interfaces(uγ_tuple, 2))
            
            # Compute convection vectors (skew-symmetric form)
            conv_x = bulk_x * uω_tuple[1] - 0.5 * (K_x * uω_tuple[1])
            conv_y = bulk_y * uω_tuple[2] - 0.5 * (K_y * uω_tuple[2])
            
            return (conv_x, conv_y)
        catch e
            @warn "Convection computation failed, using zero: $e" maxlog=1
        end
    end
    
    # Fallback: zero convection
    return (zeros(nu_x), zeros(nu_y))
end

"""
Step 1: Compute intermediate velocity u* (without pressure or with old pressure)
"""
function compute_intermediate_velocity!(s::NavierStokesProj2D, data, t::Float64)
    nu_x = data.nu_x
    nu_y = data.nu_y
    Δt = s.dt
    
    # Extract current velocity
    uωx = view(s.u, 1:nu_x)
    uγx = view(s.u, nu_x+1:2*nu_x)
    uωy = view(s.u, 2*nu_x+1:2*nu_x+nu_y)
    uγy = view(s.u, 2*nu_x+nu_y+1:2*(nu_x+nu_y))
    
    # Compute convection terms
    conv_curr = compute_convection_2d!(s, data, s.u)
    
    # Adams-Bashforth 2 for convection
    ρ = s.fluid.ρ
    ρ_val = if ρ isa Function
        # For variable density, evaluate at a representative point (domain center)
        # In a more sophisticated implementation, this would be evaluated at each cell
        x_mid = (s.fluid.mesh_p.nodes[1][1] + s.fluid.mesh_p.nodes[1][end]) / 2
        y_mid = (s.fluid.mesh_p.nodes[2][1] + s.fluid.mesh_p.nodes[2][end]) / 2
        ρ(x_mid, y_mid, 0.0)
    else
        ρ
    end
    if s.conv_prev === nothing
        # First step: use forward Euler
        conv_explicit_x = -ρ .* conv_curr[1]
        conv_explicit_y = -ρ .* conv_curr[2]
    else
        # AB2: (3/2)*conv_n - (1/2)*conv_{n-1}
        conv_explicit_x = -ρ .* (1.5 .* conv_curr[1] .- 0.5 .* s.conv_prev[1])
        conv_explicit_y = -ρ .* (1.5 .* conv_curr[2] .- 0.5 .* s.conv_prev[2])
    end
    
    # Store for next step
    s.conv_prev = conv_curr
    
    # Build viscous momentum system for intermediate velocity
    # u* = u^n + Δt * (ν∇²u* + explicit_convection + f)
    
    # For Chorin/Temam: no pressure term
    # For IncrementalPC: include old pressure gradient
    # For RotationalPC: include old pressure gradient
    
    mass_x_dt = (1.0 / Δt) * data.mass_x
    mass_y_dt = (1.0 / Δt) * data.mass_y
    
    # Implicit treatment of viscosity
    A_x = mass_x_dt - data.visc_x_ω
    A_y = mass_y_dt - data.visc_y_ω
    
    # RHS includes old velocity, explicit convection, and body forces
    f_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t)
    f_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t)
    load_x = data.Vx * f_x
    load_y = data.Vy * f_y
    
    rhs_x = mass_x_dt * Vector{Float64}(uωx) + conv_explicit_x + load_x
    rhs_y = mass_y_dt * Vector{Float64}(uωy) + conv_explicit_y + load_y
    
    # Add old pressure gradient for incremental/rotational methods
    if s.method == IncrementalPC || s.method == RotationalPC
        rhs_x .-= data.grad_x * s.p
        rhs_y .-= data.grad_y * s.p
    end
    
    # Solve for intermediate velocity
    # Note: Boundary conditions should be applied here for proper projection
    # For now, solving without BC application as a simplified first implementation
    # TODO: Add proper BC handling using apply_velocity_dirichlet_2D! from stokes.jl
    u_star_x = A_x \ rhs_x
    u_star_y = A_y \ rhs_y
    
    return vcat(u_star_x, zeros(nu_x), u_star_y, zeros(nu_y))
end

"""
Step 2: Solve Poisson equation for pressure correction
"""
function solve_pressure_poisson!(s::NavierStokesProj2D, data, u_star::AbstractVector{<:Real})
    nu_x = data.nu_x
    nu_y = data.nu_y
    np = data.np
    Δt = s.dt
    
    # Extract intermediate velocity
    u_star_ωx = view(u_star, 1:nu_x)
    u_star_ωy = view(u_star, 2*nu_x+1:2*nu_x+nu_y)
    
    # Compute divergence of u*
    div_u_star = data.div_x_ω * Vector{Float64}(u_star_ωx) + 
                 data.div_y_ω * Vector{Float64}(u_star_ωy)
    
    # Build Poisson matrix for pressure: ∇·∇p = ∇·u*/Δt
    # We need the negative of the divergence-gradient operator
    # Construct from the gradient operators with proper capacity weighting
    op_p = data.op_p
    
    # Get pressure capacity volume matrix for proper weighting
    V_p = data.cap_p.V
    
    # Laplacian with proper volume weighting
    # L = -div(grad) = -(G' + H')*(G + H) with capacity weighting
    Gp = op_p.G
    Hp = op_p.H
    
    # Construct weighted Laplacian (simplified form - full implementation 
    # would include proper metric terms and boundary handling)
    L_p = -(Gp' * V_p * Gp + Hp' * V_p * Hp)
    
    rhs_poisson = (1.0 / Δt) .* div_u_star
    
    # Apply pressure gauge
    A_poisson = copy(L_p)
    b_poisson = copy(rhs_poisson)
    apply_pressure_gauge!(A_poisson, b_poisson, s.pressure_gauge,
                          s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=0, np=np, row_start=1)
    
    # Solve Poisson equation
    if s.method == ChorinTemam
        # Solve for p^{n+1} directly
        p_new = A_poisson \ b_poisson
    else
        # Solve for pressure increment ϕ
        phi = A_poisson \ b_poisson
        # Update pressure: p^{n+1} = p^n + ϕ
        p_new = s.p .+ phi
    end
    
    return p_new
end

"""
Step 3: Correct velocity to enforce divergence-free condition
"""
function correct_velocity!(s::NavierStokesProj2D, data, u_star::AbstractVector{<:Real}, p_new::AbstractVector{<:Real})
    nu_x = data.nu_x
    nu_y = data.nu_y
    Δt = s.dt
    
    # Extract intermediate velocity ω components
    u_star_ωx = Vector{Float64}(view(u_star, 1:nu_x))
    u_star_ωy = Vector{Float64}(view(u_star, 2*nu_x+1:2*nu_x+nu_y))
    
    # Velocity correction: u^{n+1} = u* - Δt * ∇p (or ∇ϕ for incremental)
    if s.method == ChorinTemam
        grad_p_x = data.grad_x * p_new
        grad_p_y = data.grad_y * p_new
    elseif s.method == IncrementalPC
        # Use pressure increment ϕ = p_new - p_old
        phi = p_new .- s.p
        grad_p_x = data.grad_x * phi
        grad_p_y = data.grad_y * phi
    else  # RotationalPC
        # Rotational correction: ∇p^{n+1} = ∇(p^n + ϕ) - ν∇(∇·u*)
        phi = p_new .- s.p
        grad_phi_x = data.grad_x * phi
        grad_phi_y = data.grad_y * phi
        
        # Compute ∇(∇·u*) - this is the rotational correction term
        # TODO: Proper implementation requires computing the gradient of divergence
        # which needs careful stencil construction. This is a simplified placeholder.
        div_u_star = data.div_x_ω * u_star_ωx + data.div_y_ω * u_star_ωy
        # Note: This applies gradient to a scalar field which gives approximate correction
        # Full implementation would need proper second-derivative stencils
        grad_div_x = data.grad_x * div_u_star
        grad_div_y = data.grad_y * div_u_star
        
        # Get kinematic viscosity
        ρ = s.fluid.ρ
        ρ_val = if ρ isa Function
            x_mid = (s.fluid.mesh_p.nodes[1][1] + s.fluid.mesh_p.nodes[1][end]) / 2
            y_mid = (s.fluid.mesh_p.nodes[2][1] + s.fluid.mesh_p.nodes[2][end]) / 2
            ρ(x_mid, y_mid, 0.0)
        else
            ρ
        end
        ν = s.fluid.μ / ρ_val
        grad_p_x = grad_phi_x .- ν .* grad_div_x
        grad_p_y = grad_phi_y .- ν .* grad_div_y
    end
    
    # Correct velocity
    u_new_ωx = u_star_ωx .+ Δt .* grad_p_x
    u_new_ωy = u_star_ωy .+ Δt .* grad_p_y
    
    # Reconstruct full state (γ components remain from u_star or are set by BC)
    u_new = copy(u_star)
    u_new[1:nu_x] .= u_new_ωx
    u_new[2*nu_x+1:2*nu_x+nu_y] .= u_new_ωy
    
    return u_new
end

"""
    solve_NavierStokesProj2D_step!(s::NavierStokesProj2D, t::Float64)

Perform one time step using the selected projection method.
"""
function solve_NavierStokesProj2D_step!(s::NavierStokesProj2D, t::Float64)
    data = build_projection_blocks(s)
    
    # Step 1: Compute intermediate velocity
    u_star = compute_intermediate_velocity!(s, data, t)
    
    # Step 2: Solve pressure Poisson equation
    p_new = solve_pressure_poisson!(s, data, u_star)
    
    # Step 3: Correct velocity
    u_new = correct_velocity!(s, data, u_star, p_new)
    
    # Update state
    s.u .= u_new
    s.p .= p_new
    
    return nothing
end

"""
    solve_NavierStokesProj2D!(s::NavierStokesProj2D; T_end::Float64, store_states::Bool=true)

Solve the 2D Navier-Stokes equations using projection method up to time T_end.
"""
function solve_NavierStokesProj2D!(s::NavierStokesProj2D;
                                    T_end::Float64,
                                    store_states::Bool=true)
    
    times = Float64[0.0]
    u_hist = store_states ? [copy(s.u)] : Vector{Vector{Float64}}()
    p_hist = store_states ? [copy(s.p)] : Vector{Vector{Float64}}()
    
    t = 0.0
    step = 0
    
    println("[NavierStokesProj2D] Starting $(s.method) method solve up to T=$(T_end) with Δt=$(s.dt)")
    
    while t < T_end - 1e-12 * max(1.0, T_end)
        dt_step = min(s.dt, T_end - t)
        t_next = t + dt_step
        
        # Temporarily update dt if needed for final step
        dt_old = s.dt
        s.dt = dt_step
        
        solve_NavierStokesProj2D_step!(s, t_next)
        
        s.dt = dt_old
        t = t_next
        step += 1
        
        push!(times, t)
        if store_states
            push!(u_hist, copy(s.u))
            push!(p_hist, copy(s.p))
        end
        
        if step % 10 == 0
            println("[NavierStokesProj2D] Step $(step): t=$(round(t; digits=6)) max|u|=$(maximum(abs, s.u)) max|p|=$(maximum(abs, s.p))")
        end
    end
    
    println("[NavierStokesProj2D] Completed $(step) steps")
    
    return times, u_hist, p_hist
end

"""
    compute_divergence(s::NavierStokesProj2D)

Compute the divergence of the current velocity field (for validation).
Returns the L2 norm of divergence (should be close to zero for incompressible flow).
"""
function compute_divergence(s::NavierStokesProj2D)
    data = build_projection_blocks(s)
    nu_x = data.nu_x
    nu_y = data.nu_y
    
    uωx = view(s.u, 1:nu_x)
    uωy = view(s.u, 2*nu_x+1:2*nu_x+nu_y)
    
    div_u = data.div_x_ω * Vector{Float64}(uωx) + 
            data.div_y_ω * Vector{Float64}(uωy)
    
    return norm(div_u)
end
