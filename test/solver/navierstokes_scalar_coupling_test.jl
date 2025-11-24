using Penguin
using Test
using LinearAlgebra

@testset "NavierStokesScalarCoupler Picard coupling (2D)" begin
    nx, ny = 16, 8
    width, height = 1.0, 1.0
    origin = (0.0, 0.0)

    mesh_p = Penguin.Mesh((nx, ny), (width, height), origin)
    dx = width / nx
    dy = height / ny
    mesh_ux = Penguin.Mesh((nx, ny), (width, height), (origin[1] - 0.5 * dx, origin[2]))
    mesh_uy = Penguin.Mesh((nx, ny), (width, height), (origin[1], origin[2] - 0.5 * dy))
    mesh_T = mesh_p

    body = (x, y, _=0.0) -> -1.0

    capacity_ux = Capacity(body, mesh_ux)
    capacity_uy = Capacity(body, mesh_uy)
    capacity_p  = Capacity(body, mesh_p)
    capacity_T  = Capacity(body, mesh_T)

    operator_ux = DiffusionOps(capacity_ux)
    operator_uy = DiffusionOps(capacity_uy)
    operator_p  = DiffusionOps(capacity_p)

    zero_dirichlet = Dirichlet((x, y, t=0.0) -> 0.0)
    bc_ux = BorderConditions(Dict(
        :left   => zero_dirichlet,
        :right  => zero_dirichlet,
        :bottom => zero_dirichlet,
        :top    => zero_dirichlet
    ))
    bc_uy = BorderConditions(Dict(
        :left   => zero_dirichlet,
        :right  => zero_dirichlet,
        :bottom => zero_dirichlet,
        :top    => zero_dirichlet
    ))
    pressure_gauge = MeanPressureGauge()
    bc_cut = Dirichlet(0.0)

    μ = 1.0
    ρ = 1.0
    fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
    fₚ = (x, y, z=0.0, t=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (capacity_ux, capacity_uy),
                  (operator_ux, operator_uy),
                  mesh_p,
                  capacity_p,
                  operator_p,
                  μ, ρ, fᵤ, fₚ)

    nu_x = prod(operator_ux.size)
    nu_y = prod(operator_uy.size)
    np   = prod(operator_p.size)
    initial_state = zeros(2 * (nu_x + nu_y) + np)

    ns_solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut; x0=initial_state)

    T_hot = 0.5
    T_cold = -0.5
    bc_T = BorderConditions(Dict(
        :top    => Dirichlet(T_hot),
        :bottom => Dirichlet(T_cold)
    ))
    bc_T_cut = Dirichlet(0.0)

    nodes_Tx = mesh_T.nodes[1]
    nodes_Ty = mesh_T.nodes[2]
    Nx_T = length(nodes_Tx)
    Ny_T = length(nodes_Ty)
    N_scalar = Nx_T * Ny_T

    T0ω = zeros(Float64, N_scalar)
    for j in 1:Ny_T
        y = nodes_Ty[j]
        frac = (y - first(nodes_Ty)) / (last(nodes_Ty) - first(nodes_Ty))
        base = T_cold + (T_hot - T_cold) * frac
        for i in 1:Nx_T
            x = nodes_Tx[i]
            perturb = 0.02 * sinpi(x / width) * sinpi(frac)
            idx = i + (j - 1) * Nx_T
            T0ω[idx] = base + perturb
        end
    end
    T0γ = copy(T0ω)
    T0 = vcat(T0ω, T0γ)
    T0_copy = copy(T0)

    κ = 5.0e-3
    heat_source = (x, y, z=0.0, t=0.0) -> 0.0

    coupler = NavierStokesScalarCoupler(ns_solver,
                                        capacity_T,
                                        κ,
                                        heat_source,
                                        bc_T,
                                        bc_T_cut;
                                        strategy=PicardCoupling(tol_T=1e-6, tol_U=1e-6, maxiter=3, relaxation=1.0),
                                        β=1.0,
                                        gravity=(0.0, -1.0),
                                        T_ref=0.0,
                                        T0=T0,
                                        store_states=false)

    Δt = 0.01
    T_end = 0.05
    times, _, _ = solve_NavierStokesScalarCoupling!(coupler;
                                                    Δt=Δt,
                                                    T_end=T_end,
                                                    scheme=:CN)

    final_vel = coupler.velocity_state
    final_temp = coupler.scalar_state


    @test length(times) > 1
    @test !any(isnan, final_vel)
    @test !any(isnan, final_temp)
    @test maximum(abs, final_vel[1:nu_x]) < 0.5
    @test maximum(abs, final_temp) < 5.0

    ΔT = maximum(abs, final_temp .- T0_copy)
    @test ΔT > 1.0e-6
end
