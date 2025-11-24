using Penguin
using Test
using LinearAlgebra

function build_simple_navierstokes(; nx=6, ny=5)
    Lx = 1.0; Ly = 1.0
    x0 = -0.5; y0 = -0.5

    mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
    dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
    dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
    mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
    mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

    body = (x, y, _=0.0) -> -1.0

    cap_ux = Capacity(body, mesh_ux)
    cap_uy = Capacity(body, mesh_uy)
    cap_p  = Capacity(body, mesh_p)

    op_ux = DiffusionOps(cap_ux)
    op_uy = DiffusionOps(cap_uy)
    op_p  = DiffusionOps(cap_p)

    bc_zero = Dirichlet((x, y, t=0.0) -> 0.0)
    bc_ux = BorderConditions(Dict(:left=>bc_zero, :right=>bc_zero, :bottom=>bc_zero, :top=>bc_zero))
    bc_uy = BorderConditions(Dict(:left=>bc_zero, :right=>bc_zero, :bottom=>bc_zero, :top=>bc_zero))
    pressure_gauge = PinPressureGauge()
    bc_cut = Dirichlet(0.0)

    μ = 1.0
    ρ = 1.0
    fᵤ = (x, y, z=0.0) -> 0.0
    fₚ = (x, y, z=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (cap_ux, cap_uy),
                  (op_ux, op_uy),
                  mesh_p,
                  cap_p,
                  op_p,
                  μ, ρ, fᵤ, fₚ)

    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut)
    data = Penguin.navierstokes2D_blocks(solver)
    return solver, data
end

function build_simple_navierstokes_1d(; nx=8)
    Lx = 1.0
    x0 = -0.5

    mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))
    dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
    mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5 * dx,))

    body = (x, _=0.0) -> -1.0

    cap_u = Capacity(body, mesh_u)
    cap_p = Capacity(body, mesh_p)

    op_u = DiffusionOps(cap_u)
    op_p = DiffusionOps(cap_p)

    bc_zero = Dirichlet((x, t=0.0) -> 0.0)
    bc_u = BorderConditions(Dict(:left => bc_zero, :right => bc_zero))
    pressure_gauge = PinPressureGauge()
    bc_cut = Dirichlet(0.0)

    μ = 1.0
    ρ = 1.0
    fᵤ = (x, y=0.0, z=0.0) -> 0.0
    fₚ = (x, y=0.0, z=0.0) -> 0.0

    fluid = Fluid(mesh_u, cap_u, op_u, mesh_p, cap_p, op_p, μ, ρ, fᵤ, fₚ)

    solver = NavierStokesMono(fluid, bc_u, pressure_gauge, bc_cut)
    data = Penguin.navierstokes1D_blocks(solver)
    return solver, data
end

@testset "Navier–Stokes convection operator" begin
    solver, data = build_simple_navierstokes()
    nu_x = data.nu_x; nu_y = data.nu_y; np = data.np
    Ntot = 2 * (nu_x + nu_y) + np

    advecting_state = zeros(Float64, Ntot)
    # Fill with deterministic patterns (not divergence-free, but sufficient to exercise operator)
    advecting_state[1:nu_x] .= range(-0.3, 0.25; length=nu_x)
    advecting_state[nu_x+1:2nu_x] .= range(0.15, -0.1; length=nu_x)
    advecting_state[2nu_x+1:2nu_x+nu_y] .= range(-0.2, 0.4; length=nu_y)
    advecting_state[2nu_x+nu_y+1:2*(nu_x+nu_y)] .= range(0.05, -0.35; length=nu_y)

    conv = Penguin.compute_convection_vectors!(solver, data, advecting_state)
    conv_x, conv_y = conv
    ops = solver.last_conv_ops

    uωx = advecting_state[1:nu_x]
    uωy = advecting_state[2nu_x+1:2nu_x+nu_y]
    calc_x = ops.bulk[1] * uωx - ops.K_adv[1] * uωx
    calc_y = ops.bulk[2] * uωy - ops.K_adv[2] * uωy

    @test isapprox(conv_x, calc_x; atol=1e-12, rtol=1e-10)
    @test isapprox(conv_y, calc_y; atol=1e-12, rtol=1e-10)

    # Free-stream preservation: constant fields should yield zero convection
    const_state = copy(advecting_state)
    const_state .= 2.0
    conv_x_fs, conv_y_fs = Penguin.compute_convection_vectors!(solver, data, const_state)

    # Only take interior points
    conv_x_fs = reshape(conv_x_fs, data.op_ux.size)
    conv_y_fs = reshape(conv_y_fs, data.op_uy.size)
    conv_x_fs = conv_x_fs[2:end-2, 2:end-2]
    conv_y_fs = conv_y_fs[2:end-2, 2:end-2]

    @test maximum(abs, conv_x_fs) ≤ 1e-10
    @test maximum(abs, conv_y_fs) ≤ 1e-10
end

@testset "Navier–Stokes steady Picard" begin
    solver, data = build_simple_navierstokes()
    solver.x .= 0.0
    _, iters, res = Penguin.solve_NavierStokesMono_steady!(solver; tol=1e-10, maxiter=3)
    @test iters ≥ 1
    @test res ≤ 1e-10
    @test maximum(abs, solver.x) ≤ 1e-12
end

@testset "Navier–Stokes steady Newton" begin
    solver, data = build_simple_navierstokes()
    # Impose a lid-driven boundary to trigger non-trivial increments
    solver.bc_u[1].borders[:top] = Dirichlet(0.0)
    solver.bc_u[1].borders[:bottom] = Dirichlet(0.0)
    solver.bc_u[1].borders[:left] = Dirichlet(0.0)
    solver.bc_u[1].borders[:right] = Dirichlet(0.0)
    for side in (:left, :right, :bottom, :top)
        solver.bc_u[2].borders[side] = Dirichlet(0.0)
    end
    solver.x .= 0.0
    _, iters, res = Penguin.solve_NavierStokesMono_steady!(solver; tol=1e-8, maxiter=10, nlsolve_method=:newton)
    @test iters ≥ 1
    @test res ≤ 1e-8
end

@testset "Navier–Stokes outflow" begin
    solver, data = build_simple_navierstokes()
    solver.bc_u[1].borders[:right] = Outflow()
    solver.bc_u[2].borders[:right] = Outflow()
    solver.x .= 0.0
    _, iters, res = Penguin.solve_NavierStokesMono_steady!(solver; tol=1e-9, maxiter=10, nlsolve_method=:picard)
    @test iters ≥ 1
    @test res ≤ 1e-9
    @test maximum(abs, solver.x) ≤ 1e-10
end


@testset "Navier–Stokes 1D basic" begin
    solver, data = build_simple_navierstokes_1d()
    nu = data.nu
    np = data.np
    Ntot = 2 * nu + np

    state0 = zeros(Float64, Ntot)
    conv = Penguin.compute_convection_vectors!(solver, data, state0)
    @test length(conv) == 1
    @test maximum(abs, conv[1]) ≤ 1e-12

    _, iters, res = Penguin.solve_NavierStokesMono_steady!(solver; tol=1e-10, maxiter=5, relaxation=0.7, nlsolve_method=:picard)
    @test iters ≥ 1
    @test res ≤ 1e-10
    @test maximum(abs, solver.x) ≤ 1e-12

    called = Ref(false)
    precond_builder = (A, _) -> begin
        called[] = true
        nothing
    end
    Penguin.solve_NavierStokesMono_steady!(solver; tol=1e-6, maxiter=1, relaxation=1.0,
                                           nlsolve_method=:picard, precond_builder=precond_builder)
    @test called[]
end
