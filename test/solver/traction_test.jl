using Test
using LinearAlgebra
using Penguin

@testset "Traction boundary conditions" begin
    @testset "Stokes 1D traction row" begin
        nx = 6
        Lx = 1.0
        x0 = 0.0

        mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))
        dx = Lx / nx
        mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5 * dx,))

        body = (x, _=0.0) -> -1.0
        cap_u = Capacity(body, mesh_u)
        cap_p = Capacity(body, mesh_p)
        op_u = DiffusionOps(cap_u)
        op_p = DiffusionOps(cap_p)

        bc_u = BorderConditions(Dict(:bottom => Dirichlet(0.0), :top => Dirichlet(0.0)))
        pressure_gauge = PinPressureGauge()
        traction_val = 0.75
        bc_cut = Traction(traction_val)

        μ = 1.0
        ρ = 1.0
        fᵤ = (x, y=0.0, z=0.0) -> 0.0
        fₚ = (x, y=0.0, z=0.0) -> 0.0

        fluid = Fluid(mesh_u, cap_u, op_u, mesh_p, cap_p, op_p, μ, ρ, fᵤ, fₚ)
        solver = StokesMono(fluid, bc_u, pressure_gauge, bc_cut)

        data = Penguin.stokes1D_blocks(solver)
        traction_ω, traction_γ, Hp_u, g = Penguin.compute_cut_traction_data_1d(fluid, data, bc_cut; t=nothing)

        nu = data.nu
        np = data.np
        row_range = nu + 1:2nu
        A_cut = Matrix(solver.A[row_range, :])
        expected = zeros(length(row_range), size(A_cut, 2))
        expected[:, 1:nu] .= traction_ω
        expected[:, nu+1:2nu] .= traction_γ
        expected[:, 2nu+1:2nu+np] .= -Matrix(Hp_u)

        @test maximum(abs.(A_cut .- expected)) ≤ 1e-10
        @test maximum(abs.(solver.b[row_range] .- g)) ≤ 1e-10
    end

    @testset "Navier–Stokes 2D traction rows" begin
        nx, ny = 4, 3
        Lx, Ly = 1.0, 1.0
        x0, y0 = -0.5, -0.5

        mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
        dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
        dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
        mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
        mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

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

        τ = (0.25, -0.1)
        bc_cut = Traction((x, y,_=0) -> τ)

        μ = 1.0
        ρ = 1.0
        fᵤ = (x, y, z=0.0) -> 0.0
        fₚ = (x, y, z=0.0) -> 0.0

        fluid = Fluid((mesh_ux, mesh_uy), (cap_ux, cap_uy), (op_ux, op_uy),
                      mesh_p, cap_p, op_p, μ, ρ, fᵤ, fₚ)
        solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut)
        data = Penguin.navierstokes2D_blocks(solver)

        advecting_state = zeros(Float64, 2 * (data.nu_x + data.nu_y) + data.np)
        assemble_navierstokes2D_steady_picard!(solver, data, advecting_state)

        tx_ω, tx_γ, Hp_x, g_x, ty_ω, ty_γ, Hp_y, g_y = Penguin.compute_cut_traction_data_2d(fluid, data, bc_cut; t=nothing)

        nu_x = data.nu_x
        nu_y = data.nu_y
        np = data.np

        off_uωx = 0
        off_uγx = nu_x
        off_uωy = 2 * nu_x
        off_uγy = 2 * nu_x + nu_y
        off_p   = 2 * (nu_x + nu_y)

        row_uγx = nu_x
        row_uγy = 2 * nu_x + nu_y
        rows_x = row_uγx + 1:row_uγx + nu_x
        rows_y = row_uγy + 1:row_uγy + nu_y

        A = Matrix(solver.A)
        expected_x = zeros(length(rows_x), size(A, 2))
        expected_x[:, off_uωx+1:off_uωx+nu_x] .= tx_ω
        expected_x[:, off_uγx+1:off_uγx+nu_x] .= tx_γ
        expected_x[:, off_p+1:off_p+np] .= -Matrix(Hp_x)

        expected_y = zeros(length(rows_y), size(A, 2))
        expected_y[:, off_uωy+1:off_uωy+nu_y] .= ty_ω
        expected_y[:, off_uγy+1:off_uγy+nu_y] .= ty_γ
        expected_y[:, off_p+1:off_p+np] .= -Matrix(Hp_y)

        @test maximum(abs.(A[rows_x, :] .- expected_x)) ≤ 1e-10
        @test maximum(abs.(A[rows_y, :] .- expected_y)) ≤ 1e-10
        @test maximum(abs.(solver.b[rows_x] .- g_x)) ≤ 1e-10
        @test maximum(abs.(solver.b[rows_y] .- g_y)) ≤ 1e-10
    end
end
