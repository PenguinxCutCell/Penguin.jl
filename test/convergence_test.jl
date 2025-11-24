using Penguin
using Test
using SpecialFunctions
using IterativeSolvers
using LinearSolve

@testset "Convergence Test 1D" begin
    nx = 40
    lx = 4.0
    x0 = 0.0
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    center = 0.5
    radius = 0.1
    LS(x,_=0) = sqrt((x-center)^2) - radius
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Dirichlet(0.0)
    bc1 = Dirichlet(0.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc1, :bottom => bc1))
    f(x,y,z) = x
    D(x,y,z) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    solve_DiffusionSteadyMono!(solver; algorithm=UMFPACKFactorization(), log=true)
    u_analytic(x) = - (x-center)^3/6 - (center*(x-center)^2)/2 + radius^2/6 * (x-center) + center*radius^2/2
    u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytic, solver, capacity, 2, false)
    @test global_err < 1e-2
end

@testset "Convergence Test 2D" begin
    nx, ny = 40, 40
    lx, ly = 4., 4.
    x0, y0 = 0., 0.
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    LS(x,y,_=0) = (sqrt((x-2)^2 + (y-2)^2) - 1.0)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Dirichlet(0.0)
    bc1 = Dirichlet(1.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))
    f(x,y,_=0) = 4.0
    D(x,y,_=0) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    solve_DiffusionSteadyMono!(solver; method=Base.:\)
    u_analytic(x,y) = 1.0 - (x-2)^2 - (y-2)^2
    u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytic, solver, capacity, 2, false)
    @test global_err < 1e-2
end

@testset "Convergence Test 3D" begin
    nx, ny, nz = 40, 40, 40
    lx, ly, lz = 4., 4., 4.
    x0, y0, z0 = 0., 0., 0.
    mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))
    LS(x,y,z) = (sqrt((x-2)^2 + (y-2)^2 + (z-2)^2) - 1.0)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Dirichlet(0.0)
    bc1 = Dirichlet(1.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1, :forward => bc1, :backward => bc1))
    f(x,y,z) = 6.0
    D(x,y,z) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    solve_DiffusionSteadyMono!(solver; method=Base.:\)
    u_analytic(x,y,z) = 1.0 - (x-2)^2 - (y-2)^2 - (z-2)^2
    u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytic, solver, capacity, 2, false)
    @test global_err < 1e-2
end

@testset "Convergence Test Unsteady Mono 1D" begin
    nx = 40
    lx = 4.0
    x0 = 0.0
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    center = 2.0
    radius = 1.0
    LS(x,_=0) = abs(x-center) - radius
    capacity = Capacity(LS, mesh)
    operator = DiffusionOps(capacity)
    bc = Dirichlet(0.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc))
    f(x,y,z,t) = 0.0
    D(x,y,z) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    u0ₒ = zeros(nx+1)
    u0ᵧ = zeros(nx+1)
    u0 = vcat(u0ₒ, u0ᵧ)
    Δt = 0.25 * (lx/nx)^2
    Tend = 0.01
    solver = DiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, "BE")
    solve_DiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, bc, "BE"; method=IterativeSolvers.gmres)
    # Analytical solution for homogeneous Dirichlet and zero source: stays zero
    u_analytic(x,t) = 0.0
    u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence((x)->u_analytic(x,Tend), solver, capacity, 2, false)
    @test global_err < 1e-8
end

@testset "Convergence Test Diphasic 1D" begin
    # Define mesh
    nx = 100
    lx = 8.0
    x0 = 0.0
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    
    # Interface position
    xint = 4.0
    
    # Define level set functions for each phase
    body = (x, _=0) -> (x - xint)
    body_c = (x, _=0) -> -(x - xint)
    
    # Create capacities
    capacity = Capacity(body, mesh)
    capacity_c = Capacity(body_c, mesh)
    
    # Define operators
    operator = DiffusionOps(capacity)
    operator_c = DiffusionOps(capacity_c)
    
    # Define boundary conditions
    bc1 = Dirichlet(0.0)
    bc0 = Dirichlet(1.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc0, :bottom => bc1))
    
    # Interface conditions
    He = 0.5
    ic = InterfaceConditions(ScalarJump(1.0, He, 0.0), FluxJump(1.0, 1.0, 0.0))
    
    # Define source and diffusion coefficients
    f1 = (x,y,z,t)->0.0
    f2 = (x,y,z,t)->0.0
    D1 = 1.0
    D2 = 1.0
    D1_func = (x,y,z)->D1
    D2_func = (x,y,z)->D2
    
    # Define phases
    Fluide_1 = Phase(capacity, operator, f1, D1_func)
    Fluide_2 = Phase(capacity_c, operator_c, f2, D2_func)
    
    # Define initial condition
    u0ₒ1 = zeros(nx+1)
    u0ᵧ1 = zeros(nx+1)
    u0ₒ2 = ones(nx+1)
    u0ᵧ2 = ones(nx+1)
    u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)
    
    # Define time parameters
    Δt = 0.5 * (lx/nx)^2
    Tend = 0.5
    
    # Initialize solver
    solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "BE")
    
    # Solve the problem
    solve_DiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, Δt, Tend, bc_b, ic, "BE"; method=IterativeSolvers.bicgstabl)
    
    # Analytical solutions
    function T1(x)
        t = Tend
        x = x - xint
        return - He/(1+He*sqrt(D1/D2))*(erfc(x/(2*sqrt(D1*t))) - 2)
    end

    function T2(x)
        t = Tend
        x = x - xint
        return - He/(1+He*sqrt(D1/D2))*erfc(x/(2*sqrt(D2*t))) + 1
    end
    
    # Check convergence
    (ana_sols, num_sols, global_errs, full_errs, cut_errs, empty_errs) = 
        check_convergence_diph(T1, T2, solver, capacity, capacity_c, 2, false)
    
    # Extract errors
    (err1, err2, err_combined) = global_errs
    (err1_full, err2_full, err_full_combined) = full_errs
    (err1_cut, err2_cut, err_cut_combined) = cut_errs
    
    # Test assertions
    @test err1 < 1e-2
    @test err2 < 1e-2
    @test err_combined < 1e-2
    
    # Optional: Also test full and cut cell errors separately
    @test err1_full < 1e-2
    @test err2_full < 1e-2
    @test err1_cut < 5e-2  # Cut cells typically have larger errors
    @test err2_cut < 5e-2
end

@testset "Convergence Order Diphasic 1D" begin
    # Test mesh convergence order
    nx_list = [40, 80, 160]
    lx = 8.0
    x0 = 0.0
    xint = 4.0
    Tend = 0.5
    He = 0.5
    D1 = 1.0
    D2 = 1.0
    
    # Analytical solutions for diphasic heat transfer
    function T1(x)
        t = Tend
        x = x - xint
        return - He/(1+He*sqrt(D1/D2))*(erfc(x/(2*sqrt(D1*t))) - 2)
    end

    function T2(x)
        t = Tend
        x = x - xint
        return - He/(1+He*sqrt(D1/D2))*erfc(x/(2*sqrt(D2*t))) + 1
    end
    
    # Storage arrays
    h_vals = Float64[]
    err1_vals = Float64[]
    err2_vals = Float64[]
    err_combined_vals = Float64[]
    
    # For each mesh resolution
    for nx in nx_list
        # Build mesh
        mesh = Penguin.Mesh((nx,), (lx,), (x0,))
        
        # Define level set functions
        body = (x, _=0) -> (x - xint)
        body_c = (x, _=0) -> -(x - xint)
        
        # Create capacities
        capacity = Capacity(body, mesh)
        capacity_c = Capacity(body_c, mesh)
        
        # Define operators
        operator = DiffusionOps(capacity)
        operator_c = DiffusionOps(capacity_c)
        
        # Define boundary conditions
        bc1 = Dirichlet(0.0)
        bc0 = Dirichlet(1.0)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc0, :bottom => bc1))
        
        # Interface conditions
        ic = InterfaceConditions(ScalarJump(1.0, He, 0.0), FluxJump(1.0, 1.0, 0.0))
        
        # Define source and diffusion coefficients
        f1 = (x,y,z,t)->0.0
        f2 = (x,y,z,t)->0.0
        D1_func = (x,y,z)->D1
        D2_func = (x,y,z)->D2
        
        # Define phases
        Fluide_1 = Phase(capacity, operator, f1, D1_func)
        Fluide_2 = Phase(capacity_c, operator_c, f2, D2_func)
        
        # Initial condition
        u0ₒ1 = zeros(nx+1)
        u0ᵧ1 = zeros(nx+1)
        u0ₒ2 = ones(nx+1)
        u0ᵧ2 = ones(nx+1)
        u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)
        
        # Time parameters
        Δt = 0.5 * (lx/nx)^2
        
        # Initialize solver
        solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "CN")
        
        # Solve the problem
        solve_DiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, Δt, Tend, bc_b, ic, "CN"; method=IterativeSolvers.gmres)
        
        # Check convergence
        (ana_sols, num_sols, global_errs, full_errs, cut_errs, empty_errs) = 
            check_convergence_diph(T1, T2, solver, capacity, capacity_c, 2, false)
        
        # Extract errors
        (err1, err2, err_combined) = global_errs
        
        # Store mesh size and errors
        push!(h_vals, lx / nx)
        push!(err1_vals, err1)
        push!(err2_vals, err2)
        push!(err_combined_vals, err_combined)
    end
    
    # Calculate convergence orders (slopes in log-log scale)
    function calculate_order(h_vals, err_vals)
        log_h = log.(h_vals)
        log_err = log.(err_vals)
        # Fit a line to log-log data: log(err) = p*log(h) + c
        p = (log_err[end] - log_err[1]) / (log_h[end] - log_h[1])
        return p
    end
    
    order1 = calculate_order(h_vals, err1_vals)
    order2 = calculate_order(h_vals, err2_vals)
    order_combined = calculate_order(h_vals, err_combined_vals)
    
    # The method should be at least first order accurate
    @test order1 > 0.9
    @test order2 > 0.9
    @test order_combined > 0.9
    
    # For CN scheme, we expect second order convergence
    # But with interface conditions, might drop to first order near interface
    @test order1 < 2.2  # Upper bound to catch unexpected behavior
    @test order2 < 2.2
    @test order_combined < 2.2
end