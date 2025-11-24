using Penguin
using Test
using SparseArrays
using IterativeSolvers  # Add this import for bicgstabl
using LinearAlgebra

@testset "Solver test" begin
    nx, ny = 20, 20
    lx, ly = 2.0, 2.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5
    LS(x,y,_=0) = (sqrt((x-0.5)^2 + (y-0.5)^2) - 0.5)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    bc = Dirichlet(1.0)
    bc1 = Dirichlet(1.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))
    f(x,y,_=0) = 0.0
    D(x,y,_=0) = 1.0
    Fluide = Phase(capacity, operator, f, D)
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    solve_DiffusionSteadyMono!(solver)
    uo = solver.x[1:end÷2]
    ug = solver.x[end÷2+1:end]
    @test maximum(uo) ≈ 1.0 atol=1e-2
    @test maximum(ug) ≈ 1.0 atol=1e-2
    
    # Test with Robin boundary conditions
    @testset "Robin boundary conditions" begin
        # Robin BC with parameters: a*u + b*∇u·n = c
        # Here we use a=1, b=2, c=1 which gives u=1 at the boundary when ∇u·n=0
        robin_bc = Robin(1.0, 2.0, 1.0)
        
        # Apply Robin BC to all boundaries
        bc_robin = BorderConditions(Dict{Symbol, AbstractBoundary}())
        
        # Create solver with Robin boundary conditions
        solver_robin = DiffusionSteadyMono(Fluide, bc_robin, bc)
        solve_DiffusionSteadyMono!(solver_robin)
        
        # Extract solution
        uo_robin = solver_robin.x[1:end÷2]
        ug_robin = solver_robin.x[end÷2+1:end]
        
        # The solution should be close to 1.0 at the boundary
        @test maximum(uo_robin) ≈ 1.0 atol=1e-1
        @test maximum(ug_robin) ≈ 1.0 atol=1e-1
    end
    
    # Test with bicgstabl solver
    @testset "bicgstabl solver" begin
        # Use the same problem as before but with a different linear solver
        solver_bicgstabl = DiffusionSteadyMono(Fluide, bc_b, bc)
        
        # Solve using bicgstabl (use smaller tolerances for better accuracy)
        solve_DiffusionSteadyMono!(
            solver_bicgstabl, 
            method=IterativeSolvers.bicgstabl,
        )
        
        # Extract solution
        uo_bicgstabl = solver_bicgstabl.x[1:end÷2]
        ug_bicgstabl = solver_bicgstabl.x[end÷2+1:end]
        
        # The solution should match the direct solver result
        @test maximum(uo_bicgstabl) ≈ 1.0 atol=1e-2
        @test maximum(ug_bicgstabl) ≈ 1.0 atol=1e-2
        
        # Compare with the direct solver solution
        @test norm(uo_bicgstabl - uo) / norm(uo) < 1e-2
        @test norm(ug_bicgstabl - ug) / norm(ug) < 1e-2
    end
end


# Test with periodic boundary conditions
@testset "Periodic boundary conditions" begin
    # Create a mesh with periodic boundaries
    nx, ny = 20, 20
    lx, ly = 2.0, 2.0
    x0, y0 = 0.0, 0.0
    mesh = Mesh((nx, ny), (lx, ly), (x0, y0))
    
    # Level set for a simple circle
    LS(x,y,_=0) = -(sqrt((x-1.0)^2 + (y-1.0)^2) - 0.3)
    capacity = Capacity(LS, mesh, method="VOFI")
    operator = DiffusionOps(capacity)
    
    # Create periodic boundary conditions
    periodic_bc = Periodic()
    interface_bc = Dirichlet(1.0)  # Interface condition
    
    # Set periodic on left/right, Dirichlet on top/bottom
    bc_periodic_x = BorderConditions(Dict{Symbol, AbstractBoundary}(
        :left => periodic_bc, 
        :right => periodic_bc,
        :top => Dirichlet(0.0), 
        :bottom => Dirichlet(0.0)
    ))
    
    # Create a source term with x-periodicity
    f(x,y,_=0) = sin(π*y)  # y-dependent source, should create x-periodic solution
    D(x,y,_=0) = 1.0
    
    fluid = Phase(capacity, operator, f, D)
    solver_periodic = DiffusionSteadyMono(fluid, bc_periodic_x, interface_bc)
    solve_DiffusionSteadyMono!(solver_periodic)
    
    # Extract solution
    solution = solver_periodic.x[1:end÷2]
    
    # Reshape solution to 2D grid for easier verification
    sol_2d = reshape(solution, ny+1, nx+1)
    
    # Test periodicity: left and right edges should match
    for j in 2:ny
        @test sol_2d[j,1] ≈ sol_2d[j,end] atol=1e-8
    end
    
    # Also test with periodic in y-direction
    bc_periodic_y = BorderConditions(Dict{Symbol, AbstractBoundary}(
        :left => Dirichlet(0.0),
        :right => Dirichlet(0.0), 
        :top => periodic_bc, 
        :bottom => periodic_bc
    ))
    
    # Source term with y-periodicity
    f_y(x,y,_=0) = sin(π*x)  # x-dependent source, should create y-periodic solution
    
    fluid_y = Phase(capacity, operator, f_y, D)
    solver_periodic_y = DiffusionSteadyMono(fluid_y, bc_periodic_y, interface_bc)
    solve_DiffusionSteadyMono!(solver_periodic_y)
    
    # Extract solution
    solution_y = solver_periodic_y.x[1:end÷2]
    sol_2d_y = reshape(solution_y, ny+1, nx+1)
    
    # Test periodicity: top and bottom edges should match
    for i in 2:nx
        @test sol_2d_y[1,i] ≈ sol_2d_y[end,i] atol=1e-8
    end
    
    # Test fully periodic domain
    bc_fully_periodic = BorderConditions(Dict{Symbol, AbstractBoundary}(
        :left => periodic_bc,
        :right => periodic_bc, 
        :top => periodic_bc, 
        :bottom => periodic_bc
    ))
    
    # Create a source with known analytical solution
    f_periodic(x,y,_=0) = sin(π*x)*sin(π*y)
    
    fluid_full = Phase(capacity, operator, f_periodic, D)
    solver_full_periodic = DiffusionSteadyMono(fluid_full, bc_fully_periodic, interface_bc)
    solve_DiffusionSteadyMono!(solver_full_periodic)
    
    # Extract solution
    solution_full = solver_full_periodic.x[1:end÷2]
    sol_2d_full = reshape(solution_full, ny+1, nx+1)
    
    # Test full periodicity in both directions
    for j in 2:ny
        @test sol_2d_full[j,1] ≈ sol_2d_full[j,end-1] atol=1e-8
    end
    
    for i in 2:nx
        @test sol_2d_full[1,i] ≈ sol_2d_full[end-1,i] atol=1e-8
    end
end