using Penguin, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using Statistics

# Define the flower-shaped body
function flower_sdf(x, y, _=0)
    # Center the flower at (0.5, 0.5)
    cx, cy = 0.5, 0.5
    x, y = x - cx, y - cy
    
    # Convert to polar coordinates
    r = sqrt(x^2 + y^2)
    θ = atan(y, x)
    
    # SDF: negative inside, positive outside
    return -(r - (0.25 + 0.05 * cos(6 * θ)))
end

function flower_test(nx=400, ny=400)
    # Parameters for the domains
    # Flower domain Υ₁ = {(r, θ) : r ≥ 0.25 + 0.05 cos 6θ}
    # We'll place it inside a square domain Υ₂ = [0,1]×[0,1]
    
    # Mesh parameters
    lx, ly = 1.0, 1.0  # Domain size
    n = (nx+1) * (ny+1)  # Number of nodes in the mesh
    
    # Create the mesh
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))
    
    # Define the flower-shaped body
    function flower_sdf(x, y, _=0)
        # Center the flower at (0.5, 0.5)
        cx, cy = 0.5, 0.5
        x, y = x - cx, y - cy
        
        # Convert to polar coordinates
        r = sqrt(x^2 + y^2)
        θ = atan(y, x)
        
        # SDF: negative inside, positive outside
        return -(r - (0.25 + 0.05 * cos(6 * θ)))
    end
    
    # Define capacity
    capacity = Capacity(flower_sdf, mesh)
    
    # Define the operator (Laplace operator ∇²φ = 0)
    operator = DiffusionOps(capacity)
    
    # Boundary conditions:
    # φ = 1 on ∂Υ₁ (flower boundary)
    # φ = 0 on ∂Υ₂ (outer square boundary)
    bc_outer = Dirichlet(0.0)
    bc_boundary = BorderConditions(Dict(
        :left   => bc_outer,
        :right  => bc_outer,
        :top    => bc_outer,
        :bottom => bc_outer
    ))
    
    # Set up the phase with zero source term (Laplace equation)
    phase = Phase(capacity, operator, (x,y,_)->0.0, (x,y,_)->1.0)
    
    # Set up solver with Dirichlet condition φ = 1 on the immersed boundary
    solver = DiffusionSteadyMono(phase, bc_boundary, Dirichlet(1.0))
    
    # Solve the system
    solve_DiffusionSteadyMono!(solver; method=Base.:\)
    
    return solver, capacity, mesh
end

function plot_solution_(solver, mesh, sdf_func, capacity, nx, ny)
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], aspect=DataAspect(), 
             xlabel="x", ylabel="y", 
             title="Laplace Solution on Flower Domain")
    
    # Extract the solution
    n = (nx+1) * (ny+1)
    u = solver.x[1:n]
    u = reshape(u, (nx+1, ny+1))
    
    # Create a heatmap
    hm = heatmap!(ax, 0:1/nx:1, 0:1/ny:1, u, 
                 colormap=:viridis, 
                 colorrange=(0, 1))
    
    # Add colorbar
    cb = Colorbar(fig[1, 2], hm, label="φ")
    
    # Draw the flower boundary
    xs = Float64[]
    ys = Float64[]
    for θ in range(0, 2π, length=100)
        r = 0.25 + 0.05 * cos(6 * θ)
        push!(xs, 0.5 + r * cos(θ))
        push!(ys, 0.5 + r * sin(θ))
    end
    lines!(ax, xs, ys, color=:white, linewidth=2)
    
    display(fig)
    save("flower_solution.png", fig)
    return fig
end

function analyze_overshoot(solver, capacity, mesh)
    # Extract solution
    nx = length(mesh.nodes[1])
    ny = length(mesh.nodes[2])
    n = (nx+1) * (ny+1)

    u = solver.x[1:n]
    u = reshape(u, (nx+1, ny+1))

    
    # Find cells near the boundary (cut cells)
    cut_cells = findall(capacity.cell_types .== 1)
    
    # Extract values at cut cells
    cut_values = u[cut_cells]
    
    # Calculate overshoot statistics
    overshoot_cells = cut_values .> 1.0
    num_overshoot = sum(overshoot_cells)
    percent_overshoot = 100 * num_overshoot / length(cut_cells)
    max_value = maximum(cut_values)
    
    println("Overshoot analysis:")
    println("  - Number of cut cells: $(length(cut_cells))")
    println("  - Number of cells with overshoot: $num_overshoot")
    println("  - Percentage of cells with overshoot: $(round(percent_overshoot, digits=2))%")
    println("  - Maximum value: $(round(max_value, digits=6)) (should be ≤ 1.0)")
    
    # Plot overshoot along the boundary
    fig = Figure(resolution=(800, 400))
    ax = Axis(fig[1, 1], xlabel="Cell index", ylabel="Solution value", 
              title="Solution values at cut cells")
    
    # Sort by value for better visualization
    sorted_idx = sortperm(cut_values)
    
    # Plot horizontal line at y=1.0
    hlines!(ax, 1.0, color=:red, linestyle=:dash, label="Max allowed")
    
    # Plot cut cell values
    scatter!(ax, 1:length(cut_values), cut_values[sorted_idx], 
             markersize=4, color=:blue, label="Cell values")
    
    axislegend(ax, position=:lt)
    
    display(fig)
    save("boundary_overshoot.png", fig)
    
    return num_overshoot, length(cut_cells), percent_overshoot, max_value, fig
end

function bilinear_interpolate(fine_sol, i, j)
    # Get the four surrounding points from the fine grid
    v00 = fine_sol[2*i-1, 2*j-1]
    v10 = fine_sol[2*i, 2*j-1]
    v01 = fine_sol[2*i-1, 2*j]
    v11 = fine_sol[2*i, 2*j]
    
    # Return the average (for cell-centered values this is bilinear interpolation with weights 0.25)
    return 0.25 * (v00 + v10 + v01 + v11)
end

function convergence_study()
    # Grid sizes to test
    grid_sizes = [20, 40, 80, 160, 320, 640, 1280]
    
    # Store results
    results = []
    
    # First, calculate the solution for each grid size
    solutions = []
    capacities = []
    
    for N in grid_sizes
        println("Solving for grid size N = $N")
        solver, capacity, mesh = flower_test(N, N)
        
        # Store the solution and capacity
        nx, ny = N, N
        n = (nx+1) * (ny+1)
        u = solver.x[1:n]
        push!(solutions, copy(u))
        push!(capacities, capacity)
        
        # Analyze overshoot for this grid
        num_overshoot, num_cut_cells, _, _, _ = analyze_overshoot(solver, capacity, mesh)
        push!(results, (N, 0, 0, 0, 0, num_overshoot, num_cut_cells)) # Will update errors later
    end
    
    # Calculate error between successive grid levels
    for i in 2:length(grid_sizes)
        N_coarse = grid_sizes[i-1]
        N_fine = grid_sizes[i]
        
        coarse_sol = solutions[i-1]
        fine_sol = solutions[i]

        coarse_sol = reshape(coarse_sol, (N_coarse+1, N_coarse+1))
        fine_sol = reshape(fine_sol, (N_fine+1, N_fine+1))
        
        # Interpolate fine solution to coarse grid
        interp_sol = zeros(N_coarse+1, N_coarse+1)
        
        for i_coarse in 1:N_coarse
            for j_coarse in 1:N_coarse
                # Only interpolate if this is a valid domain cell in both grids
                interp_sol[i_coarse, j_coarse] = bilinear_interpolate(fine_sol, i_coarse, j_coarse)
            end
        end
        
        # Calculate error norms
        errors = abs.(coarse_sol - interp_sol)
        valid_cells = findall(capacities[i-1].cell_types .>= 1)
        
        # L-infinity norm
        err_inf = maximum(errors[valid_cells])
        
        # L1 norm (normalized by cell count)
        err_l1 = sum(errors[valid_cells]) / length(valid_cells)
        
        # Calculate convergence rates
        r_inf = i > 2 ? (results[i-1][2] / err_inf) : 0
        r_l1 = i > 2 ? (results[i-1][3] / err_l1) : 0
        
        # Update results
        results[i-1] = (N_coarse, err_inf, err_l1, r_inf, r_l1, results[i-1][6], results[i-1][7])
    end
    
    # Display table
    println("\nConvergence Results (Johansen-Colella comparison):")
    println("N     ||ξ||∞       r     ||ξ||₁       r     Overshoot/Cut cells")
    println("----------------------------------------------------------")
    
    for (N, err_inf, err_l1, r_inf, r_l1, num_overshoot, num_cut) in results[1:end-1]
        @printf("%3d  %.2e  %3.1f  %.2e  %3.1f  %d/%d\n", 
                N, err_inf, r_inf, err_l1, r_l1, num_overshoot, num_cut)
    end
    
    # Create convergence plot
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], 
              xscale=log10, yscale=log10,
              xlabel="Grid size N", 
              ylabel="Error",
              title="Convergence analysis")
    
    N_values = [result[1] for result in results[1:end-1]]
    err_inf_values = [result[2] for result in results[1:end-1]]
    err_l1_values = [result[3] for result in results[1:end-1]]
    
    scatter!(ax, N_values, err_inf_values, marker=:circle, markersize=10, label="L∞ norm")
    lines!(ax, N_values, err_inf_values, linewidth=2)
    
    scatter!(ax, N_values, err_l1_values, marker=:square, markersize=10, label="L¹ norm")
    lines!(ax, N_values, err_l1_values, linewidth=2)
    
    # Reference lines
    ref_x = [minimum(N_values), maximum(N_values)]
    ref_y2 = [minimum(err_inf_values) * (maximum(N_values)/minimum(N_values))^2, minimum(err_inf_values)]
    lines!(ax, ref_x, ref_y2, linestyle=:dash, color=:black, label="O(h²)")
    
    axislegend(ax, position=:rt)
    
    display(fig)
    save("convergence_plot.png", fig)
    
    return results, fig
end

# Wrapper function to run all analyses
function run_flower_analysis()
    println("Running flower domain Laplace equation test")
    solver, capacity, mesh = flower_test()
    
    println("\nPlotting solution...")
    plot_solution_(solver, mesh, flower_sdf, capacity,400, 400)
    
    println("\nAnalyzing overshoot...")
    analyze_overshoot(solver, capacity, mesh)
    
    println("\nPerforming convergence study...")
    convergence_study()
    
    println("\nAnalysis complete. Results and figures saved.")
end

# Run the analysis
run_flower_analysis()