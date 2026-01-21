"""
Mesh and time-step convergence study for one-phase Stefan problem.
Varies mesh resolution and initial time, recording errors to CSV.
"""

using Penguin
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using Roots
using CSV
using DataFrames

# ============================================================================
# Analytical solution functions
# ============================================================================

function find_lambda(Stefan_number)
    f = (λ) -> sqrt(pi)*λ*exp(λ^2)*erf(λ) - 1/Stefan_number
    lambda = find_zero(f, (1e-6, 10.0), Bisection())
    return lambda
end

function analytical_temperature(x, t, T₀, k, lambda)
    if t <= 0
        return T₀
    end
    return T₀ - (T₀/erf(lambda)) * (erf(x/(2*sqrt(k*t))))
end

function analytical_position(t, k, lambda)
    return 2*lambda*sqrt(t)
end

# ============================================================================
# Convergence study
# ============================================================================

# Test parameters
mesh_sizes = [8, 16, 32, 64, 128]
t_starts = [0.001, 0.01, 0.1, 0.2, 0.4]
delta_t_sim = 0.1  # Tend = Tstart + 0.1

# Stefan parameters (fixed)
Stefan_number = 1.0
lambda = find_lambda(Stefan_number)
lx = 1.0
x0 = 0.0

# Storage for results
results = DataFrame(
    nx = Int[],
    Tstart = Float64[],
    Tend = Float64[],
    dx = Float64[],
    L1_error = Float64[],
    L2_error = Float64[],
    rel_pos_error = Float64[],
    stefan_residual_max = Float64[],
    stefan_residual_last = Float64[]
)

println("Starting convergence study...")
println("Mesh sizes: $mesh_sizes")
println("Starting times: $t_starts")
println("\n")

# Loop over all combinations
total_runs = length(mesh_sizes) * length(t_starts)
run_count = 0

for nx in mesh_sizes
    for Tstart in t_starts
        global run_count += 1
        Tend = Tstart + delta_t_sim
        dx = lx / nx
        dt = 0.5 * dx^2
        
        println("[$run_count/$total_runs] nx=$nx, Tstart=$Tstart, Tend=$Tend, dx=$dx, dt=$dt")
        
        try
            # Build mesh and space-time mesh
            mesh = Penguin.Mesh((nx,), (lx,), (x0,))
            x_offset = mesh.nodes[1][1] - x0
            STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Tstart + dt], tag=mesh.tag)
            
            # Initial interface position
            xf0_phys = analytical_position(Tstart, 1.0, lambda)
            xf = xf0_phys + x_offset
            body = (x, t, _=0) -> (x - xf)
            
            # Capacity and operators
            capacity = Capacity(body, STmesh)
            operator = DiffusionOps(capacity)
            
            # Boundary conditions
            bc = Dirichlet(0.0)
            bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
                :top => Dirichlet(0.0),
                :bottom => Dirichlet(1.0)
            ))
            ρ, L = 1.0, 1.0
            stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))
            
            # Source and diffusion coefficient
            f = (x, y, z, t) -> 0.0
            K = (x, y, z) -> 1.0
            
            # Phase definition
            Fluide = Phase(capacity, operator, f, K)
            
            # Initial condition
            x_nodes = mesh.nodes[1]
            x_nodes_phys = x_nodes .- x_offset
            u0ₒ = analytical_temperature.(x_nodes_phys, Tstart, 1.0, 1.0, lambda)
            u0ᵧ = zeros((nx+1))
            u0 = vcat(u0ₒ, u0ᵧ)
            
            # Solver
            solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, dt, u0, mesh, "CN")
            
            # Solve using simplified direct method
            solver, xf_log, stefan_residuals = solve_MovingLiquidDiffusionUnsteadyMono_Simple!(
                solver, Fluide, xf, dt, Tstart, Tend, bc_b, bc, stef_cond, mesh, "CN";
                method=Base.:\, max_inner_iter=1, tol=1e-8, damping=1.0
            )
            
            # Post-processing
            xf_log_phys = xf_log .- x_offset
            xf_num = xf_log_phys[end]
            xf_exact_Tend = analytical_position(Tend, 1.0, lambda)
            
            # Interface position error
            abs_pos_err = abs(xf_num - xf_exact_Tend)
            rel_pos_err = abs_pos_err / (abs(xf_exact_Tend) > 0 ? abs(xf_exact_Tend) : eps())
            
            # Extract Stefan residuals: max and last
            stefan_residual_max = NaN
            stefan_residual_last = NaN
            if !isempty(stefan_residuals)
                # Collect all residuals across all timesteps
                all_residuals = Float64[]
                for (_, resid_vec) in stefan_residuals
                    append!(all_residuals, resid_vec)
                end
                if !isempty(all_residuals)
                    stefan_residual_max = maximum(all_residuals)
                    stefan_residual_last = all_residuals[end]  # last residual overall
                end
            end
            
            # Temperature errors
            u_num = solver.x[1:(nx+1)]
            x_num = x_nodes_phys
            mask = x_num .<= xf_num
            
            if sum(mask) > 0
                u_anal_at_num = analytical_temperature.(x_num, Tend, 1.0, 1.0, lambda)
                u_num_below = u_num[mask]
                u_anal_below = u_anal_at_num[mask]
                L1_error = sum(abs.(u_num_below .- u_anal_below)) / length(u_num_below)
                L2_error = sqrt(sum((u_num_below .- u_anal_below).^2) / length(u_num_below))
            else
                L1_error = NaN
                L2_error = NaN
            end
            
            # Store results
            push!(results, (nx, Tstart, Tend, dx, L1_error, L2_error, rel_pos_err, stefan_residual_max, stefan_residual_last))
            
            println("  → L1=$L1_error, L2=$L2_error, rel_pos_err=$rel_pos_err, stefan_max=$stefan_residual_max ✓")
            
        catch e
            println("  → ERROR: $e ✗")
            push!(results, (nx, Tstart, Tend, dx, NaN, NaN, NaN, NaN, NaN))
        end
    end
end

# Save results to CSV
output_file = "convergence_results.csv"
CSV.write(output_file, results)
println("\n" * "="^70)
println("Results saved to: $output_file")
println("="^70)

# Print summary
println("\nSummary Table (Tstart=0.001):")
subset = filter(row -> row.Tstart == 0.001, results)
println(subset)

println("\nSummary Table (Tstart=0.1):")
subset = filter(row -> row.Tstart == 0.1, results)
println(subset)

println("\nDone!")
