using Penguin
using IterativeSolvers
using LinearAlgebra, SparseArrays
using CairoMakie
using Printf

### 1D Mesh Convergence Study : Monophasic Steady Diffusion Equation with MMS
### Non-homogeneous boundary conditions

# Analytical solution function
function get_analytical_solution(Δx, lx, u_left, u_right)
    """
    Analytical solution with non-homogeneous BC:
    u(x) = u_left + (u_right - u_left) * sin(π*(x-Δx)/(lx-Δx))
    Or for a more general form:
    u(x) = u_left + (u_right - u_left) * (x-Δx)/(lx-Δx) + A*sin(π*(x-Δx)/(lx-Δx))
    """
    # Linear part to satisfy boundary conditions
    u_linear = (x) -> u_left + (u_right - u_left) * (x - Δx) / (lx - Δx)
    
    # Sinusoidal perturbation (homogeneous part)
    u_sin = (x) -> sin(π * (x - Δx) / (lx - Δx))
    
    # Combined solution: u = u_linear + A * u_sin
    # A is the amplitude of the sinusoidal perturbation
    A = 0.5
    
    return (x) -> u_linear(x) + A * u_sin(x)
end

# Source term function
function get_source_term(Δx, lx, u_left, u_right, A=0.5)
    """
    Source term for -∇²u = f
    For u = u_linear + A*sin(π*(x-Δx)/(lx-Δx))
    ∇²u_linear = 0
    ∇²u_sin = -(π/(lx-Δx))² * sin(π*(x-Δx)/(lx-Δx))
    So: f = A * (π/(lx-Δx))² * sin(π*(x-Δx)/(lx-Δx))
    """
    λ = (π / (lx - Δx))^2
    u_sin = (x) -> sin(π * (x - Δx) / (lx - Δx))
    
    return (x, y, _=0) -> A * λ * u_sin(x)
end

# Parameters
lx = 1.0
x0 = 0.0
center = 0.5
radius = 0.1

# Non-homogeneous boundary conditions
u_left = 1.0   # u(0) = 1.0
u_right = 2.0  # u(1) = 2.0
A = 0.5        # Amplitude of sinusoidal perturbation

# Define the body (same as original)
body = (x, _=0) -> -1.0

# Boundary conditions: Non-homogeneous Dirichlet
bc = Dirichlet(u_left)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :bottom => Dirichlet(u_left), 
    :top => Dirichlet(u_right)
))

# Mesh sizes to test
mesh_sizes = [5, 10, 20, 40, 80, 160]
errors_l2 = Float64[]
errors_linf = Float64[]
mesh_spacings = Float64[]

println("="^70)
println("MESH CONVERGENCE STUDY - 1D Poisson with Non-Homogeneous BC")
println("="^70)
println("Boundary conditions: u(0) = $u_left, u(1) = $u_right")
println("Sinusoidal amplitude: A = $A")

for nx in mesh_sizes
    Δx = lx / nx
    
    println("\nMesh size: nx = $nx, Δx = $Δx")
    
    # Create mesh
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    
    # Define the capacity
    capacity = Capacity(body, mesh; compute_centroids=false)
    
    # Define the operators
    operator = DiffusionOps(capacity)
    
    # Exact solution
    u_exact = get_analytical_solution(Δx, lx, u_left, u_right)
    
    # Source term
    g = get_source_term(Δx, lx, u_left, u_right, A)
    a = (x, y, _=0) -> 1.0
    
    # Create phase and solver
    Fluide = Phase(capacity, operator, g, a)
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    
    # Solve the problem
    solve_DiffusionSteadyMono!(solver; method=Base.:\)
    
    # Get numerical solution
    u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_exact, solver, capacity, 2)
    
    # Store results
    push!(mesh_spacings, Δx)
    push!(errors_l2, global_err)
    push!(errors_linf, maximum(abs.(u_ana[1:nx] .- u_num[1:nx])))
    
    println("  L2 Error:   $(@sprintf("%.6e", global_err))")
    println("  L∞ Error:   $(@sprintf("%.6e", errors_linf[end]))")
end

println("\n" * "="^70)
println("CONVERGENCE SUMMARY")
println("="^70)
println("\nMesh spacing | L2 Error      | L∞ Error      | L2 Rate | L∞ Rate")
println("-"^70)

for i in 1:length(mesh_sizes)
    rate_l2 = i > 1 ? log(errors_l2[i]/errors_l2[i-1])/log(mesh_spacings[i]/mesh_spacings[i-1]) : NaN
    rate_linf = i > 1 ? log(errors_linf[i]/errors_linf[i-1])/log(mesh_spacings[i]/mesh_spacings[i-1]) : NaN
    
    @printf("%12.6f | %13.6e | %13.6e | %7.3f | %7.3f\n", 
            mesh_spacings[i], errors_l2[i], errors_linf[i], rate_l2, rate_linf)
end

# Plot convergence
fig = Figure(size=(1000, 600))

# L2 Error plot
ax1 = Axis(fig[1, 1], xlabel="Mesh spacing (Δx)", ylabel="L2 Error",
           xscale=log10, yscale=log10, title="L2 Error Convergence")
scatter!(ax1, mesh_spacings, errors_l2, label="L2 Error", color=:blue, markersize=10)
lines!(ax1, mesh_spacings, errors_l2, color=:blue, linewidth=2)

# Add reference lines for convergence rates
if length(mesh_spacings) > 1
    # Second order reference
    ref_l2 = errors_l2[1] * (mesh_spacings ./ mesh_spacings[1]).^2
    lines!(ax1, mesh_spacings, ref_l2, linestyle=:dot, color=:gray, label="O(h²)", linewidth=2)
end

Legend(fig[1, 2], ax1; orientation=:vertical)

# L∞ Error plot
ax2 = Axis(fig[2, 1], xlabel="Mesh spacing (Δx)", ylabel="L∞ Error",
           xscale=log10, yscale=log10, title="L∞ Error Convergence")
scatter!(ax2, mesh_spacings, errors_linf, label="L∞ Error", color=:red, markersize=10)
lines!(ax2, mesh_spacings, errors_linf, color=:red, linewidth=2)

if length(mesh_spacings) > 1
    ref_linf = errors_linf[1] * (mesh_spacings ./ mesh_spacings[1]).^2
    lines!(ax2, mesh_spacings, ref_linf, linestyle=:dot, color=:gray, label="O(h²)", linewidth=2)
end

Legend(fig[2, 2], ax2; orientation=:vertical)

display(fig)

println("\nConvergence study complete!")