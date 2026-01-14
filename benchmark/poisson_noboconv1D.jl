using Penguin
using IterativeSolvers
using LinearAlgebra, SparseArrays
using CairoMakie
using Printf

### 1D Mesh Convergence Study : Monophasic Steady Diffusion Equation

# Generalized analytical solution construction
function get_analytical_solution(mesh, bc_b, Δx, lx)
    """
    Analytical solution for ∇²u = -1 with scaled mesh coordinates
    u(ξ) = -ξ²/2 + ξ/2, where ξ = (x - Δx)/(lx - Δx)
    This satisfies: u(0) = 0, u(1) = 0, ∇²u = -1
    """
    return (x) -> begin
        ξ = (x - Δx) / (lx - Δx)
        -ξ^2/2 + ξ/2
    end
end


#  Parameters (same as Poisson_nobody.jl)
lx = 1.0
x0 = 0.0
center = 0.5
radius = 0.1

# Define the body (same as original)
body = (x, _=0) -> -1.0

# Define the source term: f = -1 (so -∇²u = -1, or ∇²u = 1)
g = (x, y, _=0) -> 1.0
a = (x, y, _=0) -> 1.0

# Boundary conditions (same as original)
bc = Dirichlet(0.0)
bc1 = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(0.0)))

# Mesh sizes to test
mesh_sizes = [5, 10, 20, 40, 80, 160]
errors_l2 = Float64[]
errors_linf = Float64[]
mesh_spacings = Float64[]
u_num_list = []
u_ana_list = []
mesh_centers_list = []

println("="^60)
println("MESH CONVERGENCE STUDY")
println("="^60)

for nx in mesh_sizes
    Δx = lx / nx
    
    println("\nMesh size: nx = $nx, Δx = $Δx")
    
    # Create mesh
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    
    # Define the capacity
    capacity = Capacity(body, mesh; compute_centroids=false)
    
    # Define the operators
    operator = DiffusionOps(capacity)
    
    # Create phase
    Fluide = Phase(capacity, operator, g, a)
    
    # Define and solve
    solver = DiffusionSteadyMono(Fluide, bc_b, bc)
    solve_DiffusionSteadyMono!(solver; method=Base.:\)
    
    # Get analytical solution using the routine
    u_analytical = get_analytical_solution(mesh, bc_b, Δx, lx)
    
    # Get numerical solution
    u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytical, solver, capacity, 2)
    
    # Store results
    push!(mesh_spacings, Δx)
    push!(errors_l2, global_err)
    push!(errors_linf, maximum(abs.(u_ana[1:nx] .- u_num[1:nx])))
    push!(u_num_list, u_num)
    push!(u_ana_list, u_ana)
    push!(mesh_centers_list, mesh.centers[1])
    
    println("  L2 Error:   $(global_err)")
    println("  L∞ Error:   $(errors_linf[end])")
end

println("\n" * "="^60)
println("CONVERGENCE SUMMARY")
println("="^60)
println("\nMesh spacing | L2 Error      | L∞ Error      | L2 Rate | L∞ Rate")
println("-"^70)

for i in 1:length(mesh_sizes)
    rate_l2 = i > 1 ? log(errors_l2[i]/errors_l2[i-1])/log(mesh_spacings[i]/mesh_spacings[i-1]) : NaN
    rate_linf = i > 1 ? log(errors_linf[i]/errors_linf[i-1])/log(mesh_spacings[i]/mesh_spacings[i-1]) : NaN
    
    @printf("%12.6f | %13.6e | %13.6e | %7.3f | %7.3f\n", 
            mesh_spacings[i], errors_l2[i], errors_linf[i], rate_l2, rate_linf)
end

# Plot the numerical and analytical solutions
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="u",)
scatter!(ax, mesh_centers_list[end], u_num_list[end][1:end-1], label="Numerical Solution")
lines!(ax, mesh_centers_list[end], u_ana_list[end][1:end-1], label="Analytical Solution (from check_convergence)")
Legend(fig[1, 2], ax; orientation = :vertical)
display(fig)  

# Plot convergence
fig = Figure(size=(1000, 600))

# L2 Error plot
ax1 = Axis(fig[1, 1], xlabel="Mesh spacing (Δx)", ylabel="L2 Error",
           xscale=log10, yscale=log10, title="L2 Error Convergence")
scatter!(ax1, mesh_spacings, errors_l2, label="L2 Error", color=:blue, markersize=10)
lines!(ax1, mesh_spacings, errors_l2, color=:blue, linewidth=2)

# Add reference lines for convergence rates
if length(mesh_spacings) > 1
    # First order reference
    ref_l1 = errors_l2[1] * (mesh_spacings ./ mesh_spacings[1])
    lines!(ax1, mesh_spacings, ref_l1, linestyle=:dash, color=:gray, label="O(h)", linewidth=2)
    
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
    ref_l1 = errors_linf[1] * (mesh_spacings ./ mesh_spacings[1])
    lines!(ax2, mesh_spacings, ref_l1, linestyle=:dash, color=:gray, label="O(h)", linewidth=2)
    
    ref_l2 = errors_linf[1] * (mesh_spacings ./ mesh_spacings[1]).^2
    lines!(ax2, mesh_spacings, ref_l2, linestyle=:dot, color=:gray, label="O(h²)", linewidth=2)
end

Legend(fig[2, 2], ax2; orientation=:vertical)

display(fig)

println("\nConvergence study complete!")