using Penguin
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using CairoMakie
using Roots

"""
Calculate the dimensionless parameter λ for the one-phase Stefan problem.
"""
function find_lambda(Stefan_number)
    f = (λ) -> sqrt(pi)*λ*exp(λ^2)*erf(λ) - 1/Stefan_number
    lambda = find_zero(f, (1e-6, 10.0), Bisection())
    return lambda
end

"""
Analytical temperature distribution for the one-phase Stefan problem.
"""
function analytical_temperature(x, t, T₀, k, lambda)
    if t <= 0
        return T₀
    end
    return T₀ - (T₀/erf(lambda)) * (erf(x/(2*sqrt(k*t))))
end

"""
Analytical interface position for the one-phase Stefan problem.
"""
function analytical_position(t, k, lambda)
    return 2*lambda*sqrt(t)
end


### 1D Test Case : One-phase Stefan Problem
# Define the spatial mesh
nx = 32
lx = 1.
x0 = 0.
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))
# Mesh nodes start at x0 + Δx/2; keep a consistent physical origin for exact solutions
x_offset = mesh.nodes[1][1] - x0

# Define the Space-Time mesh
Δt = 0.5*(lx/nx)^2  # Time step based on stability condition
Tstart = 0.03
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Tstart+Δt], tag=mesh.tag)

# Calculate Stefan number and λ and set initial interface consistently with Tstart
Stefan_number = 1.0
lambda = find_lambda(Stefan_number)
# initial interface position from analytical solution at Tstart (convert to mesh coordinates)
xf0_phys = analytical_position(Tstart, 1.0, lambda)
xf = xf0_phys + x_offset
body = (x,t, _=0)->(x - xf)

# Define the capacity
capacity = Capacity(body, STmesh)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
ρ, L = 1.0, 1.0
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))

# Define the source term
f = (x,y,z,t)-> 0.0
K = (x,y,z)-> 1.0

# Define the phase
Fluide = Phase(capacity, operator, f, K)

# Initial condition
# Initial condition: bulk initialized from analytical temperature at Tstart
x_nodes = mesh.nodes[1]
x_nodes_phys = x_nodes .- x_offset
u0ₒ = analytical_temperature.(x_nodes_phys, Tstart, 1.0, 1.0, lambda)
u0ᵧ = zeros((nx+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Newton parameters
max_iter = 1
tol = eps()
reltol = eps()
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
println("Solving the Stefan problem...")
solver, residuals, xf_log, timestep_history = solve_MovingLiquidDiffusionUnsteadyMono!(
    solver, Fluide, xf, Δt, Tstart, Tend, bc_b, bc, stef_cond, mesh, "BE"; 
    Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\
)
println("Simulation complete!")


# Define analytical solution function
T_analytical = (x) -> analytical_temperature(x, Tend, 1.0, 1.0, lambda)

# Plot 1: Temperature comparison
function plot_temperature_comparison(
    T_analytical,
    u_num,
    x_num,
    t_final::Float64,
    xf_final::Float64;
    save_path::String="temperature_comparison.png"
)
    # Create fine grid for analytical solution
    x_analytical = range(minimum(x_num), maximum(x_num), 2000)
    u_analytical = T_analytical.(x_analytical)
    
    # Create figure
    fig = Figure(resolution=(900, 600), fontsize=14)
    
    ax = Axis(
        fig[1, 1],
        xlabel = "Position (x)",
        ylabel = "Temperature",
        title = "Stefan Problem: Temperature Comparison at t = $t_final"
    )
    
    # Plot solutions
    lines!(ax, x_analytical, u_analytical, color=:red, linewidth=3, 
           label="Analytical Solution")
    scatter!(ax, x_num, u_num, color=:blue, markersize=6, 
           label="Numerical Solution")
    
    # Mark interface position
    vlines!(ax, xf_final, color=:black, linewidth=2, linestyle=:dash,
            label="Interface Position")
    hlines!(ax, 0.0, color=:green, linewidth=2, linestyle=:dot,
            label="Melting Temperature")
    
    # Add legend
    axislegend(ax, position=:rt, framevisible=true, backgroundcolor=(:white, 0.7))
    
    # Save the figure if requested
    if !isempty(save_path)
        save(save_path, fig, px_per_unit=4)
    end
    
    display(fig)
    return fig
end

# Plot 2: Interface position vs time
function plot_interface_position(
    xf_numerical,
    times_numerical;
    lambda=1.0,
    vertical_offset=0.0,
    save_path::String="interface_position.png"
)
    # Analytical curve from 0 -> max time
    t_analytical = range(0.0, maximum(times_numerical), 500)
    xf_analytical = analytical_position.(t_analytical, 1.0, lambda) .+ vertical_offset
    
    fig = Figure(resolution=(900, 600), fontsize=14)
    
    ax = Axis(
        fig[1, 1],
        xlabel = "Time (t)",
        ylabel = "Interface Position (x_f)",
        title = "Stefan Problem: Interface Position vs Time"
    )
    
    # Plot solutions
    offset_label = vertical_offset != 0.0 ? " (offset=$(vertical_offset))" : ""
    lines!(ax, t_analytical, xf_analytical, color=:red, linewidth=3, 
           label="Analytical Solution$(offset_label)")
    scatter!(ax, times_numerical, xf_numerical, color=:blue, markersize=8, 
           label="Numerical Solution")
    
    # Add legend
    axislegend(ax, position=:lt, framevisible=true, backgroundcolor=(:white, 0.7))
    
    # Save the figure if requested
    if !isempty(save_path)
        save(save_path, fig, px_per_unit=4)
    end

    #xlims!(ax, 0.01-4Δt, maximum(times_numerical)+4Δt)
    #ylims!(ax, 0.10, 0.13)
    
    display(fig)
    return fig
end


# Execute the plots
println("Plotting temperature comparison...")
Δt_eff = (Tend - Tstart) / length(xf_log)  # effective step from actually performed steps
times = Tstart .+ collect(1:length(xf_log)) .* Δt_eff  # xf_log stores positions at the end of each step
# Convert from mesh coordinates to physical coordinates
# Remove ONLY the constant mesh offset (Δx/2), not any numerical error offset
xf_log_phys = xf_log .- x_offset
x_num_phys = mesh.nodes[1] .- x_offset
plot_temperature_comparison(
    T_analytical,
    solver.x[1:(nx+1)],
    x_num_phys,
    Tend,
    xf_log_phys[end]
)

println("Plotting interface position...")
times = Tstart .+ collect(1:length(xf_log)) .* Δt  # xf_log stores positions at the end of each step
plot_interface_position(
    xf_log_phys,
    times,
    lambda=lambda,
    vertical_offset=0.0
)

# Final interface position (numerical and analytical at Tend)
xf_num = xf_log_phys[end]
xf_anal_Tend = analytical_position(times[end], 1.0, lambda)
abs_pos_err = abs(xf_num - xf_anal_Tend)
rel_pos_err = abs_pos_err / (abs(xf_anal_Tend) > 0 ? abs(xf_anal_Tend) : eps())
println("Final interface position: numerical=", xf_num, ", analytical=", xf_anal_Tend)
println("Absolute position error=", abs_pos_err, ", Relative error=", rel_pos_err)

# Compute L1 error for temperature at final time but only for points below final interface
body_tend = (x, _=0) -> (x - xf_log[end])  # Use the last interface position
capacity_tend = Capacity(body_tend, mesh)

u_num = solver.x[1:(nx+1)]
x_num = x_num_phys
mask = x_num .<= xf_num
if sum(mask) == 0
    println("No mesh nodes found below final numerical interface (xf=", xf_num, "). Skipping temperature error.")
else
    u_anal_at_num = analytical_temperature.(x_num, Tend, 1.0, 1.0, lambda)
    u_num_below = u_num[mask]
    u_anal_below = u_anal_at_num[mask]
    L1_error = sum(abs.(u_num_below .- u_anal_below)) / length(u_num_below)
    println("L1 (mean abs) temperature error over x <= xf_num at t=$(Tend): ", L1_error)
end

# Compute L2 error for temperature at final time but only for points below final interface
if sum(mask) == 0
    println("No mesh nodes found below final numerical interface (xf=", xf_num, "). Skipping temperature error.")
else
    u_anal_at_num = analytical_temperature.(x_num, Tend, 1.0, 1.0, lambda)
    u_num_below = u_num[mask]
    u_anal_below = u_anal_at_num[mask]
    L2_error = sqrt(sum((u_num_below .- u_anal_below).^2) / length(u_num_below))
    println("L2 (RMS) temperature error over x <= xf_num at t=$(Tend): ", L2_error)
end 

# Summary errors
println("Summary of Errors at t=$(Tend):")
println("Final interface position relative error: ", rel_pos_err)
println("L1 temperature error (x <= xf_num): ", sum(abs.(u_num_below .- u_anal_below)) / length(u_num_below))
println("L2 temperature error (x <= xf_num): ", sqrt(sum((u_num_below .- u_anal_below).^2) / length(u_num_below)))
println("  Nodes used: $(length(u_num_below)) / $(length(u_num))")

println("Done!")
