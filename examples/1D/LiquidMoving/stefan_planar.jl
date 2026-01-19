using Penguin
using LinearAlgebra
using SparseArrays
using CairoMakie

"""
Analytical temperature distribution for the constant-velocity planar Stefan problem.
Matching `dirichlet_expand_1d.jl`: nonzero profile lives on the left of the interface (x ≤ Vt),
with `exp(+V * (x - Vt))` for the expanding case.
"""
function analytical_temperature_planar(x, t, V=1.0)
    x_interface = V * t
    if x <= x_interface
        return -1.0 + exp(-V * (x - V * t))  # Positive V for expansion
    end
    return 0.0
end

"""
Analytical interface position for constant-velocity planar Stefan problem.
"""
function analytical_position_planar(t, V=1.0)
    return V * t
end


### 1D Test Case : Constant-velocity planar Stefan problem
# Physical parameters
V = 1.0  # Interface velocity
T_eq = 0.0  # Interface temperature
T0_SHIFT = 1.0e-3  # Time shift to avoid singularity

# Define the spatial mesh
nx = 80
lx = 2.0
x0 = -0.5
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the Space-Time mesh
Tstart = 0.0  # Start at t=0
cfl = 0.1
dx = lx / nx
Δt = cfl * dx / max(abs(V), eps())
Tend = 0.2
nsteps = 20
Tend = nsteps * Δt

STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Tstart+Δt], tag=mesh.tag)

# Initial interface position with time shift
# At t=0, the effective time is t - T0_SHIFT (negative), so interface starts at negative x
xf = V * (Tstart - T0_SHIFT)
body = (x,t, _=0)->(x - xf)

# Define the capacity
capacity = Capacity(body, STmesh)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
# Bottom boundary: time-dependent based on analytical solution
# Top boundary: T = 0 (solid side)
bc_bottom = Dirichlet((x, t) -> analytical_temperature_planar(x, t - T0_SHIFT, V))
bc_top = Dirichlet(0.0)

bc_b = BorderConditions(
    Dict{Symbol, AbstractBoundary}(
        :bottom => bc_bottom,
        :top => bc_top
    )
)

# Stefan condition parameters
# Important: ρ = -1.0 for the expansion case (interface moves into solid)
ρ, L = 1.0, 1.0  # Negative density for expansion
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 0.0, ρ*L))

# Define the source term and diffusion coefficient
f = (x,y,z,t)-> 0.0
K = (x,y,z)-> 1.0

# Define the phase
Fluide = Phase(capacity, operator, f, K)

# Initial condition: bulk initialized from analytical temperature at effective time
# Effective time at t=0 is t - T0_SHIFT (negative)
x_nodes = mesh.nodes[1]
t_eff = Tstart + T0_SHIFT
u0ₒ = analytical_temperature_planar.(x_nodes, t_eff, V)
u0ᵧ = fill(T_eq, length(x_nodes))
u0 = vcat(u0ₒ, u0ᵧ)

# Newton parameters
max_iter = 10
tol = 1e-10
reltol = 1e-10
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
bc = Dirichlet(T_eq)
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
println("Solving the planar Stefan problem...")
println("Interface velocity V = $V")
println("Domain: x ∈ [$x0, $(x0+lx)]")
println("Time: t ∈ [$Tstart, $Tend]")

solver, residuals, xf_log, timestep_history = solve_MovingLiquidDiffusionUnsteadyMono!(
    solver, Fluide, xf, Δt, Tstart, Tend, bc_b, bc, stef_cond, mesh, "BE"; 
    Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\
)
println("Simulation complete!")


# Plot 1: Temperature comparison at initial and final times
function plot_temperature_evolution(
    x_nodes,
    u_initial,
    u_final,
    Tstart::Float64,
    Tend::Float64,
    xf_start::Float64,
    xf_end::Float64;
    x0=-0.5,
    lx=2.0,
    V=1.0,
    save_path::String="planar_temperature.png"
)
    # Create fine grid for analytical solutions
    x_analytical = range(x0, x0+lx, 2000)
    u_anal_start = analytical_temperature_planar.(x_analytical, Tstart, V)
    u_anal_end = analytical_temperature_planar.(x_analytical, Tend, V)
    
    # Create figure
    fig = Figure(resolution=(1200, 500), fontsize=14)
    
    # Initial time plot
    ax1 = Axis(
        fig[1, 1],
        xlabel = "Position (x)",
        ylabel = "Temperature",
        title = "Initial time t = $Tstart"
    )
    
    lines!(ax1, x_analytical, u_anal_start, color=:red, linewidth=3, 
           label="Analytical")
    scatter!(ax1, x_nodes, u_initial, color=:blue, markersize=6, 
             label="Numerical")
    vlines!(ax1, xf_start, color=:black, linewidth=2, linestyle=:dash,
            label="Interface")
    axislegend(ax1, position=:rb)
    
    # Final time plot
    ax2 = Axis(
        fig[1, 2],
        xlabel = "Position (x)",
        ylabel = "Temperature",
        title = "Final time t = $Tend"
    )
    
    lines!(ax2, x_analytical, u_anal_end, color=:red, linewidth=3, 
           label="Analytical")
    scatter!(ax2, x_nodes, u_final, color=:blue, markersize=6, 
             label="Numerical")
    vlines!(ax2, xf_end, color=:black, linewidth=2, linestyle=:dash,
            label="Interface")
    axislegend(ax2, position=:rb)
    
    # Save the figure if requested
    if !isempty(save_path)
        save(save_path, fig, px_per_unit=4)
    end
    
    display(fig)
    return fig
end

# Plot 2: Interface position vs time (should be linear with slope V)
function plot_interface_position(
    xf_numerical,
    times_numerical;
    V=1.0,
    save_path::String="planar_interface_position.png"
)
    # Analytical curve from 0 -> max time
    t_analytical = range(0.0, maximum(times_numerical), 500)
    xf_analytical = analytical_position_planar.(t_analytical, V)
    
    fig = Figure(resolution=(900, 600), fontsize=14)
    
    ax = Axis(
        fig[1, 1],
        xlabel = "Time (t)",
        ylabel = "Interface Position (x_f)",
        title = "Planar Stefan: Interface Position (V = $V)"
    )
    
    # Plot solutions
    lines!(ax, t_analytical, xf_analytical, color=:red, linewidth=3, 
           label="Analytical (x = V·t)")
    scatter!(ax, times_numerical, xf_numerical, color=:blue, markersize=8, 
             label="Numerical")
    
    # Add legend
    axislegend(ax, position=:lt, framevisible=true, backgroundcolor=(:white, 0.7))
    
    # Save the figure if requested
    if !isempty(save_path)
        save(save_path, fig, px_per_unit=4)
    end
    
    display(fig)
    return fig
end

# Plot 3: L1 error evolution over time
function plot_error_evolution(
    times,
    L1_errors,
    position_errors;
    save_path::String="planar_errors.png"
)
    fig = Figure(resolution=(1200, 500), fontsize=14)
    
    # Temperature error plot
    ax1 = Axis(
        fig[1, 1],
        xlabel = "Time (t)",
        ylabel = "L1 Temperature Error",
        yscale = log10,
        title = "Temperature Error Evolution"
    )
    
    lines!(ax1, times, L1_errors, color=:blue, linewidth=2, marker=:circle)
    
    # Position error plot
    ax2 = Axis(
        fig[1, 2],
        xlabel = "Time (t)",
        ylabel = "Absolute Position Error",
        yscale = log10,
        title = "Interface Position Error"
    )
    
    lines!(ax2, times, position_errors, color=:red, linewidth=2, marker=:circle)
    
    if !isempty(save_path)
        save(save_path, fig, px_per_unit=4)
    end
    
    display(fig)
    return fig
end


# Execute the plots
println("\nPlotting temperature evolution...")
u_initial = u0[1:(nx+1)]
u_final = solver.x[1:(nx+1)]

plot_temperature_evolution(
    x_nodes,
    u_initial,
    u_final,
    Tstart,
    Tend,
    xf,
    xf_log[end],
    x0=x0,
    lx=lx,
    V=V
)

println("Plotting interface position...")
times = Tstart .+ collect(0:length(xf_log)-1) .* Δt
# For analytical comparison, use effective time with shift
times_analytical = times .- T0_SHIFT
plot_interface_position(
    xf_log,
    times,
    V=V
)

# Compute errors at all timesteps
println("\nComputing error evolution...")
L1_errors = Float64[]
position_errors = Float64[]

for (idx, t) in enumerate(times)
    xf_num = xf_log[idx]
    t_eff = t - T0_SHIFT
    xf_anal = analytical_position_planar(t_eff, V)
    push!(position_errors, abs(xf_num - xf_anal))
end

# For L1 error, we only compute at final time to save computation
# Final interface position (numerical and analytical at Tend)
xf_num = xf_log[end]
t_eff_final = Tend - T0_SHIFT
xf_anal_Tend = analytical_position_planar(t_eff_final, V)
abs_pos_err = abs(xf_num - xf_anal_Tend)
rel_pos_err = abs_pos_err / (abs(xf_anal_Tend) > 0 ? abs(xf_anal_Tend) : eps())

println("\n" * "="^60)
println("RESULTS AT FINAL TIME t = $Tend")
println("="^60)
println("Interface Position:")
println("  Numerical:  $xf_num")
println("  Analytical: $xf_anal_Tend")
println("  Absolute error: $abs_pos_err")
println("  Relative error: $rel_pos_err")

# Compute L1 error for temperature at final time (only for points below interface)
u_num = solver.x[1:(nx+1)]
x_num = collect(range(x0, x0+lx, length(u_num)))
mask = x_num .<= xf_num

if sum(mask) == 0
    println("\nNo mesh nodes found below final interface. Skipping temperature error.")
else
    u_anal_at_num = analytical_temperature_planar.(x_num, t_eff_final, V)
    u_num_below = u_num[mask]
    u_anal_below = u_anal_at_num[mask]
    
    L1_error = sum(abs.(u_num_below .- u_anal_below)) / length(u_num_below)
    L2_error = sqrt(sum((u_num_below .- u_anal_below).^2) / length(u_num_below))
    Linf_error = maximum(abs.(u_num_below .- u_anal_below))
    
    println("\nTemperature Errors (x ≤ xf):")
    println("  L1 error:   $L1_error")
    println("  L2 error:   $L2_error")
    println("  L∞ error:   $Linf_error")
    println("  Nodes used: $(length(u_num_below)) / $(length(u_num))")
end

# Compute average interface velocity
avg_velocity = (xf_log[end] - xf_log[1]) / (times[end] - times[1])
velocity_error = abs(avg_velocity - V)
println("\nInterface Velocity:")
println("  Average numerical velocity: $avg_velocity")
println("  Expected velocity (V):      $V")
println("  Velocity error:             $velocity_error")
println("="^60)

println("\nDone!")
