using Penguin
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using CairoMakie
using Roots
using Statistics

"""
Calculate the dimensionless parameter λ for the one-phase Stefan problem.
"""
function find_lambda(Stefan_number)
    f = (λ) -> sqrt(pi)*λ*exp(λ^2)*erf(λ) - 1/Stefan_number
    lambda = find_zero(f, (1e-6, 10.0), Bisection())
    return lambda
end

"""
Analytical temperature distribution for the one-phase Stefan problem (1D).
"""
function analytical_temperature(x, t, T₀, k, lambda)
    if t <= 0
        return T₀
    end
    return T₀ - (T₀/erf(lambda)) * (erf(x/(2*sqrt(k*t))))
end

"""
Analytical interface position for the one-phase Stefan problem (1D).
"""
function analytical_position(t, k, lambda)
    return 2*lambda*sqrt(t)
end


### 2D Test Case : One-phase Stefan Problem with Height Function
# Define the spatial mesh
nx, ny = 16, 16
lx, ly = 1.0, 1.0
x0, y0 = 0.0, 0.0
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Mesh offset for exact solution consistency
x_offset = mesh.nodes[1][1][1] - x0

# Define the Space-Time mesh
Δt = 0.5*(lx/nx)^2  # Time step based on stability condition
Tstart = 0.001
Tend = 0.01
STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Tstart+Δt], tag=mesh.tag)

# Calculate Stefan number and λ and set initial interface consistently with Tstart
Stefan_number = 1.0
lambda = find_lambda(Stefan_number)

# Initial interface position from analytical solution at Tstart (convert to mesh coordinates)
xf0_phys = analytical_position(Tstart, 1.0, lambda)
xf0 = xf0_phys + x_offset

# Define the height function for the interface (uniform in y direction)
# sₙ(y) represents the x-position of the interface as a function of y
sₙ(y) = xf0
body = (x, y, t, _ = 0) -> (x - sₙ(y))

# Define the capacity
capacity = Capacity(body, STmesh; method="VOFI", integration_method=:vofijul)

# Extract initial height information
Vₙ₊₁ = capacity.A[3][1:end÷2, 1:end÷2]
Vₙ = capacity.A[3][end÷2+1:end, end÷2+1:end]
Vₙ = diag(Vₙ)
Vₙ₊₁ = diag(Vₙ₊₁)
Vₙ = reshape(Vₙ, (nx+1, ny+1))
Vₙ₊₁ = reshape(Vₙ₊₁, (nx+1, ny+1))

Hₙ⁰ = collect(vec(sum(Vₙ, dims=1)))
Hₙ₊₁⁰ = collect(vec(sum(Vₙ₊₁, dims=1)))

Interface_position = x0 .+ Hₙ⁰ ./ ((lx) / nx)
println("Interface position : $(Interface_position)")

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :top => Dirichlet(0.0),
    :bottom => Dirichlet(1.0)
))
ρ, L = 1.0, 1.0
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))

# Define the source term
f = (x, y, z, t) -> 0.0
K = (x, y, z) -> 1.0

# Define the phase
Fluide = Phase(capacity, operator, f, K)

# Initial condition: bulk initialized from analytical temperature at Tstart
x_nodes = mesh.nodes[1][:, 1]  # x-coordinates
x_nodes_phys = x_nodes .- x_offset
y_nodes = mesh.nodes[2][1, :]  # y-coordinates

# Create 2D grid of initial temperatures
u0_grid = zeros(nx+1, ny+1)
for j in 1:(ny+1)
    for i in 1:(nx+1)
        u0_grid[i, j] = analytical_temperature(x_nodes_phys[i], Tstart, 1.0, 1.0, lambda)
    end
end

u0ₒ = vec(u0_grid)
u0ᵧ = zeros((nx+1)*(ny+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Newton parameters
max_iter = 2
tol = eps()
reltol = eps()
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
println("Solving the 2D Stefan problem with height function...")
solver, residuals, xf_log, reconstruct, timestep_history = solve_MovingLiquidDiffusionUnsteadyMono2D!(
    solver, Fluide, Interface_position, Hₙ⁰, sₙ, Δt, Tstart, Tend, bc_b, bc, stef_cond, mesh, "BE";
    interpo="quad", Newton_params=Newton_params, adaptive_timestep=false, Δt_min=5e-4, method=Base.:\
)
println("Simulation complete!")


# Define analytical solution function
T_analytical = (x) -> analytical_temperature(x, Tend, 1.0, 1.0, lambda)

# Plot 1: Temperature comparison (slice through y-center)
function plot_temperature_comparison_2d(
    T_analytical,
    u_num,
    x_num,
    t_final::Float64,
    xf_final::Float64;
    save_path::String="temperature_comparison_2d.png"
)
    # Create fine grid for analytical solution
    x_analytical = range(minimum(x_num), maximum(x_num), 2000)
    u_analytical = T_analytical.(x_analytical)
    
    # Get y-center slice of numerical solution
    y_center_idx = div(ny+1, 2) + 1
    u_num_slice = u_num[(y_center_idx-1)*(nx+1)+1:y_center_idx*(nx+1)]
    
    # Create figure
    fig = Figure(resolution=(900, 600), fontsize=14)
    
    ax = Axis(
        fig[1, 1],
        xlabel = "Position (x)",
        ylabel = "Temperature",
        title = "Stefan Problem (2D, y-center slice): Temperature Comparison at t = $t_final"
    )
    
    # Plot solutions
    lines!(ax, x_analytical, u_analytical, color=:red, linewidth=3,
           label="Analytical Solution")
    scatter!(ax, x_num, u_num_slice, color=:blue, markersize=6,
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
function plot_interface_position_2d(
    xf_numerical,
    times_numerical;
    lambda=1.0,
    vertical_offset=0.0,
    save_path::String="interface_position_2d.png"
)
    # Analytical curve from 0 -> max time
    t_analytical = range(0.0, maximum(times_numerical), 500)
    xf_analytical = analytical_position.(t_analytical, 1.0, lambda) .+ vertical_offset
    
    fig = Figure(resolution=(900, 600), fontsize=14)
    
    ax = Axis(
        fig[1, 1],
        xlabel = "Time (t)",
        ylabel = "Interface Position (x_f)",
        title = "Stefan Problem (2D): Interface Position vs Time"
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
    
    display(fig)
    return fig
end

# Plot 3: 2D temperature field at final time
function plot_temperature_field_2d(
    u_num,
    t_final::Float64,
    save_path::String="temperature_field_2d.png"
)
    # Reshape to 2D grid
    u_2d = reshape(u_num, (nx+1, ny+1))
    
    # Create coordinates
    x_coords = range(x0, lx, length=nx+1)
    y_coords = range(y0, ly, length=ny+1)
    
    fig = Figure(resolution=(900, 700), fontsize=14)
    
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Temperature Field at t = $t_final")
    
    # Plot heatmap
    hm = heatmap!(ax, x_coords, y_coords, u_2d', colormap=:hot, colorrange=(0, 1))
    
    # Add colorbar
    Colorbar(fig[1, 2], hm, label="Temperature")
    
    # Save the figure if requested
    if !isempty(save_path)
        save(save_path, fig, px_per_unit=4)
    end
    
    display(fig)
    return fig
end


# Execute the plots
println("Plotting temperature comparison (y-center slice)...")
println("xf_log structure: $(length(xf_log)) timesteps")
if !isempty(xf_log) && !isempty(xf_log[1])
    println("  Each timestep has $(length(xf_log[1])) y-positions")
end

# Extract interface position at each timestep (take first value)
xf_mean_log = [xf_log[i][1] for i in 1:length(xf_log)]
println("Interface positions extracted: $(length(xf_mean_log)) timesteps")

Δt_eff = (Tend - Tstart) / length(xf_mean_log)
times = Tstart .+ collect(1:length(xf_mean_log)) .* Δt_eff
x_offset_0 = (xf_mean_log[1] - x_offset) - analytical_position(times[1], 1.0, lambda)
xf_log_phys = xf_mean_log .- x_offset .- x_offset_0
x_num_phys = mesh.nodes[1][:, 1] .- x_offset

u_num = solver.x[1:(nx+1)*(ny+1)]

plot_temperature_comparison_2d(
    T_analytical,
    u_num,
    x_num_phys,
    Tend,
    xf_log_phys[end]
)

println("Plotting interface position vs time...")
times = Tstart .+ collect(1:length(xf_log_phys)) .* Δt_eff
plot_interface_position_2d(
    xf_log_phys,
    times,
    lambda=lambda,
    vertical_offset=0.0
)

println("Plotting 2D temperature field...")
plot_temperature_field_2d(
    u_num,
    Tend
)

# Final interface position (numerical and analytical at Tend)
xf_num = xf_log_phys[end]
xf_anal_Tend = analytical_position(times[end], 1.0, lambda)
abs_pos_err = abs(xf_num - xf_anal_Tend)
rel_pos_err = abs_pos_err / (abs(xf_anal_Tend) > 0 ? abs(xf_anal_Tend) : eps())
println("Final interface position: numerical=", xf_num, ", analytical=", xf_anal_Tend)
println("Absolute position error=", abs_pos_err, ", Relative error=", rel_pos_err)

# Compute L1 and L2 errors for temperature at final time (y-center slice)
y_center_idx = div(ny+1, 2) + 1
u_num_slice = u_num[(y_center_idx-1)*(nx+1)+1:y_center_idx*(nx+1)]
x_num = x_num_phys
mask = x_num .<= xf_num
if sum(mask) == 0
    println("No mesh nodes found below final numerical interface (xf=", xf_num, "). Skipping temperature error.")
else
    u_anal_at_num = analytical_temperature.(x_num, Tend, 1.0, 1.0, lambda)
    u_num_below = u_num_slice[mask]
    u_anal_below = u_anal_at_num[mask]
    L1_error = sum(abs.(u_num_below .- u_anal_below)) / length(u_num_below)
    L2_error = sqrt(sum((u_num_below .- u_anal_below).^2) / length(u_num_below))
    println("L1 (mean abs) temperature error over x <= xf_num at t=$(Tend): ", L1_error)
    println("L2 (RMS) temperature error over x <= xf_num at t=$(Tend): ", L2_error)
end

println("Summary of Errors at t=$(Tend):")
println("Final interface position relative error: ", rel_pos_err)
if sum(mask) > 0
    println("L1 temperature error (x <= xf_num): ", sum(abs.(u_num_slice[mask] .- analytical_temperature.(x_num[mask], Tend, 1.0, 1.0, lambda))) / length(u_num_slice[mask]))
    println("L2 temperature error (x <= xf_num): ", sqrt(sum((u_num_slice[mask] .- analytical_temperature.(x_num[mask], Tend, 1.0, 1.0, lambda)).^2) / length(u_num_slice[mask])))
    println("  Nodes used: $(length(u_num_slice[mask])) / $(length(u_num_slice))")
else
    println("Temperature error not computed (no nodes below interface).")
end

println("Done!")
