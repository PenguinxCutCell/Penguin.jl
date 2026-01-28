using Penguin
using SpecialFunctions
using CairoMakie
using Roots
using LinearAlgebra

# ============================================================================
# 2D Two-Phase Stefan Problem (Planar Interface)
# ============================================================================
# This is a simplified 2D example with a planar vertical interface moving
# in the x-direction. The analytical solution is the 1D Neumann solution
# extended uniformly in y-direction.

# ------------------------------------------------------------
# Analytical helpers (same as 1D case)
# ------------------------------------------------------------
function find_lambda_two_phase(Ste1, Ste2; beta::Float64=1.0)
    f = λ -> sqrt(pi) * λ - Ste1 * exp(-λ^2) / erf(λ) - Ste2 * beta * exp(-(beta * λ)^2) / erfc(beta * λ)
    return find_zero(f, (1e-6, 10.0), Bisection())
end

analytical_interface(t, α1, λ) = 2 * λ * sqrt(α1 * t)

function analytical_temperature_left(x, t, T_L, T_m, α1, λ)
    t <= 0 && return T_L
    return T_L - (T_L - T_m) * erf(x / (2 * sqrt(α1 * t))) / erf(λ)
end

function analytical_temperature_right(x, t, T_R, T_m, α1, α2, λ)
    t <= 0 && return T_R
    β = sqrt(α1 / α2)
    ξ = (x - analytical_interface(t, α1, λ)) / (2 * sqrt(α2 * t))
    return T_R + (T_m - T_R) * erfc(ξ) / erfc(β * λ)
end

# ------------------------------------------------------------
# Problem definition
# ------------------------------------------------------------
nx, ny = 32, 32
lx, ly = 1.0, 1.0
x0, y0 = 0.0, 0.0
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
x_offset = mesh.nodes[1][1] - x0
Δx, Δy = lx / nx, ly / ny

Δt = 0.5 * Δx^2
Tstart = 0.001
Tend = 0.05
STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Tstart + Δt], tag=mesh.tag)

# Material/thermal parameters (symmetric case)
α1 = 1.0
α2 = 1.0
T_L, T_R, T_m = 1.0, 0.0, 0.0
ρ, Lh = 1.0, 1.0
Ste1 = (T_L - T_m) / Lh
Ste2 = (T_m - T_R) / Lh
beta = sqrt(α1 / α2)
lambda = find_lambda_two_phase(Ste1, Ste2; beta=beta)

# Initial planar interface position (vertical line at xf0)
xf0_phys = analytical_interface(Tstart, α1, lambda)
xf0 = xf0_phys + x_offset

# Body functions: planar interface at x = xf0 (independent of y)
body = (x, y, t, _=0) -> (x - xf0)
body_c = (x, y, t, _=0) -> -(x - xf0)

capacity = Capacity(body, STmesh)
capacity_c = Capacity(body_c, STmesh)
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

f1 = (x, y, z, t) -> 0.0
f2 = (x, y, z, t) -> 0.0
K1 = (x, y, z) -> α1
K2 = (x, y, z) -> α2

phase1 = Phase(capacity, operator, f1, K1)
phase2 = Phase(capacity_c, operator_c, f2, K2)

# Boundary conditions (left hot, right cold)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :left => Dirichlet(T_L),
    :right => Dirichlet(T_R),
    
))

ic = InterfaceConditions(ScalarJump(1.0, 1.0, T_m), FluxJump(1.0, 1.0, ρ * Lh))

# Initial condition from analytical profiles at Tstart
x_phys = mesh.nodes[1] .- x_offset
y_phys = mesh.nodes[2] .- 0.0  # y doesn't affect the solution

u0ₒ1 = zeros((nx+1)*(ny+1))
u0ₒ2 = zeros((nx+1)*(ny+1))

# Fill initial conditions (independent of y for planar interface)
idx = 1
for j in 1:(ny+1)
    for i in 1:(nx+1)
        u0ₒ1[idx] = analytical_temperature_left(x_phys[i], Tstart, T_L, T_m, α1, lambda)
        u0ₒ2[idx] = analytical_temperature_right(x_phys[i], Tstart, T_R, T_m, α1, α2, lambda)
        global idx += 1
    end
end

u0ᵧ1 = fill(T_m, (nx+1)*(ny+1))
u0ᵧ2 = fill(T_m, (nx+1)*(ny+1))
u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Solver setup
solver = MovingLiquidDiffusionUnsteadyDiph(phase1, phase2, bc_b, ic, Δt, u0, mesh, "BE")
println("Solving 2D two-phase Stefan problem (planar interface) with analytical comparison...")

# Initial interface position array (constant across y)
Interface_position = fill(xf0, ny+1)

solver, residuals, xf_log, reconstruct = solve_MovingLiquidDiffusionUnsteadyDiph2D!(
    solver, phase1, phase2, Interface_position, zeros(ny+1), y->xf0, 
    Δt, Tstart, Tend, bc_b, ic, mesh, "BE";
    interpo="linear",
    Newton_params=(100, 1e-10, 1e-10, 1.0),
    method=Base.:\
)
println("Simulation complete!")

# ------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------
T_exact_left  = x -> analytical_temperature_left(x, Tend, T_L, T_m, α1, lambda)
T_exact_right = x -> analytical_temperature_right(x, Tend, T_R, T_m, α1, α2, lambda)

xf_log_phys = [xf .- x_offset for xf in xf_log]
xf_final = mean(xf_log_phys[end])  # Should be uniform across y
xf_exact_Tend = analytical_interface(Tend, α1, lambda)

# Extract numerical solution (take mid-plane y slice for comparison)
y_mid_idx = (ny+1) ÷ 2
u1_bulk = reshape(solver.x[1:(nx+1)*(ny+1)], (ny+1, nx+1))
u2_bulk = reshape(solver.x[2*(nx+1)*(ny+1)+1:3*(nx+1)*(ny+1)], (ny+1, nx+1))

u1_slice = u1_bulk[y_mid_idx, :]
u2_slice = u2_bulk[y_mid_idx, :]

# Build piecewise numerical field
u_num = similar(x_phys)
mask_left = x_phys .<= xf_final
u_num[mask_left] .= u1_slice[mask_left]
u_num[.!mask_left] .= u2_slice[.!mask_left]

# Plot temperature comparison (mid-plane slice)
function plot_temperature_slice(T_left, T_right, u_num, x_num, xf_num, xf_exact; save_path="temperature_2d_simple.png")
    x_fine = range(minimum(x_num), maximum(x_num), 2000)
    u_exact = [xi <= xf_exact ? T_left(xi) : T_right(xi) for xi in x_fine]

    fig = Figure(resolution=(900, 600), fontsize=14)
    ax = Axis(fig[1, 1], xlabel="Position (x)", ylabel="Temperature",
        title="2D Stefan (y-mid slice): Temperature at t = $(round(Tend, digits=4))")

    lines!(ax, x_fine, u_exact, color=:red, linewidth=3, label="Analytical")
    scatter!(ax, x_num, u_num, color=:blue, markersize=6, label="Numerical")
    vlines!(ax, xf_exact, color=:black, linewidth=2, linestyle=:dash, label="Interface (analytical)")
    vlines!(ax, xf_num, color=:orange, linewidth=2, linestyle=:dot, label="Interface (numerical)")
    hlines!(ax, T_m, color=:green, linewidth=2, linestyle=:dot, label="T = T_m")
    axislegend(ax, position=:rt, framevisible=true, backgroundcolor=(:white, 0.7))

    !isempty(save_path) && save(save_path, fig, px_per_unit=4)
    display(fig)
    return fig
end

println("Plotting temperature comparison (mid-plane slice)...")
plot_temperature_slice(T_exact_left, T_exact_right, u_num, x_phys, xf_final, xf_exact_Tend)

# Plot interface position evolution (should be uniform in y)
function plot_interface_evolution(xf_vals, Δt, Tstart; lambda=1.0, α1=1.0, save_path="interface_2d_simple.png")
    # Extract mean interface position at each time
    xf_mean = [mean(xf) for xf in xf_vals]
    times = Tstart .+ collect(1:length(xf_vals)) .* Δt
    t_exact = range(0.0, maximum(times), 500)
    xf_exact = analytical_interface.(t_exact, α1, lambda)

    fig = Figure(resolution=(900, 600), fontsize=14)
    ax = Axis(fig[1, 1], xlabel="Time (t)", ylabel="Interface position",
        title="2D Stefan: Interface position vs time")
    lines!(ax, t_exact, xf_exact, color=:red, linewidth=3, label="Analytical")
    scatter!(ax, times, xf_mean, color=:blue, markersize=8, label="Numerical (mean)")
    axislegend(ax, position=:lt, framevisible=true, backgroundcolor=(:white, 0.7))

    !isempty(save_path) && save(save_path, fig, px_per_unit=4)
    display(fig)
    return fig
end

println("Plotting interface position...")
plot_interface_evolution(xf_log_phys, Δt, Tstart; lambda=lambda, α1=α1)

# 2D Heatmap visualization at final time
function plot_2d_heatmap(solver, mesh, xf_final, nx, ny; save_path="heatmap_2d_simple.png")
    u1_bulk = reshape(solver.x[1:(nx+1)*(ny+1)], (ny+1, nx+1))
    u2_bulk = reshape(solver.x[2*(nx+1)*(ny+1)+1:3*(nx+1)*(ny+1)], (ny+1, nx+1))
    
    # Build combined field
    x_centers = mesh.centers[1]
    y_centers = mesh.centers[2]
    
    u_combined = zeros(ny, nx)
    for j in 1:ny
        for i in 1:nx
            if x_centers[i] <= xf_final
                u_combined[j, i] = u1_bulk[j, i]
            else
                u_combined[j, i] = u2_bulk[j, i]
            end
        end
    end
    
    fig = Figure(resolution=(900, 700), fontsize=14)
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Temperature field at t=$(round(Tend,digits=4))",
              aspect=DataAspect())
    
    hm = heatmap!(ax, x_centers, y_centers, u_combined, colormap=:thermal)
    vlines!(ax, xf_final, color=:white, linewidth=2, linestyle=:dash, label="Interface")
    Colorbar(fig[1, 2], hm, label="Temperature")
    
    !isempty(save_path) && save(save_path, fig, px_per_unit=4)
    display(fig)
    return fig
end

println("Plotting 2D temperature field...")
plot_2d_heatmap(solver, mesh, xf_final, nx, ny)

# Error diagnostics
pos_err = abs(xf_final - xf_exact_Tend)
println("\n" * "="^70)
println("INTERFACE POSITION ERROR")
println("="^70)
println("Final interface position: numerical = $xf_final, analytical = $xf_exact_Tend")
println("Absolute error = $pos_err")
println("Relative error = $(pos_err / abs(xf_exact_Tend))")

# Temperature error on mid-plane slice
mask = x_phys .<= xf_final
if any(mask)
    u_exact_left = T_exact_left.(x_phys[mask])
    u_num_left = u_num[mask]
    L2_error = sqrt(sum((u_num_left .- u_exact_left).^2) / length(u_num_left))
    println("\nL2 temperature error (left phase, mid-slice): $L2_error")
end

# Volume-integrated error (summed over all y-slices)
body_tend = (x, y, _=0) -> (x - (xf_final + x_offset))
capacity_tend = Capacity(body_tend, mesh)
cell_vols = diag(capacity_tend.V)

total_volume = 0.0
L1_vol_error = 0.0

idx = 1
for j in 1:(ny+1)
    for i in 1:(nx+1)
        cell_vol = cell_vols[idx]
        
        if cell_vol > 0.0
            total_volume += cell_vol
            x_cell = x_phys[i]
            
            if x_cell <= xf_final
                u_exact_val = T_exact_left(x_cell)
                u_num_val = u1_bulk[j, i]
            else
                u_exact_val = T_exact_right(x_cell)
                u_num_val = u2_bulk[j, i]
            end
            
            L1_vol_error += abs(u_num_val - u_exact_val) * cell_vol
        end
        idx += 1
    end
end

L1_vol_error = L1_vol_error / total_volume

println("\nVOLUME-INTEGRATED L1 ERROR NORM")
println("="^70)
println("Total computational volume: ", total_volume)
println("L1 error (volume-integrated): ", L1_vol_error)
println("="^70)

println("\nDone!")
