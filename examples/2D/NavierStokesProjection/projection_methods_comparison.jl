using Penguin
using CairoMakie
using LinearAlgebra
using Printf

"""
Comparison of three projection methods for 2D Navier-Stokes equations:
1. Chorin/Temam (original, non-incremental)
2. Incremental Pressure-Correction (Godunov/Van Kan)
3. Rotational Incremental Pressure-Correction (most accurate)

All methods use Adams-Bashforth 2 for explicit convection treatment.

This example solves a simple 2D flow problem to demonstrate the differences
between the three projection methods.
"""

###########
# Geometry
###########
nx, ny = 32, 32
Lx, Ly = 1.0, 1.0
x0, y0 = 0.0, 0.0

###########
# Meshes (staggered grid)
###########
mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

###########
# Capacities & operators
###########
# Domain with no obstacles (negative body fills entire domain)
full_body = (x, y, t=0.0) -> -1.0

capacity_ux = Capacity(full_body, mesh_ux; compute_centroids=false)
capacity_uy = Capacity(full_body, mesh_uy; compute_centroids=false)
capacity_p = Capacity(full_body, mesh_p; compute_centroids=false)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
# Lid-driven cavity style: moving top wall
lid_speed = 1.0

ux_top = Dirichlet((x, y, t=0.0) -> lid_speed)
ux_bottom = Dirichlet((x, y, t=0.0) -> 0.0)
ux_left = Dirichlet((x, y, t=0.0) -> 0.0)
ux_right = Dirichlet((x, y, t=0.0) -> 0.0)

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)

bc_ux = BorderConditions(Dict(
    :left => ux_left,
    :right => ux_right,
    :bottom => ux_bottom,
    :top => ux_top
))

bc_uy = BorderConditions(Dict(
    :left => uy_zero,
    :right => uy_zero,
    :bottom => uy_zero,
    :top => uy_zero
))

pressure_gauge = PinPressureGauge()
bc_cut = Dirichlet(0.0)

###########
# Fluid properties
###########
Re = 100.0
U_ref = lid_speed
L_ref = Lx
ν = U_ref * L_ref / Re
ρ = 1.0
μ = ρ * ν

# No body forces
fᵤ = (x, y, t=0.0) -> 0.0
fₚ = (x, y, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

###########
# Time stepping
###########
CFL = 0.5
dt = CFL * min(dx, dy) / lid_speed
T_end = 0.1  # Short simulation for comparison

println("=" ^ 70)
println("2D Navier-Stokes Projection Methods Comparison")
println("=" ^ 70)
println("Grid: $(nx) × $(ny)")
println("Reynolds number: $(Re)")
println("Time step: $(dt)")
println("Final time: $(T_end)")
println("Number of steps: $(Int(ceil(T_end/dt)))")
println()

###########
# Solve with each projection method
###########
methods = [
    (ChorinTemam, "Chorin/Temam (Original)"),
    (IncrementalPC, "Incremental Pressure-Correction"),
    (RotationalPC, "Rotational Pressure-Correction")
]

results = Dict()

for (method_enum, method_name) in methods
    println("=" ^ 70)
    println("Method: $(method_name)")
    println("=" ^ 70)
    
    # Create solver
    solver = NavierStokesProj2D(
        fluid,
        (bc_ux, bc_uy),
        pressure_gauge,
        bc_cut,
        method_enum;
        dt=dt
    )
    
    # Solve
    times, u_hist, p_hist = solve_NavierStokesProj2D!(solver; T_end=T_end, store_states=true)
    
    # Store results
    results[method_name] = (times=times, u_hist=u_hist, p_hist=p_hist, solver=solver)
    
    println("Final max velocity: $(maximum(abs, solver.u))")
    println("Final max pressure: $(maximum(abs, solver.p))")
    println()
end

###########
# Compare results
###########
println("=" ^ 70)
println("Comparison Summary")
println("=" ^ 70)

# Compare final states
ref_method = "Rotational Pressure-Correction"
ref_u = results[ref_method].solver.u
ref_p = results[ref_method].solver.p

for (method_enum, method_name) in methods
    if method_name == ref_method
        continue
    end
    
    u = results[method_name].solver.u
    p = results[method_name].solver.p
    
    u_diff = norm(u - ref_u) / norm(ref_u)
    p_diff = norm(p - ref_p) / norm(ref_p)
    
    println("$(method_name) vs $(ref_method):")
    println("  Relative velocity error: $(@sprintf("%.6e", u_diff))")
    println("  Relative pressure error: $(@sprintf("%.6e", p_diff))")
    println()
end

###########
# Visualization
###########
println("Creating visualization...")

fig = Figure(resolution=(1200, 800))

# Plot pressure fields at final time for each method
for (idx, (method_enum, method_name)) in enumerate(methods)
    p_field = results[method_name].solver.p
    
    # Reshape pressure for plotting
    # Get actual pressure mesh dimensions
    p_nx = length(results[method_name].solver.fluid.mesh_p.nodes[1])
    p_ny = length(results[method_name].solver.fluid.mesh_p.nodes[2])
    p_2d = reshape(p_field, (p_nx, p_ny))
    
    ax = Axis(fig[1, idx],
              title=method_name,
              xlabel="x",
              ylabel="y",
              aspect=DataAspect())
    
    hm = heatmap!(ax, mesh_p.nodes[1], mesh_p.nodes[2], p_2d',
                  colormap=:balance)
    Colorbar(fig[1, idx+3], hm, label="Pressure")
end

# Add title
Label(fig[0, :], "Pressure Fields at t = $(T_end)",
      fontsize=20, font=:bold)

# Save figure
save("projection_methods_comparison.png", fig)
println("Saved: projection_methods_comparison.png")

println()
println("=" ^ 70)
println("Comparison complete!")
println("=" ^ 70)
