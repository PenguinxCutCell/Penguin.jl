using Penguin
using IterativeSolvers
using LinearAlgebra, SparseArrays
using CairoMakie
using LsqFit
using Statistics

"""
    run_mesh_convergence_1d(
      nx_list::Vector{Int},
      target_center::Float64,
      radius::Float64,
      u_analytical_factory::Function;
      lx::Float64 = 1.0,
      bc_value::Float64 = 0.0,
      norm = 2,
      relative::Bool = false,
      plot_position = nothing,
      theta::Union{Float64, Nothing} = nothing
    )

Perform a mesh‐convergence study for the 1D Poisson example:
returns h_vals, err_vals, estimated order p.
If `theta` is provided (0 <= theta < 1), the center is adjusted for each mesh
so that the interface is located at `(k + theta)*h`.
"""
function run_mesh_convergence_1d(
    nx_list::Vector{Int},
    target_center::Float64,
    radius::Float64,
    u_analytical_factory::Function;
    lx::Float64 = 1.0,
    bc_value::Float64 = 0.0,
    norm = 2,
    relative::Bool = false,
    plot_position = nothing,
    theta::Union{Float64, Nothing} = nothing
)
  h_vals = Float64[]
  err_vals = Float64[]
  err_full_vals = Float64[]
  err_cut_vals = Float64[]
  err_empty_vals = Float64[]

  # For the finest mesh, save solution for visualization
  finest_mesh = nothing
  finest_u_ana = nothing 
  finest_u_num = nothing
  finest_err = nothing
  finest_capacity = nothing

  for nx in nx_list
    # 1D mesh
    mesh = Penguin.Mesh((nx,), (lx,), (0.0,))
    h = lx / nx

    # Determine center
    if isnothing(theta)
        current_center = target_center
    else
        # We want the right interface (center + radius) to be at relative position theta
        # x_int = (k + theta) * h
        # Find k based on target_center
        x_int_approx = target_center + radius
        k = floor(Int, x_int_approx / h)
        
        x_int_new = (k + theta) * h
        current_center = x_int_new - radius
    end

    # Get analytical solution for this center
    u_analytical = u_analytical_factory(current_center, radius)

    # level‐set
    body = (x, _=0) -> abs(x - current_center) - radius
    # assemble
    capacity = Capacity(body, mesh)
    operator = DiffusionOps(capacity)
    #volume_redefinition!(capacity, operator)
    # BCs
    bc = Dirichlet(bc_value)
    bc_b = BorderConditions(Dict(:top => bc, :bottom => bc))
    # define phase (source g=x, diffusion a=1.0)
    phase = Phase(capacity, operator, (x,y,z)-> x, (x,y,z)->1.0)
    # solver
    solver = DiffusionSteadyMono(phase, bc_b, bc)
    solve_DiffusionSteadyMono!(solver; method=Base.:\)
    # error
    u_ana, u_num, global_err, full_err, cut_err, empty_err =
      check_convergence(u_analytical, solver, capacity, norm, relative)
    
    push!(h_vals, 1.0/nx)
    push!(err_vals, global_err)
    push!(err_full_vals, full_err)
    push!(err_cut_vals, cut_err)
    push!(err_empty_vals, empty_err)
    
    @info "nx=$nx, h=$(round(1/nx, digits=4)), center=$(round(current_center, digits=4)), global_err=$(round(global_err, sigdigits=4))"
    
    # Save finest mesh solution
    if nx == maximum(nx_list)
      finest_mesh = mesh
      finest_u_ana = u_ana
      finest_u_num = u_num
      finest_err = abs.(u_ana - u_num)
      finest_capacity = capacity
    end
  end

  # Model for curve fitting
  function fit_model(x, p)
      p[1] .* x .+ p[2]
  end

  # Fit each on log scale: log(err) = p*log(h) + c
  log_h = log.(h_vals)

  function do_fit(log_err, use_last_n=3)
      # Use only the last n points (default 3)
      n = min(use_last_n, length(log_h))
      idx = length(log_h) - n + 1 : length(log_h)
      
      # Fit using only those points
      fit_result = curve_fit(fit_model, log_h[idx], log_err[idx], [-1.0, 0.0])
      return fit_result.param[1], fit_result.param[2]  # (p_est, c_est)
  end

  # Get convergence rates
  p_global, c_global = do_fit(log.(err_vals), 3)
  p_full, c_full = do_fit(log.(err_full_vals), 3)
  p_cut, c_cut = do_fit(log.(err_cut_vals), 3)

  # Round for display
  p_global_r = round(p_global, digits=1)
  p_full_r = round(p_full, digits=1)
  p_cut_r = round(p_cut, digits=1)

  println("\nEstimated order of convergence (last 3 points):")
  println("  - Global = ", p_global_r)
  println("  - Full   = ", p_full_r)
  println("  - Cut    = ", p_cut_r)

  # Create a figure layout
  if isnothing(plot_position)
      fig = Figure(resolution=(1000,800))
      pos = fig[1,1:2]
  else
      fig = nothing
      pos = plot_position
  end
  
  # Plot 1: Convergence plot (like in the 2D benchmark)
  title_str = isnothing(theta) ? "Convergence (center=$target_center)" : "Convergence (theta=$theta)"
  ax1 = Axis(pos;
    xlabel = "h",
    ylabel = "L$norm error",
    title  = title_str,
    xscale = log10,
    yscale = log10,
    xminorticksvisible = true, 
    xminorgridvisible = true,
    xminorticks = IntervalsBetween(5),
    yminorticksvisible = true,
    yminorgridvisible = true,
    yminorticks = IntervalsBetween(5),
  )

  # Plot data points and reference lines
  scatter!(ax1, h_vals, err_vals, label="All cells ($(p_global_r))", markersize=10)
  lines!(ax1, h_vals, err_vals, color=:black)
  
  scatter!(ax1, h_vals, err_full_vals, label="Full cells ($(p_full_r))", markersize=10)
  lines!(ax1, h_vals, err_full_vals, color=:black)
  
  scatter!(ax1, h_vals, err_cut_vals, label="Cut cells ($(p_cut_r))", markersize=10)
  lines!(ax1, h_vals, err_cut_vals, color=:black)

  # Add reference slopes
  lines!(ax1, h_vals, 10.0*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
  lines!(ax1, h_vals, 1.0*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
  
  axislegend(ax1, position=:rb)
  if !isnothing(fig)
      display(fig)
  end

  return (
    h_vals,
    err_vals,
    err_full_vals,
    err_cut_vals,
    err_empty_vals,
    p_global_r,
    p_full_r,
    p_cut_r,
    fig
  )
end

# Example usage
target_center = 0.5
radius = 0.1
thetas = [0.0, 0.1, 0.5000001, 0.9, 1.0]
nx_list = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120]

# Factory for analytical solution
u_analytical_factory = (c, r) -> (x -> - (x-c)^3/6 - (c*(x-c)^2)/2 +
                   r^2/6*(x-c) + c*r^2/2)

# Create a figure for side-by-side plots
fig = Figure(resolution=(1500, 500))

for (i, theta) in enumerate(thetas)
    println("\nRunning convergence for theta = $theta")
    
    # Run convergence and plot in the i-th column
    results = run_mesh_convergence_1d(
      nx_list, target_center, radius, u_analytical_factory;
      lx=1.0, bc_value=0.0, norm=2, relative=false,
      plot_position = fig[1, i],
      theta = theta
    )
    
    # Unpack results
    h_vals, err_vals, err_full_vals, err_cut_vals, err_empty_vals, 
    p_global, p_full, p_cut, _ = results

    # Write the output to a file
    filename = "convergence_1d_theta_$(theta).txt"
    open(filename, "w") do io
        println(io, "h, err, err_full, err_cut, err_empty")
        for (h, err, err_full, err_cut, err_empty) in zip(h_vals, err_vals, err_full_vals, err_cut_vals, err_empty_vals)
            println(io, "$h, $err, $err_full, $err_cut, $err_empty")
        end
    end
end

display(fig)
save("poisson_1d_benchmark_comparison_theta.png", fig)