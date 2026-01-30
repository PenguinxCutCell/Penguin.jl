using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers, SpecialFunctions
using Roots, CairoMakie
using CSV, DataFrames

function run_moving_heat_benchmark(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    x_0_initial::Float64,
    y_0_initial::Float64,
    velocity_x::Float64,
    velocity_y::Float64;
    lx::Float64=4.0,
    ly::Float64=4.0,
    D::Float64=1.0,
    Tend::Float64=0.1,
    norm=Inf
)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]
    time_vals = Float64[]  # To track computational time

    # Pre-compute Bessel function zeros for analytical solution
    alphas = j0_zeros(100)

    for (nx, ny) in zip(nx_list, ny_list)
        println("Running simulation with nx=$nx, ny=$ny")
        
        # Start timing
        start_time = time()
        
        # Build mesh
        x0, y0 = 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

        # Define the translating body
        function translating_body(x, y, t)
            # Calculate current position of center
            x_t = x_0_initial + velocity_x * t
            y_t = y_0_initial + velocity_y * t
            
            # Return signed distance function to disk
            return sqrt((x - x_t)^2 + (y - y_t)^2) - radius
        end
            
        # Time step based on mesh size
        Δt = 0.5*(lx/nx)^2
        Tstart = 0.01  # Start at small positive time to avoid t=0 singularity
        
        # Create space-time mesh
        STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Δt], tag=mesh.tag)
        
        # Define capacity with moving body
        capacity = Capacity(translating_body, STmesh)
        
        # Initialize velocity field to match disk translation
        N = (nx + 1) * (ny + 1) * 2
        uₒx = ones(N) * velocity_x  # Constant velocity in x-direction
        uₒy = ones(N) * velocity_y  # Constant velocity in y-direction
        uₒt = zeros(N)  # No time component needed for this problem
        uₒ = (uₒx, uₒy, uₒt)

        # For boundary velocities
        uᵧ = zeros(3*N) * 1.0
        uᵧx = ones(N) * velocity_x  # Constant velocity in x-direction
        uᵧy = ones(N) * velocity_y  # Constant velocity in y-direction
        uᵧt = zeros(N)  # No time component needed for this problem
        uᵧ = vcat(uᵧx, uᵧy, uᵧt)
        
        # Define the operators
        operator = ConvectionOps(capacity, uₒ, uᵧ)
        
        # Define the boundary conditions
        bc = Dirichlet(0.0)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
            :left => bc, :right => bc, :top => bc, :bottom => bc
        ))
        
        # Dirichlet condition on the disk boundary
        ic = Dirichlet(1.0)
        
        # Define the phase
        Fluide = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->D)

        # Create initial condition
        T0ₒ = zeros((nx+1)*(ny+1))
        T0ᵧ = ones((nx+1)*(ny+1))
        T0 = vcat(T0ₒ, T0ᵧ)
        
        # Define the solver
        solver = MovingAdvDiffusionUnsteadyMono(Fluide, bc_b, ic, Δt, Tstart, T0, mesh, "BE")

        # Solve the problem
        solve_MovingAdvDiffusionUnsteadyMono!(solver, Fluide, translating_body, 
                                             Δt, Tstart, Tend, bc_b, ic, 
                                             mesh, "CN", uₒ, uᵧ; method=Base.:\)
        
        # Create analytical solution function for final time
        u_analytical = (x, y) -> analytical_solution(x, y, Tend, x_0_initial, y_0_initial, 
                                                    velocity_x, velocity_y, radius, D, alphas)
        
        # Define body at final time for error computation
        body_tend = (x, y, _=0) -> begin
            # Calculate center position at Tend
            x_t = x_0_initial + velocity_x * Tend
            y_t = y_0_initial + velocity_y * Tend
            # Return signed distance function to disk at Tend
            return sqrt((x - x_t)^2 + (y - y_t)^2) - radius
        end
        
        capacity_tend = Capacity(body_tend, mesh; compute_centroids=false)
        
        # Compute errors
        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity_tend, norm)

        # Record compute time
        compute_time = time() - start_time
        
        # Record results
        push!(h_vals, 1.0 / min(nx, ny))
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        push!(time_vals, compute_time)
        
        println("  Complete. Error: $global_err, Time: $compute_time seconds")
    end

    # Perform curve fitting for convergence analysis
    function fit_model(x, p)
        p[1] .* x .+ p[2]
    end

    # Fit each on log scale: log(err) = p*log(h) + c
    log_h = log.(h_vals)
    
    function do_fit(log_err)
        fit_result = curve_fit(fit_model, log_h, log_err, [-1.0, 0.0])
        return fit_result.param[1], fit_result.param[2]  # (p_est, c_est)
    end

    p_global, _ = do_fit(log.(err_vals))
    p_full,   _ = do_fit(log.(err_full_vals))
    p_cut,    _ = do_fit(log.(err_cut_vals))

    # Round convergence rates
    p_global = round(p_global, digits=2)
    p_full   = round(p_full, digits=2)
    p_cut    = round(p_cut, digits=2)

    println("Estimated order of convergence (global) = ", p_global)
    println("Estimated order of convergence (full)   = ", p_full)
    println("Estimated order of convergence (cut)    = ", p_cut)

    # Create convergence plot
    fig = Figure(resolution=(900, 700), fontsize=14)
    ax = Axis(
        fig[1, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Convergence for Moving Heat Problem",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(10),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(10),
    )

    scatter!(ax, h_vals, err_vals,       label="Global error ($p_global)", markersize=12)
    lines!(ax, h_vals, err_vals,         color=:black)
    scatter!(ax, h_vals, err_full_vals,  label="Full error ($p_full)",   markersize=12)
    lines!(ax, h_vals, err_full_vals,    color=:black)
    scatter!(ax, h_vals, err_cut_vals,   label="Cut error ($p_cut)",     markersize=12)
    lines!(ax, h_vals, err_cut_vals,     color=:black)

    # Add reference lines for O(h²) and O(h)
    lines!(ax, h_vals, 10.0*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax, h_vals, 1.0*h_vals.^1.0,  label="O(h)", color=:black, linestyle=:dashdot)
    axislegend(ax, position=:rb)
    
    # Add timing plot
    ax2 = Axis(
        fig[2, 1],
        xlabel = "Mesh size (h)",
        ylabel = "Compute time (s)",
        title = "Computational Performance",
        xscale = log10,
        yscale = log10
    )
    
    scatter!(ax2, h_vals, time_vals, markersize=12, color=:purple)
    lines!(ax2, h_vals, time_vals, color=:purple)
    
    # Reference line for O(h^-3) scaling (typical for 2D problems)
    lines!(ax2, h_vals, 0.1*h_vals.^(-3), label="O(h⁻³)", color=:black, linestyle=:dash)
    axislegend(ax2, position=:lt)
    
    save("moving_heat_benchmark.pdf", fig)
    save("moving_heat_benchmark.png", fig, px_per_unit=3)
    display(fig)

    # Create a results dataframe and save to CSV
    results_df = DataFrame(
        mesh_size = h_vals,
        nx = nx_list,
        ny = ny_list,
        global_error = err_vals,
        full_error = err_full_vals,
        cut_error = err_cut_vals,
        empty_error = err_empty_vals,
        compute_time = time_vals
    )
    
    CSV.write("moving_heat_benchmark_results.csv", results_df)

    return (
        h_vals,
        err_vals,
        err_full_vals,
        err_cut_vals,
        err_empty_vals,
        p_global,
        p_full,
        p_cut,
        time_vals,
        fig
    )
end

# Helper function to compute Bessel function zeros
function j0_zeros(N; guess_shift=0.25)
    zs = zeros(Float64, N)
    # The m-th zero of J₀ is *roughly* near (m - guess_shift)*π for large m.
    for m in 1:N
        # approximate location
        x_left  = (m - guess_shift - 0.5)*pi
        x_right = (m - guess_shift + 0.5)*pi
        # ensure left>0
        x_left = max(x_left, 1e-6)
        
        # Find zero using Roots.jl
        αm = find_zero(besselj0, (x_left, x_right))
        zs[m] = αm
    end
    return zs
end

# Analytical solution function (shifted to follow the moving center)
function analytical_solution(x, y, t, x_0_initial, y_0_initial, velocity_x, velocity_y, 
                           radius, D, alphas=j0_zeros(1000))
    # Calculate center position at time t
    x_t = x_0_initial + velocity_x * t
    y_t = y_0_initial + velocity_y * t
    
    # Calculate distance from center at time t
    r = sqrt((x - x_t)^2 + (y - y_t)^2)
    
    # If outside the disk, return 0 (or NaN)
    if r >= radius
        return 0.0
    end
    
    # If in the disk: sum the series
    s = 0.0
    for m in 1:length(alphas)
        αm = alphas[m]
        s += exp(-αm^2 * D * t / radius^2) * besselj0(αm * (r / radius)) / (αm * besselj1(αm))
    end
    return 1.0 - 2.0*s
end

# Example usage:
nx_list = [8, 16, 32, 64, 128]
ny_list = nx_list
radius = 1.0
x_0_initial = 2.01
y_0_initial = 2.01
velocity_x = 0.0
velocity_y = 0.0
D = 1.0
Tend = 0.1

results = run_moving_heat_benchmark(
    nx_list, ny_list, radius, x_0_initial, y_0_initial, 
    velocity_x, velocity_y, D=D, Tend=Tend, norm=2
)