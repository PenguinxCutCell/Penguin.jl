using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using SpecialFunctions
using Roots

function run_mesh_convergence_ft(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    norm
)
    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]

    for (nx, ny) in zip(nx_list, ny_list)
        println("Running Front Tracking with mesh $(nx)×$(ny)...")
        # Build mesh
        x0, y0 = 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

        # Create front tracker with resolution proportional to mesh size
        front = FrontTracker()
        resolution = max(500, min(nx, ny) * 10)
        create_circle!(front, center[1], center[2], radius, resolution)
        
        # Define capacity using front tracking
        capacity = Capacity(front, mesh)
        operator = DiffusionOps(capacity)

        # Same boundary conditions and solver parameters as original
        bc_boundary = Robin(3.0, 1.0, 3.0*400)
        bc0 = Dirichlet(400.0)
        bc_b = BorderConditions(Dict(
            :left   => bc0,
            :right  => bc0,
            :top    => bc0,
            :bottom => bc0
        ))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        u0ₒ = ones((nx+1)*(ny+1)) * 270.0
        u0ᵧ = zeros((nx+1)*(ny+1)) * 270.0
        u0 = vcat(u0ₒ, u0ᵧ)

        Δt = 0.25*(lx/nx)^2
        Tend = 0.1

        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "BE")

        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, "CN"; method=Base.:\)

        # Compute errors
        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm)

        push!(h_vals, 1.0 / min(nx, ny))
        push!(err_vals, global_err)
        push!(err_full_vals, full_err)
        push!(err_cut_vals, cut_err)
        push!(err_empty_vals, empty_err)
        
        println("  Global error: $global_err")
        println("  Full cells error: $full_err")
        println("  Cut cells error: $cut_err")
    end

    # Model for curve_fit
    function fit_model(x, p)
        p[1] .* x .+ p[2]
    end

    # Fit on log scale
    log_h = log.(h_vals)

    function do_fit(log_err)
        fit_result = curve_fit(fit_model, log_h, log_err, [-1.0, 0.0])
        return fit_result.param[1], fit_result.param[2]
    end

    # Get convergence rates
    p_global, _ = do_fit(log.(err_vals))
    p_full, _ = do_fit(log.(err_full_vals))
    p_cut, _ = do_fit(log.(err_cut_vals))

    # Round for display
    p_global = round(p_global, digits=2)
    p_full = round(p_full, digits=2)
    p_cut = round(p_cut, digits=2)

    println("\nFRONT TRACKING RESULTS:")
    println("Estimated order of convergence (global) = ", p_global)
    println("Estimated order of convergence (full)   = ", p_full)
    println("Estimated order of convergence (cut)    = ", p_cut)

    return (
        h_vals,
        err_vals,
        err_full_vals,
        err_cut_vals,
        err_empty_vals,
        p_global,
        p_full,
        p_cut,
    )
end

function run_mesh_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    norm
)

    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]

    for (nx, ny) in zip(nx_list, ny_list)
        # Build mesh
        x0, y0 = 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

        # Define the body
        circle = (x,y,_=0) -> (sqrt((x-center[1])^2 + (y-center[2])^2) - radius)
            
        # Define capacity/operator
        capacity = Capacity(circle, mesh)
        operator = DiffusionOps(capacity)

        # BC + solver
        bc_boundary = Robin(3.0,1.0,3.0*400)
        bc0 = Dirichlet(400.0)
        bc_b = BorderConditions(Dict(
            :left   => bc0,
            :right  => bc0,
            :top    => bc0,
            :bottom => bc0
        ))
        phase = Phase(capacity, operator, (x,y,z,t)->0.0, (x,y,z)->1.0)

        u0ₒ = ones((nx+1)*(ny+1)) * 270.0
        u0ᵧ = zeros((nx+1)*(ny+1)) * 270.0
        u0 = vcat(u0ₒ, u0ᵧ)

        Δt = 0.25*(lx/nx)^2
        Tend = 0.1

        solver = DiffusionUnsteadyMono(phase, bc_b, bc_boundary, Δt, u0, "BE") # Start by a backward Euler scheme to prevent oscillation due to CN scheme

        solve_DiffusionUnsteadyMono!(solver, phase, Δt, Tend, bc_b, bc_boundary, "CN"; method=Base.:\)

        # Compute errors
        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm)

        # Representative mesh size ~ 1 / min(nx, ny)
        push!(h_vals, 1.0 / min(nx, ny))

        push!(err_vals,       global_err)
        push!(err_full_vals,  full_err)
        push!(err_cut_vals,   cut_err)
        push!(err_empty_vals, empty_err)
    end

    # Model for curve_fit
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

    # Round
    p_global = round(p_global, digits=2)
    p_full   = round(p_full, digits=2)
    p_cut    = round(p_cut, digits=2)

    println("Estimated order of convergence (global) = ", p_global)
    println("Estimated order of convergence (full)   = ", p_full)
    println("Estimated order of convergence (cut)    = ", p_cut)

    # Plot in log-log scale
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "h",
        ylabel = "L$norm error",
        title  = "Convergence plot",
        xscale = log10,
        yscale = log10
    )

    scatter!(ax, h_vals, err_vals,       label="Global error ($p_global)", markersize=12)
    lines!(ax, h_vals, err_vals,         color=:black)
    scatter!(ax, h_vals, err_full_vals,  label="Full error ($p_full)",   markersize=12)
    lines!(ax, h_vals, err_full_vals,    color=:black)
    scatter!(ax, h_vals, err_cut_vals,   label="Cut error ($p_cut)",     markersize=12)
    lines!(ax, h_vals, err_cut_vals,     color=:black)

    lines!(ax, h_vals, 10.0*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax, h_vals, 1.0*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    axislegend(ax, position=:rb)
    display(fig)

    return (
        h_vals,
        err_vals,
        err_full_vals,
        err_cut_vals,
        err_empty_vals,
        p_global,
        p_full,
        p_cut,
    )
end

function compare_methods(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function
)
    # Run Front Tracking convergence
    println("Running Front Tracking convergence tests...")
    ft_results = run_mesh_convergence_ft(
        nx_list, ny_list, radius, center, u_analytical, norm=2
    )
    
    h_vals_ft, err_vals_ft, err_full_vals_ft, err_cut_vals_ft, 
    err_empty_vals_ft, p_global_ft, p_full_ft, p_cut_ft = ft_results
    
    # Run the original VOFI method
    println("\nRunning VOFI convergence tests...")
    vofi_results = run_mesh_convergence(
        nx_list, ny_list, radius, center, u_analytical, norm=2
    )
    
    h_vals_vofi, err_vals_vofi, err_full_vals_vofi, err_cut_vals_vofi, 
    err_empty_vals_vofi, p_global_vofi, p_full_vofi, p_cut_vofi = vofi_results
    
    # Create comparison plot
    fig_comp = Figure(size=(900, 600))
    ax_comp = Axis(
        fig_comp[1, 1],
        xlabel = "h",
        ylabel = "L2 error",
        title  = "Front Tracking vs VOFI for Heat Conduction",
        xscale = log10,
        yscale = log10,
        xminorticksvisible = true, 
        xminorgridvisible = true,
        xminorticks = IntervalsBetween(5),
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(5),
    )

    # Plot global errors
    scatter!(ax_comp, h_vals_ft, err_vals_ft, label="FT Global ($(p_global_ft))", 
             markersize=12, color=:blue)
    lines!(ax_comp, h_vals_ft, err_vals_ft, color=:blue)
    scatter!(ax_comp, h_vals_vofi, err_vals_vofi, label="VOFI Global ($(p_global_vofi))", 
             markersize=12, color=:red)
    lines!(ax_comp, h_vals_vofi, err_vals_vofi, color=:red)

    # Plot cut cell errors
    scatter!(ax_comp, h_vals_ft, err_cut_vals_ft, label="FT Cut ($(p_cut_ft))", 
             markersize=8, color=:blue, marker=:diamond)
    lines!(ax_comp, h_vals_ft, err_cut_vals_ft, color=:blue, linestyle=:dash)
    scatter!(ax_comp, h_vals_vofi, err_cut_vals_vofi, label="VOFI Cut ($(p_cut_vofi))", 
             markersize=8, color=:red, marker=:diamond)
    lines!(ax_comp, h_vals_vofi, err_cut_vals_vofi, color=:red, linestyle=:dash)

    # Reference lines
    lines!(ax_comp, h_vals_ft, 10.0*h_vals_ft.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax_comp, h_vals_ft, 1.0*h_vals_ft.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    axislegend(ax_comp, position=:rb)
    
    display(fig_comp)
    
    # Create a table to compare error values and convergence rates
    println("\n====== CONVERGENCE COMPARISON TABLE ======")
    println("                  | FRONT TRACKING |    VOFI")
    println("------------------+----------------+---------")
    println("Global convergence|      $(p_global_ft)      |    $(p_global_vofi)")
    println("Full cells        |      $(p_full_ft)      |    $(p_full_vofi)")
    println("Cut cells         |      $(p_cut_ft)      |    $(p_cut_vofi)")
    
    # Show the error ratio (FT/VOFI) at each resolution
    println("\n====== ERROR RATIO (FrontTracking/VOFI) ======")
    println("Mesh Size | Global Error | Cut Cells Error")
    println("----------+--------------+----------------")
    for i in 1:length(nx_list)
        ratio_global = err_vals_ft[i] / err_vals_vofi[i]
        ratio_cut = err_cut_vals_ft[i] / err_cut_vals_vofi[i]
        println("$(nx_list[i])x$(ny_list[i]) | $(round(ratio_global, digits=3)) | $(round(ratio_cut, digits=3))")
    end
    
    return fig_comp
end

# Define the analytical solution (same as original)
function radial_heat_(x, y)
    t=0.1
    R=1.0
    k=3.0
    a=1.0

    function j0_zeros_robin(N, k, R; guess_shift = 0.25)
        # Define the function for alpha J1(alpha) - k R J0(alpha) = 0
        eq(alpha) = alpha * besselj1(alpha) - k * R * besselj0(alpha)
    
        zs = zeros(Float64, N)
        for m in 1:N
            # Approximate location around (m - guess_shift)*π
            x_left  = (m - guess_shift - 0.5) * π
            x_right = (m - guess_shift + 0.5) * π
            x_left  = max(x_left, 1e-6)  # Ensure bracket is positive
            zs[m]   = find_zero(eq, (x_left, x_right))
        end
        return zs
    end

    alphas = j0_zeros_robin(1000, k, R)
    N=length(alphas)
    r = sqrt((x - center[1])^2 + (y - center[2])^2)
    if r >= R
        # Not physically in the domain, so return NaN
        return NaN
    end
    
    # If in the disk: sum the series
    s = 0.0
    for m in 1:N
        αm = alphas[m]
        An = 2.0 * k * R / ((k^2 * R^2 + αm^2) * besselj0(αm))
        s += An * exp(- a * αm^2 * t/R^2) * besselj0(αm * (r / R))
    end
    return (1.0 - s) * (400 - 270) + 270
end

# Run the comparison
nx_list = [20, 40, 80, 160]  # Reduced to save computation time
ny_list = [20, 40, 80, 160]
radius, center = 1.0, (2.01, 2.01)

fig = compare_methods(nx_list, ny_list, radius, center, radial_heat_)
CairoMakie.save("heat_ft_vs_vofi_comparison.png", fig)