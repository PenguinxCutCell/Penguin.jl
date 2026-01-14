using Penguin
using LinearSolve
using Printf
using Statistics

# ---------------------------------------------------------
# 2D MMS Poisson, homogeneous Dirichlet on all borders
# u_exact = sin(pi*(x-Δx)/(Lx-Δx)) * sin(pi*(y-Δy)/(Ly-Δy))
# -Δu = f = ( (pi/(Lx-Δx))^2 + (pi/(Ly-Δy))^2 ) * u_exact
# ---------------------------------------------------------

function run_mms_convergence_2d(nx_list; lx=1.0, ly=1.0, x0=0.0, y0=0.0,
                                method_capacity="ImplicitIntegration",
                                solver_alg=KrylovJL_GMRES())

    h_vals   = Float64[]
    err_vals = Float64[]

    for nx in nx_list
        ny = nx
        Δx = lx/nx
        Δy = ly/ny
        lx_eff = lx - Δx
        ly_eff = ly - Δy

        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

        # Full domain
        body(x,y,_=0) = -1.0

        capacity = Capacity(body, mesh; method=method_capacity, compute_centroids=false)
        operator = DiffusionOps(capacity)

        # MMS exact (aligned with your 1D convention)
        u_exact(x,y) = sin(pi*(x - Δx)/lx_eff) * sin(pi*(y - Δy)/ly_eff)

        λx = (pi/lx_eff)^2
        λy = (pi/ly_eff)^2
        f(x,y,_=0) = (λx + λy) * u_exact(x,y)
        D(x,y,_=0) = 1.0

        # Homogeneous Dirichlet everywhere
        bc0 = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(:left=>bc0, :right=>bc0, :bottom=>bc0, :top=>bc0))

        phase  = Phase(capacity, operator, f, D)
        solver = DiffusionSteadyMono(phase, bc_b, bc0)

        solve_DiffusionSteadyMono!(solver; algorithm=solver_alg, log=false)

        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(u_exact, solver, capacity, 2, false)

        push!(h_vals, 1.0/nx)
        push!(err_vals, global_err)

        @printf("nx=%4d  h=%.4e  L2=%.6e\n", nx, h_vals[end], err_vals[end])
    end

    orders = [log(err_vals[i]/err_vals[i+1]) / log(h_vals[i]/h_vals[i+1])
              for i in 1:length(err_vals)-1]

    println("\nEstimated order (pairwise):")
    for i in 1:length(orders)
        @printf("  %4d -> %4d : %.3f\n", nx_list[i], nx_list[i+1], orders[i])
    end
    if !isempty(orders)
        w = min(2, length(orders))
        @printf("  Mean order (last %d): %.3f\n", w, mean(@view orders[end-w+1:end]))
    end

    return h_vals, err_vals, orders
end

# Example run
nx_list = [20, 40, 80, 160]
h, err, p = run_mms_convergence_2d(nx_list; lx=1.0, ly=1.0)
