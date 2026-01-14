using Penguin
using LinearSolve
using Printf
using Statistics

# ---------------------------------------------------------
# 2D MMS Poisson on full domain (no body), with shift map
# u_hat(ξ) = sin(pi*ξ1) * sin(pi*ξ2), ξ in [0,1]^2
# u_exact(x) = u_hat(map_to_unit(x))
# -Δu = f = ( (pi/Leff[1])^2 + (pi/Leff[2])^2 ) * u_exact
# ---------------------------------------------------------

function make_shift_map(nx, L, x0; shift_left::Bool=true)
    d = length(nx)
    Δ   = ntuple(i -> L[i] / nx[i], d)
    xL  = ntuple(i -> x0[i] + (shift_left ? Δ[i] : 0.0), d)
    Leff = ntuple(i -> L[i] - (shift_left ? Δ[i] : 0.0), d)

    map_to_unit = function (x)
        ntuple(i -> (x[i] - xL[i]) / Leff[i], d)
    end

    return map_to_unit, xL, Leff, Δ
end

wrap_u(u_hat, map_to_unit) = function (x...)
    ξ = map_to_unit(ntuple(i -> x[i], length(x)))
    return u_hat(ξ...)
end

function run_mms_convergence_2d(nx_list; lx=1.0, ly=1.0, x0=0.0, y0=0.0,
                                method_capacity="VOFI",
                                solver_alg=KrylovJL_GMRES())

    h_vals   = Float64[]
    err_vals = Float64[]

    for nx in nx_list
        ny = nx
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

        # Full domain
        body(x,y,_=0) = -1.0

        capacity = Capacity(body, mesh; method=method_capacity, compute_centroids=false)
        operator = DiffusionOps(capacity)

        map_to_unit, xL, Leff, Δ = make_shift_map((nx, ny), (lx, ly), (x0, y0); shift_left=true)
        u_hat = (ξ1, ξ2) -> sin(pi * ξ1) * sin(pi * ξ2)
        u_exact = wrap_u(u_hat, map_to_unit)

        λ = (pi/Leff[1])^2 + (pi/Leff[2])^2
        f(x, y, _=0) = λ * u_exact(x, y)
        D(x,y,_=0) = 1.0

        bc0 = Dirichlet(0.0)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
            :left   => Dirichlet((y, _=0) -> u_exact(xL[1], y)),
            :right  => Dirichlet((y, _=0) -> u_exact(x0 + lx, y)),
            :bottom => Dirichlet((x, _=0) -> u_exact(x, xL[2])),
            :top    => Dirichlet((x, _=0) -> u_exact(x, y0 + ly)),
        ))

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
