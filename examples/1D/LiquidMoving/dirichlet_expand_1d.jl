using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using Printf
using Statistics
using CairoMakie

# 1D Dirichlet expanding interface test case (Basilisk adaptation)
const V = 1.0
const T_EQ = 0.0
const T0_SHIFT = 1.0e-3

function t_exact(x, v, t)
    if x < v * t
        return -1.0 + exp(v * (x - v * t))
    end
    return 0.0
end

function run_case(nx; L=1.0, x0=-0.5, v=V, t0=T0_SHIFT, cfl=0.1, t_end=0.2, scheme="BE")
    dx = L / nx
    dt = cfl * dx / max(abs(v), eps())
    nsteps = max(1, Int(round(t_end / dt)))
    t_end = nsteps * dt

    mesh = Penguin.Mesh((nx,), (L,), (x0,))
    t_init = 0.0
    xf0 = v * (t_init - t0)
    body = (x, t, _=0) -> (x - xf0)

    stmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + dt], tag=mesh.tag)
    capacity = Capacity(body, stmesh)
    operator = DiffusionOps(capacity)

    f = (x, y, z, t) -> 0.0
    k = (x, y, z) -> 1.0
    phase = Phase(capacity, operator, f, k)

    bc_interface = Dirichlet(T_EQ)
    bc_bottom = Dirichlet((x, t) -> t_exact(x, v, t - t0))
    bc_top = Dirichlet(T_EQ)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
        :bottom => bc_bottom,
        :top => bc_top,
    ))

    x_nodes = range(x0, stop=x0 + L, length=nx + 1)
    t_eff = t_init - t0
    u0_bulk = [t_exact(x, v, t_eff) for x in x_nodes]
    u0_iface = fill(T_EQ, length(x_nodes))
    u0 = vcat(u0_bulk, u0_iface)

    ρ = -1.0
    Lh = 1.0
    stefan_cond = InterfaceConditions(nothing, FluxJump(1.0, 0.0, ρ * Lh))
    Newton_params = (20, 1e-10, 1e-10, 1.0)

    solver = MovingLiquidDiffusionUnsteadyMono(phase, bc_b, bc_interface, dt, u0, mesh, scheme)
    solver, residuals, xf_log, timestep_history = solve_MovingLiquidDiffusionUnsteadyMono!(
        solver, phase, xf0, dt, t_init, t_end, bc_b, bc_interface, stefan_cond, mesh, scheme;
        Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\,
    )

    times = collect(range(t_init, step=dt, length=length(solver.states)))
    l1_errors = Float64[]
    for (step, t) in enumerate(times)
        t_eff = t - t0
        u_num = solver.states[step][1:length(x_nodes)]
        u_ex = [t_exact(x, v, t_eff) for x in x_nodes]
        mask = x_nodes .<= v * t_eff
        if any(mask)
            err = sum(abs.(u_num[mask] .- u_ex[mask])) * dx
            l1 = err / (sum(mask) * dx)
        else
            l1 = 0.0
        end
        push!(l1_errors, l1)
    end

    return (solver=solver, x=x_nodes, times=times, l1_errors=l1_errors, dt=dt, t_end=t_end,
        xf_log=xf_log, timestep_history=timestep_history)
end

function write_profile(path, x_nodes, u_num, v, t_eff)
    open(path, "w") do io
        for (x, u) in zip(x_nodes, u_num)
            if x <= v * t_eff
                u_ex = t_exact(x, v, t_eff)
                @printf(io, "%.8g %.8g %.8g %.8g\n", x, u, u_ex, abs(u - u_ex))
            end
        end
    end
end

function main()
    cfl = 0.5
    t_end = 0.2
    levels = 0:2
    nx_list = Int[]
    l1_avg_list = Float64[]
    l1_final_list = Float64[]
    last_result = nothing
    for j in levels
        nx = 2^(5 + j)
        result = run_case(nx; cfl=cfl, t_end=t_end)
        l1_window = length(result.l1_errors) > 3 ? result.l1_errors[2:end-2] : result.l1_errors
        l1_avg = mean(l1_window)
        l1_final = l1_window[end]
        @printf("nx=%d dt=%.4g L1_avg=%.4e L1_final=%.4e\n",
                nx, result.dt, l1_avg, l1_final)

        log_path = @sprintf("log%d", j)
        init_path = @sprintf("init%d", j)
        out_path = @sprintf("out%d", j)

        open(log_path, "w") do io
            for t in result.times
                xf = V * (t - T0_SHIFT)
                @printf(io, "%.8g %.8g\n", t, xf)
            end
        end

        write_profile(init_path,
            result.x,
            result.solver.states[1][1:length(result.x)],
            V,
            -T0_SHIFT,
        )
        write_profile(out_path,
            result.x,
            result.solver.states[end][1:length(result.x)],
            V,
            result.t_end - T0_SHIFT,
        )

        push!(nx_list, nx)
        push!(l1_avg_list, l1_avg)
        push!(l1_final_list, l1_final)
        last_result = result
    end

    if length(nx_list) > 1
        println("Order of convergence (mesh-to-mesh):")
        for i in 2:length(nx_list)
            rate_avg = log(l1_avg_list[i-1] / l1_avg_list[i]) / log(2.0)
            rate_final = log(l1_final_list[i-1] / l1_final_list[i]) / log(2.0)
            @printf("nx %d -> %d | p_avg=%.3f p_final=%.3f\n",
                    nx_list[i-1], nx_list[i], rate_avg, rate_final)
        end
    end

    if last_result !== nothing
        x_nodes = last_result.x
        t_eff = last_result.t_end - T0_SHIFT
        u_num = last_result.solver.states[end][1:length(x_nodes)]
        u_ex = [t_exact(x, V, t_eff) for x in x_nodes]

        fig = Figure(resolution=(900, 600))
        ax = Axis(fig[1, 1], xlabel="x", ylabel="T", title="Final Temperature Profile")
        lines!(ax, x_nodes, u_ex, color=:black, linewidth=2, label="Exact")
        lines!(ax, x_nodes, u_num, color=:dodgerblue, linewidth=2, label="Numerical")
        axislegend(ax, position=:lt)
        display(fig)

        times = last_result.times
        xf_exact = [V * (t - T0_SHIFT) for t in times]
        xf_num = last_result.xf_log
        xf_times = range(0.0, last_result.t_end, length=length(xf_num))
        fig2 = Figure(resolution=(900, 600))
        ax2 = Axis(fig2[1, 1], xlabel="t", ylabel="x_f", title="Interface Position vs Time")
        lines!(ax2, times, xf_exact, color=:black, linewidth=2, label="Exact")
        lines!(ax2, xf_times, xf_num, color=:dodgerblue, linewidth=2, label="Numerical")
        axislegend(ax2, position=:lt)
        display(fig2)
    end
end

main()
