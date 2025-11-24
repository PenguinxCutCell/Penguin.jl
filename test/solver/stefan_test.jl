using Penguin
using Test
using Statistics

const ρ_test = 1.0
const L_test = 1.0
const TM_test = 0.0

function mean_radius(markers::Vector{Tuple{Float64,Float64}})
    if isempty(markers)
        return 0.0
    end
    xs = [m[1] for m in markers]
    ys = [m[2] for m in markers]
    if length(xs) > 1 && isapprox(xs[1], xs[end]; atol=1e-12) && isapprox(ys[1], ys[end]; atol=1e-12)
        pop!(xs)
        pop!(ys)
    end
    center_x = mean(xs)
    center_y = mean(ys)
    return mean(sqrt.((xs .- center_x) .^ 2 .+ (ys .- center_y) .^ 2))
end

function build_circle_problem(T∞; radius=0.8, nx=24, ny=24, nmarkers=120, t_init=0.6, nsteps=4)
    lx, ly = 8.0, 8.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (-lx/2, -ly/2))
    front = FrontTracker()
    create_circle!(front, 0.0, 0.0, radius, nmarkers)

    body = (x, y, t, _=0) -> -sdf(front, x, y)
    Δt = 0.08 * (lx / nx)^2
    STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + Δt], tag=mesh.tag)
    capacity = Capacity(body, STmesh; compute_centroids=false)
    operator = DiffusionOps(capacity)

    bc_outer = Dirichlet(T∞)
    bc_interface = Dirichlet(TM_test)
    bc_b = BorderConditions(Dict(:left => bc_outer, :right => bc_outer, :top => bc_outer, :bottom => bc_outer))
    stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ_test * L_test))

    K = (x, y, z) -> 1.0
    f = (x, y, z, t) -> 0.0
    phase = Phase(capacity, operator, f, K)

    npoints = (nx + 1) * (ny + 1)
    bulk_ic = fill(T∞, npoints)
    ghost_ic = fill(TM_test, npoints)
    u0 = vcat(bulk_ic, ghost_ic)

    solver = redirect_stdout(devnull) do
        StefanMono2D(phase, bc_b, bc_interface, Δt, u0, mesh, "BE")
    end
    t_final = t_init + nsteps * Δt
    return (; solver, phase, front, mesh, bc_b, bc_interface, stef_cond, Δt, t_init, t_final)
end

function run_geometric_radius(T∞; kwargs...)
    prob = build_circle_problem(T∞; kwargs...)
    result = redirect_stdout(devnull) do
        solve_StefanMono2D_geom!(prob.solver, prob.phase, prob.front, prob.Δt, prob.t_init, prob.t_final,
                                 prob.bc_b, prob.bc_interface, prob.stef_cond, prob.mesh, "BE";
                                 Newton_params=(15, 1e-6, 1e-6, 0.7),
                                 smooth_factor=0.3,
                                 window_size=4,
                                 method=Base.:\)
    end
    _, _, xf_log, _, _, _ = result
    initial_markers = xf_log[minimum(keys(xf_log))]
    final_markers = xf_log[maximum(keys(xf_log))]
    return mean_radius(initial_markers), mean_radius(final_markers)
end

function run_ls_radius(T∞; kwargs...)
    prob = build_circle_problem(T∞; kwargs...)
    result = redirect_stdout(devnull) do
        solve_StefanMono2D!(prob.solver, prob.phase, prob.front, prob.Δt, prob.t_init, prob.t_final,
                             prob.bc_b, prob.bc_interface, prob.stef_cond, prob.mesh, "BE";
                             Newton_params=(12, 1e-6, 1e-6, 0.6),
                             enable_stencil_fusion=false,
                             smooth_factor=0.3,
                             window_size=4,
                             method=Base.:\)
    end
    _, _, xf_log, _, _, _ = result
    initial_markers = xf_log[minimum(keys(xf_log))]
    final_markers = xf_log[maximum(keys(xf_log))]
    return mean_radius(initial_markers), mean_radius(final_markers)
end

@testset "Stefan front-tracking geometric growth" begin
    mktempdir() do tmp
        cd(tmp) do
            r0, rf = run_geometric_radius(-0.4)
            @test rf > r0
        end
    end
end

@testset "Stefan front-tracking geometric melt" begin
    mktempdir() do tmp
        cd(tmp) do
            r0, rf = run_geometric_radius(0.4)
            @test rf < r0
        end
    end
end

@testset "Stefan front-tracking solver consistency" begin
    mktempdir() do tmp
        cd(tmp) do
            r0_geom, rf_geom = run_geometric_radius(-0.3)
            r0_ls, rf_ls = run_ls_radius(-0.3)
            @test isapprox(r0_geom, r0_ls; atol=1e-6)
            @test isapprox(rf_geom, rf_ls; atol=0.15)
        end
    end
end
