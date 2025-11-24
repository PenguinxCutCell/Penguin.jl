using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using DataFrames

# Problem setup shared across runs ------------------------------------------------
nx = 80
lx = 1.0
x0 = 0.0
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

xf₀ = 0.05 * lx
body = (x, t, _=0) -> (x - xf₀)

Δt = 0.5 * (lx / nx)^2
Tstart = 0.0
Tend = 0.01
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag = mesh.tag)

capacity = Capacity(body, STmesh)
operator = DiffusionOps(capacity)

bc = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :top => Dirichlet(0.0),
    :bottom => Dirichlet(1.0),
))

ρ, L = 1.0, 1.0
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ * L))

source_term = (x, y, z, t) -> 0.0
conductivity = (x, y, z) -> 1.0

phase = Phase(capacity, operator, source_term, conductivity)

u0ₒ = zeros(nx + 1)
u0ᵧ = zeros(nx + 1)
u0 = vcat(u0ₒ, u0ᵧ)

max_iter = 1000
tol = eps()
reltol = eps()
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

strategies = Dict(
    :fixed => (;),
    :adagrad => (; eps = 1e-10, min_lr = 1e-6),
    :rmsprop => (; beta2 = 0.9, eps = 1e-10, min_lr = 1e-6),
    :nadam => (; beta1 = 0.95, beta2 = 0.999, eps = 1e-10, min_lr = 1e-6),
    :barzilai_borwein => (; min_lr = 1e-6, max_lr = 10.0),
)

results = DataFrame(strategy = Symbol[], total_iterations = Int[], final_residual = Float64[], final_xf = Float64[])

function dump_residuals(strategy, residuals)
    if isempty(residuals)
        return
    end

    rows = Vector{NamedTuple{(:strategy, :timestep, :iteration, :residual), Tuple{Symbol, Int, Int, Float64}}}()
    for timestep in sort(collect(keys(residuals)))
        values = residuals[timestep]
        for (it, value) in enumerate(values)
            residual_value = value isa AbstractArray ? norm(value) : value
            push!(rows, (strategy = strategy, timestep = timestep, iteration = it, residual = residual_value))
        end
    end

    if !isempty(rows)
        df = DataFrame(rows)
        filename = "residuals_$(Symbol(strategy)).csv"
        CSV.write(filename, df)
        println("Wrote residual history for $(strategy) to $(filename)")
    end
end

for (strategy, opts) in strategies
    println("\n=== Running strategy: $(strategy) ===")
    solver = MovingLiquidDiffusionUnsteadyMono(phase, bc_b, bc, Δt, u0, mesh, "BE")
    solver, residuals, xf_log, timestep_history = solve_MovingLiquidDiffusionUnsteadyMono!(
        solver,
        phase,
        xf₀,
        Δt,
        Tstart,
        Tend,
        bc_b,
        bc,
        stef_cond,
        mesh,
        "CN";
        Newton_params = Newton_params,
        adaptive_timestep = false,
        method = IterativeSolvers.gmres,
        learning_rate_strategy = strategy,
        learning_rate_options = opts,
    )

    dump_residuals(strategy, residuals)

    total_iterations = sum(length(v) for v in values(residuals))
    final_residual = NaN
    if !isempty(residuals)
        last_key = maximum(keys(residuals))
        last_vec = residuals[last_key]
        final_residual = isempty(last_vec) ? NaN : (last_vec[end] isa AbstractArray ? norm(last_vec[end]) : last_vec[end])
    end
    final_xf = isempty(xf_log) ? NaN : xf_log[end]

    push!(results, (; strategy, total_iterations, final_residual, final_xf))
end

println("\nSummary of learning rate strategies:")
show(results, allcols = true, allrows = true)
println()
