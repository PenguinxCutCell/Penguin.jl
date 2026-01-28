using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CSV
using DataFrames

# ------------------------------------------------------------------------------
# Shared problem setup (single time step)
# ------------------------------------------------------------------------------
nx = 40
lx = 1.0
x0 = 0.0
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

xf₀ = 0.05 * lx
body = (x, t, _=0) -> (x - xf₀)

Δt = 0.5 * (lx / nx)^2
Tstart = 0.0
Tend = Δt  # only the first time step

STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag = mesh.tag)
capacity = Capacity(body, STmesh)
operator = DiffusionOps(capacity)

phase = Phase(
    capacity,
    operator,
    (x, y, z, t) -> 0.0,  # source term
    (x, y, z) -> 1.0,     # conductivity
)

bc = Dirichlet(0.0)
bc_b = BorderConditions(Dict(
    :top => Dirichlet(0.0),
    :bottom => Dirichlet(1.0),
))

ρ, L = 1.0, 1.0
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ * L))

u0 = vcat(zeros(nx + 1), zeros(nx + 1))

max_iter = 1000
tol = eps()
reltol = eps()
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# ------------------------------------------------------------------------------
# Parameter sweeps: fixed baseline + Adagrad + Barzilai-Borwein
# ------------------------------------------------------------------------------
cases = [
    (:fixed, (; label = "fixed_default")),
    (:barzilai_borwein, (; min_lr = 1e-6, max_lr = 10.0, label = "Newton")),
   
]

# ------------------------------------------------------------------------------
# Helper to run a case and capture first-step diagnostics
# ------------------------------------------------------------------------------
function run_case(strategy::Symbol, opts::NamedTuple)
    label = get(opts, :label, string(strategy))
    println("\n=== Strategy: $(strategy) | Label: $(label) ===")
    opts_for_solver = Base.structdiff(opts, (; label = label))

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
        learning_rate_options = opts_for_solver,
    )

    first_key = isempty(residuals) ? nothing : minimum(keys(residuals))
    residual_vector = first_key === nothing ? Float64[] : residuals[first_key]
    total_iters = length(residual_vector)
    final_residual = total_iters == 0 ? NaN : (residual_vector[end] isa AbstractArray ? norm(residual_vector[end]) : residual_vector[end])
    final_xf = isempty(xf_log) ? NaN : xf_log[end]
    effective_alpha = solver isa Solver ? NaN : NaN  # placeholder if we later expose α history

    rows = Vector{NamedTuple{(:label, :iteration, :residual), Tuple{String, Int, Float64}}}()
    for (iter, value) in enumerate(residual_vector)
        residual_scalar = value isa AbstractArray ? norm(value) : value
        push!(rows, (label, iter, residual_scalar))
    end
    if !isempty(rows)
        df = DataFrame(rows)
        CSV.write("residuals_step1_$(label).csv", df)
        println("Saved step-1 residuals to residuals_step1_$(label).csv")
    end

    return (
        strategy = strategy,
        label = label,
        options = opts_for_solver,
        iterations_first_step = total_iters,
        residual_first_step = final_residual,
        interface_after_step = final_xf,
    )
end

summaries = Vector{NamedTuple}()
for (strategy, opts) in cases
    push!(summaries, run_case(strategy, opts))
end

summary_df = DataFrame(summaries)
CSV.write("learning_rate_step1_summary.csv", summary_df)

println("\nSummary table (first time step only):")
show(summary_df, allcols = true, allrows = true)
println()

# ------------------------------------------------------------------------------
# Visualization of residual histories
# ------------------------------------------------------------------------------
try
    using CairoMakie

    combined_df = DataFrame(label = String[], iteration = Int[], residual = Float64[])
    for (strategy, opts) in cases
        label = get(opts, :label, string(strategy))
        file = "residuals_step1_$(label).csv"
        if isfile(file)
            df = CSV.read(file, DataFrame)
            df.label .= label
            append!(combined_df, df)
        else
            @warn "Skipping plot for $(label); residual CSV not found."
        end
    end

    if nrow(combined_df) > 0
        fig = Figure(resolution = (1600, 900), fontsize = 14)
        ax = Axis(fig[1, 1];
            xlabel = "Newton iteration",
            ylabel = "Residual norm",
            yscale = Makie.log10,
            title = "First-step Newton residuals across learning-rate settings",
            xminorticksvisible = true,
            yminorticksvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
        )

        grouped = groupby(combined_df, :label)
        for df in grouped
            lines!(ax, df.iteration, df.residual; label = df.label[1])
        end

        axislegend(ax, position = :rb, framevisible = false)
        save("learning_rate_step1_residuals.png", fig)
        println("Saved residual comparison plot to learning_rate_step1_residuals.png")
    else
        println("No residual data available for plotting.")
    end
catch err
    @warn "Could not generate residual plot." exception = (err, catch_backtrace())
end
