using Penguin
using IterativeSolvers
using LinearAlgebra
using CairoMakie

# -- Domain setup (mirrors Basilisk's stream.c example) -----------------------
origin = (-0.5, -0.5)
extent = (1.0, 1.0)
mesh = Mesh((64, 64), extent, origin)

# Uniform fluid domain (no embedded boundary)
body = (x, y, _=0.0) -> -1.0
capacity = Capacity(body, mesh)

# -- Initial condition: pair of Gaussian vortices ----------------------------
operator = DiffusionOps(capacity)
n = prod(operator.size)

dd = 0.1
a = 1.0
b = 10.0
σ² = dd / b

function double_vortex(x, y)
    g1 = exp(-((x - dd)^2 + y^2) / σ²)
    g2 = exp(-((x + dd)^2 + y^2) / σ²)
    return a * (g1 + g2)
end

ω_bulk = [double_vortex(c[1], c[2]) for c in capacity.C_ω]
ω0 = vcat(ω_bulk, zeros(n))

# -- Boundary conditions ------------------------------------------------------
zero_dirichlet = Dirichlet(0.0)
border_bc = BorderConditions(Dict(
    :left => zero_dirichlet,
    :right => zero_dirichlet,
    :bottom => zero_dirichlet,
    :top => zero_dirichlet,
))

# -- Solver parameters --------------------------------------------------------
ν = 1.0e-4          # mimic (almost) inviscid evolution
Δt = 2.5e-3
t_end = 5.0

solver = StreamVorticity(capacity, ν, Δt;
    bc_stream = zero_dirichlet,
    bc_vorticity = zero_dirichlet,
    bc_stream_border = border_bc,
    bc_vorticity_border = border_bc,
    ω0 = ω0,
)

# -- Time integration ---------------------------------------------------------
steps = ceil(Int, (t_end - solver.time) / solver.Δt)
run_StreamVorticity!(solver, steps; method = gmres, scheme = "BE")

# -- Post-processing: vorticity snapshots ------------------------------------
outdir = joinpath(@__DIR__, "outputs", "double_vortex")
isdir(outdir) || mkpath(outdir)

function vorticity_matrix(state)
    ω_full = hasproperty(state, :ω) ? state.ω : state
    ω_bulk = ω_full[1:n]
    return reshape(ω_bulk, length(mesh.nodes[1]), length(mesh.nodes[2]))'
end

num_panels = 4
indices = round.(Int, range(1, length(solver.states), length = num_panels))
times = [solver.states[i].time for i in indices]
fields = [vorticity_matrix(solver.states[i]) for i in indices]

xs = mesh.nodes[1]
ys = mesh.nodes[2]

fig = Figure(resolution = (900, 600))

lim = maximum(abs, reduce(vcat, fields))

for (col, (field, t)) in enumerate(zip(fields, times))
    ax = Axis(fig[1, col], aspect = DataAspect(),
              title = "t = $(round(t, digits = 3))")
    heatmap!(ax, xs, ys, field; colormap = :balance, colorrange = (-lim, lim))
end

Colorbar(fig[1, num_panels + 1], colormap = :balance, limits = (-lim, lim), label = "ω")

save(joinpath(outdir, "double_vortex_snapshots.png"), fig)
println("Saved vorticity snapshots to $(joinpath(outdir, "double_vortex_snapshots.png")).")
