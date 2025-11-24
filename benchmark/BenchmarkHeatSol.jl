# BenchmarkHeatIterative.jl

using Penguin
using IterativeSolvers
using LinearSolve
using CairoMakie
using Statistics

#–– 1) Problem setup (identical to your Heat.jl) ––
nx, ny = 80, 80
lx, ly = 4.0, 4.0
mesh = Penguin.Mesh((nx, ny), (lx, ly), (0.0, 0.0))

radius, center = ly/4, (lx/2, ly/2) .+ (0.01, 0.01)
circle = (x,y,_=0) -> sqrt((x-center[1])^2 + (y-center[2])^2) - radius

capacity = Capacity(circle, mesh)
operator = DiffusionOps(capacity)

bc0 = Dirichlet(0.0)
bc = Dirichlet(1.0)
bc_b = BorderConditions(Dict(
  :left   => bc0,
  :right  => bc0,
  :top    => bc0,
  :bottom => bc0
))

f      = (x,y,z,t) -> 0.0
Dcoef  = (x,y,z)   -> 1.0
Fluide = Phase(capacity, operator, f, Dcoef)

u0ₒ = zeros((nx+1)*(ny+1))
u0ᵧ = ones((nx+1)*(ny+1))
u0  = vcat(u0ₒ, u0ᵧ)

Δt   = 0.25 * (lx/nx)^2
Tend = 0.011

#–– 2) The list of iterative‐solver methods to try ––
methods = Dict(
  "CG"      => cg,
  "BiCGSTAB"=> bicgstabl,
  "MINRES"  => minres,
  "IDRS"    => idrs,
  "GMRES"   => gmres,
   #"LSMR"    => lsmr,
  "LSQR"    => lsqr
)

#–– 3) Storage for residual histories (first time‐step only) ––
res_hist = Dict{String, Vector{Float64}}()
stats = Dict{String, Tuple{Int,Float64}}()  # (niter, final_res)
#–– 4) Loop over methods ––
solvers = Dict{String, Any}()
for (name, meth) in methods
    println("→ Testing $name…")
    solver = DiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, "BE")
    try
        solve_DiffusionUnsteadyMono!(
            solver, Fluide, Δt, Tend, bc_b, bc, "BE";
            method = meth,
            reltol   = eps(Float64),
            log    = true
        )
    catch err
        @warn "Solver failed for $name with atol=eps(). Retrying with abstol=1e-10..." exception=err
        solve_DiffusionUnsteadyMono!(
            solver, Fluide, Δt, Tend, bc_b, bc, "BE";
            method = meth,
            atol = eps(Float64),
            btol = eps(Float64),
            log    = true
        )
    end

    ch = solver.ch[1]
    resid = ch[:resnorm]
    niter = length(resid)
    final_res = resid[end]

    println("    iterations = $niter, final residual = $(final_res)")
    res_hist[name] = resid
    stats[name] = (niter, final_res)
    solvers[name] = solver
end

#–– 5) Plot all residual histories together ––
fig = Figure(resolution = (800, 500))
ax  = Axis(fig[1, 1],
    xlabel = "Iteration",
    ylabel = "Residual norm",
    yscale = log10,
    title  = "Iteration‐residual histories (timestep 1)",
    yminorticksvisible = true,
    yminorgridvisible = true,
    yminorticks = IntervalsBetween(10),
    yticks = LogTicks(WilkinsonTicks(9,k_min=3)),
)

for (name, resid) in res_hist
    lines!(ax, 1:length(resid), resid, label = name)
end
axislegend(ax; position = :rb)

display(fig)

using ColorSchemes

# For plotting residuals at different timesteps, pick a solver (e.g., the first one)
first_solver = first(values(solvers))
timesteps = [1, Int(cld(length(first_solver.ch),2)), length(first_solver.ch)]
tlabels = ["First", "Middle", "Last"]

fig = Figure(resolution = (1200, 400))
for (j, tstep) in enumerate(timesteps)
    ax = Axis(fig[1, j],
        xlabel = "Iteration",
        ylabel = "Residual norm",
        yscale = log10,
        title  = "Residuals at timestep $tstep",
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = IntervalsBetween(10),
        yticks = LogTicks(WilkinsonTicks(9,k_min=3)),
    )
    n_methods = length(methods)
    colors = [get(ColorSchemes.viridis, (i-1)/(n_methods-1)) for i in 1:n_methods]
    i = 1
    for (name, meth) in methods
        ch = solvers[name].ch[tstep]
        resid = ch[:resnorm]
        lines!(ax, 1:length(resid), resid, color=colors[i], label=name)
        i += 1
    end
    axislegend(ax; position = :rb)
end

display(fig)
save("iterativesolvers_residuals.pdf", fig)