using Penguin
using IterativeSolvers
using LinearAlgebra, SparseArrays

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


### 1D Test Case : Monophasic Steady Diffusion Equation with MMS
# Define the mesh
nx = 10
lx = 1.
x0 = 0.
Δx = lx / nx
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
center = 0.5
radius = 0.1
body = (x, _=0) -> -1.0

# Define the capacity
capacity = Capacity(body, mesh; compute_centroids=false)

# Define the operators
operator = DiffusionOps(capacity)

map_to_unit, xL_tuple, Leff_tuple, Δ_tuple = make_shift_map((nx,), (lx,), (x0,); shift_left=true)
# define û on [0,1]
u_hat = ξ -> sin(pi * ξ[1])
# physical exact solution
u_exact = wrap_u(u_hat, map_to_unit)

# Source term: f = λ * u_exact, where λ = (π/(lx-Δx))²
λ = (pi/Leff_tuple[1])^2
g = (x, y, _=0) -> λ * u_exact(x)
a = (x, y, _=0) -> 1.0

# Boundary conditions: Homogeneous Dirichlet (u = 0 on boundaries)
bc = Dirichlet(0.0)
bc1 = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(0.0)))

Fluide = Phase(capacity, operator, g, a)

# Define the solver
solver = DiffusionSteadyMono(Fluide, bc_b, bc)

# Solve the problem
solve_DiffusionSteadyMono!(solver; method=Base.:\)

# Analytical solution
u_analytical = u_exact

u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytical, solver, capacity, 2)

# Plot the numerical and analytical solutions
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="u",)
scatter!(ax, mesh.centers[1], u_num[1:end-1], label="Numerical Solution")
lines!(ax, mesh.centers[1], u_ana[1:end-1], label="Analytical Solution (from check_convergence)")
lines!(ax, mesh.centers[1], u_analytical.(mesh.centers[1]), linestyle=:dash, color=:red, label="Analytical Solution (function)")
Legend(fig[1, 2], ax; orientation = :vertical)
display(fig)  

# Plot the error
err = u_ana - u_num
err[capacity.cell_types .== 0] .= NaN
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="Log10 of the error", title="Monophasic Steady Diffusion Equation")
scatter!(ax, mesh.centers[1], log10.(abs.(err[1:nx])), label="Log10 of the error")
display(fig)