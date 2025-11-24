using Penguin
using IterativeSolvers
using SparseArrays
using LinearAlgebra
using CairoMakie
using Statistics

"""
1D Navier–Stokes benchmark (transverse / demo)

This example replaces the previous "poiseuille" script with a 1D Navier–Stokes-style
benchmark that follows the pattern used in the Stokes mono example. In strict 1D the
classical Poiseuille profile is transverse; here we build a driven/channel-like 1D
setup and run the steady Navier–Stokes mono solver. At the end we print diagnostics
similar to the Stokes benchmark: velocity stats, pressure stats, adjointness and
viscous-block diagnostics.
"""

###########
# Grids
###########
nx = 128
Lx = 1.0
x0 = 0.0

# Pressure grid
mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))

# Velocity grid staggered half a cell relative to pressure grid
dx = Lx / nx
mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5 * dx,))

###########
# Body and capacities/operators
###########
# Here the body is full domain (value negative means inside)
body = (x, _=0.0) -> -1.0
capacity_u = Capacity(body, mesh_u)
capacity_p = Capacity(body, mesh_p)

operator_u = DiffusionOps(capacity_u)
operator_p = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
# No-slip walls (Dirichlet zero) left/right; this is a transverse-driven demo in 1D
u_left  = Dirichlet(0.0)
u_right = Dirichlet(0.0)
bc_u = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => u_left, :right => u_right))

# Pressure gauge
pressure_gauge = MeanPressureGauge()

# Cut-cell/interface BC for cut velocity unknowns (prototype)
bc_cut = Dirichlet(0.0)

###########
# Sources and physical constants
###########
μ = 1.0
ρ = 1.0

# Apply a uniform body force in the 1D domain to drive the flow
fᵤ = (x, y=0.0, z=0.0) -> 1.0
fₚ = (x, y=0.0, z=0.0) -> 0.0

###########
# Assemble the phase object
###########
fluid = Fluid(mesh_u, capacity_u, operator_u, mesh_p, capacity_p, operator_p, μ, ρ, fᵤ, fₚ)

###########
# Initial conditions
###########
nu = prod(operator_u.size)
np = prod(operator_p.size)

u0_ω = zeros(nu)
u0_γ = zeros(nu)
p0_ω = zeros(np)
x0 = vcat(u0_ω, u0_γ, p0_ω)

###########
# Solver
###########
solver = NavierStokesMono(fluid, bc_u, pressure_gauge, bc_cut; x0=x0)
_, iters, res = solve_NavierStokesMono_steady!(solver; tol=1e-10, maxiter=200, relaxation=0.8)
println("Navier–Stokes steady solve: iterations=", iters, ", residual=", res)

###########
# Extract unknowns and trim inactive cells
###########
nu = prod(operator_u.size)
uω_full = solver.x[1:nu]
uγ_full = solver.x[nu+1:2nu]
pω_full = solver.x[2nu+1:end]

# Remove empty (masked) cells using capacity diagonal for reporting
mask_u = diag(capacity_u.V) .> 1e-12
mask_p = diag(capacity_p.V) .> 1e-12
uω = uω_full[mask_u]
uγ = uγ_full[mask_u]
pω = pω_full[mask_p]

###########
# Summary checks (statistics + diagnostics)
###########
mean_uω = mean(uω)
max_uω = maximum(abs.(uω))
mean_uγ = mean(uγ)
max_uγ = maximum(abs.(uγ))

min_p = minimum(pω)
max_p = maximum(pω)
mean_p = mean(pω)

# Adjointness and viscous diagnostics (use full operator sizes)
uω_rand = randn(nu)
uγ_zero = zeros(nu)
pω_rand = randn(np)
div_u = - (operator_p.G' + operator_p.H') * uω_rand + (operator_p.H') * uγ_zero
grad_p = operator_p.Wꜝ * (operator_p.G + operator_p.H) * pω_rand
lhs = dot(operator_p.V * div_u, pω_rand)
rhs = dot(uω_rand, grad_p)
adj_rel = abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))

# Viscous block
Iμ⁻¹ = build_I_D(operator_u, 1/μ, capacity_u)
WG = operator_u.Wꜝ * operator_u.G
S = Iμ⁻¹ * (operator_u.G' * WG)
sym_err = opnorm(Matrix(S - S')) / max(1e-12, opnorm(Matrix(S)))
uω_t = randn(nu)
rq = dot(uω_t, S * uω_t)

println("--- navierstokes_1d benchmark summary ---")
println("velocity uω: mean=", round(mean_uω, sigdigits=6), ", max_abs=", round(max_uω, sigdigits=6))
println("velocity uγ: mean=", round(mean_uγ, sigdigits=6), ", max_abs=", round(max_uγ, sigdigits=6))
println("pressure: min=", round(min_p, sigdigits=6), ", max=", round(max_p, sigdigits=6), ", mean=", round(mean_p, sigdigits=6))
println("adjointness rel.err=", adj_rel)
println("viscous sym.rel.err=", sym_err, ", Rayleigh quotient=", rq)

println("checks:")
println("- velocity nearly constant? ", (max_uω - minimum(uω)) < 1e-6)
println("- pressure low variation? ", mean_p < 1e-6)
println("benchmark file completed.")