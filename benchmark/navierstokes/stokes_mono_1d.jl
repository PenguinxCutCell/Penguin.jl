using Penguin
using IterativeSolvers
using SparseArrays
using LinearAlgebra
using CairoMakie
using Statistics

"""
1D Monophasic Stokes benchmark

This benchmark is a lightly adapted copy of examples/1D/Stokes/stokes_mono.jl
placed under benchmark/navierstokes. At the end it prints simple checks:
- velocity mean and max (should be near expected Dirichlet profile)
- pressure min/max and mean (check gauge and low variation)
- adjointness and viscous-block Rayleigh quotient diagnostics

This script intentionally avoids heavy plotting when used in automated CI,
but keeps the plotting code commented for interactive runs.
"""

###########
# Grids
###########
nx = 160
lx = 4.0
x0 = 0.0

# Pressure grid
mesh_p = Penguin.Mesh((nx,), (lx,), (x0,))

# Velocity grid staggered half a cell relative to pressure grid
dx = lx / nx
mesh_u = Penguin.Mesh((nx,), (lx,), (x0 - 0.5*dx,))

###########
# Body and capacities/operators
###########
# Level-set body (e.g., interface/reference position); not strictly required for Stokes but kept for consistency
xint = 2.0 
body = (x, _=0) -> (x-xint)^2 - 1.0

# Capacities (per-field)
capacity_u = Capacity(body, mesh_u)
capacity_p = Capacity(body, mesh_p)

# Operators (per-field)
operator_u = DiffusionOps(capacity_u)
operator_p = DiffusionOps(capacity_p)


###########
# Boundary conditions
###########
# In 1D examples here, :bottom and :top correspond to the left and right boundaries
u_left  = Dirichlet(0.0)
u_right = Dirichlet(0.0)
bc_u = BorderConditions(Dict{Symbol, AbstractBoundary}(:bottom => u_left, :top => u_right))

# Pressure gauge: pin one cell to zero (leaves rest free)
pressure_gauge = PinPressureGauge()

# Cut-cell/interface boundary condition for uγ (prototype: Dirichlet constant)
u_bc = Dirichlet(1.0)

###########
# Sources and physical constants
###########
fᵤ = (x, y=0.0, z=0.0) -> 0.0   # Body force for velocity equation (per unit volume)
fₚ = (x, y=0.0, z=0.0) -> 0.0   # Mass source for continuity (usually 0)

μ = 1.0  # Dynamic viscosity
ρ = 1.0  # Density

###########
# Phase type: Fluid
###########

###########
# Assemble the phase object
###########
fluid = Fluid(mesh_u, capacity_u, operator_u, mesh_p, capacity_p, operator_p, μ, ρ, fᵤ, fₚ)

###########
# Initial conditions (placeholders)
###########
nu = prod(operator_u.size)
np = prod(operator_p.size)

u0_ω = zeros(nu)
u0_γ = zeros(nu)
p0_ω = zeros(np)

# Global initial vector: [uω; uγ; pω]
x0 = vcat(u0_ω, u0_γ, p0_ω)

###########
# Solver (use exported type)
###########
solver = StokesMono(fluid, bc_u, pressure_gauge, u_bc; x0=x0)
solve_StokesMono!(solver)

# Extract unknowns
u = prod(operator_u.size)
u = prod(operator_u.size)
nu = prod(operator_u.size) # ensure nu in scope
uω = solver.x[1:nu]
uγ = solver.x[nu+1:2nu]
pω = solver.x[2nu+1:end]

# Remove empty cells
uω = uω[diag(capacity_u.V) .> 1e-12]
uγ = uγ[diag(capacity_u.V) .> 1e-12]
pω = pω[diag(capacity_p.V) .> 1e-12]

# Basic checks and summary prints
mean_uω = mean(uω)
max_uω = maximum(abs.(uω))
mean_uγ = mean(uγ)
max_uγ = maximum(abs.(uγ))

min_p = minimum(pω)
max_p = maximum(pω)
mean_p = mean(pω)

# Adjointness and viscous diagnostics (copied from example)
# 1) Discrete adjointness check: < -div uω, p >_V ≈ < uω, grad p >_W
uω_rand = randn(nu)
uγ_zero = zeros(nu)
pω_rand = randn(np)
div_u = - (operator_p.G' + operator_p.H') * uω_rand + (operator_p.H') * uγ_zero
grad_p = operator_p.Wꜝ * (operator_p.G + operator_p.H) * pω_rand
lhs = dot(operator_p.V * div_u, pω_rand)
rhs = dot(uω_rand, grad_p)
adj_rel = abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))

# 2) Viscous block symmetry/positivity (Rayleigh quotient for G' W† G)
Iμ⁻¹ = build_I_D(operator_u, 1/μ, capacity_u)
WG = operator_u.Wꜝ * operator_u.G
S = Iμ⁻¹ * (operator_u.G' * WG)
sym_err = opnorm(Matrix(S - S')) / max(1e-12, opnorm(Matrix(S)))
uω_t = randn(nu)
rq = dot(uω_t, S * uω_t)

# Print concise benchmark summary (machine-friendly lines and human-friendly)
println("--- stokes_mono_benchmark summary ---")
println("velocity uω: mean=", round(mean_uω, sigdigits=6), ", max_abs=", round(max_uω, sigdigits=6))
println("velocity uγ: mean=", round(mean_uγ, sigdigits=6), ", max_abs=", round(max_uγ, sigdigits=6))
println("pressure: min=", round(min_p, sigdigits=6), ", max=", round(max_p, sigdigits=6), ", mean=", round(mean_p, sigdigits=6))
println("adjointness rel.err=", adj_rel)
println("viscous sym.rel.err=", sym_err, ", Rayleigh quotient (>=0 expected)=", rq)

# Simple checks to help detect obvious failures
println("checks:")
println("- velocity nearly constant? ", (max_uω-minimum(uω)) < 1e-6)
println("- pressure low variation? ", (max_p - min_p) < 1e-6)

# Optionally save figures when running interactively (commented for CI)
# using CairoMakie
# fig = Figure(resolution=(900, 500))
# ax1 = Axis(fig[1, 1], xlabel="x", ylabel="uₓ", title="Velocity (staggered)")
# lines!(ax1, mesh_u.nodes[1], uω, color=:blue, label="uω")
# lines!(ax1, mesh_u.nodes[1], uγ, color=:orange, linestyle=:dash, label="uγ")
# ax2 = Axis(fig[1, 2], xlabel="x", ylabel="p", title="Pressure (cell-centered)")
# lines!(ax2, mesh_p.nodes[1], pω, color=:green, label="pω")
# save("benchmark_stokes1d_u_p.png", fig)

println("benchmark file completed.")
