using Penguin
using IterativeSolvers
using SparseArrays
using LinearAlgebra
using CairoMakie

"""
1D Monophasic Stokes (steady)

This example assembles a staggered 1D Stokes system with unknowns
`[uₓ^ω, uₓ^γ, p^ω]` and equations:
- Momentum: `-(1/μ) G' W† G uₓ^ω -(1/μ) G' W† H uₓ^γ - W† G p^ω = V fᵤ`
- Continuity: `-(G' + H') uₓ^ω + H' uₓ^γ = 0`
Velocity uses Dirichlet at domain boundaries; pressure uses a single gauge (p at left).
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
# Fluid type is now defined in src/phase.jl and exported

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
# Solver (stub)
###########
# StokesMono and its solver are now defined in src/solver/stokes.jl and exported

###########
# Build and run the (stub) solver
###########
solver = StokesMono(fluid, bc_u, pressure_gauge, u_bc; x0=x0)
solve_StokesMono!(solver)

println(solver.x)

println("StokesMono solution ready. Unknowns = ", length(solver.x))

###########
# Plot velocity and pressure
###########
uω = solver.x[1:nu]
uγ = solver.x[nu+1:2nu]
pω = solver.x[2nu+1:end]

x_u = mesh_u.nodes[1]
x_p = mesh_p.nodes[1]

fig = Figure(resolution=(900, 500))
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="uₓ", title="Velocity (staggered)")
lines!(ax1, x_u, uω, color=:blue, label="uω")
lines!(ax1, x_u, uγ, color=:orange, linestyle=:dash, label="uγ")
axislegend(ax1, position=:rb)

ax2 = Axis(fig[1, 2], xlabel="x", ylabel="p", title="Pressure (cell-centered)")
lines!(ax2, x_p, pω, color=:green, label="pω")
axislegend(ax2, position=:rb)

display(fig)
save("stokes1d_u_p.png", fig)

###########
# Diagnostics: sparsity, adjointness, spectrum
###########
A_full = solver.A
nu = prod(operator_u.size)
np = prod(operator_p.size)

# Trim zero rows/cols for clearer visualization (matches solver trimming)
A, btrim, rows_idx, cols_idx = remove_zero_rows_cols_separate!(A_full, solver.b)
rows, cols = size(A)

# Map block separators after trimming
k1 = count(<=(nu), cols_idx)       # columns up to end of uω
k2 = count(<=(2nu), cols_idx)      # columns up to end of uγ
h1 = count(<=(nu), rows_idx)       # rows up to end of momentum block
h2 = count(<=(2nu), rows_idx)      # rows up to end of tie block

# 1) Sparsity pattern with block separators
I, J, V = findnz(A)
fig_spy = Figure(resolution=(700, 600))
ax_spy = Axis(fig_spy[1, 1], xlabel="columns", ylabel="rows", title="Sparsity pattern of A (trimmed)")
# Flip y for matrix-like view
scatter!(ax_spy, J, rows .- I .+ 1; markersize=2, color=:black)
vlines!(ax_spy, [k1, k2]; color=:red, linestyle=:dash)
hlines!(ax_spy, rows .- [h1, h2] .+ 1; color=:red, linestyle=:dash)
display(fig_spy)
save("stokes1d_A_spy.png", fig_spy)

# 2) Discrete adjointness check: < -div uω, p >_V ≈ < uω, grad p >_W
# Use uγ = 0 to isolate the uω-divergence pairing in continuity
uω_rand = randn(nu)
uγ_zero = zeros(nu)
pω_rand = randn(np)
div_u = - (operator_p.G' + operator_p.H') * uω_rand + (operator_p.H') * uγ_zero
grad_p = operator_p.Wꜝ * (operator_p.G + operator_p.H) * pω_rand
lhs = dot(operator_p.V * div_u, pω_rand)
rhs = dot(uω_rand, grad_p) # pairing only with uω
println("Adjointness check (uγ=0): |lhs-rhs|/max(1,|lhs|,|rhs|) = ", abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs)))

# 3) Viscous block symmetry/positivity (Rayleigh quotient for G' W† G)
Iμ⁻¹ = build_I_D(operator_u, 1/μ, capacity_u)
WG = operator_u.Wꜝ * operator_u.G
S = Iμ⁻¹ * (operator_u.G' * WG)
sym_err = opnorm(Matrix(S - S')) / max(1e-12, opnorm(Matrix(S)))
uω_t = randn(nu)
rq = dot(uω_t, S * uω_t)
println("Viscous block symmetry rel. error = ", sym_err, "; Rayleigh quotient (>=0 expected) = ", rq)

# 4) Singular values (to visualize saddle-point structure)
try
    using LinearAlgebra
    Svals = svdvals(Matrix(A))
    fig_sv = Figure(resolution=(700, 400))
    ax_sv = Axis(fig_sv[1, 1], xlabel="index", ylabel="log10(σ)", title="Singular values of A")
    lines!(ax_sv, 1:length(Svals), log10.(Svals .+ eps()));
    display(fig_sv)
    save("stokes1d_A_svals.png", fig_sv)
catch e
    @warn "SVD failed for A (dense)." exception=e
end

# 5) Row nonzeros histogram (structure diagnostic)
row_counts = zeros(Int, rows)
for r in I
    row_counts[r] += 1
end
fig_hist = Figure(resolution=(700, 400))
ax_hist = Axis(fig_hist[1, 1], xlabel="row index", ylabel="nnz per row", title="Row-wise nnz in A")
lines!(ax_hist, 1:rows, row_counts)
display(fig_hist)
save("stokes1d_A_row_nnz.png", fig_hist)
