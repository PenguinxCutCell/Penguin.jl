using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays

### 3D Test Case : One-phase Stefan Problem : Growing Planar Interface
# This example extends the 2D stefan.jl to 3D, simulating a growing planar
# interface using the height function method.

# Define the spatial mesh
nx, ny, nz = 16, 16, 16  # Lower resolution for 3D to manage computational cost
lx, ly, lz = 1., 1., 1.
x0, y0, z0 = 0., 0., 0.
domain = ((x0, lx), (y0, ly), (z0, lz))
mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))

Δx, Δy, Δz = lx/nx, ly/ny, lz/nz

# Define the body - planar interface with small perturbation
# Interface position as a function of y and z
sₙ(y, z) = 0.2*lx + 0.01*cos(2π*y) * cos(2π*z)
body = (x, y, z, t) -> (x - sₙ(y, z))

# Define the Space-Time mesh
Δt = 0.001
Tend = 0.01
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)

# Extract initial Height Vₙ₊₁ and Vₙ from capacity
# In 3D, capacity.A[4] contains the volume data (index 4 for 4D spacetime)
dims = (nx+1, ny+1, nz+1, 2)  # spatial dims + time stencil

Vₙ₊₁_block = capacity.A[4][1:end÷2, 1:end÷2]
Vₙ_block = capacity.A[4][end÷2+1:end, end÷2+1:end]
Vₙ = diag(Vₙ_block)
Vₙ₊₁ = diag(Vₙ₊₁_block)

# Reshape to 3D spatial arrays
Vₙ = reshape(Vₙ, (nx+1, ny+1, nz+1))
Vₙ₊₁ = reshape(Vₙ₊₁, (nx+1, ny+1, nz+1))

# Compute height profiles by summing along x-direction (dim 1)
# Result is (ny+1, nz+1) matrix
Hₙ⁰ = dropdims(sum(Vₙ, dims=1), dims=1)
Hₙ₊₁⁰ = dropdims(sum(Vₙ₊₁, dims=1), dims=1)

# Compute interface positions from heights
# For 3D: xf(y,z) = x0 + H(y,z) / (Δy * Δz)
Interface_position = x0 .+ Hₙ⁰ ./ (Δy * Δz)
println("Interface position range: min=$(minimum(Interface_position)), max=$(maximum(Interface_position))")

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(1.0)    # Interface temperature
bc1 = Dirichlet(0.0)   # Bottom boundary

# Border conditions - set all boundaries except interface
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :bottom => bc1,
    :top => Neumann(0.0),
    :left => Neumann(0.0),
    :right => Neumann(0.0),
    :front => Neumann(0.0),
    :back => Neumann(0.0)
))

# Physical parameters
ρ, L = 1.0, 1.0  # density and latent heat
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))

# Define the source term and diffusion coefficient
f = (x, y, z, t) -> 0.0
K = (x, y, z) -> 1.0

Fluide = Phase(capacity, operator, f, K)

# Initial condition - zero temperature everywhere
n_total = (nx+1)*(ny+1)*(nz+1)
u0ₒ = zeros(n_total)
u0ᵧ = zeros(n_total)
u0 = vcat(u0ₒ, u0ᵧ)

# Newton parameters
max_iter = 10000
tol = 1e-6
reltol = 1e-6
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
println("\nStarting 3D Stefan problem solver...")
solver, residuals, xf_log, reconstruct, timestep_history = solve_MovingLiquidDiffusionUnsteadyMono3D!(
    solver, Fluide, Interface_position, Hₙ⁰, sₙ, Δt, Tend, bc_b, bc, stef_cond, mesh, "BE"; 
    interpo="linear",  # Use linear interpolation for simplicity in 3D
    Newton_params=Newton_params, 
    adaptive_timestep=false, 
    Δt_min=5e-4, 
    method=Base.:\
)

println("\n=== Simulation completed ===")
println("Number of time steps: $(length(solver.states))")
println("Number of Newton iterations logged: $(length(keys(residuals)))")
println("Interface position history entries: $(length(xf_log))")

# Print final interface statistics if available
if !isempty(xf_log)
    final_xf = xf_log[end]
    if isa(final_xf, AbstractMatrix)
        println("Final interface position range: min=$(minimum(final_xf)), max=$(maximum(final_xf))")
    else
        println("Final interface position: $(final_xf)")
    end
end

# Print some diagnostics about the solution
if !isempty(solver.states)
    final_state = solver.states[end]
    temp_bulk = final_state[1:n_total]
    println("Final bulk temperature: min=$(minimum(temp_bulk)), max=$(maximum(temp_bulk))")
end
