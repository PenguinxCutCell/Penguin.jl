using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CairoMakie

### 1D Test Case: Concentration Diffusion with Moving Interface
# Movement of interface between two phases due to concentration gradients

# Define the spatial mesh
nx = 160
lx = 1.0
x0 = 0.0
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the interface location
xint = 0.05*lx  # Interface location
body = (x, t, _=0) -> (x - xint)      # For phase 1
body_c = (x, t, _=0) -> -(x - xint)   # For phase 2

# Define the Space-Time mesh
Δt = 0.001
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity for both phases
capacity1 = Capacity(body, STmesh)
capacity2 = Capacity(body_c, STmesh)

# Define the diffusion operators
operator1 = DiffusionOps(capacity1)
operator2 = DiffusionOps(capacity2)

# Define the boundary conditions
Ccold = 0.0
Chot = 1.0
Cinterface = 0.5  # Concentration at the interface
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(Ccold), :bottom => Dirichlet(Chot)))

# Define the diffusion coefficients
D1 = (x,y,z) -> 1.0
D2 = (x,y,z) -> 1.0

# Define the source term
f = (x,y,z,t) -> 0.0

# Define the phases
phase1 = Phase(capacity1, operator1, f, D1)
phase2 = Phase(capacity2, operator2, f, D2)

# Define the interface condition
# flux_factor controls interface movement (similar to latent heat in Stefan problem)
flux_factor = 1.0
ic = InterfaceConditions(ScalarJump(1.0, 1.0, Cinterface), FluxJump(D1(0,0,0), D2(0,0,0), flux_factor))

# Initial conditions
# Phase 1
C1_bulk = ones(nx+1) * Chot
C1_interface = ones(nx+1) * Cinterface
# Phase 2
C2_bulk = zeros(nx+1)
C2_interface = ones(nx+1) * Cinterface

# Combined initial conditions
u0 = vcat(C1_bulk, C1_interface, C2_bulk, C2_interface)

# Newton parameters
max_iter = 1000
tol = 1e-8
reltol = 1e-8
α = 0.8
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = DiffusionUnsteadyConcentration(phase1, phase2, bc_b, ic, Δt, u0, "CN", Newton_params)

# Solve the problem
solver, residuals, xf_log = solve_DiffusionUnsteadyConcentration!(solver, phase1, phase2, xint, Δt, Tend, bc_b, ic, mesh; Newton_params=Newton_params, method=Base.:\)

# Visualize the results
animate_concentration_profile(solver, xf_log, mesh; filename="concentration_profile.mp4", fps=10)