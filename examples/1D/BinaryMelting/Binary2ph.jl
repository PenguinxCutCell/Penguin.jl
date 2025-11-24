using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CairoMakie

### 1D Test Case: Binary Melting with Temperature and Concentration Fields (Two-Phase)
# Melting of a solid ice phase (x>h(t)) by a liquid phase (x<h(t)) with temperature and concentration fields

# Define the spatial mesh
nx = 160
lx = 1.0
x0 = 0.0
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the interface location
xint = 0.05*lx  # Interface location
body = (x, t, _=0) -> (x - xint)      # For liquid phase
body_c = (x, t, _=0) -> -(x - xint)   # For solid phase

# Define the Space-Time mesh
Δt = 0.001
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity for both phases
capacity_liquid = Capacity(body, STmesh)
capacity_solid = Capacity(body_c, STmesh)

# Define the diffusion operators
operator_liquid = DiffusionOps(capacity_liquid)
operator_solid = DiffusionOps(capacity_solid)

# Define the boundary conditions
# Temperature BCs
Tcold = 0.0
Thot = 1.0
Tinterface = 0.0  # Temperature at the interface
bc_b_T = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(Tcold), :bottom => Dirichlet(Thot)))

# Concentration BCs
Ccold = 0.0
Chot = 1.0
Cinterface = 0.5  # Concentration at the interface
bc_b_C = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(Ccold), :bottom => Dirichlet(Chot)))

# Define the physical parameters
ρ = 1.0         # Density
L = 1.0         # Latent heat
cp = 1.0        # Heat capacity
k = 1.0         # Partition coefficient
λ = 1.0         # 
m = λ*1.0/1.0   # Liquidus slope

# Define the diffusion coefficients
# Thermal diffusivity for both phases
D_T_liquid = (x,y,z) -> 1.0
D_T_solid = (x,y,z) -> 1.0
# Mass diffusivity for both phases
D_C_liquid = (x,y,z) -> 1.0
D_C_solid = (x,y,z) -> 1.0

# Define the source terms
f_T = (x,y,z,t) -> 0.0
f_C = (x,y,z,t) -> 0.0

# Define the phases
# Temperature phases
T_liquid = Phase(capacity_liquid, operator_liquid, f_T, D_T_liquid)
T_solid = Phase(capacity_solid, operator_solid, f_T, D_T_solid)

# Concentration phases
C_liquid = Phase(capacity_liquid, operator_liquid, f_C, D_C_liquid)
C_solid = Phase(capacity_solid, operator_solid, f_C, D_C_solid)

# Define the interface conditions
# For temperature: temperature jump and energy (Stefan) condition
ic_T = InterfaceConditions(ScalarJump(1.0, 1.0, Tinterface), FluxJump(D_T_liquid(0,0,0), D_T_solid(0,0,0), ρ*L))

# For concentration: concentration jump and mass conservation
ic_C = InterfaceConditions(ScalarJump(1.0, 1.0, Cinterface), FluxJump(D_C_liquid(0,0,0), D_C_solid(0,0,0), 0.0))

# Initial conditions
# Phase 1 (liquid) - Temperature
u0_T_liquid_ₒ = ones(nx+1) * Thot
u0_T_liquid_ᵧ = ones(nx+1) * Tinterface
# Phase 1 (liquid) - Concentration 
u0_C_liquid_ₒ = ones(nx+1) * Chot
u0_C_liquid_ᵧ = ones(nx+1) * Cinterface

# Phase 2 (solid) - Temperature
u0_T_solid_ₒ = zeros(nx+1)
u0_T_solid_ᵧ = ones(nx+1) * Tinterface
# Phase 2 (solid) - Concentration
u0_C_solid_ₒ = ones(nx+1) * Ccold
u0_C_solid_ᵧ = ones(nx+1) * Cinterface

# Create the initial condition vectors
u0_T = vcat(u0_T_liquid_ₒ, u0_T_liquid_ᵧ, u0_T_solid_ₒ, u0_T_solid_ᵧ)
u0_C = vcat(u0_C_liquid_ₒ, u0_C_liquid_ᵧ, u0_C_solid_ₒ, u0_C_solid_ᵧ)

# Newton parameters
max_iter = 1000
tol = 1e-8
reltol = 1e-8
α = 0.8
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = DiffusionUnsteadyBinary(T_liquid, T_solid, C_liquid, C_solid, bc_b_T, bc_b_C, ic_T, ic_C, Δt, u0_T, u0_C, "CN", Newton_params)

# Solve the problem
solver, residuals, xf_log = solve_DiffusionUnsteadyBinary!(solver, T_liquid, T_solid, C_liquid, C_solid, xint,Δt, Tend, bc_b_T, bc_b_C, ic_T, ic_C, mesh; Newton_params=Newton_params, method=Base.:\)