using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays

### 1D Test Case : Diphasic Unsteady Diffusion Equation inside a moving body
# Define the spatial mesh
nx = 160
lx = 8.0
x0 = 0.0
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
xf = 4.0   # Interface position
c = 0.0        # Interface velocity
body = (x,t, _=0)->(x - xf)
body_c = (x,t, _=0)->-(x - xf)

# Define the Space-Time mesh
Δt = 0.01
Tend = 0.5
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)
capacity_c = Capacity(body_c, STmesh)

# Define the diffusion operator
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc1 = Dirichlet(0.0)
bc0 = Dirichlet(1.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc0, :bottom => bc1))

ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Define the source term
f1 = (x,y,z,t)->0.0
f2 = (x,y,z,t)->0.0

K1= (x,y,z)->1.0
K2= (x,y,z)->1.0

# Define the phase
Fluide1 = Phase(capacity, operator, f1, K1)
Fluide2 = Phase(capacity_c, operator_c, f2, K2)

# Initial condition
u0ₒ1 = zeros(nx+1)
u0ᵧ1 = zeros(nx+1)
u0ₒ2 = ones(nx+1)
u0ᵧ2 = ones(nx+1)

u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Define the solver
solver = MovingDiffusionUnsteadyDiph(Fluide1, Fluide2, bc_b, ic, Δt, u0, mesh, "BE")

# Solve the problem
solve_MovingDiffusionUnsteadyDiph!(solver, Fluide1, Fluide2, body, body_c, Δt, Tend, bc_b, ic, mesh, "BE"; method=Base.:\)

# Plot the solution
#plot_solution(solver, mesh, body, capacity; state_i=10)

# Animation
animate_solution(solver, mesh, body)

state_i = 10
u1ₒ = solver.states[state_i][1:nx+1]
u1ᵧ = solver.states[state_i][nx+2:2*(nx+1)]
u2ₒ = solver.states[state_i][2*(nx+1)+1:3*(nx+1)]
u2ᵧ = solver.states[state_i][3*(nx+1)+1:end]

x = range(x0, stop = lx, length = nx+1)
using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="u", title="Diphasic Unsteady Diffusion Equation")

for (idx, i) in enumerate(1:10:length(solver.states))
    u1ₒ = solver.states[i][1:nx+1]
    u1ᵧ = solver.states[i][nx+2:2*(nx+1)]
    u2ₒ = solver.states[i][2*(nx+1)+1:3*(nx+1)]
    u2ᵧ = solver.states[i][3*(nx+1)+1:end]

    # Display labels only for first iteration
    scatter!(ax, x, u1ₒ, color=:blue,  markersize=3,
        label = (idx == 1 ? "Bulk Field - Phase 1" : nothing))
    scatter!(ax, x, u1ᵧ, color=:red,   markersize=3,
        label = (idx == 1 ? "Interface Field - Phase 1" : nothing))
    scatter!(ax, x, u2ₒ, color=:green, markersize=3,
        label = (idx == 1 ? "Bulk Field - Phase 2" : nothing))
    scatter!(ax, x, u2ᵧ, color=:orange, markersize=3,
        label = (idx == 1 ? "Interface Field - Phase 2" : nothing))
end

axislegend(ax, position=:rb)
display(fig)