using Penguin
using IterativeSolvers

### 1D Test Case : Diphasic Steady Diffusion Equation
# Define the mesh
nx = 40
lx = 1.0
x0 = 0.
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
pos = 0.5
body = (x, _=0) -> x - pos
body_c = (x, _=0) -> pos - x

# Define the capacity
capacity = Capacity(body, mesh)
capacity_c = Capacity(body_c, mesh)

# Define the operators
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc = Dirichlet(1.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc, :bottom => bc1))

ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Fedkiw test case : 1) ScalarJump(1.0, 1.0, -1.0), FluxJump(1.0, 1.0, 0.0), f=0, u(0)=0, u(1)=2
#                     2) ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, -1.0), f=0, u(0)=0, u(1)=2

# Define the source term
f1 = (x, y, _=0) -> 0.0
f2 = (x, y, _=0) -> 0.0

D1 = (x, y, _=0) -> 1.0
D2 = (x, y, _=0) -> 1.0

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, D1)
Fluide_2 = Phase(capacity_c, operator_c, f2, D2)

# Define the solver 
solver = DiffusionSteadyDiph(Fluide_1, Fluide_2, bc_b, ic)

# Solve the problem
solve_DiffusionSteadyDiph!(solver; method=Base.:\)

# Plot the solution
plot_solution(solver, mesh, body, capacity)

using LinearAlgebra

"""
    two_phase_solution(k1, k2, f1, f2, α1, α2, g, h, xI, x0, xL, T0, TL)

Compute the piecewise solution T1(x) and T2(x) for the 1D two-phase Poisson equation:
    d/dx(k_i dT_i/dx) = f_i, for i=1,2,
with interface jumps [α_i T_i] = g and [k_i dT_i/dx] = h at x = xI.
Domain: [x0, xI] for T1 and [xI, xL] for T2, with boundary conditions T1(x0) = T0 and T2(xL) = TL.

Returns a tuple of functions (T1, T2) that evaluate the temperature in each region.
"""
function two_phase_solution(k1, k2, f1, f2, α1, α2, g, h, xI, x0, xL, T0, TL)
    # Assemble the 4x4 linear system M * [A1, B1, A2, B2] = RHS
    M = [
        x0   1    0     0;
        0    0    xL    1;
        α1*xI α1  -α2*xI -α2;
        k1   0   -k2    0
    ]

    RHS = [
        T0 - f1/(2*k1)*x0^2;
        TL - f2/(2*k2)*xL^2;
        g - ((α1*f1)/(2*k1) - (α2*f2)/(2*k2))*xI^2;
        h - (f1 - f2)*xI
    ]

    coeffs = M \ RHS
    A1, B1, A2, B2 = coeffs

    # Define the piecewise temperature functions
    T1 = x -> f1/(2*k1)*x^2 + A1*x + B1
    T2 = x -> f2/(2*k2)*x^2 + A2*x + B2

    return T1, T2
end

# Example usage:
D1, D2 = 1.0, 1.0
f1, f2 = 0.0, 0.0
α1, α2 = 1.0, 1.0
g, h = 0.0, 0.0
xI, x0, xL = pos, x0, lx
T0, TL = 0.0, 1.0
T1, T2 = two_phase_solution(D1, D2, f1, f2, α1, α2, g, h, xI, x0, xL, T0, TL)
x = mesh.nodes[1]

u1 = T1.(x)
u2 = T2.(x)

u1[capacity.cell_types .== 0] .= NaN
u2[capacity_c.cell_types .== 0] .= NaN

u1_num = solver.x[1:nx+1]
u2_num = solver.x[2*(nx+1)+1:3*(nx+1)]

push!(solver.states, solver.x)

u1_num[capacity.cell_types .== 0] .= NaN
u2_num[capacity_c.cell_types .== 0] .= NaN

fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="u", title="Diphasic Unsteady Diffusion Equation")
scatter!(ax, x, u1, color=:blue, label="Bulk Field - Phase 1")
scatter!(ax, x, u1_num, color=:red, label="Bulk Field - Phase 1 - Numerical")
scatter!(ax, x, u2, color=:green, label="Bulk Field - Phase 2")
scatter!(ax, x, u2_num, color=:orange, label="Bulk Field - Phase 2 - Numerical")
axislegend(ax, position=:rb)
display(fig)

(ana_sols, num_sols, global_errs, full_errs, cut_errs, empty_errs) = check_convergence_diph(T1, T2, solver, capacity, capacity_c, 2, false)

