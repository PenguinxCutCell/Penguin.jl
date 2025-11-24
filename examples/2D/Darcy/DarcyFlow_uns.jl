using Penguin
using IterativeSolvers

### 2D Test Case : Unsteady Darcy Flow with a disk
# Define the mesh
nx, ny = 80, 80
lx, ly = 4., 4.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/4, (lx/2, ly/2) .+ (0.01, 0.01)
circle = (x,y,_=0)->-(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# Define the capacity
capacity = Capacity(circle, mesh)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions for pressure for left and right faces. Others faces don't have BC
bc_10 = Dirichlet(10.0)
bc_20 = Dirichlet(20.0)

bc_p = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc_10, :bottom => bc_20))

ic = Neumann(0.0)

# Define the source term
f = (x,y,z,t)-> 0.0

# Define the phase
K = (x,y,z)-> 1.0
Fluide = Phase(capacity, operator, f, K)

# Initial condition
u0ₒ = zeros((nx+1)*(ny+1))
u0ᵧ = zeros((nx+1)*(ny+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
Δt = 0.01
Tend = 1.0
solver = DarcyFlowUnsteady(Fluide, bc_p, ic, Δt, u0, "CN")

# Solve the problem
solve_DarcyFlowUnsteady!(solver, Fluide, Δt, Tend, bc_p, ic, "CN"; method=Base.:\)

# Plot the pressure solution
#animate_solution(solver, mesh, circle)

# Solve the velocity problem
u = solve_darcy_velocity(solver, Fluide; state_i=100)

# Plot the velocity solution
xrange, yrange = range(x0, stop=lx, length=nx+1), range(y0, stop=ly, length=ny+1)
ux, uy = u[1:div(end,2)], u[div(end,2)+1:end]
ux, uy= reshape(ux, (nx+1, ny+1)), reshape(uy, (nx+1, ny+1))
mag = sqrt.(ux.^2 + uy.^2)


function plot_darcy_velocity(ux, uy, mag)
    fig = Figure(size=(800, 800))
    ax1 = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y")
    hm1 = heatmap!(ax1, ux, colormap=:viridis)
    Colorbar(fig[1, 2], label="u_x", hm1)

    ax2 = Axis(fig[2, 1], aspect=DataAspect(), xlabel="x", ylabel="y")
    hm2 = heatmap!(ax2, uy, colormap=:viridis)
    Colorbar(fig[2, 2], label="u_y", hm2)

    ax3 = Axis(fig[3, 1], aspect=DataAspect(), xlabel="x", ylabel="y")
    hm3 = heatmap!(ax3, mag, colormap=:viridis)
    Colorbar(fig[3, 2], label="u", hm3)

    display(fig)
end


using CairoMakie

plot_darcy_velocity(ux, uy, mag)