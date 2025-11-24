using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays

### 2D Test Case : Darcy Flow with a disk
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
f = (x,y,z)-> 0.0

# Define the phase
K = (x,y,z)-> 1.0
Fluide = Phase(capacity, operator, f, K)

# Define the solver
solver = DarcyFlow(Fluide, bc_p, ic)

# Solve the pressure problem
solve_DarcyFlow!(solver; method=Base.:\)

# Plot the pressure solution
#plot_solution(solver, mesh, circle, capacity)

# Solve the velocity problem
u = solve_darcy_velocity(solver, Fluide)

# Compute drag and lift forces
function compute_drag_lift_forces(
    p_gamma::Vector{Float64},      # Pressure at each boundary point
    normals::Vector{NTuple{2, Float64}},  # Boundary normal vectors
    ds::Vector{Float64}            # Boundary segment lengths
)
    drag = 0.0
    lift = 0.0
    for i in 1:length(ds)
        drag += p_gamma[i] * normals[i][1] * ds[i]
        lift += p_gamma[i] * normals[i][2] * ds[i]
    end
    return drag, lift
end

function compute_drag_lift_coeffs(
    drag::Float64, lift::Float64,
    density::Float64, uref::Float64, length_ref::Float64
)
    Cd = 2 * drag / (density * uref^2 * length_ref)
    Cl = 2 * lift / (density * uref^2 * length_ref)
    return Cd, Cl
end

"""
function compute_drag_lift_forces(solver, capacity)
    pᵧ = solver.x[div(end,2)+1:end]
    Γ = capacity.Γ

    # Extract the diagonal of Γ and reshape it
    Γ = reshape(diag(Γ), (nx+1, ny+1))  # Length of the interface in each cell
    pᵧ = reshape(pᵧ, (nx+1, ny+1))  # Pressure on the interface in each cell

    # Compute drag and lift forces
    Fᵧ = sum(Γ .* pᵧ)

    return Fᵧ
end
"""

# Compute drag and lift forces
pᵧ = solver.x[div(end,2)+1:end]
ds = diag(capacity.Γ)



# Plot the velocity solution
xrange, yrange = range(x0, stop=lx, length=nx+1), range(y0, stop=ly, length=ny+1)
ux, uy = u[1:div(end,2)], u[div(end,2)+1:end]
ux, uy= reshape(ux, (nx+1, ny+1)), reshape(uy, (nx+1, ny+1))
mag = sqrt.(ux.^2 + uy.^2)

function plot_darcy_arrow(ux, uy, mag, xrange, yrange; arrowsize=10, lengthscale=0.02)
    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], backgroundcolor=:white, xlabel="x", ylabel="y")
    ar = arrows!(ax, xrange, yrange, ux, uy, arrowsize=arrowsize, arrowcolor=mag, linecolor=mag, lengthscale=lengthscale)
    Colorbar(fig[1, 2], label="u", ar)
    display(fig)
end

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

plot_darcy_arrow(ux, uy, vec(mag), xrange, yrange)
#plot_darcy_velocity(ux, uy, mag)
