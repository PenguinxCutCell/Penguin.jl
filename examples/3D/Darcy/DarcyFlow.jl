using Penguin
using IterativeSolvers

### 3D Test Case : Darcy Flow with a Sphere
# Define the mesh
nx, ny, nz = 40, 40, 40
lx, ly, lz = 4., 4., 4.
x0, y0, z0 = 0., 0., 0.
domain = ((x0, lx), (y0, ly), (z0, lz))
mesh = Penguin.Mesh((nx, ny, nz), (lx, ly, lz), (x0, y0, z0))

# Define the body
radius, center = ly/4, (lx/2, ly/2, lz/2) #.+ (0.01, 0.01, 0.01)
sphere = (x,y,z)->-(sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - radius)

# Define the capacity
capacity = Capacity(sphere, mesh)

# Define the operators
operator_p = DiffusionOps(capacity)

# Define the boundary conditions for pressure for left and right faces. Others faces don't have BC
bc_10 = Dirichlet(10.0)
bc_20 = Dirichlet(20.0)

bc_p = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc_10, :bottom => bc_20))

ic = Dirichlet(1.0)

# Define the source term
f = (x,y,z)-> 0.0

# Define the phase
K = (x,y,z)-> 1.0
Fluide = Phase(capacity, operator_p, f, K)

# Define the solver
solver = DarcyFlow(Fluide, bc_p, ic)

# Solve the problem
solve_DarcyFlow!(solver; method=IterativeSolvers.gmres)

# Write the solution to a VTK file
write_vtk("darcy_3d", mesh, solver)

plot_solution(solver, mesh, sphere, capacity)

function ∇(operator::AbstractOperators, p::Vector{Float64})
    ∇ = operator.Wꜝ * (operator.G * p[1:div(end,2)] + operator.H * p[div(end,2)+1:end])

    return ∇
end

# Compute the velocity field from the pressure field
u = ∇(operator_p, solver.x)

ux = u[1:div(end,3)]
uy = u[div(end,3)+1:2*div(end,3)]
uz = u[2*div(end,3)+1:end]

ux = reshape(ux, (nx+1, ny+1, nz+1))
uy = reshape(uy, (nx+1, ny+1, nz+1))
uz = reshape(uz, (nx+1, ny+1, nz+1))


using CairoMakie

"""
    plot_middle_velocities(ux, uy, uz, mesh)

Plot middle-plane slices for 3D velocity components (ux, uy, uz) in a single figure.
Assumes `ux`, `uy`, `uz` are sized (nx+1, ny+1, nz+1).
Slices at y-mid index => shape (nx+1, nz+1).
"""
function plot_middle_velocities(ux, uy, uz, mesh, nx, ny, nz)
    #x_range = LinRange(mesh.x0[1], mesh.x0[1] + mesh.h[1][1]*nx, nx+1)
    #z_range = LinRange(mesh.x0[3], mesh.x0[3] + mesh.h[3][1]*nz, nz+1)

    # Middle slice along y
    i_mid = div(ny+1, 2)

    # Extract slices (nx+1, nz+1). For Makie's heatmap, transpose them if needed
    ux_middle = ux[:, i_mid, :]
    uy_middle = uy[:, i_mid, :]
    uz_middle = uz[:, i_mid, :]

    # Build figure with three subplots
    fig = Figure(size=(1500, 400))

    # 1) ux slice
    ax1 = Axis(fig[1, 1], title="ux @ y-mid")
    hm1 = heatmap!(ax1, ux_middle', colormap=:viridis)
    Colorbar(fig[1, 2], hm1)

    # 2) uy slice
    ax2 = Axis(fig[1, 3], title="uy @ y-mid")
    hm2 = heatmap!(ax2, uy_middle', colormap=:viridis)
    Colorbar(fig[1, 4], hm2)

    # 3) uz slice
    ax3 = Axis(fig[1, 5], title="uz @ y-mid")
    hm3 = heatmap!(ax3, uz_middle', colormap=:viridis)
    Colorbar(fig[1, 6], hm3)

    display(fig)
end

plot_middle_velocities(ux, uy, uz, mesh, nx, ny, nz)

function plot_middle_velocity_magnitude(ux, uy, uz, mesh, nx, ny, nz)
    velocity = sqrt.(ux.^2 .+ uy.^2 .+ uz.^2)
    i_mid = div(ny+1, 2)
    vel_mid = velocity[:, i_mid, :]

    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1], title="Velocity magnitude @ y-mid", aspect=DataAspect())
    hm = heatmap!(ax, vel_mid', colormap=:viridis)
    Colorbar(fig[1, 2], hm)
    display(fig)
end

plot_middle_velocity_magnitude(ux, uy, uz, mesh, nx, ny, nz)
