using Penguin
using IterativeSolvers

### 2D Test Case : Monophasic Unsteady Diffusion Equation inside a Disk
# Define the mesh
nx, ny = 80, 80
lx, ly = 4.0, 4.0
x0, y0 = 0.0, 0.0
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/4, (lx/2, ly/2)
circle = (x,y,_=0)->(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# Define the capacity
capacity = Capacity(circle, mesh)

# Define the operators
operator = DiffusionOps(capacity)

cell_types = capacity.cell_types

# Define the boundary conditions 
bc = Robin(3.0,1.0,3.0*400)
bc0 = Dirichlet(400.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc0, :right => bc0, :top => bc0, :bottom => bc0))

# Define the source term
f = (x,y,z,t)->0.0

# Define the phase
a = (x,y,z)->1.0
Fluide = Phase(capacity, operator, f, a)

# Initial condition
u0ₒ = ones((nx+1)*(ny+1)) * 270.0
u0ᵧ = zeros((nx+1)*(ny+1)) * 400.0
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
Δt = 0.25*(lx/nx)^2
Tend = 0.1     
solver = DiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, "BE") # Start by a backward Euler scheme to prevent oscillation due to CN scheme

# Solve the problem
solve_DiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, bc, "BE"; method=Base.:\)

# Write the solution to a VTK file
#write_vtk("heat", mesh, solver)

# Plot the solution
plot_solution(solver, mesh, circle, capacity; state_i=10)

"""
x=range(x0, stop=lx, length=nx+1)
y=range(y0, stop=ly, length=ny+1)
z=reshape(solver.states[end][1:div(end,2)], (nx+1, ny+1))

# Plot the isocontours with CairoMakie
using CairoMakie

fig=Figure(size=(800, 800))
ax = Axis(fig[1, 1], xlabel = "x", ylabel="y", title="Temperature field")
contour!(ax, x, y, z, levels=range(270, stop=400, length=100), colormap=:viridis)
display(fig)
"""

# Animation
animate_solution(solver, mesh, circle)


# Analytical solution
using SpecialFunctions
using Roots

function radial_heat_(x, y)
    t=0.1
    R=1.0
    k=3.0
    a=1.0

    function j0_zeros_robin(N, k, R; guess_shift = 0.25)
        # Define the function for alpha J1(alpha) - k R J0(alpha) = 0
        eq(alpha) = alpha * besselj1(alpha) - k * R * besselj0(alpha)
    
        zs = zeros(Float64, N)
        for m in 1:N
            # Approximate location around (m - guess_shift)*π
            x_left  = (m - guess_shift - 0.5) * π
            x_right = (m - guess_shift + 0.5) * π
            x_left  = max(x_left, 1e-6)  # Ensure bracket is positive
            zs[m]   = find_zero(eq, (x_left, x_right))
        end
        return zs
    end

    alphas = j0_zeros_robin(1000, k, R)
    N=length(alphas)
    r = sqrt((x - center[1])^2 + (y - center[2])^2)
    if r >= R
        # Not physically in the domain, so return NaN or handle as you wish.
        return NaN
    end
    
    # If in the disk: sum the series
    s = 0.0
    for m in 1:N
        αm = alphas[m]
        An = 2.0 * k * R / ((k^2 * R^2 + αm^2) * besselj0(αm))
        s += An * exp(- a * αm^2 * t/R^2) * besselj0(αm * (r / R))
    end
    return (1.0 - s) * (400 - 270) + 270
end


# Check the error convergence

u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(radial_heat_, solver, capacity, 2)

u_ana[capacity.cell_types .== 0] .= NaN
u_ana = reshape(u_ana, (nx+1, ny+1))

u_num[capacity.cell_types .== 0] .= NaN
u_num = reshape(u_num, (nx+1, ny+1))

err = u_ana - u_num

using CairoMakie
fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel="y", title="Analytical solution")
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel="y", title="Numerical solution")
hm1 = heatmap!(ax1, u_ana, colormap=:viridis)
hm2 = heatmap!(ax2, u_num, colormap=:viridis)
Colorbar(fig[1, 3], hm1, label="u(x)")
Colorbar(fig[1, 4], hm2, label="u_num(x)")
display(fig)
readline()

# Plot error heatmap
err = reshape(err, (nx+1, ny+1))
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel="y", title="Log error")
hm = heatmap!(ax, log10.(abs.(err)), colormap=:viridis)
Colorbar(fig[1, 2], hm, label="log10(|u(x) - u_num(x)|)")
display(fig)
