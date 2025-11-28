using Penguin
using IterativeSolvers
using VTKOutputs

### 2D Test Case : Monophasic Unsteady Diffusion Equation inside a Disk
# Define the mesh
nx, ny = 80, 80
lx, ly = 4., 4.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/4, (lx/2, ly/2) .+ (0.01, 0.01)
circle = (x,y,_=0)->(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# Define the capacity
capacity = Capacity(circle, mesh)

# Define the operators
operator = DiffusionOps(capacity)

cell_types = capacity.cell_types

# Define the boundary conditions 
bc = Dirichlet((x,y,z,t)->sin(π*x) * sin(π*y))
bc0 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc0, :right => bc0, :top => bc0, :bottom => bc0))

# Define the source term
f = (x,y,z,t)->0.0
D = (x,y,z)->1.0

# Define the phase
Fluide = Phase(capacity, operator, f, D)

# Initial condition
u0ₒ = zeros((nx+1)*(ny+1))
u0ᵧ = ones((nx+1)*(ny+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
Δt = 0.25 * (lx/nx)^2
Tend = 0.01
solver = DiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, "BE") # Start by a backward Euler scheme to prevent oscillation due to CN scheme

# Solve the problem
solve_DiffusionUnsteadyMono!(solver, Fluide, Δt, Tend, bc_b, bc, "BE"; method=Base.:\)

# Write the solution to a VTK file
write_vtk("heat", mesh, solver)

# Plot the solution
#plot_solution(solver, mesh, circle, capacity; state_i=1)

# Animation
animate_solution(solver, mesh, circle)

# Analytical solution
using SpecialFunctions
using Roots

function radial_heat_xy(x, y)
    t=Tend
    R=1.0

    function j0_zeros(N; guess_shift=0.25)
        zs = zeros(Float64, N)
        # The m-th zero of J₀ is *roughly* near (m - guess_shift)*π for large m.
        # We'll bracket around that approximate location and refine via find_zero.
        for m in 1:N
            # approximate location
            x_left  = (m - guess_shift - 0.5)*pi
            x_right = (m - guess_shift + 0.5)*pi
            # ensure left>0
            x_left = max(x_left, 1e-6)
            
            # We'll use bisection or Brent's method from Roots.jl
            αm = find_zero(besselj0, (x_left, x_right))
            zs[m] = αm
        end
        return zs
    end

    alphas = j0_zeros(1000)
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
        s += exp(-αm^2 * t) * besselj0(αm * (r / R)) / (αm * besselj1(αm))
    end
    return 1.0 - 2.0*s
end

u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(radial_heat_xy, solver, capacity, 2)

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
Colorbar(fig[1, 4], hm2, label="u(x)")
display(fig)
readline()

# Plot error heatmap
err = reshape(err, (nx+1, ny+1))
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel="y", title="Log error")
hm = heatmap!(ax, log10.(abs.(err)), colormap=:viridis)
Colorbar(fig[1, 2], hm, label="log10(|u(x) - u_num(x)|)")
display(fig)
