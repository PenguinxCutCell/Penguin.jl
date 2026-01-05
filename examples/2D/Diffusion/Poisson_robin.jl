using Penguin
using IterativeSolvers
using SparseArrays
using LinearAlgebra
using CairoMakie

### 2D Test Case : Monophasic Steady Diffusion Equation inside a Disk
# Define the mesh
nx, ny = 512, 512
lx, ly = 4., 4.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/4, (lx/2, ly/2) .+ (0.01, 0.01)
circle = (x, y, _=0) -> (sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# Define the capacity
capacity = Capacity(circle, mesh)

# Count cells whose volume is less than 1e-3 of a full cell
cell_volumes = capacity.V[diagind(capacity.V)]
full_cell_vol = (lx/nx) * (ly/ny)
tiny_mask = cell_volumes .< (1e-3 * full_cell_vol)
println("Cells with relative volume < 1e-3: $(count(tiny_mask))")
println("Total volume of these cells: $(sum(cell_volumes[tiny_mask]))")

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions 
bc = Robin(1.0,1.0,1.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))

# Define the source term
f = (x,y,_)-> 1.0 #sin(x)*cos(10*y)
K = (x,y,_)-> 1.0

Fluide = Phase(capacity, operator, f, K)

# Define the solver
solver = DiffusionSteadyMono(Fluide, bc_b, bc)

# Solve the problem
solve_DiffusionSteadyMono!(solver; method=Base.:\)

# Plot the solution
plot_solution(solver, mesh, circle, capacity)

# Write the solution to a VTK file
#write_vtk("poisson_2d", mesh, solver)

# Analytical solution
u_analytical(x,y) = 7/4 - ((x-center[1])^2 + (y-center[2])^2)/4
∇x_analytical(x,y) = - (x-center[1])/2
∇y_analytical(x,y) = - (y-center[2])/2

u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence(u_analytical, solver, capacity, 2, false)

u_ana[capacity.cell_types .== 0] .= NaN
u_ana = reshape(u_ana, (nx+1, ny+1))

u_num[capacity.cell_types .== 0] .= NaN
u_num = reshape(u_num, (nx+1, ny+1))

err = u_ana .- u_num

using CairoMakie
x,y = mesh.nodes[1], mesh.nodes[2]
fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel="y", title="Numerical solution")
hm =heatmap!(ax1, x, y, u_num, colormap=:viridis)
Colorbar(fig[1, 2], hm, label="Value")
display(fig)
save("numerical_solution.png", fig)

# Plot error heatmap
err = reshape(err, (nx+1, ny+1))
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel="y", title="Log error")
x,y = mesh.nodes[1], mesh.nodes[2]
xlims!(ax, 2, 3.25)
ylims!(ax, 2, 3.25)
hm = heatmap!(ax, x,y,log10.(abs.(err)), colormap=:viridis)
Colorbar(fig[1, 2], hm, label="Log10 Error")
display(fig)
save("error_heatmap.png", fig)

# Plot interface value 
ϕγ = solver.x[end÷2+1:end]
ϕγ = reshape(ϕγ, (nx+1, ny+1))
ϕγ[capacity.cell_types .!= -1] .= NaN
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "x", ylabel="y", title="Value along the interface")
hm = heatmap!(ax, x,y, ϕγ, colormap=:viridis)
Colorbar(fig[1, 2], hm, label="u(interface)")
display(fig)
save("interface_values.png", fig)

# Plot interface value against polar angle (degrees)
# Use cell centroids of cut cells to avoid zeroed interface centroids
ϕγ_vec = solver.x[end÷2+1:end]
idx_cut = findall(capacity.cell_types .== -1)
centroids_cut = capacity.C_ω[idx_cut]
xγ = getindex.(centroids_cut, 1)
yγ = getindex.(centroids_cut, 2)
θ = rad2deg.(atan.(yγ .- center[2], xγ .- center[1]))
θ = map(t -> t < 0 ? t + 360.0 : t, θ)
ϕγ_cut = ϕγ_vec[idx_cut]
order = sortperm(θ)
θ = θ[order]
ϕγ_cut = ϕγ_cut[order]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "θ (deg)", ylabel="uγ", title="Interface value vs angle")
scatter!(ax, θ, ϕγ_cut; markersize=5, label="Numerical")
lines!(ax, θ, ones(length(θ)) * (7/4 - radius^2/4); color=:red, linestyle=:dash, label="Analytical")
xlims!(ax, 0, 360)
axislegend(ax, position=:rt)
display(fig)
save("interface_values_theta_deg.png", fig)

# Gradient error
∇_num = ∇(operator, solver.x)
∇_numx = ∇_num[1:end÷2]
∇_numy = ∇_num[end÷2+1:end]

∇x_ana = [∇x_analytical(x,y) for x in range(x0, stop=lx, length=nx+1), y in range(y0, stop=ly, length=ny+1)]
∇y_ana = [∇y_analytical(x,y) for x in range(x0, stop=lx, length=nx+1), y in range(y0, stop=ly, length=ny+1)]

∇x_ana[capacity.cell_types .== 0] .= NaN
∇x_ana = reshape(∇x_ana, (nx+1, ny+1))

∇y_ana[capacity.cell_types .== 0] .= NaN
∇y_ana = reshape(∇y_ana, (nx+1, ny+1))

∇_numx[capacity.cell_types .== 0] .= NaN
∇_numx = reshape(∇_numx, (nx+1, ny+1))

∇_numy[capacity.cell_types .== 0] .= NaN
∇_numy = reshape(∇_numy, (nx+1, ny+1))

∇x_num = reshape(∇_numx, (nx+1, ny+1))
∇y_num = reshape(∇_numy, (nx+1, ny+1))

∇x_err = ∇x_ana .- ∇x_num
∇y_err = ∇y_ana .- ∇y_num

using CairoMakie
fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = "x", ylabel="y", title="Analytical gradient in x")
ax2 = Axis(fig[1, 2], xlabel = "x", ylabel="y", title="Numerical gradient in x")
ax3 = Axis(fig[1, 3], xlabel = "x", ylabel="y", title="Log10 Error in gradient in x")
hm1 = heatmap!(ax1, ∇x_ana, colormap=:viridis)
hm2 = heatmap!(ax2, ∇x_num, colormap=:viridis)
hm3 = heatmap!(ax3, log10.(abs.(∇x_err)), colormap=:viridis)
Colorbar(fig[1, 4], hm1, label="∇x(x)")
Colorbar(fig[1, 5], hm2, label="∇x(x)")
Colorbar(fig[1, 6], hm3, label="∇x(x)")
display(fig)
