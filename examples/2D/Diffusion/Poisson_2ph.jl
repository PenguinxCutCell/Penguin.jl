using Penguin
using IterativeSolvers, LinearAlgebra
using VTKOutputs

### 2D Test Case : Diphasic Steady Diffusion Equation inside a Disk
# Define the mesh
nx, ny = 80, 80
lx, ly = 4., 4.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/4, (lx/2, ly/2) #.+ (0.01, 0.01)
circle = (x,y,_=0)->sqrt((x-center[1])^2 + (y-center[2])^2) - radius
circle_c = (x,y,_=0)->-(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# Define the capacity
capacity = Capacity(circle, mesh)
capacity_c = Capacity(circle_c, mesh)

# Define the operators
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc = Dirichlet(1.0)
bc1 = Dirichlet(0.0)
bc2 = Dirichlet(2.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc1, :right => bc1, :top => bc1, :bottom => bc1))

ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Define the source term
f1 = (x,y,_)->1.0
f2 = (x,y,_)->1.0

D1, D2 = (x,y,_)->1.0, (x,y,_)->1.0

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, D1)
Fluide_2 = Phase(capacity_c, operator_c, f2, D2)

# Define the solver
solver = DiffusionSteadyDiph(Fluide_1, Fluide_2, bc_b, ic)

# Solve the problem
solve_DiffusionSteadyDiph!(solver; method=Base.:\)

println(maximum(abs.(solver.x)))

# Plot the solution usign Makie
#plot_solution(solver, mesh, circle, capacity)

# Write the solution to a VTK file
write_vtk("poisson_2d", mesh, solver)


# Analytical solution
function u_analytical(x,y)
    # Sum from m,n=1 to 100 m,n odd of 16 L²/((π^4 mn) * (m² + n²)) * sin(mπx/L) * sin(nπy/L)
    sum = 0.0
    for m in 1:2:500
        for n in 1:2:500
            sum += 16 * lx^2 / (π^4 * m * n * (m^2 + n^2)) * sin(m*π*x/lx) * sin(n*π*y/ly)
        end
    end
    return sum
end

# Weighted Lp or L∞ norm helper
function lp_norm(errors, indices, pval, capacity)
    if pval == Inf
        return maximum(abs.(errors[indices]))
    else
        part_sum = 0.0
        for i in indices
            Vi = capacity.V[i,i]
            part_sum += (abs(errors[i])^pval) * Vi
        end
        return (part_sum / sum(capacity.V))^(1/pval)
    end
end

function check_convergence_diph(u_analytical::Function, solver, capacity::Capacity{2}, capacity_c::Capacity{2}, p::Real=2, relative::Bool=false)
    # 1) Compute pointwise error
    cell_centroids = capacity.C_ω
    cell_centroids_c = capacity_c.C_ω
    u_ana = map(c -> u_analytical(c[1], c[2]), cell_centroids)
    u_ana_c = map(c -> u_analytical(c[1], c[2]), cell_centroids_c)

    u_num1 = solver.x[1:end÷2]
    u_num2 = solver.x[end÷2+1:end]

    u_num1ₒ = u_num1[1:end÷2]
    u_num1ᵧ = u_num1[end÷2+1:end]
    u_num2ₒ = u_num2[1:end÷2]
    u_num2ᵧ = u_num2[end÷2+1:end]

    err = u_ana .- u_num1ₒ
    err_c = u_ana_c .- u_num2ₒ

    # 2) Retrieve cell types and separate full, cut, empty
    cell_types = capacity.cell_types
    idx_all    = findall((cell_types .== 1) .| (cell_types .== -1))
    idx_full   = findall(cell_types .== 1)
    idx_cut    = findall(cell_types .== -1)
    idx_empty  = findall(cell_types .== 0)

    cell_types_c = capacity_c.cell_types
    idx_all_c    = findall((cell_types_c .== 1) .| (cell_types_c .== -1))
    idx_full_c   = findall(cell_types_c .== 1)
    idx_cut_c    = findall(cell_types_c .== -1)
    idx_empty_c  = findall(cell_types_c .== 0)

    # 4) Compute norms (relative or not)
    if relative
        global_err = relative_lp_norm(err, idx_all, p, capacity, u_ana)
        full_err   = relative_lp_norm(err, idx_full,  p, capacity, u_ana)
        cut_err    = relative_lp_norm(err, idx_cut,   p, capacity, u_ana)
        empty_err  = relative_lp_norm(err, idx_empty, p, capacity, u_ana)

        global_err_c = relative_lp_norm(err_c, idx_all_c, p, capacity_c, u_ana_c)
        full_err_c   = relative_lp_norm(err_c, idx_full_c,  p, capacity_c, u_ana_c)
        cut_err_c    = relative_lp_norm(err_c, idx_cut_c,   p, capacity_c, u_ana_c)
        empty_err_c  = relative_lp_norm(err_c, idx_empty_c, p, capacity_c, u_ana_c)
    else
        global_err = lp_norm(err, idx_all, p, capacity)
        full_err   = lp_norm(err, idx_full,  p, capacity)
        cut_err    = lp_norm(err, idx_cut,   p, capacity)
        empty_err  = lp_norm(err, idx_empty, p, capacity)

        global_err_c = lp_norm(err_c, idx_all_c, p, capacity_c)
        full_err_c   = lp_norm(err_c, idx_full_c,  p, capacity_c)
        cut_err_c    = lp_norm(err_c, idx_cut_c,   p, capacity_c)
        empty_err_c  = lp_norm(err_c, idx_empty_c, p, capacity_c)
    end

    println("All cells L$p norm - Phase 1       = $global_err")
    println("Full cells L$p norm - Phase 1      = $full_err")
    println("Cut cells L$p norm - Phase 1       = $cut_err")
    println("Empty cells L$p norm - Phase 1     = $empty_err")

    println("All cells L$p norm - Phase 2       = $global_err_c")
    println("Full cells L$p norm - Phase 2      = $full_err_c")
    println("Cut cells L$p norm - Phase 2       = $cut_err_c")
    println("Empty cells L$p norm - Phase 2     = $empty_err_c")

    u_ana = (u_ana, u_ana_c)
    u_num = (u_num1ₒ, u_num2ₒ)

    return (u_ana, u_num, global_err, full_err, cut_err, empty_err)
end

u_ana, u_num, global_err, full_err, cut_err, empty_err = check_convergence_diph(u_analytical, solver, capacity, capacity_c, 2, false)

"""


cell_centroids = capacity.C_ω
cell_centroids_c = capacity_c.C_ω
u_ana = map(c -> u_analytical(c[1], c[2]), cell_centroids)
u_ana_c = map(c -> u_analytical(c[1], c[2]), cell_centroids_c)

u_num1 = solver.x[1:end÷2]
u_num2 = solver.x[end÷2+1:end]

u_num1ₒ = u_num1[1:end÷2]
u_num1ᵧ = u_num1[end÷2+1:end]
u_num2ₒ = u_num2[1:end÷2]
u_num2ᵧ = u_num2[end÷2+1:end]

err = u_ana .- u_num1ₒ
err_c = u_ana_c .- u_num2ₒ

u_ana[capacity.cell_types .== 0] .= NaN
u_ana_c[capacity_c.cell_types .== 0] .= NaN
u_ana = reshape(u_ana, (nx+1, ny+1))
u_ana_c = reshape(u_ana_c, (nx+1, ny+1))

u_num1ₒ[capacity.cell_types .== 0] .= NaN
u_num1ₒ = reshape(u_num1ₒ, (nx+1, ny+1))

u_num2ₒ[capacity_c.cell_types .== 0] .= NaN
u_num2ₒ = reshape(u_num2ₒ, (nx+1, ny+1))

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Phase 1")
ax1 = Axis(fig[1, 2], xlabel="x", ylabel="y", title="Phase 2")
ax2 = Axis(fig[2, 1], xlabel="x", ylabel="y", title="Analytical solution - Phase 1")
ax3 = Axis(fig[2, 2], xlabel="x", ylabel="y", title="Analytical solution - Phase 2")
hm = heatmap!(ax, u_num1ₒ, colormap=:viridis)
hm1 = heatmap!(ax1, u_num2ₒ, colormap=:viridis)
hm2 = heatmap!(ax2, u_ana, colormap=:viridis)
hm3 = heatmap!(ax3, u_ana_c, colormap=:viridis)
Colorbar(fig[1, 3], hm, label="uₙ(x) - Phase 1")
Colorbar(fig[1, 4], hm1, label="uₙ(x) - Phase 2")
Colorbar(fig[2, 3], hm2, label="uₐ(x) - Phase 1")
Colorbar(fig[2, 4], hm3, label="uₐ(x) - Phase 2")
display(fig)

# Plot error heatmap
err = reshape(err, (nx+1, ny+1))
err_c = reshape(err_c, (nx+1, ny+1))
err[capacity.cell_types .== 0] .= NaN
err_c[capacity_c.cell_types .== 0] .= NaN

err[end, :] .= NaN
err_c[end-3:end, :] .= NaN


fig = Figure()

ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Log error - Phase 1")
ax1 = Axis(fig[1, 2], xlabel="x", ylabel="y", title="Log error - Phase 2")
hm = heatmap!(ax, log10.(abs.(err)), colormap=:viridis)
hm1 = heatmap!(ax1, log10.(abs.(err_c)), colormap=:viridis)
Colorbar(fig[1, 3], hm, label="log10(|u(x) - u_num(x)|) - Phase 1")
Colorbar(fig[1, 4], hm1, label="log10(|u(x) - u_num(x)|) - Phase 2")
display(fig)
"""