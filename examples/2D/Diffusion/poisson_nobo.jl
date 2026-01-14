using Penguin
using LinearSolve

# ----------------------------
# MMS Poisson on full domain
# ----------------------------
nx, ny = 80, 80
lx, ly = 4.0, 4.0
x0, y0 = 0.0, 0.0
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Full fluid: no immersed boundary
body = (x,y,_=0) -> -1.0
capacity = Capacity(body, mesh; method="VOFI")  # OK for full domain
operator = DiffusionOps(capacity)

# Exact solution (scaled to the domain)
Δx = lx / nx
Δy = ly / ny
lx_eff = lx - Δx
ly_eff = ly - Δy

# MMS exact solution (matches the "truncated" boundary location)
u_exact(x, y) = sin(pi*(x - Δx)/lx_eff) * sin(pi*(y - Δy)/ly_eff)

λ = (pi^2/(lx_eff^2) + pi^2/(ly_eff^2))  # eigenvalue for the Poisson equation
f(x, y, _=0) = λ * u_exact(x, y)
D(x, y, _=0) = 1.0

# Homogeneous Dirichlet (since u_exact = 0 on all borders)
bc0 = Dirichlet((x, y, _=0) -> u_exact(x, y))
bc_b = BorderConditions(Dict(:left=>bc0, :right=>bc0, :bottom=>bc0, :top=>bc0))

phase = Phase(capacity, operator, f, D)
solver = DiffusionSteadyMono(phase, bc_b, bc0)

solve_DiffusionSteadyMono!(solver; method=Base.:\)

# Error
u_ana, u_num, global_err, full_err, cut_err, empty_err =
    check_convergence(u_exact, solver, capacity, 2, false)

println("L2 global error = ", global_err)

# Plot the solution
using CairoMakie
fig=Figure()
ax=Axis(fig[1,1], xlabel="x", ylabel="y", title="Numerical Solution")
hm = heatmap!(ax, mesh.centers[1], mesh.centers[2], reshape(u_num, nx+1, ny+1)', colormap=:viridis)
Colorbar(fig[1,2], hm; label="u")

ax2=Axis(fig[2,1], xlabel="x", ylabel="y", title="Analytical Solution")
hm2 = heatmap!(ax2, mesh.centers[1], mesh.centers[2], reshape(u_ana, nx+1, ny+1)', colormap=:viridis)
Colorbar(fig[2,2], hm2; label="u")

ax3=Axis(fig[3,1], xlabel="x", ylabel="y", title="Error (Numerical - Analytical)")
hm3 = heatmap!(ax3, mesh.centers[1], mesh.centers[2], reshape(u_num - u_ana, nx+1, ny+1)', colormap=:viridis)
Colorbar(fig[3,2], hm3; label="Error")
display(fig)