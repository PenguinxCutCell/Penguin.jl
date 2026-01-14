using Penguin
using LinearSolve

function make_shift_map(nx, L, x0; shift_left::Bool=true)
    d = length(nx)
    Δ   = ntuple(i -> L[i] / nx[i], d)
    xL  = ntuple(i -> x0[i] + (shift_left ? Δ[i] : 0.0), d)
    Leff = ntuple(i -> L[i] - (shift_left ? Δ[i] : 0.0), d)

    map_to_unit = function (x)
        ntuple(i -> (x[i] - xL[i]) / Leff[i], d)
    end

    return map_to_unit, xL, Leff, Δ
end

wrap_u(u_hat, map_to_unit) = function (x...)
    ξ = map_to_unit(ntuple(i -> x[i], length(x)))
    return u_hat(ξ...)
end

# ----------------------------
# MMS Poisson on full domain (2D, no body)
# ----------------------------
nx, ny = 80, 80
lx, ly = 4.0, 4.0
x0, y0 = 0.0, 0.0
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Full fluid: no immersed boundary
body = (x, y, _=0) -> -1.0
capacity = Capacity(body, mesh; method="VOFI")
operator = DiffusionOps(capacity)

# Shift map + exact solution on [0,1]^2
map_to_unit, xL, Leff, Δ = make_shift_map((nx, ny), (lx, ly), (x0, y0); shift_left=true)
u_hat = (ξ1, ξ2) -> sin(pi * ξ1) * sin(pi * ξ2)
u_exact = wrap_u(u_hat, map_to_unit)

λ = (pi/Leff[1])^2 + (pi/Leff[2])^2
f = (x, y, _=0) -> λ * u_exact(x, y)
D = (x, y, _=0) -> 1.0

bc0 = Dirichlet((x, y, _=0) ->0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :left   => Dirichlet((y, _=0) -> u_exact(xL[1], y)),
    :right  => Dirichlet((y, _=0) -> u_exact(x0 + lx, y)),
    :bottom => Dirichlet((x, _=0) -> u_exact(x, xL[2])),
    :top    => Dirichlet((x, _=0) -> u_exact(x, y0 + ly)),
))

phase = Phase(capacity, operator, f, D)
solver = DiffusionSteadyMono(phase, bc_b, bc0)

solve_DiffusionSteadyMono!(solver; method=Base.:\)

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