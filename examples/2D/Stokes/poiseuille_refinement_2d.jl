using Penguin
using LinearAlgebra
using CairoMakie

# Mesh sizes to test
resolutions = [(16,16), (32,32), (64,64), (128,128), (256,256), (512,512)]


# Domain dimensions
Lx, Ly = 2.0, 1.0
x0, y0 = 0.0, 0.0

Umax = 1.0
parabola = (x, y) -> 4Umax * (y - y0) * (Ly - (y - y0)) / (Ly^2)

results = Float64[]
spacings = Float64[]

for (nx, ny) in resolutions
    println("Solving Poiseuille with grid $(nx)x$(ny)...")

    mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
    dx, dy = Lx/nx, Ly/ny
    mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
    mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

    body = (x, y, _=0) -> -1.0
    capacity_ux = Capacity(body, mesh_ux)
    capacity_uy = Capacity(body, mesh_uy)
    capacity_p  = Capacity(body, mesh_p)

    operator_ux = DiffusionOps(capacity_ux)
    operator_uy = DiffusionOps(capacity_uy)
    operator_p  = DiffusionOps(capacity_p)

    ux_left  = Dirichlet(parabola)
    ux_right = Dirichlet(parabola)
    ux_bot   = Dirichlet((x, y)-> 0.0)
    ux_top   = Dirichlet((x, y)-> 0.0)

    uy_zero = Dirichlet((x, y)-> 0.0)

    bc_ux = BorderConditions(Dict(
        :left=>ux_left, :right=>ux_right, :bottom=>ux_bot, :top=>ux_top
    ))
    bc_uy = BorderConditions(Dict(
        :left=>uy_zero, :right=>uy_zero, :bottom=>uy_zero, :top=>uy_zero
    ))
    pressure_gauge = PinPressureGauge()
    u_bc = Dirichlet(0.0)

    μ = 1.0
    ρ = 1.0
    fᵤ = (x, y, z=0.0) -> 0.0
    fₚ = (x, y, z=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (capacity_ux, capacity_uy),
                  (operator_ux, operator_uy),
                  mesh_p,
                  capacity_p,
                  operator_p,
                  μ, ρ, fᵤ, fₚ)

    nu = prod(operator_ux.size)
    np = prod(operator_p.size)
    x0_vec = zeros(4*nu + np)

    solver = StokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, u_bc; x0=x0_vec)
    solve_StokesMono!(solver; method=Base.:\)

    uωx = solver.x[1:nu]
    xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
    LIux = LinearIndices((length(xs), length(ys)))
    icol = Int(cld(length(xs), 2))
    ux_profile = [uωx[LIux[icol, j]] for j in 1:length(ys)]
    ux_analytic = [parabola(0.0, y) for y in ys]

    err = ux_profile .- ux_analytic
    ℓ2 = sqrt(sum(abs2, err) / length(err))
    push!(results, ℓ2)
    push!(spacings, dx)
    println("  L2 profile error = ", ℓ2)
end

println("\nGrid spacing vs L2 error: ")
for (h, e) in zip(spacings, results)
    println("  h = ", h, ", error = ", e)
end

if length(results) >= 2
    println("\nEstimated convergence rates:")
    for i in 2:length(results)
        rate = log(results[i]/results[i-1]) / log(spacings[i]/spacings[i-1])
        println("  Between grids $(resolutions[i-1]) -> $(resolutions[i]) : order ≈ ", rate)
    end
end

fig = Figure(resolution=(600, 450))
ax = Axis(fig[1,1], xlabel="h", ylabel="L₂ error", title="Poiseuille refinement", xscale=log10, yscale=log10)
sorted = sortperm(spacings, rev=false)
h_sorted = spacings[sorted]
err_sorted = results[sorted]
scatterlines!(ax, h_sorted, err_sorted, marker=:circle, label="error")

if length(err_sorted) >= 2
    reference_order = 2.0
    reference_order_1 = 1.0
    anchor_h = h_sorted[end]
    anchor_err = err_sorted[end]
    ref_values = anchor_err .* (h_sorted ./ anchor_h) .^ reference_order
    ref_values_1 = anchor_err .* (h_sorted ./ anchor_h) .^ reference_order_1
    lines!(ax, h_sorted, ref_values, linestyle=:dash, color=:gray, label="slope $(reference_order)")
    lines!(ax, h_sorted, ref_values_1, linestyle=:dot, color=:black, label="slope $(reference_order_1)")
end
axislegend(ax, position=:lt)
display(fig)
