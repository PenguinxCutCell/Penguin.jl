using Penguin
using IterativeSolvers
using CairoMakie

# 2D oscillating flat plate inside a box using the moving Stokes solver

nx, ny = 48, 48
Lx, Ly = 2.0, 2.0
x0, y0 = -1.0, -1.0

mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
mesh_u = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
mesh_u_tuple = (mesh_u, mesh_u)

plate_half_thickness = 0.03 * Ly
ampl = 0.1 * Ly
freq = 2π
ymid = 0.0

y_plate(t) = ymid + ampl * sin(freq * t)
body = (x, y, t, _=0.0) -> -(abs(y - y_plate(t)) - plate_half_thickness)
body_velocity = (x, y, t) -> (0.0, ampl * freq * cos(freq * t))

Δt = 1.0 * (Lx / nx)^2
Tstart = 0.0
Tend = 0.1

STmesh_ux = Penguin.SpaceTimeMesh(mesh_u, [Tstart, Tstart + Δt])
STmesh_uy = Penguin.SpaceTimeMesh(mesh_u, [Tstart, Tstart + Δt])
STmesh_p = Penguin.SpaceTimeMesh(mesh_p, [Tstart, Tstart + Δt])

capacity_ux = Capacity(body, STmesh_ux; method="VOFI", compute_centroids=true, integration_method=:vofijul)
capacity_uy = Capacity(body, STmesh_uy; method="VOFI", compute_centroids=true, integration_method=:vofijul)
capacity_p = Capacity(body, STmesh_p; method="VOFI", compute_centroids=true, integration_method=:vofijul)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p = DiffusionOps(capacity_p)

μ = 1.0
ρ = 1.0
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

bc_common = BorderConditions(Dict(
    :left => Dirichlet(0.0),
    :right => Dirichlet(0.0),
    :top => Dirichlet(0.0),
    :bottom => Dirichlet(0.0)
))

bc_tuple = (bc_common, bc_common)
bc_cut = (
    Dirichlet((x, y, t, _=0) -> body_velocity(x, y, t)[2]),
    Dirichlet((x, y, t, _=0) -> 0.0)
)
pressure_gauge = PinPressureGauge()

fluid = Fluid((STmesh_ux, STmesh_uy), (capacity_ux, capacity_uy), (operator_ux, operator_uy),
              STmesh_p, capacity_p, operator_p,
              μ, ρ, fᵤ, fₚ)

solver = MovingStokesMono(fluid, bc_tuple, bc_cut;
                          pressure_gauge=pressure_gauge,
                          scheme="BE", t_eval=Tstart + Δt)

times = solve_MovingStokesMono!(solver, fluid, body,
                                mesh_u_tuple, mesh_p,
                                bc_tuple, bc_cut,
                                Δt, Tstart, Tend;
                                pressure_gauge=pressure_gauge,
                                scheme="BE",
                                method=gmres,
                                reltol=1e-12,
                                maxiter=10000,
                                geometry_method="ImplicitIntegration")

println("Stored states: ", length(solver.states))

state = solver.states[end]
nx_nodes = length(mesh_u.nodes[1])
ny_nodes = length(mesh_u.nodes[2])
nu_x = nx_nodes * ny_nodes
nu_y = nu_x
uωx = state[1:nu_x]
uωy = state[2nu_x+1:2nu_x+nu_y]

xs = mesh_u.nodes[1]
ys = mesh_u.nodes[2]

Ux = reshape(uωx, (nx_nodes, ny_nodes))
Uy = reshape(uωy, (nx_nodes, ny_nodes))
speed = sqrt.(Ux.^2 .+ Uy.^2)

fig = Figure(resolution=(900, 400))
ax1 = Axis(fig[1, 1], title="|u| at t=$(round(times[end], digits=3))",
           xlabel="x", ylabel="y")
heatmap!(ax1, xs, ys, speed'; colormap=:viridis)

ax2 = Axis(fig[1, 2], title="uy slice at x=0", xlabel="y", ylabel="uy")
mid_i = argmin(abs.(xs .- 0.0))
lines!(ax2, ys, Uy[mid_i, :], color=:steelblue, linewidth=2)

display(fig)
println("Max speed = ", maximum(speed))

function create_plate_animation(solver, mesh, times; filename="moving_stokes_plate.gif")
    nx_nodes = length(mesh.nodes[1])
    ny_nodes = length(mesh.nodes[2])
    nu = nx_nodes * ny_nodes
    xs = mesh.nodes[1]
    ys = mesh.nodes[2]

    vel_obs = Observable(reshape(solver.states[1][1:nu], (nx_nodes, ny_nodes)))
    plate_obs = Observable(y_plate(times[1]))

    fig = Figure(resolution=(600, 500))
    ax = Axis(fig[1, 1], title="|u|", xlabel="x", ylabel="y")
    hm = heatmap!(ax, xs, ys, abs.(vel_obs[]); colormap=:viridis)
    lines!(ax, xs, fill(plate_obs[], length(xs)), color=:red, linestyle=:dash, linewidth=2)

    record(fig, filename, 1:length(solver.states)) do i
        state = solver.states[i]
        uωx = reshape(state[1:nu], (nx_nodes, ny_nodes))
        uωy = reshape(state[2nu+1:2nu+nu], (nx_nodes, ny_nodes))
        speed = sqrt.(uωx.^2 .+ uωy.^2)
        vel_obs[] = speed
        plate_obs[] = y_plate(times[min(i, length(times))])
        hm[1] = xs
        hm[2] = ys
        hm[3] = speed
        ax.title = "t=$(round(times[min(i, length(times))], digits=3))"
    end
    println("Saved animation to $(filename)")
end

create_plate_animation(solver, mesh_u, times, filename="moving_stokes_plate.gif")
