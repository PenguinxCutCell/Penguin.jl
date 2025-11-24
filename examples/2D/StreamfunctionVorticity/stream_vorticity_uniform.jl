using Penguin
using IterativeSolvers
using LinearAlgebra
using Statistics
# Domain and mesh definition
nx, ny = 64, 64
Lx, Ly = 1.0, 1.0
mesh = Mesh((nx, ny), (Lx, Ly), (0.0, 0.0))

# Trivial embedded geometry: the whole domain is fluid
body = (x, y, _=0.0) -> -1.0
capacity = Capacity(body, mesh)

# Pre-compute operators for initial conditions
operator = DiffusionOps(capacity)
n = prod(operator.size)

# Initial vorticity: random noise (zero-mean) to mimic decaying turbulence
using Random
Random.seed!(2024)

ω_bulk = 2 .* rand(n) .- 1.0
ω_bulk .-= mean(ω_bulk)
ω_interface = zeros(n)
ω0 = vcat(ω_bulk, ω_interface)

# Boundary conditions
dirichlet_zero = Dirichlet(0.0)
border_bc = BorderConditions(Dict(
    :left => dirichlet_zero,
    :right => dirichlet_zero,
    :bottom => dirichlet_zero,
    :top => dirichlet_zero,
))

ν = 0.01
Δt = 1.0e-3

solver = StreamVorticity(capacity, ν, Δt;
    bc_stream = dirichlet_zero,
    bc_vorticity = dirichlet_zero,
    bc_stream_border = border_bc,
    bc_vorticity_border = border_bc,
    ω0 = ω0,
)

# Run a few implicit steps to diffuse the vorticity blob
run_StreamVorticity!(solver, 20; method = gmres)

bulk_norm = norm(solver.ω[1:n])
println("Final vorticity L2 norm (uniform domain): $(bulk_norm)")

# --- Visualization and animation ---
using CairoMakie

outdir = joinpath(@__DIR__, "outputs")
isdir(outdir) || mkpath(outdir)

mesh_x = mesh.nodes[1]
mesh_y = mesh.nodes[2]
cx = mesh.nodes[1]
cy = mesh.nodes[2]
function vorticity_field_from_state(state)
    # Extract ω vector from NamedTuple or accept a raw vector.
    ωfull = hasproperty(state, :ω) ? state.ω : state
    n = prod(operator.size)
    ωbulk = ωfull[1:n]
    nx_nodes = length(mesh.nodes[1])
    ny_nodes = length(mesh.nodes[2])
    mat = reshape(ωbulk, (nx_nodes, ny_nodes))
    return mat'
end


# Static plot
fig = Figure(resolution = (900, 450))
ax1 = Axis(fig[1, 1], title = "Vorticity (final)", aspect = DataAspect())
ax2 = Axis(fig[1, 2], title = "Velocity quiver (final)", aspect = DataAspect())

vort_mat = vorticity_field_from_state(solver.states[end])
hm = heatmap!(ax1, mesh_x, mesh_y, vort_mat, colormap = :viridis)
Colorbar(fig[1, 3], hm, label = "ω")

U = reshape(solver.velocity[1], (length(cx), length(cy)))'
V = reshape(solver.velocity[2], (length(cx), length(cy)))'
quiver!(ax2, cx, cy, U, V, linewidth = 1.0, color = :black)

save(joinpath(outdir, "stream_vorticity_uniform_final.png"), fig)

# Animation
# Choose number of steps to integrate and animate
steps = 400
println("Running streamfunction–vorticity for $steps steps (Δt=$(solver.Δt))...")
run_StreamVorticity!(solver, steps; method = gmres)

# Compute kinetic energy from stored ψ states
function kinetic_energy_from_state(state)
    ψ = hasproperty(state, :ψ) ? state.ψ : state
    gradψ = ∇(operator, ψ)
    ∂ψ∂x = view(gradψ, 1:n)
    ∂ψ∂y = view(gradψ, n+1:2n)
    u = ∂ψ∂y
    v = -∂ψ∂x
    return 0.5 * (sum(abs2, u) + sum(abs2, v))
end

energies = [kinetic_energy_from_state(s) for s in solver.states]
println("Initial kinetic energy = ", energies[2])
println("Final kinetic energy = ", energies[end])

# Animation over saved states (subsample if too many)
num_states = length(solver.states)
max_frames = 40
frame_indices = round.(Int, range(1, num_states, length=min(max_frames, num_states)))

giffile = joinpath(outdir, "stream_vorticity_uniform.gif")


record(fig, giffile, 1:length(frame_indices); framerate = 12) do idx
    i = frame_indices[idx]
    ax1 = Axis(fig[1, 1], title = "Vorticity (t=$(round(solver.states[i].time, digits=4)))", aspect = DataAspect())
    vort_mat = vorticity_field_from_state(solver.states[i])
    heatmap!(ax1, mesh_x, mesh_y, vort_mat, colormap = :viridis)
    ax2 = Axis(fig[1, 2], title = "Velocity quiver", aspect = DataAspect())
    ψi = solver.states[i].ψ
    gradψ = ∇(operator, ψi)
    ∂ψ∂x = view(gradψ, 1:n)
    ∂ψ∂y = view(gradψ, n+1:2n)
    Ui = reshape(∂ψ∂y, (length(cx), length(cy)))'
    Vi = -reshape(∂ψ∂x, (length(cx), length(cy)))'
    quiver!(ax2, cx, cy, Ui, Vi,  color = :black)
end

println("Saved figures and animations to: ", outdir)
