using Penguin
using IterativeSolvers
using LinearAlgebra

# Square box with an immersed circular boundary
nx, ny = 96, 96
Lx, Ly = 1.0, 1.0
mesh = Mesh((nx, ny), (Lx, Ly), (0.0, 0.0))

radius = 0.2
centre = (0.5 * Lx, 0.5 * Ly)
circle = (x, y, _=0.0) -> sqrt((x - centre[1])^2 + (y - centre[2])^2) - radius
capacity = Capacity(circle, mesh)

operator = DiffusionOps(capacity)
n = prod(operator.size)

# Localised vorticity ring hugging the circular interface
ω_bulk = zeros(n)
for (i, coord) in enumerate(capacity.C_ω)
    x, y = coord[1], coord[2]
    if circle(x, y) < 0
        r = sqrt((x - centre[1])^2 + (y - centre[2])^2)
        ω_bulk[i] = cospi(clamp((r / radius), 0.0, 1.0))
    end
end
ω_interface = zeros(n)
ω0 = vcat(ω_bulk, ω_interface)

dirichlet_zero = Dirichlet(0.0)
border_bc = BorderConditions(Dict(
    :left => dirichlet_zero,
    :right => dirichlet_zero,
    :bottom => dirichlet_zero,
    :top => dirichlet_zero,
))

ν = 0.005
Δt = 5.0e-4

solver = StreamVorticity(capacity, ν, Δt;
    bc_stream = dirichlet_zero,
    bc_vorticity = dirichlet_zero,
    bc_stream_border = border_bc,
    bc_vorticity_border = border_bc,
    ω0 = ω0,
)

run_StreamVorticity!(solver, 40; method = gmres)

u, v = solver.velocity
println("Velocity extrema with circular cut cells: |u|ₘₐₓ=$(maximum(abs.(u))) |v|ₘₐₓ=$(maximum(abs.(v)))")

# --- Visualization and animation ---
using CairoMakie
using Dates

# ensure output directory
outdir = joinpath(@__DIR__, "outputs")
isdir(outdir) || mkpath(outdir)

# Extract grid coordinates for plotting
x = mesh.nodes[1]
y = mesh.nodes[2]
n = prod(operator.size)

# Helper to build vorticity bulk field as a matrix (nx+1, ny+1)
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

# Static snapshot of final state
fig = Figure(resolution = (900, 450))
ax1 = Axis(fig[1, 1], title = "Vorticity (final)", aspect = DataAspect())
ax2 = Axis(fig[1, 2], title = "Velocity quiver (final)", aspect = DataAspect())

vort_mat = vorticity_field_from_state(solver.states[end])
hm = heatmap!(ax1, x, y, vort_mat, colormap = :viridis)
Colorbar(fig[1, 3], hm, label = "ω")

# quiver on cell-centred locations
cx = mesh.nodes[1]
cy = mesh.nodes[2]
U = reshape(solver.velocity[1], (length(cx), length(cy)))'
V = reshape(solver.velocity[2], (length(cx), length(cy)))'
quiver!(ax2, cx, cy, U, V,  linewidth = 1.0, color = :black)

save(joinpath(outdir, "stream_vorticity_circle_final.png"), fig)

# Animation: record vorticity evolution and velocity quiver every few frames
frames = 1:length(solver.states)
giffile = joinpath(outdir, "stream_vorticity_circle.gif")


# create a small GIF for quick preview
record(fig, giffile, frames; framerate = 10) do i
    # same content as mp4 recording
    ax1 = Axis(fig[1, 1], title = "Vorticity (t=$(round(solver.states[i].time, digits=4)))", aspect = DataAspect())
    vort_mat = vorticity_field_from_state(solver.states[i])
    heatmap!(ax1, x, y, vort_mat, colormap = :viridis)
    ax2 = Axis(fig[1, 2], title = "Velocity quiver", aspect = DataAspect())
    ψi = solver.states[i].ψ
    gradψ = ∇(operator, ψi)
    ∂ψ∂x = view(gradψ, 1:n)
    ∂ψ∂y = view(gradψ, n+1:2n)
    Ui = reshape(∂ψ∂y, (length(cx), length(cy)))'
    Vi = -reshape(∂ψ∂x, (length(cx), length(cy)))'
    quiver!(ax2, cx, cy, Ui, Vi, color = :black)
end

println("Saved figures and animations to: ", outdir)
