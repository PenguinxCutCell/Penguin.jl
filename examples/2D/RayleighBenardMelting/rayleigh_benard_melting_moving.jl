using Penguin
using LinearAlgebra
try
    using CairoMakie
catch
    @warn "CairoMakie not available; skipping plots."
end

# Rayleigh-Benard instability with a melting boundary driven by a moving liquid front.

Tm = 0.3
T_hot = 1.0
Ra = 1.0e3
Pr = 1.0

Lx = 1.0
Ly = 1.0
nx = 64
ny = 32
origin = (0.0, 0.0)

mesh = Penguin.Mesh((nx, ny), (Lx, Ly), origin)
dx = Lx / nx
dy = Ly / ny
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (origin[1] - 0.5 * dx, origin[2]))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (origin[1], origin[2] - 0.5 * dy))
mesh_p = mesh
mesh_T = mesh

ΔT = T_hot - Tm
ν = sqrt(Pr * ΔT / Ra)
κ_const = ν / Pr
κ = (x, y, z=0.0, t=0.0) -> κ_const
β = 1.0
gravity = (-1.0, 0.0)

zero_dirichlet = Dirichlet((x, y, t=0.0) -> 0.0)
bc_ux = BorderConditions(Dict(:left => zero_dirichlet, :right => zero_dirichlet, :bottom => zero_dirichlet, :top => zero_dirichlet))
bc_uy = BorderConditions(Dict(:left => zero_dirichlet, :right => zero_dirichlet, :bottom => zero_dirichlet, :top => zero_dirichlet))
bc_cut_u = Dirichlet(0.0)

bc_T = BorderConditions(Dict(
    :bottom => Dirichlet(T_hot),
    :top => Dirichlet(Tm),
    :left => Neumann(0.0),
    :right => Neumann(0.0)
))
bc_cut_T = Dirichlet(Tm)

pressure_gauge = PinPressureGauge()
scalar_source = (x, y, z=0.0, t=0.0) -> 0.0
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

xf0 = 0.6 * Lx
xf_profile = fill(xf0, length(mesh.nodes[2]))
Δy = mesh.nodes[2][2] - mesh.nodes[2][1]
Hₙ⁰ = (xf_profile .- mesh.nodes[1][1]) .* Δy
sₙ = lin_interpol(mesh.nodes[2], xf_profile)

nodes_Tx = mesh_T.nodes[1]
nodes_Ty = mesh_T.nodes[2]
Nx_T = length(nodes_Tx)
Ny_T = length(nodes_Ty)
T0ω = zeros(Float64, Nx_T * Ny_T)
for j in 1:Ny_T
    for i in 1:Nx_T
        x = nodes_Tx[i]
        idx = i + (j - 1) * Nx_T
        T0ω[idx] = x <= xf0 ? T_hot - (T_hot - Tm) * (x / xf0) : Tm
    end
end
T0γ = copy(T0ω)
T0 = vcat(T0ω, T0γ)

setup = RayleighBenardMeltingSetup(
    mesh_ux, mesh_uy, mesh_p, mesh_T,
    bc_ux, bc_uy, bc_T, bc_cut_u, bc_cut_T, pressure_gauge;
    ν=ν, ρ=1.0, κ=κ, scalar_source=scalar_source,
    strategy=PicardCoupling(tol_T=1e-4, tol_U=1e-4, maxiter=2, relaxation=0.8),
    β=β, gravity=gravity, T_ref=Tm,
    fᵤ=fᵤ, fₚ=fₚ,
    store_states=false,
    initial_temperature=T0
)

ic = InterfaceConditions(nothing, FluxJump(1.0, 0.0, 1.0))

Δt = 2e-2
T_end = 0.05

temps, velocities, residuals, xf_log, reconstruct, timestep_history = solve_MovingRayleighBenardMelting2D!(
    setup, xf_profile, Hₙ⁰, Δt, T_end, ic, mesh, "CN";
    sₙ=sₙ, interpo="quad", adaptive_timestep=false
)

final_xf = isempty(xf_log) ? xf_profile : xf_log[end]
println("Simulation complete over $(length(timestep_history)-1) steps.")
println("Final interface range: min=$(minimum(final_xf)), max=$(maximum(final_xf))")
println("Stored $(length(temps)) temperature snapshots and $(length(velocities)) velocity snapshots.")

if @isdefined(CairoMakie) && !isempty(temps)
    Nx = length(mesh_T.nodes[1])
    Ny = length(mesh_T.nodes[2])
    T_final = temps[end][1:(Nx * Ny)]
    T_grid = reshape(T_final, (Nx, Ny))

    fig = Figure(resolution=(900, 600))
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Temperature and interface at t=$(round(T_end, digits=3))")
    hm = heatmap!(ax, mesh_T.centers[1], mesh_T.centers[2], T_grid'; colormap=:thermal)
    hlines!(ax, mesh_T.centers[2], final_xf; color=:white, linewidth=2.0)
    Colorbar(fig[1, 2], hm; label="Temperature")
    display(fig)
    save(joinpath(@__DIR__, "rayleigh_benard_melting_final.png"), fig, px_per_unit=3)
end
