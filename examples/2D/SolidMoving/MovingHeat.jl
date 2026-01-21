using Penguin
using IterativeSolvers, SpecialFunctions
using Roots
using CairoMakie
using Statistics

### 2D Test Case : Monophasic Unsteady Diffusion Equation inside a moving Disk
# Define the mesh
nx, ny = 128, 128
lx, ly = 16.0, 16.0
x0, y0 = -8.0, -8.0
Δx, Δy = lx/(nx), ly/(ny)
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body and similarity parameter
center = [0.0, 0.0]
# similarity parameter s0 and far-field temperature T_inf
s0 = 1.56
T_inf = -0.5
Tm = 0.0

# Sharp interface radius R(t) = s0 * sqrt(t)
body = (x,y,t) -> -(sqrt((x - center[1])^2 + (y - center[2])^2) - s0 * sqrt(t))

# Define F(s) function using the exponential integral E₁
function F(s)
    return expint(s^2/4)  # E₁(s²/4)
end

# Analytical temperature (similarity solution)
function analytical_temperature(r, t)
    if t <= 0
        return Tm
    end
    R = s0 * sqrt(t)
    if r < R
        return Tm
    else
        s = r / sqrt(t)
        return T_inf * (1 - F(s) / F(s0))
    end
end

# Define the Space-Time mesh
Δt = 0.5*(lx/nx)^2
Tstart = 1.0
Tend = 2.0
STmesh = Penguin.SpaceTimeMesh(mesh, [Tstart, Tstart+Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh; method="VOFI", compute_centroids=false)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(T_inf)
bc1 = Dirichlet(Tm)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc))

# Define the source term
f = (x,y,z,t)-> 0.0 #sin(x)*cos(10*y)
K = (x,y,z)-> 1.0

Fluide = Phase(capacity, operator, f, K)

# Initial condition
u0ₒ = ones((nx+1)*(ny+1))*T_inf
u0ᵧ = ones((nx+1)*(ny+1))*Tm
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
solver = MovingDiffusionUnsteadyMono(Fluide, bc_b, bc1, Δt, u0, mesh, "BE")

# Solve the problem
solve_MovingDiffusionUnsteadyMono!(solver, Fluide, body, Δt, Tstart, Tend, bc_b, bc1, mesh, "BE"; method=Base.:\,  geometry_method="VOFI", integration_method=:vofijul, compute_centroids=false)


# Plot the solution
#plot_solution(solver, mesh, body, capacity; state_i=1)


function plot_three_snapshots(solver, mesh, body, Tend; times=nothing, filename="moving_heat_snapshots.pdf")
    xi = mesh.centers[1]
    yi = mesh.centers[2]
    npts = (nx+1)*(ny+1)
    center = [0.0, 0.0]
    Δx, Δy = lx/nx, ly/ny

    # Default physical times: start, midpoint, end
    if times === nothing
        times = [Tstart, 0.5*(Tstart + Tend), Tend]
    end

    # Compute state time-step and map physical times to indices
    dt_state = length(solver.states) > 1 ? (Tend - Tstart) / (length(solver.states) - 1) : 0.0
    time_indices = [clamp(round(Int, (t - Tstart) / dt_state) + 1, 1, length(solver.states)) for t in times]
    # actual times corresponding to chosen indices
    times_selected = Tstart .+ (time_indices .- 1) .* dt_state

    # Temperature limits for consistent color scale (ignore non-finite)
    all_temps = Float64[]
    for Tstate in solver.states
        Tw = Tstate[1:npts]
        Tw_mat = reshape(Tw, (nx+1, ny+1))[1:end-1, 1:end-1]
        push!(all_temps, filter(isfinite, vec(Tw_mat))...)
    end
    temp_limits = isempty(all_temps) ? (0.0, 1.0) : (minimum(all_temps), maximum(all_temps))

    fig = Figure(size=(1200, 400))
    hm = nothing
    for (i, idx) in enumerate(time_indices)
        ax = Axis(fig[1, i],
            title = "t = $(round(times_selected[i], digits=3))",
            xlabel = "x", ylabel = "y",
            aspect = DataAspect())

        current_time = times_selected[i]
        body_end = (x, y, _=0) -> -(sqrt((x - center[1])^2 + (y - center[2])^2) - s0 * sqrt(current_time))
        capacity_end = Capacity(body_end, mesh; compute_centroids=false)

        Tstate = solver.states[idx]
        Tw = Tstate[1:npts]
        Tw[capacity_end.cell_types .== 0] .= 1.0

        Tmat = reshape(Tw, (nx+1, ny+1))[1:end-1, 1:end-1]
        Tmat_clean = replace(Tmat) do v; isfinite(v) ? v : 0.0 end
        hm = heatmap!(ax, xi, yi, Tmat_clean, colormap=:viridis, colorrange=temp_limits)

        # Interface at the selected physical time
        n_interface_points = 200
        theta = range(0, 2π, length=n_interface_points)
        interface_radius = s0 * sqrt(current_time)
        adjusted_center_x = center[1] - Δx
        adjusted_center_y = center[2] - Δy
        interface_x = adjusted_center_x .+ interface_radius .* cos.(theta)
        interface_y = adjusted_center_y .+ interface_radius .* sin.(theta)
        lines!(ax, interface_x, interface_y, color = :white, linewidth = 2)
    end

    if hm !== nothing
        Colorbar(fig[1, 4], hm, label = "Temperature")
    end
    save(filename, fig)
    println("Saved 3-snapshot figure to $filename")
    return fig
end

# Usage:
plot_three_snapshots(solver, mesh, body, Tend)

# Animation
function create_moving_heat_animation(solver, mesh, body, Tend)
    # Extraire les dimensions du maillage
    xi = mesh.centers[1]
    yi = mesh.centers[2]
    nx1, ny1 = length(xi), length(yi)
    npts = (nx+1) * (ny+1)  # Nombre total de points dans les états du solveur
    
    # Extraire les limites de température pour une échelle de couleur cohérente
    all_temps = Float64[]
    for Tstate in solver.states
        Tw = Tstate[1:npts]  # Températures à l'extérieur de l'interface
        # Reshape puis retirer la dernière ligne et colonne pour correspondre aux centres
        Tw_mat = reshape(Tw, (nx+1, ny+1))[1:end-1, 1:end-1]
        push!(all_temps, extrema(Tw_mat)...)
    end
    temp_limits = (minimum(all_temps), maximum(all_temps))
    
    # Obtenir le nombre de pas de temps
    n_timesteps = length(solver.states)
    
    # Créer le répertoire des résultats si nécessaire
    results_dir = joinpath(pwd(),"stefre")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Créer la figure pour l'animation
    fig = Figure(size=(900, 700))
    ax = Axis(fig[1, 1], 
            title="Temperature & Moving Interface", 
            xlabel="x", ylabel="y", 
            aspect=DataAspect(),
            limits=(x0, x0+lx, y0, y0+ly))

    # Persistent heatmap and colorbar
    hm = heatmap!(ax, xi, yi, zeros(nx1, ny1), colormap=:thermal)
    Colorbar(fig[1, 2], hm, label="Temperature")
    # placeholder for interface line (update per frame)
    lineplot = lines!(ax, [NaN], [NaN], color=:white, linewidth=3)

    # Enregistrer l'animation MP4
    record(fig, joinpath(results_dir, "moving_heat_animation.mp4"), 1:n_timesteps) do i
        # Calculer le temps actuel
        current_time = (i-1) * Δt
        ax.title = "Temperature & Moving Interface, t=$(round(current_time, digits=4))"

        # Extraire et reformater la température pour ce pas de temps
        body_end = (x, y,_=0) -> -(sqrt((x - center[1])^2 + (y - center[2])^2) - s0 * sqrt(current_time))
        capacity_end = Capacity(body_end, mesh; compute_centroids=false)
        Tstate = solver.states[i]
        Tw = Tstate[1:npts]
        Tw[capacity_end.cell_types .== 0] .= 1.0

        Tmat = reshape(Tw, (nx+1, ny+1))[1:end-1, 1:end-1]
        Tmat_clean = replace(Tmat) do v; isfinite(v) ? v : 0.0 end
        hm[3] = Tmat_clean

        # Générer et tracer l'interface
        n_interface_points = 100
        theta = range(0, 2π, length=n_interface_points)
        interface_radius = s0 * sqrt(current_time)
        adjusted_center_x = center[1] - Δx
        adjusted_center_y = center[2] - Δy
        interface_x = adjusted_center_x .+ interface_radius .* cos.(theta)
        interface_y = adjusted_center_y .+ interface_radius .* sin.(theta)
        lineplot[1] = interface_x
        lineplot[2] = interface_y

        println("Frame $i of $n_timesteps")
    end
    
    # Créer également une version GIF (réutilise hm et lineplot)
    record(fig, joinpath(results_dir, "moving_heat_animation.gif"), 1:n_timesteps) do i
        # Calculer le temps actuel
        current_time = (i-1) * Δt
        ax.title = "Temperature & Moving Interface, t=$(round(current_time, digits=4))"

        body_end = (x, y,_=0) -> -(sqrt((x - center[1])^2 + (y - center[2])^2) - s0 * sqrt(current_time))
        capacity_end = Capacity(body_end, mesh; compute_centroids=false)
        Tstate = solver.states[i]
        Tw = Tstate[1:npts]
        Tw[capacity_end.cell_types .== 0] .= 1.0

        Tmat = reshape(Tw, (nx+1, ny+1))[1:end-1, 1:end-1]
        Tmat_clean = replace(Tmat) do v; isfinite(v) ? v : 0.0 end
        hm[3] = Tmat_clean

        n_interface_points = 100
        theta = range(0, 2π, length=n_interface_points)
        interface_radius = s0 * sqrt(current_time)
        adjusted_center_x = center[1] - Δx
        adjusted_center_y = center[2] - Δy
        interface_x = adjusted_center_x .+ interface_radius .* cos.(theta)
        interface_y = adjusted_center_y .+ interface_radius .* sin.(theta)
        lineplot[1] = interface_x
        lineplot[2] = interface_y
    end
    
    println("\nAnimation saved to:")
    println("- MP4: $(joinpath(results_dir, "moving_heat_animation.mp4"))")
    println("- GIF: $(joinpath(results_dir, "moving_heat_animation.gif"))")
    
    return fig
end

# Appel de la fonction pour créer l'animation
anim = create_moving_heat_animation(solver, mesh, body, Tend)
