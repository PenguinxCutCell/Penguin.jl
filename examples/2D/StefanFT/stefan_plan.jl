using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using CairoMakie
using Statistics
using SpecialFunctions
using FrontCutTracking

### Problème de Stefan avec interface plane
### Solidification avec une interface plane se déplaçant dans la direction x

# Définition des paramètres physiques
L = 1.0      # Chaleur latente
c = 1.0      # Capacité calorifique 
TM = 0.0     # Température de fusion (à l'interface)
T_hot = 0.5  # Température chaude (côté gauche)
T_cold = -0.5 # Température froide (côté droit)

# Calcul du nombre de Stefan
Ste = (c * abs(TM - T_cold)) / L
println("Nombre de Stefan: $Ste")

# Position initiale et solution similitude
x0 = 0.0     # Position initiale de l'interface
t_init = 1.0  # Temps initial
t_final = 2.0 # Temps final

# Paramètre de similitude pour interface plane
# Résolution de l'équation de transcendance pour λ
function find_lambda(Ste)
    f(λ) = λ * erf(λ) * exp(λ^2) - Ste/sqrt(π)
    
    # Méthode de Newton pour trouver λ
    λ = 0.5  # Valeur initiale
    max_iter = 100
    tol = 1e-10
    
    for i in 1:max_iter
        # Calculer f(λ) et f'(λ)
        fx = λ * erf(λ) * exp(λ^2) - Ste/sqrt(π)
        dfx = erf(λ) * exp(λ^2) + λ * 2/sqrt(π) * exp(-λ^2) * exp(λ^2) + λ * erf(λ) * exp(λ^2) * 2*λ
        
        # Mettre à jour λ
        λ_new = λ - fx/dfx
        
        if abs(λ_new - λ) < tol
            return λ_new
        end
        λ = λ_new
    end
    
    return λ  # Retourner la meilleure approximation
end

# Calcul de la valeur de λ basée sur le nombre de Stefan
λ = find_lambda(Ste)
println("Paramètre λ calculé: $λ")

# Position de l'interface au temps t
function interface_position(t)
    return 2.0 * λ * sqrt(t)
end

# Fonction température analytique pour le problème monophasique
function analytical_temperature(x, t)
    # Solution de similitude pour problème monophasique
    interface_pos = interface_position(t)
    
    if x < interface_pos
        # Zone solide (seule phase modélisée)
        η = x / (2.0 * sqrt(t))
        return T_cold + (TM - T_cold) * erf(η) / erf(λ)
    else
        # Zone liquide (non modélisée dans le cas monophasique)
        return TM
    end
end

println("Position initiale de l'interface à t=$t_init: X=$(interface_position(t_init))")
println("Position finale de l'interface à t=$t_final: X=$(interface_position(t_final))")

# Définition du maillage spatial
nx, ny = 100, 20
lx, ly = 10.0, 2.0
x0_domain, y0_domain = -1.0, -1.0
Δx, Δy = lx/nx, ly/ny
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0_domain, y0_domain))

println("Maillage créé avec dimensions: $(nx) x $(ny), Δx=$(Δx), Δy=$(Δy), domaine=[$(x0_domain), $(x0_domain+lx)], [$(y0_domain), $(y0_domain+ly)]")

# Création de la ligne pour le front-tracking
nmarkers = 20
front = FrontTracker()
# Créer une ligne verticale à x = interface_position(t_init)
interface_x = 0.0  # Position initiale de l'interface

# Fonction pour créer une ligne verticale
function create_vertical_line!(front, x_pos, y_min, y_max, n_points)
    y_points = range(y_min, y_max, n_points)
    markers = [(x_pos, y) for y in y_points]
    set_markers!(front, markers)
    return front
end

# Créer la ligne verticale
set_domain!(front, x0_domain, x0_domain+lx, y0_domain, y0_domain+ly; fluid_side=:left)  # fluid on the left of the interface
create_vertical_line!(front, interface_x, y0_domain, y0_domain+ly, nmarkers)

# Définir la position initiale du front sous forme de SDF
# Pour une ligne verticale, la SDF est simplement x - x_interface
function plane_sdf(x, y, x_interface)
    return x - x_interface
end

# Définir la fonction body pour l'interface
body = (x, y, t, _=0) -> plane_sdf(x, y, interface_x)

# Définir le maillage spatio-temporel
Δt = 0.05
STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + Δt], tag=mesh.tag)

# Définir la capacité
capacity = Capacity(body, STmesh; compute_centroids=false)

# Définir l'opérateur de diffusion
operator = DiffusionOps(capacity)

# Définir les conditions aux limites
# Conditions de Dirichlet sur les bords gauche et droit
bc_left = Dirichlet(T_hot)
bc_right = Dirichlet(T_cold)
# Conditions de Neumann sur les bords haut et bas (flux nul)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :bottom => Dirichlet(1.0), 
))

# Condition à l'interface (température de fusion)
bc = Dirichlet(TM)

# Condition de Stefan à l'interface (saut de flux)
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, L))

# Terme source (pas de source)
f = (x,y,z,t) -> 0.0
K = (x,y,z) -> 1.0  # Conductivité thermique

# Définition de la phase
Fluide = Phase(capacity, operator, f, K)

# Condition initiale
u0ₒ = zeros((nx+1)*(ny+1))
for i in 1:nx+1
    for j in 1:ny+1
        idx = i + (j - 1) * (nx + 1)
        x = mesh.nodes[1][i]
        y = mesh.nodes[2][j]
        u0ₒ[idx] = analytical_temperature(x, t_init)
    end
end
u0ᵧ = ones((nx+1)*(ny+1))*TM  # Température à l'interface
u0 = vcat(u0ₒ, u0ᵧ)

# Visualiser le champ de température initial
fig_init = Figure(size=(800, 600))
ax_init = Axis(fig_init[1, 1], 
              title="Champ de température initial", 
              xlabel="x", 
              ylabel="y",
              aspect=DataAspect())
hm = heatmap!(ax_init, mesh.nodes[1], mesh.nodes[2], reshape(u0ₒ, (nx+1, ny+1)),
             colormap=:thermal)
Colorbar(fig_init[1, 2], hm, label="Température")

# Visualiser l'interface initiale
marker_x = [m[1] for m in get_markers(front)]
marker_y = [m[2] for m in get_markers(front)]
lines!(ax_init, marker_x, marker_y, 
      color=:white, linewidth=2, 
      label="Interface initiale")

display(fig_init)

# Paramètres de Newton
Newton_params = (100, 1e-6, 1e-6, 1.0)  # max_iter, tol, reltol, α

# Initialiser le solveur
solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Résoudre le problème
solver, residuals, xf_log, timestep_history = solve_StefanMono2Dunclosed!(
    solver, Fluide, front, Δt, t_init, t_final, bc_b, bc, stef_cond, mesh, "BE";
    Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\
)

# Fonctions pour visualiser les résultats
function plot_simulation_results(solver, residuals, xf_log, timestep_history)
    # Créer un répertoire pour les résultats
    results_dir = joinpath(pwd(), "stefan_plan_results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # 1. Tracer les résidus pour chaque pas de temps
    fig_residuals = Figure(size=(900, 600))
    ax_residuals = Axis(fig_residuals[1, 1], 
                       title="Historique de convergence", 
                       xlabel="Itération", 
                       ylabel="Norme du résidu (échelle log)",
                       yscale=log10)
    
    for (timestep, residual_vec) in sort(collect(residuals))
        lines!(ax_residuals, 1:length(residual_vec), residual_vec, 
              label="Pas de temps $timestep", 
              linewidth=2)
    end
    
    Legend(fig_residuals[1, 2], ax_residuals)
    save(joinpath(results_dir, "residus.png"), fig_residuals)
    
    # 2. Tracer l'évolution de l'interface
    fig_interface = Figure(size=(1000, 600))
    ax_interface = Axis(fig_interface[1, 1], 
                      title="Évolution de l'interface", 
                      xlabel="x", 
                      ylabel="y",
                      aspect=DataAspect())
    
    all_timesteps = sort(collect(keys(xf_log)))
    num_timesteps = length(all_timesteps)
    colors = cgrad(:viridis, num_timesteps)
    
    for (i, timestep) in enumerate(all_timesteps)
        markers = xf_log[timestep]
        marker_x = [m[1] for m in markers]
        marker_y = [m[2] for m in markers]
        
        lines!(ax_interface, marker_x, marker_y, 
              color=colors[i], 
              linewidth=2)
        
        if i == 1 || i == num_timesteps || i % max(1, div(num_timesteps, 5)) == 0
            time_value = timestep_history[min(timestep, length(timestep_history))][1]
            # Pour une interface verticale, placer le texte sur le côté
            text!(ax_interface, mean(marker_x) + 0.2, mean(marker_y), 
                 text="t=$(round(time_value, digits=2))",
                 align=(:left, :center),
                 fontsize=10)
        end
    end
    
    Colorbar(fig_interface[1, 2], limits=(1, num_timesteps),
            colormap=:viridis, label="Pas de temps")
    
    save(joinpath(results_dir, "evolution_interface.png"), fig_interface)
    
    # 3. Tracer la position de l'interface en fonction du temps
    times = [hist[1] for hist in timestep_history]
    positions = Float64[]
    
    for timestep in all_timesteps
        markers = xf_log[timestep]
        # Pour une interface verticale, prendre la moyenne des coordonnées x
        mean_x = sum(m[1] for m in markers) / length(markers)
        push!(positions, mean_x)
    end
    
    # Comparer avec la solution analytique
    analytical_positions = [interface_position(t) for t in times[1:length(positions)]]
    
    fig_position = Figure(size=(800, 600))
    ax_position = Axis(fig_position[1, 1], 
                     title="Position de l'interface en fonction du temps", 
                     xlabel="Temps", 
                     ylabel="Position X")
    
    scatter!(ax_position, times[1:length(positions)], positions, 
            label="Simulation")
    lines!(ax_position, times[1:length(positions)], analytical_positions, 
          label="Solution analytique", 
          linestyle=:dash, 
          linewidth=2)
    
    Legend(fig_position[1, 2], ax_position)
    save(joinpath(results_dir, "position_interface.png"), fig_position)
    
    return results_dir
end

# Visualiser les champs de température
function plot_temperature_heatmaps(solver, mesh, timestep_history, xf_log)
    results_dir = joinpath(pwd(), "stefan_plan_results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    xi = mesh.nodes[1]
    yi = mesh.nodes[2]
    nx1, ny1 = length(xi), length(yi)
    npts = nx1 * ny1
    
    for (i, Tstate) in enumerate(solver.states)
        # Extraire et formater la température
        Tw = Tstate[1:npts]
        Tmat = reshape(Tw, (nx1, ny1))
        
        # Créer la figure
        fig = Figure(size=(800, 600))
        ax = Axis(fig[1, 1],
                 title="Température, pas $i, t=$(round(timestep_history[i][1], digits=3))",
                 xlabel="x", ylabel="y", aspect=DataAspect())
        
        hm = heatmap!(ax, xi, yi, Tmat, colormap=:thermal)
        Colorbar(fig[1, 2], hm, label="T")
        
        # Ajouter l'interface si disponible
        if i <= length(xf_log)
            markers = xf_log[i]
            marker_x = [m[1] for m in markers]
            marker_y = [m[2] for m in markers]
            lines!(ax, marker_x, marker_y, 
                  color=:white, linewidth=2, 
                  label="Interface")
        end
        
        display(fig)
        save(joinpath(results_dir, "temperature_$(i).png"), fig)
    end
end

# Appeler les fonctions de visualisation
results_dir = plot_simulation_results(solver, residuals, xf_log, timestep_history)
plot_temperature_heatmaps(solver, mesh, timestep_history, xf_log)

println("\nVisualisations des résultats enregistrées dans: $results_dir")