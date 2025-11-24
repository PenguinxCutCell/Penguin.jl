using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations
using Statistics
using ProgressMeter

# Importer les fonctions d'analyse existantes
include("fixedpoint_analysis.jl")

"""
Analyser la fonction de point fixe pour les mouvements vers la gauche et vers la droite de l'interface
"""
function analyze_interface_direction(mesh_size=80, Δt=0.001, alpha=1.0;
                                   xf_base=0.5, displacement=0.2,
                                   num_samples=100, max_iter=20)
    
    # Définir les paramètres du domaine
    nx = mesh_size
    lx = 1.0
    x0 = 0.0
    
    # Définir les plages pour les mouvements vers la gauche et vers la droite
    xf_left = max(0.1, xf_base - displacement)
    xf_right = min(0.9, xf_base + displacement)
    
    # Plages d'échantillonnage
    left_range = (xf_left - 0.1, xf_base)
    right_range = (xf_base, xf_right + 0.1)
    
    # Échantillonner et évaluer la fonction de point fixe pour les deux directions
    println("Analyse du mouvement vers la gauche (interface se déplace de $xf_base vers $xf_left)...")
    left_samples = range(left_range[1], left_range[2], length=num_samples)
    left_next = zeros(num_samples)
    left_residuals = Float64[]
    
    for (i, xf) in enumerate(left_samples)
        res, next_xf = evaluate_fixed_point_function(xf, mesh_size, Δt, alpha)
        left_next[i] = next_xf
        push!(left_residuals, res)
    end
    
    println("Analyse du mouvement vers la droite (interface se déplace de $xf_base vers $xf_right)...")
    right_samples = range(right_range[1], right_range[2], length=num_samples)
    right_next = zeros(num_samples)
    right_residuals = Float64[]
    
    for (i, xf) in enumerate(right_samples)
        res, next_xf = evaluate_fixed_point_function(xf, mesh_size, Δt, alpha)
        right_next[i] = next_xf
        push!(right_residuals, res)
    end
    
    # Calculer les constantes de Lipschitz
    dx_left = diff(left_samples)
    dy_left = diff(left_next)
    slopes_left = dy_left ./ dx_left
    L_left = maximum(abs.(slopes_left))
    
    dx_right = diff(right_samples)
    dy_right = diff(right_next)
    slopes_right = dy_right ./ dx_right
    L_right = maximum(abs.(slopes_right))
    
    # Trouver les points fixes pour chaque direction
    fixed_left = find_fixed_points(left_samples, left_next)
    fixed_right = find_fixed_points(right_samples, right_next)
    
    # Créer la figure de comparaison
    fig = Figure(resolution=(1200, 900))
    
    # Axes pour la fonction de point fixe
    ax_fx = Axis(fig[1, 1:2],
                xlabel="Position de l'interface", 
                ylabel="f(x)",
                title="Comparaison de la fonction de point fixe: déplacement gauche vs. droite")
    
    # Tracer les fonctions de point fixe
    scatter!(ax_fx, left_samples, left_next, markersize=6, color=:blue, label="Mouvement vers la gauche")
    lines!(ax_fx, left_samples, left_next, linewidth=2, color=:blue)
    
    scatter!(ax_fx, right_samples, right_next, markersize=6, color=:red, label="Mouvement vers la droite")
    lines!(ax_fx, right_samples, right_next, linewidth=2, color=:red)
    
    # Tracer la diagonale y=x
    diagonal = range(min(minimum(left_samples), minimum(right_samples)), 
                    max(maximum(left_samples), maximum(right_samples)), 
                    length=100)
    lines!(ax_fx, diagonal, diagonal, linewidth=2, linestyle=:dash, color=:black, label="y = x")
    
    # Marquer les points fixes éventuels
    if !isempty(fixed_left)
        itp_left = linear_interpolation(left_samples, left_next, extrapolation_bc=Flat())
        fixed_y_left = itp_left.(fixed_left)
        scatter!(ax_fx, fixed_left, fixed_y_left, markersize=12, marker=:star5, color=:blue,
                label="Point fixe (mouvement gauche)")
    end
    
    if !isempty(fixed_right)
        itp_right = linear_interpolation(right_samples, right_next, extrapolation_bc=Flat())
        fixed_y_right = itp_right.(fixed_right)
        scatter!(ax_fx, fixed_right, fixed_y_right, markersize=12, marker=:star5, color=:red,
                label="Point fixe (mouvement droite)")
    end
    
    # Axes pour les dérivées/pentes
    ax_slopes = Axis(fig[2, 1:2],
                    xlabel="Position de l'interface", 
                    ylabel="f'(x)",
                    title="Comparaison des pentes: déplacement gauche vs. droite")
    
    # Tracer les pentes
    x_mid_left = (left_samples[1:end-1] .+ left_samples[2:end]) ./ 2
    scatter!(ax_slopes, x_mid_left, slopes_left, markersize=6, color=:blue, label="Pentes (gauche)")
    lines!(ax_slopes, x_mid_left, slopes_left, linewidth=2, color=:blue)
    
    x_mid_right = (right_samples[1:end-1] .+ right_samples[2:end]) ./ 2
    scatter!(ax_slopes, x_mid_right, slopes_right, markersize=6, color=:red, label="Pentes (droite)")
    lines!(ax_slopes, x_mid_right, slopes_right, linewidth=2, color=:red)
    
    # Ajouter la ligne L=1
    hlines!(ax_slopes, [1.0], color=:black, linestyle=:dash, linewidth=2, label="L = 1 (seuil de convergence)")
    hlines!(ax_slopes, [-1.0], color=:black, linestyle=:dash, linewidth=2)
    
    # Marquer les constantes de Lipschitz
    max_slope_idx_left = findmax(abs.(slopes_left))[2]
    max_slope_x_left = x_mid_left[max_slope_idx_left]
    max_slope_y_left = slopes_left[max_slope_idx_left]
    
    max_slope_idx_right = findmax(abs.(slopes_right))[2]
    max_slope_x_right = x_mid_right[max_slope_idx_right]
    max_slope_y_right = slopes_right[max_slope_idx_right]
    
    scatter!(ax_slopes, [max_slope_x_left], [max_slope_y_left], marker=:diamond, markersize=10, color=:blue,
            label="L_max = $(round(L_left, digits=3)) (gauche)")
    
    scatter!(ax_slopes, [max_slope_x_right], [max_slope_y_right], marker=:diamond, markersize=10, color=:red,
            label="L_max = $(round(L_right, digits=3)) (droite)")
    
    # Axes pour les résidus
    ax_res = Axis(fig[3, 1:2],
                 xlabel="Position de l'interface", 
                 ylabel="Résidus",
                 title="Comparaison des résidus: déplacement gauche vs. droite")
    
    # Tracer les résidus
    scatter!(ax_res, left_samples, left_residuals, markersize=6, color=:blue, label="Résidus (gauche)")
    lines!(ax_res, left_samples, left_residuals, linewidth=2, color=:blue)
    
    scatter!(ax_res, right_samples, right_residuals, markersize=6, color=:red, label="Résidus (droite)")
    lines!(ax_res, right_samples, right_residuals, linewidth=2, color=:red)
    
    # Ajouter la ligne zéro pour les résidus
    hlines!(ax_res, [0.0], color=:black, linestyle=:dash, linewidth=2)
    
    # Simulation d'itération pour les deux directions
    ax_iter = Axis(fig[4, 1:2],
                  xlabel="Itération", 
                  ylabel="Position de l'interface",
                  title="Simulation d'itération: déplacement gauche vs. droite")
    
    # Créer des fonctions d'interpolation pour les itérations
    itp_left = linear_interpolation(left_samples, left_next, extrapolation_bc=Flat())
    itp_right = linear_interpolation(right_samples, right_next, extrapolation_bc=Flat())
    
    # Simuler les itérations pour le mouvement vers la gauche
    iter_left = [xf_base - displacement/2]  # Starting point between base and target
    x_curr = iter_left[1]
    
    for i in 1:max_iter
        if x_curr < minimum(left_samples) || x_curr > maximum(left_samples)
            break
        end
        
        x_next = itp_left(x_curr)
        push!(iter_left, x_next)
        
        # Vérifier la convergence
        if abs(x_next - x_curr) < 1e-6
            break
        end
        
        x_curr = x_next
    end
    
    # Simuler les itérations pour le mouvement vers la droite
    iter_right = [xf_base + displacement/2]  # Starting point between base and target
    x_curr = iter_right[1]
    
    for i in 1:max_iter
        if x_curr < minimum(right_samples) || x_curr > maximum(right_samples)
            break
        end
        
        x_next = itp_right(x_curr)
        push!(iter_right, x_next)
        
        # Vérifier la convergence
        if abs(x_next - x_curr) < 1e-6
            break
        end
        
        x_curr = x_next
    end
    
    # Tracer les itérations
    scatter!(ax_iter, 0:length(iter_left)-1, iter_left, markersize=8, color=:blue, 
            label="Itérations (gauche)")
    lines!(ax_iter, 0:length(iter_left)-1, iter_left, linewidth=2, color=:blue)
    
    scatter!(ax_iter, 0:length(iter_right)-1, iter_right, markersize=8, color=:red,
            label="Itérations (droite)")
    lines!(ax_iter, 0:length(iter_right)-1, iter_right, linewidth=2, color=:red)
    
    # Ajouter les lignes de point fixe si elles existent
    if !isempty(fixed_left)
        hlines!(ax_iter, [fixed_left[1]], color=:blue, linestyle=:dash, linewidth=2,
               label="Point fixe (gauche)")
    end
    
    if !isempty(fixed_right)
        hlines!(ax_iter, [fixed_right[1]], color=:red, linestyle=:dash, linewidth=2,
               label="Point fixe (droite)")
    end
    
    # Ajouter les légendes
    axislegend(ax_fx, position=:lt)
    axislegend(ax_slopes, position=:lt)
    axislegend(ax_res, position=:lt)
    axislegend(ax_iter, position=:lt)
    
    # Ajouter un résumé et des recommandations
    if L_left < 1.0 && L_right < 1.0
        status = "Les deux directions convergent (L < 1)"
        color_status = :green
    elseif L_left < 1.0
        status = "Seul le mouvement vers la gauche converge"
        color_status = :orange
    elseif L_right < 1.0
        status = "Seul le mouvement vers la droite converge"
        color_status = :orange
    else
        status = "Aucune direction ne converge (L > 1)"
        color_status = :red
    end
    
    summary_text = "Analyse du déplacement de l'interface - nx: $mesh_size, Δt: $Δt, α: $alpha\n" *
                  "Mouvement gauche: L = $(round(L_left, digits=3)), " *
                  "Mouvement droite: L = $(round(L_right, digits=3))\n" *
                  "Statut: $status"
    
    Label(fig[5, 1:2], summary_text, fontsize=16, color=color_status)
    
    # Enregistrer la figure
    save("interface_direction_comparison.png", fig)
    
    return fig, (L_left, L_right), (fixed_left, fixed_right), (iter_left, iter_right)
end

"""
Fonction utilitaire pour trouver les points fixes dans une séquence de points
"""
function find_fixed_points(x_values, f_values)
    fixed_points = Float64[]
    
    # Recherche directe des points fixes
    for i in 1:length(x_values)
        if abs(f_values[i] - x_values[i]) < 1e-6
            push!(fixed_points, x_values[i])
        end
    end
    
    # Utiliser l'interpolation pour trouver plus précisément les points fixes
    if isempty(fixed_points)
        # Créer une fonction g(x) = f(x) - x
        g_vals = f_values - x_values
        
        # Chercher les changements de signe dans g(x)
        for i in 1:length(g_vals)-1
            if g_vals[i] * g_vals[i+1] <= 0
                # Changement de signe détecté, approximer le point fixe par interpolation linéaire
                x1, x2 = x_values[i], x_values[i+1]
                g1, g2 = g_vals[i], g_vals[i+1]
                
                # Interpolation linéaire pour trouver où g(x) = 0
                x_fixed = x1 - g1 * (x2 - x1) / (g2 - g1)
                push!(fixed_points, x_fixed)
            end
        end
    end
    
    return fixed_points
end

"""
Fonction utilitaire pour évaluer la fonction de point fixe pour une position donnée
"""
function evaluate_fixed_point_function(xf, mesh_size, Δt, alpha)
    # Définir les paramètres du domaine
    nx = mesh_size
    lx = 1.0
    x0 = 0.0
    mesh = Penguin.Mesh((nx,), (lx,), (x0,))
    
    # Définir le maillage espace-temps
    STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)
    
    # Définir la fonction de corps pour la position actuelle
    body = (x, t, _=0) -> (x - xf)
    
    # Définir la capacité, l'opérateur, etc.
    capacity = Capacity(body, STmesh)
    operator = DiffusionOps(capacity)
    f = (x, y, z, t) -> 0.0  # Terme source
    K = (x, y, z) -> 1.0     # Coefficient de diffusion
    
    # Définir la phase
    phase = Phase(capacity, operator, f, K)
    
    # Définir les conditions aux limites
    bc = Dirichlet(0.0)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
    ρL = 1.0  # Paramètre de chaleur latente
    stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρL))
    
    # Condition initiale
    u0ₒ = zeros(nx+1)
    u0ᵧ = zeros(nx+1)
    u0 = vcat(u0ₒ, u0ᵧ)
    
    # Définir le solveur
    solver = MovingLiquidDiffusionUnsteadyMono(phase, bc_b, bc, Δt, u0, mesh, "BE")
    
    # Résoudre une itération
    solve_system!(solver, method=Base.:\)
    
    # Obtenir le champ de température
    Tᵢ = solver.x
    
    # Extraire les dimensions et la taille de l'opérateur
    dims = phase.operator.size
    len_dims = length(dims)
    cap_index = len_dims
    
    # Créer les indices 1D ou 2D
    if len_dims == 2
        # Cas 1D
        nx, _ = dims
        n = nx
    elseif len_dims == 3
        # Cas 2D
        nx, ny, _ = dims
        n = nx*ny
    end
    
    # Mettre à jour les volumes / calculer la nouvelle position d'interface
    Vn_1 = phase.capacity.A[cap_index][1:end÷2, 1:end÷2]
    Vn   = phase.capacity.A[cap_index][end÷2+1:end, end÷2+1:end]
    Hₙ   = sum(diag(Vn))
    Hₙ₊₁ = sum(diag(Vn_1))
    
    # Calculer le flux d'interface
    W! = phase.operator.Wꜝ[1:n, 1:n]
    G = phase.operator.G[1:n, 1:n]
    H = phase.operator.H[1:n, 1:n]
    V = phase.operator.V[1:n, 1:n]
    Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
    Id = Id[1:n, 1:n]
    Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
    Interface_term = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
    Interface_term = 1/ρL * sum(Interface_term)
    
    # Calculer le résidu
    res = Hₙ₊₁ - Hₙ - Interface_term
    
    # Appliquer la règle de mise à jour de Newton pour obtenir la position suivante
    #xf_next = xf + alpha * res
    xf_next = Hₙ + alpha * res
    
    return res, xf_next
end

"""
Étudier l'impact de la taille du maillage sur les mouvements droite/gauche
"""
function analyze_mesh_size_direction(mesh_sizes=[20, 40, 80, 160]; 
                                    Δt=0.001, alpha=1.0, 
                                    base_position=0.5, displacement=0.2)
    # Stocker les résultats
    L_left_values = Float64[]
    L_right_values = Float64[]
    
    # Créer la figure
    fig = Figure(resolution=(1200, 800))
    
    # Axes pour les constantes de Lipschitz
    ax_L = Axis(fig[1, 1:2],
               xlabel="Taille du maillage", 
               ylabel="Constante de Lipschitz",
               title="Impact de la taille du maillage sur la stabilité des déplacements gauche/droite")
    
    # Axes pour le ratio des constantes de Lipschitz
    ax_ratio = Axis(fig[2, 1],
                   xlabel="Taille du maillage", 
                   ylabel="Ratio L_gauche / L_droite",
                   title="Asymétrie dans la convergence")
    
    # Axes pour la zone de stabilité
    ax_stability = Axis(fig[2, 2],
                       xlabel="Taille du maillage", 
                       ylabel="Paramètre",
                       title="Zone de stabilité")
    
    # Analyser chaque taille de maillage
    for nx in mesh_sizes
        println("\nAnalyse pour taille de maillage = $nx")
        
        # Analyser les deux directions
        _, (L_left, L_right), _, _ = analyze_interface_direction(
            nx, Δt, alpha,
            xf_base=base_position, 
            displacement=displacement,
            num_samples=100
        )
        
        # Stocker les résultats
        push!(L_left_values, L_left)
        push!(L_right_values, L_right)
    end
    
    # Calculer les ratios
    L_ratios = L_left_values ./ L_right_values
    
    # Tracer les constantes de Lipschitz
    scatter!(ax_L, mesh_sizes, L_left_values, markersize=10, color=:blue, 
            label="Mouvement gauche")
    lines!(ax_L, mesh_sizes, L_left_values, linewidth=2, color=:blue)
    
    scatter!(ax_L, mesh_sizes, L_right_values, markersize=10, color=:red, 
            label="Mouvement droite")
    lines!(ax_L, mesh_sizes, L_right_values, linewidth=2, color=:red)
    
    # Ajouter la ligne L=1
    hlines!(ax_L, [1.0], color=:black, linestyle=:dash, linewidth=2, 
           label="L=1 (seuil de convergence)")
    
    # Tracer les ratios
    scatter!(ax_ratio, mesh_sizes, L_ratios, markersize=10, color=:purple)
    lines!(ax_ratio, mesh_sizes, L_ratios, linewidth=2, color=:purple)
    
    # Ajouter la ligne de symétrie
    hlines!(ax_ratio, [1.0], color=:black, linestyle=:dash, linewidth=2, 
           label="Symétrie parfaite")
    
    # Visualiser la zone de stabilité
    stable_left = L_left_values .< 1.0
    stable_right = L_right_values .< 1.0
    
    mesh_x_repeat = repeat(mesh_sizes, 2)
    direction_y = [fill(0.25, length(mesh_sizes)); fill(0.75, length(mesh_sizes))]
    stability_color = [stable_left; stable_right]
    
    for i in 1:length(mesh_x_repeat)
        color = stability_color[i] ? (:green, 0.7) : (:red, 0.7)
        scatter!(ax_stability, [mesh_x_repeat[i]], [direction_y[i]], 
                markersize=20, color=color)
    end
    
    # Ajouter des étiquettes pour la direction
    text!(ax_stability, "Mouvement gauche", position=(mean(mesh_sizes), 0.25), 
         fontsize=14, align=(:center, :center))
    
    text!(ax_stability, "Mouvement droite", position=(mean(mesh_sizes), 0.75), 
         fontsize=14, align=(:center, :center))
    
    # Ajouter une légende
    axislegend(ax_L, position=:rt)
    axislegend(ax_ratio, position=:rt)
    
    # Ajouter un résumé
    better_direction = if all(L_left_values .< L_right_values)
        "le mouvement vers la gauche"
    elseif all(L_right_values .< L_left_values)
        "le mouvement vers la droite"
    else
        "variable selon la taille du maillage"
    end
    
    mesh_for_convergence_left = mesh_sizes[L_left_values .< 1.0]
    mesh_for_convergence_right = mesh_sizes[L_right_values .< 1.0]
    
    recommendations = "Résumé de l'analyse direction/maillage:\n" *
                     "- Direction plus stable: $better_direction\n" *
                     "- Maillages pour convergence gauche: $(isempty(mesh_for_convergence_left) ? "aucun" : join(mesh_for_convergence_left, ", "))\n" *
                     "- Maillages pour convergence droite: $(isempty(mesh_for_convergence_right) ? "aucun" : join(mesh_for_convergence_right, ", "))"
    
    Label(fig[3, 1:2], recommendations, fontsize=14)
    
    # Enregistrer la figure
    save("mesh_direction_analysis.png", fig)
    
    return fig, (L_left_values, L_right_values, L_ratios)
end

"""
Étudier l'impact du déplacement de l'interface
"""
function analyze_displacement_magnitude(displacements=[0.05, 0.1, 0.2, 0.3]; 
                                      mesh_size=80, Δt=0.001, alpha=1.0, 
                                      base_position=0.5)
    # Stocker les résultats
    L_left_values = Float64[]
    L_right_values = Float64[]
    
    # Créer la figure
    fig = Figure(resolution=(1200, 800))
    
    # Axes pour les constantes de Lipschitz
    ax_L = Axis(fig[1, 1:2],
               xlabel="Amplitude du déplacement", 
               ylabel="Constante de Lipschitz",
               title="Impact de l'amplitude du déplacement sur la stabilité gauche/droite")
    
    # Axes pour le ratio des constantes de Lipschitz
    ax_ratio = Axis(fig[2, 1],
                   xlabel="Amplitude du déplacement", 
                   ylabel="Ratio L_gauche / L_droite",
                   title="Asymétrie vs amplitude du déplacement")
    
    # Axes pour la zone de stabilité
    ax_stability = Axis(fig[2, 2],
                       xlabel="Amplitude du déplacement", 
                       ylabel="Paramètre",
                       title="Zone de stabilité vs amplitude")
    
    # Analyser chaque amplitude de déplacement
    for d in displacements
        println("\nAnalyse pour amplitude de déplacement = $d")
        
        # Analyser les deux directions
        _, (L_left, L_right), _, _ = analyze_interface_direction(
            mesh_size, Δt, alpha,
            xf_base=base_position, 
            displacement=d,
            num_samples=100
        )
        
        # Stocker les résultats
        push!(L_left_values, L_left)
        push!(L_right_values, L_right)
    end
    
    # Calculer les ratios
    L_ratios = L_left_values ./ L_right_values
    
    # Tracer les constantes de Lipschitz
    scatter!(ax_L, displacements, L_left_values, markersize=10, color=:blue, 
            label="Mouvement gauche")
    lines!(ax_L, displacements, L_left_values, linewidth=2, color=:blue)
    
    scatter!(ax_L, displacements, L_right_values, markersize=10, color=:red, 
            label="Mouvement droite")
    lines!(ax_L, displacements, L_right_values, linewidth=2, color=:red)
    
    # Ajouter la ligne L=1
    hlines!(ax_L, [1.0], color=:black, linestyle=:dash, linewidth=2, 
           label="L=1 (seuil de convergence)")
    
    # Tracer les ratios
    scatter!(ax_ratio, displacements, L_ratios, markersize=10, color=:purple)
    lines!(ax_ratio, displacements, L_ratios, linewidth=2, color=:purple)
    
    # Ajouter la ligne de symétrie
    hlines!(ax_ratio, [1.0], color=:black, linestyle=:dash, linewidth=2, 
           label="Symétrie parfaite")
    
    # Visualiser la zone de stabilité
    stable_left = L_left_values .< 1.0
    stable_right = L_right_values .< 1.0
    
    disp_x_repeat = repeat(displacements, 2)
    direction_y = [fill(0.25, length(displacements)); fill(0.75, length(displacements))]
    stability_color = [stable_left; stable_right]
    
    for i in 1:length(disp_x_repeat)
        color = stability_color[i] ? (:green, 0.7) : (:red, 0.7)
        scatter!(ax_stability, [disp_x_repeat[i]], [direction_y[i]], 
                markersize=20, color=color)
    end
    
    # Ajouter des étiquettes pour la direction
    text!(ax_stability, "Mouvement gauche", position=(mean(displacements), 0.25), 
         fontsize=14, align=(:center, :center))
    
    text!(ax_stability, "Mouvement droite", position=(mean(displacements), 0.75), 
         fontsize=14, align=(:center, :center))
    
    # Ajouter une légende
    axislegend(ax_L, position=:rt)
    axislegend(ax_ratio, position=:rt)
    
    # Ajouter un résumé
    max_stable_left = displacements[findall(stable_left)]
    max_stable_right = displacements[findall(stable_right)]
    
    max_stable_disp_left = isempty(max_stable_left) ? "aucun" : "$(maximum(max_stable_left))"
    max_stable_disp_right = isempty(max_stable_right) ? "aucun" : "$(maximum(max_stable_right))"
    
    recommendations = "Résumé de l'analyse d'amplitude:\n" *
                     "- Déplacement stable maximal vers la gauche: $max_stable_disp_left\n" *
                     "- Déplacement stable maximal vers la droite: $max_stable_disp_right\n" *
                     "- Ratio L_gauche/L_droite: $(round(mean(L_ratios), digits=2)) en moyenne"
    
    Label(fig[3, 1:2], recommendations, fontsize=14)
    
    # Enregistrer la figure
    save("displacement_magnitude_analysis.png", fig)
    
    return fig, (L_left_values, L_right_values, L_ratios)
end

"""
Dashboard complet pour l'étude de l'asymétrie directionnelle
"""
function interface_direction_dashboard()
    # Paramètres de base
    mesh_size_base = 80
    Δt_base = 0.001
    alpha_base = 1.0
    
    # Créer une figure pour le dashboard
    fig = Figure(resolution=(1400, 1000))
    
    # Ajouter un titre
    Label(fig[1, 1:2], "Analyse de l'asymétrie directionnelle\nDéplacement de l'interface dans le problème de Stefan", 
          fontsize=24, font=:bold)
    
    # 1. Analyser un cas de base
    println("\n=== Analyse du cas de base ===")
    _, (L_left_base, L_right_base), (fixed_left, fixed_right), _ = analyze_interface_direction(
        mesh_size_base, Δt_base, alpha_base,
        xf_base=0.5, displacement=0.1
    )
    
    # 2. Analyser l'effet de la taille du maillage
    println("\n=== Analyse de l'effet de la taille du maillage ===")
    mesh_sizes = [20, 40, 80, 160]
    L_left_mesh = Float64[]
    L_right_mesh = Float64[]
    
    for nx in mesh_sizes
        _, (L_left, L_right), _, _ = analyze_interface_direction(
            nx, Δt_base, alpha_base,
            xf_base=0.5, displacement=0.1,
            num_samples=50  # Réduire pour accélérer
        )
        push!(L_left_mesh, L_left)
        push!(L_right_mesh, L_right)
    end
    
    # 3. Analyser l'effet du pas de temps
    println("\n=== Analyse de l'effet du pas de temps ===")
    time_steps = [0.0005, 0.001, 0.002, 0.005]
    L_left_dt = Float64[]
    L_right_dt = Float64[]
    
    for dt in time_steps
        _, (L_left, L_right), _, _ = analyze_interface_direction(
            mesh_size_base, dt, alpha_base,
            xf_base=0.5, displacement=0.1,
            num_samples=50  # Réduire pour accélérer
        )
        push!(L_left_dt, L_left)
        push!(L_right_dt, L_right)
    end
    
    # 4. Analyser l'effet du paramètre de relaxation alpha
    println("\n=== Analyse de l'effet du paramètre de relaxation ===")
    alphas = [0.5, 0.75, 1.0, 1.25, 1.5]
    L_left_alpha = Float64[]
    L_right_alpha = Float64[]
    
    for a in alphas
        _, (L_left, L_right), _, _ = analyze_interface_direction(
            mesh_size_base, Δt_base, a,
            xf_base=0.5, displacement=0.1,
            num_samples=50  # Réduire pour accélérer
        )
        push!(L_left_alpha, L_left)
        push!(L_right_alpha, L_right)
    end
    
    # 5. Analyser l'effet de l'amplitude du déplacement
    println("\n=== Analyse de l'effet de l'amplitude du déplacement ===")
    displacements = [0.05, 0.1, 0.2, 0.3]
    L_left_disp = Float64[]
    L_right_disp = Float64[]
    
    for d in displacements
        _, (L_left, L_right), _, _ = analyze_interface_direction(
            mesh_size_base, Δt_base, alpha_base,
            xf_base=0.5, displacement=d,
            num_samples=50  # Réduire pour accélérer
        )
        push!(L_left_disp, L_left)
        push!(L_right_disp, L_right)
    end
    
    # Panel 1: Effet de la taille du maillage
    ax_mesh = Axis(fig[2, 1], 
                  xlabel="Taille du maillage", 
                  ylabel="Constante de Lipschitz",
                  title="Effet de la taille du maillage")
    
    scatter!(ax_mesh, mesh_sizes, L_left_mesh, markersize=8, color=:blue, 
            label="Gauche")
    lines!(ax_mesh, mesh_sizes, L_left_mesh, linewidth=2, color=:blue)
    
    scatter!(ax_mesh, mesh_sizes, L_right_mesh, markersize=8, color=:red, 
            label="Droite")
    lines!(ax_mesh, mesh_sizes, L_right_mesh, linewidth=2, color=:red)
    
    hlines!(ax_mesh, [1.0], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax_mesh, position=:rt)
    
    # Panel 2: Effet du pas de temps
    ax_dt = Axis(fig[2, 2], 
                xlabel="Pas de temps", 
                ylabel="Constante de Lipschitz",
                title="Effet du pas de temps")
    
    scatter!(ax_dt, time_steps, L_left_dt, markersize=8, color=:blue, 
            label="Gauche")
    lines!(ax_dt, time_steps, L_left_dt, linewidth=2, color=:blue)
    
    scatter!(ax_dt, time_steps, L_right_dt, markersize=8, color=:red, 
            label="Droite")
    lines!(ax_dt, time_steps, L_right_dt, linewidth=2, color=:red)
    
    hlines!(ax_dt, [1.0], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax_dt, position=:rt)
    
    # Panel 3: Effet du paramètre de relaxation alpha
    ax_alpha = Axis(fig[3, 1], 
                   xlabel="Paramètre α", 
                   ylabel="Constante de Lipschitz",
                   title="Effet du paramètre de relaxation")
    
    scatter!(ax_alpha, alphas, L_left_alpha, markersize=8, color=:blue, 
            label="Gauche")
    lines!(ax_alpha, alphas, L_left_alpha, linewidth=2, color=:blue)
    
    scatter!(ax_alpha, alphas, L_right_alpha, markersize=8, color=:red, 
            label="Droite")
    lines!(ax_alpha, alphas, L_right_alpha, linewidth=2, color=:red)
    
    hlines!(ax_alpha, [1.0], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax_alpha, position=:rt)
    
    # Panel 4: Effet de l'amplitude du déplacement
    ax_disp = Axis(fig[3, 2], 
                  xlabel="Amplitude du déplacement", 
                  ylabel="Constante de Lipschitz",
                  title="Effet de l'amplitude du déplacement")
    
    scatter!(ax_disp, displacements, L_left_disp, markersize=8, color=:blue, 
            label="Gauche")
    lines!(ax_disp, displacements, L_left_disp, linewidth=2, color=:blue)
    
    scatter!(ax_disp, displacements, L_right_disp, markersize=8, color=:red, 
            label="Droite")
    lines!(ax_disp, displacements, L_right_disp, linewidth=2, color=:red)
    
    hlines!(ax_disp, [1.0], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax_disp, position=:rt)
    
    # Panel 5: Résumé et recommandations
    has_asymmetry = any(abs.(L_left_mesh./L_right_mesh .- 1) .> 0.1) || 
                   any(abs.(L_left_dt./L_right_dt .- 1) .> 0.1) ||
                   any(abs.(L_left_alpha./L_right_alpha .- 1) .> 0.1) ||
                   any(abs.(L_left_disp./L_right_disp .- 1) .> 0.1)
    
    better_direction = if mean(L_left_mesh) < mean(L_right_mesh) && 
                         mean(L_left_dt) < mean(L_right_dt) &&
                         mean(L_left_alpha) < mean(L_right_alpha) &&
                         mean(L_left_disp) < mean(L_right_disp)
        "le mouvement vers la gauche (interface se rétrécit)"
    elseif mean(L_right_mesh) < mean(L_left_mesh) &&
           mean(L_right_dt) < mean(L_left_dt) &&
           mean(L_right_alpha) < mean(L_left_alpha) &&
           mean(L_right_disp) < mean(L_left_disp)
        "le mouvement vers la droite (interface s'étend)"
    else
        "variable selon les paramètres"
    end
    
    # Trouver les meilleures combinaisons
    best_mesh_left_idx = argmin(L_left_mesh)
    best_mesh_right_idx = argmin(L_right_mesh)
    
    best_dt_left_idx = argmin(L_left_dt)
    best_dt_right_idx = argmin(L_right_dt)
    
    best_alpha_left_idx = argmin(L_left_alpha)
    best_alpha_right_idx = argmin(L_right_alpha)
    
    summary_text = "Résumé de l'analyse directionnelle:\n\n" *
                  "Asymétrie significative: $(has_asymmetry ? "Oui" : "Non")\n" *
                  "Direction généralement plus stable: $better_direction\n\n" *
                  "Paramètres optimaux pour mouvement vers la gauche:\n" *
                  "- Maillage: $(mesh_sizes[best_mesh_left_idx])\n" *
                  "- Pas de temps: $(time_steps[best_dt_left_idx])\n" *
                  "- Paramètre α: $(alphas[best_alpha_left_idx])\n\n" *
                  "Paramètres optimaux pour mouvement vers la droite:\n" *
                  "- Maillage: $(mesh_sizes[best_mesh_right_idx])\n" *
                  "- Pas de temps: $(time_steps[best_dt_right_idx])\n" *
                  "- Paramètre α: $(alphas[best_alpha_right_idx])"
    
    ax_text = Axis(fig[4, 1:2])
    hidedecorations!(ax_text)
    hidespines!(ax_text)
    text!(ax_text, summary_text, position=(0.05, 0.95), fontsize=14, 
         align=(:left, :top))
    
    # Enregistrer le dashboard
    save("interface_direction_dashboard.png", fig)
    
    return fig
end

# Exécuter les analyses
println("=== Analyse de l'impact de la direction du déplacement de l'interface ===")

"""
# 1. Analyser un cas spécifique de déplacement gauche vs droite
fig_dir, lipschitz_values, fixed_points, iterations = analyze_interface_direction(
    80,  # Taille du maillage
    0.001,  # Pas de temps
    1.0,  # Paramètre alpha
    xf_base=0.5,  # Position de base
    displacement=0.1  # Amplitude du déplacement
)

# 2. Étudier l'effet de la taille du maillage sur l'asymétrie directionnelle
fig_mesh, mesh_results = analyze_mesh_size_direction(
    [40, 80, 160],  # Tailles de maillage
    Δt=0.001,
    alpha=1.0,
    base_position=0.05,
    displacement=0.01
)
"""

# 3. Étudier l'effet de l'amplitude du déplacement
fig_disp, disp_results = analyze_displacement_magnitude(
    [0.01, 0.02, 0.05, 0.1],  # Amplitudes de déplacement
    mesh_size=80,
    Δt=0.001,
    alpha=1.0,
    base_position=0.05
)

# 4. Créer un dashboard complet
fig_dashboard = interface_direction_dashboard()

println("\nAnalyse terminée !")
println("Figures enregistrées:")
println("- interface_direction_comparison.png")
println("- mesh_direction_analysis.png")
println("- displacement_magnitude_analysis.png")
println("- interface_direction_dashboard.png")