using Penguin
using CairoMakie
using LibGEOS
using LinearAlgebra
using SparseArrays  # Add for sparse matrix support

"""
Démonstration du modèle d'intercept pour le Front Tracking
Visualise les segments de l'interface et leurs déplacements normaux
"""

# 1. Définir le maillage
nx, ny = 20, 20
lx, ly = 10.0, 10.0
x0, y0 = -5.0, -5.0
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
x_range = mesh.nodes[1]
y_range = mesh.nodes[2]

# 2. Créer l'interface (un cercle)
front = FrontTracker()
radius = 3.0
center_x, center_y = 0.5, 0.0
n_markers = 12  # Utiliser peu de marqueurs pour bien visualiser les segments

create_circle!(front, center_x, center_y, radius, n_markers)

# 3. Calculer les paramètres des segments et la jacobienne
segments, segment_normals, segment_intercepts, segment_lengths, segment_midpoints = 
    compute_segment_parameters(front)

# Afficher des informations pour le débogage
println("Nombre de marqueurs: ", length(get_markers(front)))
println("Nombre de segments: ", length(segments))
println("Nombre de normales: ", length(segment_normals))

jacobian, _, _, _, _ = compute_intercept_jacobian(mesh, front)

# 4. Visualisation
fig = Figure(size=(1500, 1000))

# 4.1 Interface et paramètres des segments
ax1 = Axis(fig[1, 1], title="Interface et paramètres des segments",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Tracer le maillage
for x in x_range
    lines!(ax1, [x, x], [y_range[1], y_range[end]], color=:lightgray, linestyle=:dash, linewidth=0.5)
end
for y in y_range
    lines!(ax1, [x_range[1], x_range[end]], [y, y], color=:lightgray, linestyle=:dash, linewidth=0.5)
end

# Tracer l'interface
markers = get_markers(front)
n_unique_markers = front.is_closed ? length(markers) - 1 : length(markers)
marker_x = [m[1] for m in markers]
marker_y = [m[2] for m in markers]

lines!(ax1, marker_x, marker_y, color=:blue, linewidth=2, label="Interface")
scatter!(ax1, marker_x, marker_y, color=:red, markersize=6, label="Marqueurs")

# Tracer les normales des segments - s'assurer qu'elles sont au milieu des segments
scale_factor = 0.5  # Facteur d'échelle pour la visualisation
for i in 1:length(segments)
    midpoint = segment_midpoints[i]
    normal = segment_normals[i]
    
    # Tracer la normale
    arrows!(ax1, [midpoint[1]], [midpoint[2]], 
           [scale_factor * normal[1]], [scale_factor * normal[2]], 
           color=:green, linewidth=1.5, arrowsize=15)
    
    # Afficher l'indice du segment
    text!(ax1, midpoint[1] + 0.2 * normal[1], midpoint[2] + 0.2 * normal[2], 
         text="$i", fontsize=10, align=(:center, :center))
end

# 4.2 Visualiser la jacobienne
ax2 = Axis(fig[1, 2], title="Jacobienne d'intercept",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Tracer le maillage
for x in x_range
    lines!(ax2, [x, x], [y_range[1], y_range[end]], color=:lightgray, linestyle=:dash, linewidth=0.5)
end
for y in y_range
    lines!(ax2, [x_range[1], x_range[end]], [y, y], color=:lightgray, linestyle=:dash, linewidth=0.5)
end

# Créer une matrice pour visualiser l'intensité de la jacobienne
intensity = zeros(nx, ny)
for i in 1:nx
    for j in 1:ny
        # Somme des contributions de tous les segments qui coupent cette cellule
        total_contribution = 0.0
        for (segment_idx, jac_value) in jacobian[(i,j)]
            total_contribution += abs(jac_value)
        end
        intensity[i, j] = total_contribution
    end
end

# Visualiser l'intensité par un heatmap
hm = heatmap!(ax2, x_range, y_range, intensity, colormap=:viridis, alpha=0.7)
Colorbar(fig[1, 3], hm, label="Somme des longueurs d'intersection")

# Tracer l'interface sur le heatmap
lines!(ax2, marker_x, marker_y, color=:white, linewidth=2, label="Interface")
scatter!(ax2, marker_x, marker_y, color=:red, markersize=4, label="Marqueurs")

# 4.3 Visualiser l'effet d'une perturbation
# Choisir un segment à perturber
segment_to_perturb = 3
perturbation_size = 0.5

# Créer une copie de l'interface avec le segment perturbé
perturbed_markers = copy(markers)
for i in 1:length(perturbed_markers)
    # Trouver les segments qui utilisent ce marqueur
    for (s_idx, (start_idx, end_idx)) in enumerate(segments)
        if s_idx == segment_to_perturb && (i == start_idx || i == end_idx)
            # Appliquer la perturbation le long de la normale
            normal = segment_normals[s_idx]
            perturbed_markers[i] = (
                perturbed_markers[i][1] + perturbation_size * normal[1],
                perturbed_markers[i][2] + perturbation_size * normal[2]
            )
        end
    end
end

# Créer une nouvelle interface avec les marqueurs perturbés
perturbed_front = FrontTracker(perturbed_markers, front.is_closed)

# 4.4 Utiliser update_front_with_intercept_displacements! pour une comparaison
# Créer un vecteur de perturbation (zéros partout sauf pour le segment à perturber)
perturbation_vector = zeros(length(segments))
perturbation_vector[segment_to_perturb] = perturbation_size

# Créer une copie de l'interface originale
function_perturbed_front = FrontTracker(markers, front.is_closed)

# Appliquer la fonction update_front_with_intercept_displacements!
update_front_with_intercept_displacements!(function_perturbed_front, 
                                         perturbation_vector, 
                                         segment_normals,
                                         segment_lengths)

# Récupérer les marqueurs de l'interface mise à jour
function_perturbed_markers = get_markers(function_perturbed_front)
function_marker_x = [m[1] for m in function_perturbed_markers]
function_marker_y = [m[2] for m in function_perturbed_markers]

# Tracer l'interface originale et les deux interfaces perturbées
ax3 = Axis(fig[2, 1:2], title="Comparaison des méthodes de perturbation",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Tracer le maillage
for x in x_range
    lines!(ax3, [x, x], [y_range[1], y_range[end]], color=:lightgray, linestyle=:dash, linewidth=0.5)
end
for y in y_range
    lines!(ax3, [x_range[1], x_range[end]], [y, y], color=:lightgray, linestyle=:dash, linewidth=0.5)
end

# Tracer l'interface originale
lines!(ax3, marker_x, marker_y, color=:blue, linewidth=2, label="Interface originale")
scatter!(ax3, marker_x, marker_y, color=:blue, markersize=4)

# Tracer l'interface perturbée manuellement
perturbed_marker_x = [m[1] for m in perturbed_markers]
perturbed_marker_y = [m[2] for m in perturbed_markers]
lines!(ax3, perturbed_marker_x, perturbed_marker_y, color=:red, linewidth=2, 
      label="Perturbation manuelle")
scatter!(ax3, perturbed_marker_x, perturbed_marker_y, color=:red, markersize=4)

# Tracer l'interface perturbée avec la fonction
lines!(ax3, function_marker_x, function_marker_y, color=:green, linewidth=2, 
      label="Avec update_front_with_intercept_displacements!")
scatter!(ax3, function_marker_x, function_marker_y, color=:green, markersize=4)

# Mettre en évidence le segment perturbé
midpoint = segment_midpoints[segment_to_perturb]
normal = segment_normals[segment_to_perturb]
arrows!(ax3, [midpoint[1]], [midpoint[2]], 
       [scale_factor * normal[1]], [scale_factor * normal[2]], 
       color=:darkgreen, linewidth=2, arrowsize=20)

text!(ax3, midpoint[1] + 0.3 * normal[1], midpoint[2] + 0.3 * normal[2], 
     text="$segment_to_perturb", fontsize=12, align=(:center, :center))

# 4.5 Ajouter un zoom sur la région perturbée
ax4 = Axis(fig[2, 3], title="Zoom sur la région perturbée",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Calculer la zone de zoom autour du segment perturbé
zoom_center_x = midpoint[1]
zoom_center_y = midpoint[2]
zoom_radius = 2.0  # Rayon de la zone de zoom

# Définir les limites de l'axe pour le zoom
limits!(ax4, 
       zoom_center_x - zoom_radius, zoom_center_x + zoom_radius,
       zoom_center_y - zoom_radius, zoom_center_y + zoom_radius)

# Tracer le maillage dans la zone de zoom
for x in x_range
    if x_range[1] <= x <= x_range[end]
        lines!(ax4, [x, x], [zoom_center_y - zoom_radius, zoom_center_y + zoom_radius], 
              color=:lightgray, linestyle=:dash, linewidth=0.5)
    end
end
for y in y_range
    if y_range[1] <= y <= y_range[end]
        lines!(ax4, [zoom_center_x - zoom_radius, zoom_center_x + zoom_radius], [y, y], 
              color=:lightgray, linestyle=:dash, linewidth=0.5)
    end
end

# Tracer les interfaces dans la zone de zoom
lines!(ax4, marker_x, marker_y, color=:blue, linewidth=2, label="Interface originale")
scatter!(ax4, marker_x, marker_y, color=:blue, markersize=6)

lines!(ax4, perturbed_marker_x, perturbed_marker_y, color=:red, linewidth=2, 
      label="Perturbation manuelle")
scatter!(ax4, perturbed_marker_x, perturbed_marker_y, color=:red, markersize=6)

lines!(ax4, function_marker_x, function_marker_y, color=:green, linewidth=2, 
      label="Avec update_front_with_intercept_displacements!")
scatter!(ax4, function_marker_x, function_marker_y, color=:green, markersize=6)

# Mettre en évidence le segment perturbé dans le zoom
arrows!(ax4, [midpoint[1]], [midpoint[2]], 
       [scale_factor * normal[1]], [scale_factor * normal[2]], 
       color=:darkgreen, linewidth=2, arrowsize=20)

# Légendes
axislegend(ax3, position=:rb)
axislegend(ax4, position=:rb)

# Afficher la figure
display(fig)
save("intercept_model_comparison.png", fig)
println("Visualisation sauvegardée dans 'intercept_model_comparison.png'")