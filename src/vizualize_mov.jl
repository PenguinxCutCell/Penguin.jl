function analyze_convergence_rates_newton(residuals)
    figure = Figure(resolution=(800, 600))
    ax = Axis(figure[1,1], xlabel = "Newton Iterations", ylabel = "Convergence Rate", 
             title = "Convergence Rate Analysis")
    
    sorted_keys = sort(collect(keys(residuals)))
    
    for k in sorted_keys
        if length(residuals[k]) > 2
            # Calculate convergence rates
            rates = [log(residuals[k][i+1]/residuals[k][i])/log(residuals[k][i]/residuals[k][i-1]) 
                    for i in 2:length(residuals[k])-1]
            
            # Plot convergence rates
            scatter!(ax, 2:(length(residuals[k])-1), rates, 
                    label = "Step $k", markersize=8)
            
            # Add horizontal line at y=2 for quadratic convergence reference
            hlines!(ax, [2.0], color=:black, linestyle=:dash, linewidth=1)
            text!(ax, length(residuals[k])-2, 3.2, text="Quadratic convergence")
        end
    end
    
    axislegend(ax, position=:rt)
    return figure
end

function plot_timestep_history(timestep_history)
    times = cumsum([h[1] for h in timestep_history])
    timesteps = [h[1] for h in timestep_history]
    cfls = [h[2] for h in timestep_history]
    
    fig = Figure(resolution=(1000, 600))
    
    ax1 = Axis(fig[1, 1], xlabel="Simulation Time", ylabel="Δt",
              title="Adaptive Time Step Evolution")
    lines!(ax1, times, timesteps, linewidth=2, color=:blue)
    
    ax2 = Axis(fig[2, 1], xlabel="Simulation Time", ylabel="CFL",
              title="CFL Number Evolution")
    lines!(ax2, times, cfls, linewidth=2, color=:red)
    
    return fig
end

"""
    plot_interface_evolution(xf_log=nothing, reconstruct=nothing, mesh=nothing;
                            time_steps=:auto,
                            n_times=8, 
                            specific_times=nothing,
                            y_resolution=100,
                            color_gradient=true,
                            add_markers=false,
                            line_styles=true,
                            legend_position=:rt)

Trace l'évolution de la position de l'interface au cours du temps.

Paramètres:
- `xf_log` : Vecteur contenant les positions discrètes de l'interface (points) à chaque pas de temps
- `reconstruct` : Vecteur de fonctions d'interpolation de l'interface à chaque pas de temps
- `mesh` : Maillage utilisé (nécessaire si reconstruct est fourni)
- `time_steps` : Pas de temps à afficher (:auto, :all, ou vecteur d'indices)
- `n_times` : Nombre de pas de temps à afficher si time_steps=:auto
- `specific_times` : Indices spécifiques des pas de temps à afficher
- `y_resolution` : Nombre de points pour l'échantillonnage en y (si reconstruct est utilisé)
- `color_gradient` : Utiliser un dégradé de couleurs pour représenter la progression temporelle
- `add_markers` : Ajouter des marqueurs sur les courbes d'interface
- `line_styles` : Utiliser différents styles de ligne pour les interfaces (si color_gradient=false)
- `legend_position` : Position de la légende (:rt, :lt, :rb, :lb)
- `aspect_ratio` : Ratio d'aspect du graphique

Retourne:
- Une figure CairoMakie
"""
function plot_interface_evolution(xf_log=nothing, reconstruct=nothing, mesh=nothing;
                                time_steps=:auto,
                                n_times=8, 
                                specific_times=nothing,
                                y_resolution=100,
                                color_gradient=true,
                                add_markers=false,
                                line_styles=true,
                                legend_position=:rt)    
    # Vérifier qu'au moins une source de données est fournie
    if xf_log === nothing && reconstruct === nothing
        error("Au moins un des paramètres xf_log ou reconstruct doit être fourni")
    end
    
    # Déterminer le nombre total de pas de temps disponibles
    if reconstruct !== nothing
        total_steps = length(reconstruct)
        use_reconstruct = true
    else
        total_steps = length(xf_log)
        use_reconstruct = false
    end
    
    # Déterminer quels pas de temps afficher
    if time_steps == :auto
        # Sélectionner n_times pas de temps répartis uniformément
        steps_to_plot = round.(Int, range(1, total_steps, length=min(n_times, total_steps)))
    elseif time_steps == :all
        # Afficher tous les pas de temps
        steps_to_plot = 1:total_steps
    elseif isa(time_steps, Vector)
        # Utiliser les pas de temps spécifiés
        steps_to_plot = filter(t -> 1 <= t <= total_steps, time_steps)
    elseif specific_times !== nothing
        # Utiliser les pas de temps spécifiques fournis
        steps_to_plot = filter(t -> 1 <= t <= total_steps, specific_times)
    else
        # Par défaut, utiliser n_times pas répartis uniformément
        steps_to_plot = round.(Int, range(1, total_steps, length=min(n_times, total_steps)))
    end
    
    # Créer la figure
    fig = Figure(resolution=(900, 600))
    ax = Axis(fig[1, 1], 
              xlabel = "Position x", 
              ylabel = "Position y", 
              title = "Évolution de l'interface au cours du temps")    
    # Configurer les couleurs et styles
    if color_gradient
        colormap = cgrad(:viridis, length(steps_to_plot), categorical=false)
    else
        # Palette de couleurs distinctes
        colors = distinguishable_colors(min(10, length(steps_to_plot)), [RGB(1,1,1), RGB(0,0,0)])
        # Styles de ligne
        styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    end
    
    # Tracer l'interface pour chaque pas de temps sélectionné
    for (idx, step) in enumerate(steps_to_plot)
        if use_reconstruct
            # Utiliser les fonctions d'interpolation reconstruct
            if mesh === nothing
                error("Le paramètre mesh est nécessaire lorsque reconstruct est utilisé")
            end
            
            # Générer les points y pour l'échantillonnage
            y_vals = range(minimum(mesh.nodes[2]), maximum(mesh.nodes[2]), length=y_resolution)
            
            # Évaluer la fonction d'interface à ces points
            s_func = reconstruct[step]
            x_vals = s_func.(y_vals)
        else
            # Utiliser directement les positions discrètes de xf_log
            x_vals = xf_log[step][1:end-1]  # [1:end-1] car c'est ce qui est utilisé dans l'exemple
            
            # Créer un vecteur y correspondant
            n_points = length(x_vals)
            if mesh !== nothing
                # Si le maillage est disponible, utiliser ses coordonnées y
                y_vals = range(minimum(mesh.nodes[2]), maximum(mesh.nodes[2]), length=n_points)
            else
                # Sinon, utiliser une grille normalisée
                y_vals = range(0, 1, length=n_points)
            end
        end
        
        # Déterminer la couleur et le style
        if color_gradient
            color_val = colormap[idx]
            line_style = :solid
        else
            color_idx = mod1(idx, length(colors))
            style_idx = mod1(idx, length(styles))
            color_val = colors[color_idx]
            line_style = line_styles ? styles[style_idx] : :solid
        end
        
        # Tracer la ligne d'interface
        lines!(ax, x_vals, y_vals, 
              color=color_val, 
              linestyle=line_style, 
              linewidth=2, 
              label="Pas $step")
        
        # Ajouter des marqueurs si demandé
        if add_markers
            # Réduire la densité des marqueurs si trop de points
            if length(x_vals) > 20
                stride = max(1, div(length(x_vals), 20))
                scatter!(ax, x_vals[1:stride:end], y_vals[1:stride:end], 
                       color=color_val, 
                       markersize=6)
            else
                scatter!(ax, x_vals, y_vals, 
                       color=color_val, 
                       markersize=6)
            end
        end
    end
    
    # Ajouter une barre de couleur si gradient
    if color_gradient
        cb = Colorbar(fig[1, 2], 
                     colormap=colormap, 
                     limits=(minimum(steps_to_plot), maximum(steps_to_plot)), 
                     label="Pas de temps")
    end
    
    # Ajouter la légende
    if length(steps_to_plot) <= 15  # Légende uniquement si pas trop de courbes
        axislegend(ax, position=legend_position, 
                  nbanks=min(3, ceil(Int, length(steps_to_plot)/5)),
                  framevisible=true)
    end
    
    return fig
end



"""
    plot_newton_residuals(residuals; 
                        time_steps=:auto, 
                        n_times=10,
                        color_mode=:gradient,
                        use_log=true,
                        ylimits=nothing,
                        line_width=2.0,
                        resolution=(1000, 700),
                        fontsize=18)

Trace les résidus des itérations de Newton pour différents pas de temps.

Paramètres:
- `residuals` : Dictionnaire contenant les résidus pour chaque pas de temps
- `time_steps` : `:auto`, `:all`, ou liste d'indices des pas de temps à afficher
- `n_times` : Nombre de pas de temps à afficher si time_steps=:auto
- `color_mode` : Mode de couleur (:gradient, :distinct, :monochrome)
- `use_log` : Utiliser l'échelle logarithmique pour les résidus
- `ylimits` : Limites manuelles de l'axe y (rien pour auto)
- `line_width` : Épaisseur des lignes
- `resolution` : Résolution de la figure (largeur, hauteur)
- `fontsize` : Taille de police

Retourne:
- Une figure CairoMakie
"""
function plot_newton_residuals(residuals; 
                             time_steps=:auto, 
                             n_times=10,
                             color_mode=:gradient,
                             use_log=true,
                             ylimits=nothing,
                             line_width=2.0,
                             resolution=(1000, 700),
                             fontsize=18)
    
    # Créer la figure avec une résolution et taille de police améliorées
    fig = Figure(resolution=resolution, fontsize=fontsize)
    
    # Créer un layout avec de l'espace pour la colorbar si nécessaire
    if color_mode == :gradient
        gl = fig[1, 1] = GridLayout()
        ax = Axis(gl[1, 1],
                xlabel = "Itération de Newton", 
                ylabel = use_log ? "Résidus (log₁₀)" : "Résidus", 
                title = "Convergence des itérations de Newton")
    else
        ax = Axis(fig[1, 1],
                xlabel = "Itération de Newton", 
                ylabel = use_log ? "Résidus (log₁₀)" : "Résidus", 
                title = "Convergence des itérations de Newton")
    end
    
    # Préparer les données
    sorted_keys = sort(collect(keys(residuals)))
    
    # Sélectionner les pas de temps à afficher
    if time_steps == :auto
        # Sélectionner n_times pas de temps uniformément répartis
        if length(sorted_keys) > n_times
            step = max(1, floor(Int, length(sorted_keys) / n_times))
            selected_keys = sorted_keys[1:step:end]
            if length(selected_keys) > n_times
                selected_keys = selected_keys[1:n_times]
            end
        else
            selected_keys = sorted_keys
        end
    elseif time_steps == :all
        selected_keys = sorted_keys
    elseif isa(time_steps, Vector)
        selected_keys = filter(k -> k in time_steps && haskey(residuals, k), time_steps)
    else
        selected_keys = sorted_keys
    end
    
    # Configurer les couleurs en fonction du mode choisi
    if color_mode == :gradient
        colormap = cgrad(:turbo, length(selected_keys), categorical=false)
    elseif color_mode == :distinct
        colors = distinguishable_colors(min(length(selected_keys), 15), 
                                      [RGB(1,1,1), RGB(0,0,0)], 
                                      dropseed=true)
    end
    
    # Tracer les résidus
    min_val = Inf
    max_val = -Inf
    
    for (i, k) in enumerate(selected_keys)
        if haskey(residuals, k) && !isempty(residuals[k])
            # Préparer les données
            res_values = abs.(residuals[k])
            
            if use_log
                # Éviter les problèmes avec les valeurs nulles ou négatives
                res_values = [r > 0 ? log10(r) : -16.0 for r in res_values]
            end
            
            # Mettre à jour les valeurs min/max
            min_val = min(min_val, minimum(res_values))
            max_val = max(max_val, maximum(res_values))
            
            # Déterminer la couleur en fonction du mode
            if color_mode == :gradient
                color_val = colormap[i]
                label = i % 5 == 0 || i == 1 || i == length(selected_keys) ? "Pas $k" : ""
            elseif color_mode == :distinct
                color_val = colors[mod1(i, length(colors))]
                label = "Pas $k"
            else # :monochrome
                color_val = :black
                alpha_val = 0.3 + 0.7 * (i / length(selected_keys))
                label = i % 5 == 0 || i == 1 || i == length(selected_keys) ? "Pas $k" : ""
            end
            
            # Tracer
            if color_mode == :monochrome
                lines!(ax, res_values, color=(color_val, alpha_val), 
                      linewidth=line_width, label=label)
            else
                lines!(ax, res_values, color=color_val, 
                      linewidth=line_width, label=label)
            end
        end
    end
    
    # Ajuster les limites de l'axe y avec une marge de 10%
    if ylimits === nothing
        range_val = max_val - min_val
        margin = 0.1 * range_val
        ax.limits = (nothing, nothing, min_val - margin, max_val + margin)
    else
        ax.limits = (nothing, nothing, ylimits[1], ylimits[2])
    end
    
    
    # Ajouter la colorbar si en mode gradient
    if color_mode == :gradient
        cb = Colorbar(fig[1, 2], 
                    colormap=colormap, 
                    limits=(1, length(selected_keys)), 
                    label="Pas de temps",
                    width=20,
                    ticklabelsize=fontsize-2)
        
        # Ajuster la taille relative de la colorbar
        colsize!(fig.layout, 2, Relative(0.05))
    end
    
    # Ajouter la légende si pas trop de courbes
    if (color_mode != :gradient) || (length(selected_keys) <= 15)
        leg = axislegend(ax, position=:rt, 
                        nbanks=min(2, ceil(Int, length(selected_keys)/8)),
                        framevisible=true,
                        labelsize=fontsize-2,
                        patchsize=(30, 10),
                        margin=(10, 10, 10, 10))
    end
    
    # Ajuster l'espacement
    colgap!(fig.layout, 20)
    
    return fig
end


"""
    analyze_interface_spectrum(reconstruct, mesh, times;
                             window=nothing, 
                             padding_factor=2,
                             n_points=200,
                             plot_amplitude=true,
                             plot_spectrum=true,
                             plot_spectrogram=true)

Analyse spectrale de l'évolution de l'interface au cours du temps.

Paramètres:
- `reconstruct` : Vecteur de fonctions d'interpolation de l'interface
- `mesh` : Maillage utilisé pour les coordonnées
- `times` : Vecteur des temps correspondant à chaque état (si non fourni, utilise indices)
- `window` : Fonction de fenêtrage (hanning, hamming, etc.) ou rien
- `padding_factor` : Facteur de zero-padding pour améliorer la résolution fréquentielle
- `n_points` : Nombre de points pour l'échantillonnage de l'interface
- `plot_amplitude` : Afficher le graphique d'évolution de l'amplitude
- `plot_spectrum` : Afficher le spectre de fréquence
- `plot_spectrogram` : Afficher le spectrogramme (évolution du spectre dans le temps)

Retourne:
- Un dictionnaire contenant les figures et les données d'analyse
"""
function analyze_interface_spectrum(reconstruct, mesh, times=nothing;
                                  window=nothing, 
                                  padding_factor=2,
                                  n_points=200,
                                  plot_amplitude=true,
                                  plot_spectrum=true,
                                  plot_spectrogram=true)
    
    # Générer le vecteur des temps si non fourni
    if times === nothing
        times = collect(1:length(reconstruct))
    elseif length(times) != length(reconstruct)
        error("Le vecteur des temps doit avoir la même longueur que le vecteur reconstruct")
    end
    
    # Définir les coordonnées y pour l'échantillonnage
    y_coords = range(minimum(mesh.nodes[2]), maximum(mesh.nodes[2]), length=n_points)
    
    # Calculer l'amplitude à chaque pas de temps
    amplitude = Float64[]
    interface_positions = Vector{Vector{Float64}}()
    
    for i in 1:length(reconstruct)
        s = reconstruct[i]
        values = s.(y_coords)
        push!(interface_positions, values)
        push!(amplitude, (maximum(values) - minimum(values)) / 2)
    end
    
    # Période d'échantillonnage en temps
    Δt = times[2] - times[1]
    fs = 1/Δt  # Fréquence d'échantillonnage
    
    # Résultats à retourner
    results = Dict{Symbol, Any}()
    results[:amplitude] = amplitude
    results[:times] = times
    results[:interface_positions] = interface_positions
    
    # 1. Tracer l'évolution de l'amplitude au cours du temps
    if plot_amplitude
        fig_amplitude = Figure(resolution=(900, 600))
        ax = Axis(fig_amplitude[1, 1], 
                 xlabel = "Temps", 
                 ylabel = "Amplitude", 
                 title = "Évolution de l'amplitude de l'interface")
        
        lines!(ax, times, amplitude, linewidth=2, color=:blue)
        scatter!(ax, times, amplitude, markersize=4, color=:blue)
        
        results[:fig_amplitude] = fig_amplitude
    end
    
    # 2. Analyse spectrale de l'amplitude
    if plot_spectrum
        # Appliquer une fenêtre (pour réduire les fuites spectrales)
        if window !== nothing
            windowed_amplitude = window(length(amplitude)) .* amplitude
        else
            windowed_amplitude = amplitude
        end
        
        # Méthode 1: Effectuer explicitement le zero-padding
        n_fft = padding_factor * length(windowed_amplitude)
        padded_amplitude = [windowed_amplitude; zeros(n_fft - length(windowed_amplitude))]
        fft_result = fft(padded_amplitude)
        
        # Fréquences correspondantes
        freqs = fftfreq(length(padded_amplitude), fs)
        
        # Ne conserver que les fréquences positives (jusqu'à Nyquist)
        n_half = div(length(freqs), 2) + 1
        freqs = freqs[1:n_half]
        power = abs.(fft_result[1:n_half]).^2 / length(padded_amplitude)
        
        # Identifier les pics principaux
        p = sortperm(power, rev=true)
        top_peaks = p[1:min(5, length(p))]
        
        # Tracer le spectre
        fig_spectrum = Figure(resolution=(900, 600))
        ax = Axis(fig_spectrum[1, 1], 
                 xlabel = "Fréquence", 
                 ylabel = "Puissance spectrale", 
                 title = "Spectre de l'amplitude de l'interface")
        
        lines!(ax, freqs[2:end], power[2:end], linewidth=2, color=:blue)  # Ignorer DC
        
        # Annoter les pics principaux
        for i in top_peaks
            if i > 1  # Ignorer la composante DC
                scatter!(ax, [freqs[i]], [power[i]], markersize=8, color=:red)
                text!(ax, "$(round(freqs[i], digits=4))", position=(freqs[i], power[i]*1.1),
                     fontsize=14, align=(:center, :bottom))
            end
        end
        
        ax.yscale = log10
        
        results[:fig_spectrum] = fig_spectrum
        results[:frequencies] = freqs
        results[:power_spectrum] = power
        results[:dominant_frequencies] = freqs[top_peaks]
    end
    
    # 3. Analyse spectrale de l'évolution de l'interface (spectrogramme)
    if plot_spectrogram && length(interface_positions) > 10
        # Préparer la matrice pour le spectrogramme
        interface_matrix = zeros(length(y_coords), length(interface_positions))
        
        for (i, pos) in enumerate(interface_positions)
            interface_matrix[:, i] = pos
        end
        
        # Soustraire la position moyenne à chaque colonne pour se concentrer sur les fluctuations
        for j in 1:size(interface_matrix, 2)
            interface_matrix[:, j] .-= mean(interface_matrix[:, j])
        end
        
        # Calculer le spectrogramme
        # Coordonnée spatiale (y) en fonction du temps
        fig_spectrogram = Figure(resolution=(1000, 800))
        ax = Axis(fig_spectrogram[1, 1], 
                 xlabel = "Temps", 
                 ylabel = "Coordonnée y", 
                 title = "Évolution de l'interface")
        
        hm = heatmap!(ax, times, y_coords, interface_matrix, colormap=:turbo)
        Colorbar(fig_spectrogram[1, 2], hm, label="Déviation par rapport à la position moyenne")
        
        results[:fig_spectrogram] = fig_spectrogram
    end
    
    # 4. Analyse de la décroissance exponentielle (optionnelle)
    if length(amplitude) > 10
        # Si l'amplitude semble décroître, tenter un fit exponentiel
        if amplitude[end] < 0.8 * amplitude[1]
            # Modèle: A(t) = A₀ * exp(-γ*t)
            # En log: log(A(t)) = log(A₀) - γ*t
            model(t, p) = p[1] .* exp.(-p[2] .* t)
            p0 = [amplitude[1], 0.01]  # Valeurs initiales
            
            try
                fit = curve_fit(model, times, amplitude, p0)
                
                if fit.converged
                    A0 = fit.param[1]
                    gamma = fit.param[2]
                    
                    # Tracer le fit sur le graphique d'amplitude
                    if plot_amplitude
                        t_fit = range(minimum(times), maximum(times), length=100)
                        lines!(results[:fig_amplitude][1, 1], t_fit, model(t_fit, fit.param), 
                              linewidth=2, color=:red, linestyle=:dash,
                              label="Fit: A₀=$(round(A0, digits=4))·exp(-$(round(gamma, digits=4))·t)")
                        
                        axislegend(results[:fig_amplitude][1, 1], position=:rt)
                    end
                    
                    results[:decay_parameters] = (A0=A0, gamma=gamma)
                    results[:half_life] = log(2)/gamma
                    println("Temps de demi-vie estimé: $(round(log(2)/gamma, digits=5))")
                end
            catch e
                println("Impossible d'ajuster un modèle exponentiel: $e")
            end
        end
    end
    
    # 5. Analyse de la longueur d'onde dominante
    if length(interface_positions) > 0
        # Déterminer la longueur d'onde moyenne à partir d'une FFT spatiale
        spatial_fft = zeros(ComplexF64, length(interface_positions[end]))
        
        for pos in interface_positions
            pos_centered = pos .- mean(pos)
            if window !== nothing
                pos_windowed = window(length(pos_centered)) .* pos_centered
            else
                pos_windowed = pos_centered
            end
            # Utilisez la fft régulière sans second argument
            spatial_fft .+= abs.(fft(pos_windowed))
        end
        
        spatial_fft ./= length(interface_positions)
        
        # Fréquences spatiales
        L = maximum(mesh.nodes[2]) - minimum(mesh.nodes[2])  # Taille du domaine en y
        k_values = fftfreq(length(y_coords), n_points/L)
                
        # Ne conserver que les fréquences positives
        half_length = div(length(k_values), 2) + 1
        k_positive = k_values[1:half_length]
        spec_positive = spatial_fft[1:half_length]
        
        # Identifier la longueur d'onde dominante
        p = sortperm(abs.(spec_positive), rev=true)
        dominant_k = k_positive[p[2]]  # p[1] est souvent la composante DC
        dominant_wavelength = 1.0 / dominant_k
        
        # Tracer le spectre spatial
        fig_wavenumber = Figure(resolution=(900, 600))
        ax = Axis(fig_wavenumber[1, 1], 
                 xlabel = "Nombre d'onde (1/longueur)", 
                 ylabel = "Puissance spectrale", 
                 title = "Spectre spatial de l'interface")
        
        lines!(ax, k_positive[2:end], abs.(spec_positive[2:end]), linewidth=2, color=:blue)
        
        # Marquer la fréquence dominante
        scatter!(ax, [dominant_k], [abs(spec_positive[p[2]])], markersize=8, color=:red)
        text!(ax, "λ = $(round(dominant_wavelength, digits=3))", 
             position=(dominant_k, abs(spec_positive[p[2]])*1.1),
             fontsize=14, align=(:center, :bottom))
        
        ax.yscale = log10
        
        results[:fig_wavenumber] = fig_wavenumber
        results[:dominant_wavelength] = dominant_wavelength
        println("Longueur d'onde dominante: $(round(dominant_wavelength, digits=5))")
    end
    
    return results
end
