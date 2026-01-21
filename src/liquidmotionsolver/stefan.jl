function compute_volume_jacobian(mesh::Penguin.Mesh{2}, front::FrontTracker, epsilon::Float64=1e-6)
    # Extract mesh data
    x_faces = vcat(mesh.nodes[1][1], mesh.nodes[1][2:end])
    y_faces = vcat(mesh.nodes[2][1], mesh.nodes[2][2:end])
    
    # Call Julia function directly
    return FrontCutTracking.compute_volume_jacobian(front, x_faces, y_faces, epsilon)
end

"""
Génère une heatmap des résidus sur la grille 2D avec visualisation des stencils
"""
function plot_residual_heatmap(mesh, residual_values, cells_idx, stencil_centers=nothing, enable_stencil_fusion=false)
    nx, ny = size(mesh.centers[1])[1], size(mesh.centers[2])[1]
    
    # Créer une matrice 2D pour les résidus (initialisation à NaN pour les cellules sans résidu)
    residual_matrix = fill(NaN, nx, ny)
    
    # Remplir avec les valeurs de résidus disponibles
    if enable_stencil_fusion && stencil_centers !== nothing
        # Version fusion stencil
        stencil_indices = collect(keys(stencil_centers))
        for (eq_idx, (i, j)) in enumerate(stencil_indices)
            if 1 <= i <= nx && 1 <= j <= ny
                residual_matrix[i, j] = residual_values[eq_idx]
            end
        end
    else
        # Version standard sans fusion
        for (eq_idx, (i, j)) in enumerate(cells_idx)
            if 1 <= i <= nx && 1 <= j <= ny
                residual_matrix[i, j] = residual_values[eq_idx]
            end
        end
    end
    
    # Créer la figure
    fig = Figure(size=(800, 700))
    
    # Heatmap des résidus (log scale pour mieux voir les variations)
    ax1 = Axis(fig[1, 1], 
              title="Carte des résidus", 
              xlabel="Colonne", ylabel="Rangée",
              aspect=DataAspect())
    
    # Trouver les limites d'échelle qui évitent les NaN
    residual_values_filtered = filter(!isnan, residual_matrix[:])
    if !isempty(residual_values_filtered)
        max_abs_val = maximum(abs.(residual_values_filtered))
        colorrange = (-max_abs_val, max_abs_val)
    else
        colorrange = (-1, 1)  # Valeurs par défaut
    end
    
    # Créer la heatmap avec une échelle de couleur centrée sur zéro
    hm = heatmap!(ax1, 1:nx, 1:ny, residual_matrix, 
                 colormap=:balance,  # Colormap rouge-bleu centré sur zéro
                 colorrange=colorrange,
                 nan_color=:lightgrey)  # Gris clair pour les cellules sans résidu
    
    # Marquer les cellules affectées
    if enable_stencil_fusion && stencil_centers !== nothing
        # Afficher les centres des stencils
        stencil_centers_list = collect(keys(stencil_centers))
        stencil_i = [c[1] for c in stencil_centers_list]
        stencil_j = [c[2] for c in stencil_centers_list]
        scatter!(ax1, stencil_j, stencil_i, 
                color=:black, markersize=4, marker=:circle)
        
        # Afficher les connexions de stencil
        for (i, j) in stencil_centers_list
            for (si, sj) in stencil_centers[(i, j)]
                # Sauter la cellule centrale
                if (si, sj) != (i, j)
                    lines!(ax1, [j, sj], [i, si], 
                          color=:gray, linewidth=0.5, linestyle=:dash, alpha=0.3)
                end
            end
        end
    else
        # Marquer simplement les cellules affectées
        cell_i = [c[1] for c in cells_idx]
        cell_j = [c[2] for c in cells_idx]
        scatter!(ax1, cell_j, cell_i, 
                color=:black, markersize=4, marker=:cross)
    end
    
    # Ajouter une barre de couleur
    Colorbar(fig[1, 2], hm, label="Valeur du résidu")
    
    # Statistiques sur les résidus
    non_nan_values = filter(!isnan, residual_matrix[:])
    if !isempty(non_nan_values)
        max_val = maximum(non_nan_values)
        min_val = minimum(non_nan_values)
        mean_val = mean(non_nan_values)
        std_val = std(non_nan_values)
        
        stats_text = """
        Max: $(round(max_val, digits=6))
        Min: $(round(min_val, digits=6))
        Moyenne: $(round(mean_val, digits=6))
        Écart-type: $(round(std_val, digits=6))
        """
        
        Label(fig[2, 1:2], stats_text, tellwidth=false)
    end
    
    return fig
end
"""
Smooth marker displacements using weighted averaging of neighbors
This helps maintain interface regularity and stability

Parameters:
- displacements: Vector of displacement values to smooth
- markers: Vector of marker positions (Tuple{Float64,Float64})
- is_closed: Boolean indicating if the interface is a closed curve
- smoothing_factor: Weight given to neighbor values (0.0-1.0)
- window_size: Number of neighbors to consider on each side
"""
function smooth_displacements!(displacements::Vector{Float64}, 
                              markers::Vector{Tuple{Float64,Float64}}, 
                              is_closed::Bool=true,
                              smoothing_factor::Float64=0.5,
                              window_size::Int=2)
    n = length(displacements)
    if n <= 1
        return displacements  # Nothing to smooth with single marker
    end
    
    # Create a copy of original displacements
    original_displacements = copy(displacements)
    
    for i in 1:n
        # Calculate weighted sum of neighbors
        neighbor_sum = 0.0
        weight_sum = 0.0
        
        for j in -window_size:window_size
            if j == 0
                continue  # Skip the marker itself
            end
            
            # Handle wrapping for closed curves
            idx = i + j
            if is_closed
                # Apply modulo to wrap around for closed curves
                idx = mod1(idx, n)
            else
                # Skip out of bounds indices for open curves
                if idx < 1 || idx > n
                    continue
                end
            end
            
            # Calculate weight based on distance (closer markers have higher weight)
            distance = sqrt((markers[i][1] - markers[idx][1])^2 + 
                           (markers[i][2] - markers[idx][2])^2)
            weight = 1.0 / (distance + 1e-10)  # Avoid division by zero
            
            # Add weighted contribution
            neighbor_sum += weight * original_displacements[idx]
            weight_sum += weight
        end
        
        # Calculate weighted average
        if weight_sum > 0
            neighbor_avg = neighbor_sum / weight_sum
            
            # Apply smoothing with blend factor
            displacements[i] = (1.0 - smoothing_factor) * original_displacements[i] + 
                               smoothing_factor * neighbor_avg
        end
    end
    
    return displacements
end

function compute_geometric_segment_displacements(front::FrontTracker,
                                                 mesh::Penguin.Mesh{2},
                                                 Vₙ::AbstractMatrix{<:Real},
                                                 Vₙ₊₁::AbstractMatrix{<:Real},
                                                 interface_flux::AbstractMatrix{<:Real},
                                                 ρL::Real,
                                                 α::Real)
    intercept_jacobian, segments, _segment_normals, _segment_intercepts, segment_lengths =
        compute_intercept_jacobian(mesh.nodes, front; density=1.0)

    cells_idx = Tuple{Int, Int}[]
    residual_values = Float64[]
    segment_accumulator = zeros(Float64, length(segments))
    segment_weights = zeros(Float64, length(segments))

    ρL_val = Float64(ρL)
    α_val = Float64(α)

    for (cell, entries) in intercept_jacobian
        if isempty(entries)
            continue
        end

        i, j = cell
        volume_change = Float64(Vₙ₊₁[i, j] - Vₙ[i, j])
        flux = Float64(interface_flux[i, j])
        residual = ρL_val * volume_change - flux

        push!(cells_idx, cell)
        push!(residual_values, residual)

        # Compute total length of all segment pieces in this cell
        total_length = 0.0
        for (_, length) in entries
            total_length += max(Float64(length), 1e-12)
        end

        if total_length <= 1e-12
            continue
        end

        # Normalized displacement for this cell: D_{i,j} = residual / (ρL * L_interface)
        # Units: [J] / ([J/m²] * [m]) = [J] / [J/m] = [m]
        # This is the displacement needed for the interface in this cell
        D_cell = residual / (ρL_val * total_length)

        # Each segment piece in this cell contributes this displacement, weighted by its length
        for (segment_idx, length) in entries
            segment_length = max(Float64(length), 1e-12)
            # Weight the displacement contribution by the segment piece length
            # The displacement itself is D_cell, weighted by how much of the segment is in this cell
            segment_accumulator[segment_idx] += D_cell * segment_length
            segment_weights[segment_idx] += segment_length
        end
    end

    # Average the contributions to each segment weighted by intersection lengths
    # dJ = (|CD|*D_{i,j+1} + |DE|*D_{i,j}) / (|CD| + |DE|)
    segment_displacements = zeros(Float64, length(segments))
    for idx in eachindex(segment_displacements)
        if segment_weights[idx] > 0
            segment_displacements[idx] = α_val * (segment_accumulator[idx] / segment_weights[idx])
        end
    end

    return segment_displacements, segment_lengths, cells_idx, residual_values
end

function smooth_segment_displacements!(segment_displacements::Vector{Float64},
                                      segment_lengths::Vector{Float64},
                                      is_closed::Bool;
                                      iterations::Int=1)
    n = length(segment_displacements)
    if n == 0 || iterations <= 0
        return segment_displacements
    end

    buffer = similar(segment_displacements)

    for _ in 1:iterations
        for i in 1:n
            prev = i - 1
            next = i + 1

            if is_closed
                prev = prev < 1 ? n : prev
                next = next > n ? 1 : next
            else
                prev = max(prev, 1)
                next = min(next, n)
            end

            len_prev = prev == i ? 0.0 : max(segment_lengths[prev], 1e-12)
            len_self = max(segment_lengths[i], 1e-12)
            len_next = next == i ? 0.0 : max(segment_lengths[next], 1e-12)

            weight_sum = len_prev + len_self + len_next

            value = len_self * segment_displacements[i]
            if prev != i
                value += len_prev * segment_displacements[prev]
            end
            if next != i
                value += len_next * segment_displacements[next]
            end

            buffer[i] = weight_sum > 0 ? value / weight_sum : segment_displacements[i]
        end

        segment_displacements .= buffer
    end

    return segment_displacements
end

function segment_to_marker_displacements(segment_displacements::Vector{Float64},
                                         segment_lengths::Vector{Float64},
                                         markers::AbstractVector{<:Tuple{Float64,Float64}},
                                         is_closed::Bool)
    n_markers = length(markers)
    n_segments = length(segment_displacements)
    displacements = zeros(Float64, n_markers)

    for i in 1:n_markers
        prev_seg = i - 1
        if prev_seg < 1
            prev_seg = is_closed ? n_segments : 0
        end

        next_seg = i
        if next_seg > n_segments
            next_seg = is_closed ? 1 : 0
        end

        numerator = 0.0
        denominator = 0.0

        # Marker displacement is weighted by INVERSE segment lengths
        # s_γ = (dJ/|CE| + dK/|EH|) / (1/|CE| + 1/|EH|)
        if prev_seg > 0 && segment_lengths[prev_seg] > 1e-14
            len_prev = segment_lengths[prev_seg]
            inv_len_prev = 1.0 / len_prev
            numerator += segment_displacements[prev_seg] * inv_len_prev
            denominator += inv_len_prev
        end

        if next_seg > 0 && segment_lengths[next_seg] > 1e-14
            len_next = segment_lengths[next_seg]
            inv_len_next = 1.0 / len_next
            numerator += segment_displacements[next_seg] * inv_len_next
            denominator += inv_len_next
        end

        if denominator > 0
            displacements[i] = numerator / denominator
        else
            displacements[i] = 0.0
        end
    end

    return displacements
end

function StefanMono2D(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Stefan problem")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, [], [])    
    if scheme == "CN"
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "CN")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "CN")
    else # BE
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "BE")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "BE")
    end
    
    BC_border_mono!(s.A, s.b, bc_b, mesh; t=0.0)
    
    s.x = Tᵢ
    return s
end

function solve_StefanMono2D!(s::Solver, phase::Phase, front::FrontTracker, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh::Penguin.Mesh{2}, scheme::String; 
                            method=Base.:\,
                            Newton_params=(100, 1e-6, 1e-6, 1.0),
                            jacobian_epsilon=1e-6, smooth_factor=0.5, window_size=10,
                            gmorlm="LM", # "GN" pour Gauss-Newton ou "LM" pour Levenberg-Marquardt
                            lm_init_lambda=1e-4, # Paramètre d'amortissement initial pour LM
                            lm_lambda_factor=10.0, # Facteur de multiplication/division pour lambda
                            lm_min_lambda=1e-10, # Lambda minimum
                            lm_max_lambda=1e6, # Lambda maximum
                            enable_stencil_fusion=true, # Activer/désactiver la fusion des résidus
                            stencil_weights=nothing, # Poids optionnels pour le stencil (sinon uniforme)
                            fusion_strategy="3x3", # Nouvelle option: "3x3", "5x5"
                            algorithm=nothing,
                            kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving Stefan problem with Front Tracking (Julia implementation):")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Unpack parameters
    max_iter = Newton_params[1]
    tol = Newton_params[2]
    reltol = Newton_params[3]
    α = Newton_params[4]
    
    # Extract interface flux parameter
    ρL = ic.flux.value
    
    # Initialize tracking variables
    t = Tₛ
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))
    position_increments = Dict{Int, Vector{Float64}}()  # Nouveau dictionnaire pour les incréments de position

    
    # Determine how many dimensions
    dims = phase.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Create the 1D or 2D indices
    nx, ny, _ = dims
    n = nx * ny  # Total number of nodes in the mesh
    
    # Store initial state
    Tᵢ = s.x
    push!(s.states, s.x)
    
    # Get initial interface markers
    markers = get_markers(front)
    
    # Store initial interface position
    xf_log[1] = markers
    
    # Afficher l'état de la fusion des résidus et la stratégie
    if enable_stencil_fusion
        println("Résidus fusion activée: stratégie $fusion_strategy")
    else
        println("Résidus fusion désactivée: résolution standard")
    end
    
    # Définir les poids du stencil selon la stratégie
    if stencil_weights === nothing
        if fusion_strategy == "3x3"
            # Stencil 3x3 par défaut (uniforme ou pondéré)
            stencil_weights = fill(1.0/9, 3, 3)  # Uniforme
        elseif fusion_strategy == "5x5"
            # Stencil 5x5 uniforme
            stencil_weights = fill(1.0/25, 5, 5)  # Uniforme
        end
    end
    
    # Définir une fonction pour obtenir les cellules du stencil
    function get_stencil_cells(i, j, nx, ny, fusion_strategy)
        stencil_cells = []
        
        if fusion_strategy == "5x5"
            # Stencil 5x5: cellule courante + voisinage plus étendu 
            for di in -2:2
                for dj in -2:2
                    si = i + di
                    sj = j + dj
                    if 1 <= si <= nx && 1 <= sj <= ny
                        push!(stencil_cells, (si, sj))
                    end
                end
            end
        else
            # Stencil 3x3 standard (par défaut)
            for di in -1:1
                for dj in -1:1
                    si = i + di
                    sj = j + dj
                    if 1 <= si <= nx && 1 <= sj <= ny
                        push!(stencil_cells, (si, sj))
                    end
                end
            end
        end
        
        return stencil_cells
    end

    # Boucle principale pour tous les pas de temps
    timestep = 1
    phase_2d = phase  # Initialize phase_2d for later use
    last_displacements = nothing  # Stocker les déplacements du pas de temps précédent

    while t < Tₑ
        # Affichage du pas de temps actuel
        if timestep == 1
            println("\nFirst time step: t = $(round(t, digits=6))")
        else
            println("\nTime step $(timestep), t = $(round(t, digits=6))")
        end

        # Get current markers and calculate normals
        markers = get_markers(front)
        normals = compute_marker_normals(front, markers)
        
        # Si nous avons des déplacements du pas précédent, initialisons l'interface
        if last_displacements !== nothing && timestep > 1
            # Calculer un facteur d'extrapolation (peut être ajusté pour stabilité)
            extrapolation_factor = 0.8  # Valeur prudente inférieure à 1
            
            println("Initializing interface using previous displacements (extrapolation factor: $extrapolation_factor)")
            
            # Nombre de marqueurs (en excluant le point dupliqué si l'interface est fermée)
            n_markers = length(markers) - (front.is_closed ? 1 : 0)
            
            # Créer des nouveaux marqueurs extrapolés
            new_markers = copy(markers)
            for i in 1:n_markers
                # Appliquer les déplacements précédents avec le facteur d'extrapolation
                normal = normals[i]
                new_markers[i] = (
                    markers[i][1] + extrapolation_factor * last_displacements[i] * normal[1],
                    markers[i][2] + extrapolation_factor * last_displacements[i] * normal[2]
                )
            end
            
            # Si l'interface est fermée, mettre à jour le marqueur dupliqué
            if front.is_closed
                new_markers[end] = new_markers[1]
            end
            
            # Mettre à jour le front avec les marqueurs extrapolés
            set_markers!(front, new_markers)
            
            # Recalculer les normales avec la nouvelle position
            markers = get_markers(front)
            normals = compute_marker_normals(front, markers)
        end
        
        # Update time for this step
        t += Δt
        tₙ = t - Δt
        tₙ₊₁ = t
        time_interval = [tₙ, tₙ₊₁]
        
        # Calculate total number of markers (excluding duplicated closing point if interface is closed)
        n_markers = length(markers) - (front.is_closed ? 1 : 0)
        
        # Initialize displacement vector and residual vector
        displacements = zeros(n_markers)
        residual_norm_history = Float64[]
        position_increment_history = Float64[]
        
        # Variables pour Levenberg-Marquardt
        lambda = lm_init_lambda
        prev_residual_norm = Inf

        # Gauss-Newton iterations
        for iter in 1:max_iter
            # 1. Solve temperature field with current interface position
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            Tᵢ = s.x
            
            # Get capacity matrices
            V_matrices = phase.capacity.A[cap_index]
            Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]
            Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]
            Vₙ₊₁_matrix = diag(Vₙ₊₁_matrix)  # Convert to diagonal matrix for easier handling
            Vₙ_matrix = diag(Vₙ_matrix)  # Convert to diagonal matrix for easier handling
            Vₙ₊₁_matrix = reshape(Vₙ₊₁_matrix, (nx, ny))  # Reshape to 2D matrix
            Vₙ_matrix = reshape(Vₙ_matrix, (nx, ny))  # Reshape to 2D matrix
            
            # 2. Calculate the interface flux
            W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
            G = phase.operator.G[1:end÷2, 1:end÷2]
            H = phase.operator.H[1:end÷2, 1:end÷2]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:end÷2, 1:end÷2]  # Adjust for 2D case
            
            Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
            interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            
            # Reshape to get flux per cell - IMPORTANT: use the same reshape consistently
            interface_flux_2d = reshape(interface_flux, (nx, ny))

            # Compute volume Jacobian for the mesh
            volume_jacobian = compute_volume_jacobian(mesh, front, jacobian_epsilon)
            
            # 3. Build least squares system - Modified for stencil fusion
            cells_idx = []
            
            # Precompute affected cells and their indices for residual vector
            for i in 1:nx
                for j in 1:ny
                    if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                        push!(cells_idx, (i, j))
                    end
                end
            end

            # Si la fusion par stencil est activée
            if enable_stencil_fusion
                # Dictionnaire pour stocker les stencils centrés sur chaque cellule
                stencil_centers = Dict()
                
                # Pour chaque cellule affectée, identifier les fresh/dead cells si nécessaire
                fresh_cells = []
                dead_cells = []
                
                if fusion_strategy == "fresh_dead"
                    # Identifier les fresh et dead cells
                    for (i, j) in cells_idx
                        volume_new = Vₙ₊₁_matrix[i, j]
                        volume_old = Vₙ_matrix[i, j]
                        
                        # Fresh cell: n'existait pas avant mais existe maintenant
                        if volume_old < 1e-10 && volume_new > 1e-10
                            push!(fresh_cells, (i, j))
                        end
                        
                        # Dead cell: existait avant mais n'existe plus maintenant
                        if volume_old > 1e-10 && volume_new < 1e-10
                            push!(dead_cells, (i, j))
                        end
                    end
                    
                    # N'utiliser que les fresh/dead cells comme centres de stencil
                    for cell in vcat(fresh_cells, dead_cells)
                        stencil_centers[cell] = get_stencil_cells(cell[1], cell[2], nx, ny, "3x3")
                    end
                    
                    println("Nombre de fresh cells: $(length(fresh_cells))")
                    println("Nombre de dead cells: $(length(dead_cells))")
                else
                    # Pour les autres stratégies, créer un stencil pour chaque cellule affectée
                    for (i, j) in cells_idx
                        stencil_centers[(i, j)] = get_stencil_cells(i, j, nx, ny, fusion_strategy)
                    end
                end
                
                println("Nombre de cellules affectées: $(length(cells_idx))")
                println("Nombre de stencils: $(length(stencil_centers))")
            
                
                # Reconstruire les indices des équations basés sur les stencils
                row_indices = Int[]
                col_indices = Int[]
                values = Float64[]
                stencil_indices = collect(keys(stencil_centers))
                # Construire la matrice jacobienne pour chaque stencil
                for (eq_idx, center_idx) in enumerate(stencil_indices)
                    # Pour chaque cellule dans ce stencil
                    i_center, j_center = center_idx
                    
                    # Pour chaque marqueur affectant les cellules du stencil
                    # Commencer par la cellule centrale qui est la plus importante
                    if haskey(volume_jacobian, (i_center, j_center))
                        for (marker_idx, jac_value) in volume_jacobian[(i_center, j_center)]
                            if 0 <= marker_idx < n_markers
                                push!(row_indices, eq_idx)
                                push!(col_indices, marker_idx + 1)  # 1-based indexing
                                
                                # Déterminer l'indice central selon la taille du stencil
                                center_index = fusion_strategy == "5x5" ? 3 : 2
                                
                                # Utiliser un poids pour la cellule centrale
                                push!(values, ρL * jac_value * stencil_weights[center_index, center_index])
                            end
                        end
                    end
                    
                    # Ensuite, parcourir les voisins dans le stencil
                    for (si, sj) in stencil_centers[(i_center, j_center)]
                        # Sauter la cellule centrale qui a déjà été traitée
                        if (si, sj) == (i_center, j_center)
                            continue
                        end
                        
                        if haskey(volume_jacobian, (si, sj))
                            for (marker_idx, jac_value) in volume_jacobian[(si, sj)]
                                if 0 <= marker_idx < n_markers
                                    push!(row_indices, eq_idx)
                                    push!(col_indices, marker_idx + 1)  # 1-based indexing
                                    
                                    # Appliquer les poids du stencil selon la position relative
                                    # Adapter le décalage selon la taille du stencil
                                    if fusion_strategy == "5x5"
                                        # Pour un stencil 5x5, on ajoute 3 pour centrer
                                        stencil_i = si - i_center + 3
                                        stencil_j = sj - j_center + 3
                                    else
                                        # Pour un stencil 3x3, on ajoute 2 comme avant
                                        stencil_i = si - i_center + 2
                                        stencil_j = sj - j_center + 2
                                    end
                                    
                                    weight = stencil_weights[stencil_i, stencil_j]
                                    push!(values, ρL * jac_value * weight)
                                end
                            end
                        end
                    end
                end
                # Créer la matrice jacobienne J pour le système
                m = length(stencil_indices)  # Nombre d'équations (stencils)
                J = sparse(row_indices, col_indices, values, m, n_markers)
                
                # Calculer le vecteur résidu F fusionné
                F = zeros(m)
                mismatches = 0
                total_stencils = 0
                
                # Arrays for diagnostics
                flux_nonzero = zeros(Bool, nx, ny)
                volume_nonzero = zeros(Bool, nx, ny)
                both_nonzero = zeros(Bool, nx, ny)
                
                # Pour chaque stencil, calculer le résidu combiné
                for (eq_idx, center_idx) in enumerate(stencil_indices)
                    total_stencils += 1
                    stencil_volume_change = 0.0
                    stencil_flux = 0.0
                    i_center, j_center = center_idx
                    
                    # Calculer la contribution de la cellule centrale (plus importante)
                    if 1 <= i_center <= nx && 1 <= j_center <= ny
                        volume_change = Vₙ₊₁_matrix[i_center, j_center] - Vₙ_matrix[i_center, j_center]
                        flux = interface_flux_2d[i_center, j_center]
                        
                        # Appliquer le poids central (généralement plus élevé)
                        stencil_volume_change += volume_change * stencil_weights[2, 2]
                        stencil_flux += flux * stencil_weights[2, 2]
                        
                        # Record diagnostics
                        volume_nonzero[i_center, j_center] = abs(volume_change) > 1e-10
                        flux_nonzero[i_center, j_center] = abs(flux) > 1e-10
                        both_nonzero[i_center, j_center] = volume_nonzero[i_center, j_center] && 
                                                          flux_nonzero[i_center, j_center]
                    end
                    
                    # Ajouter la contribution des cellules voisines
                    for (si, sj) in stencil_centers[(i_center, j_center)]
                        # Sauter la cellule centrale déjà traitée
                        if (si, sj) == (i_center, j_center)
                            continue
                        end
                        
                        if 1 <= si <= nx && 1 <= sj <= ny
                            volume_change = Vₙ₊₁_matrix[si, sj] - Vₙ_matrix[si, sj]
                            flux = interface_flux_2d[si, sj]
                            
                            # Appliquer le poids selon la position dans le stencil
                            if fusion_strategy == "5x5"
                                # Pour un stencil 5x5, on ajoute 3 pour centrer
                                stencil_i = si - i_center + 3
                                stencil_j = sj - j_center + 3
                                
                                # Ensure indices are within bounds (1-5)
                                stencil_i = max(1, min(5, stencil_i))
                                stencil_j = max(1, min(5, stencil_j))
                            else
                                # Pour un stencil 3x3, on ajoute 2 comme avant
                                stencil_i = si - i_center + 2
                                stencil_j = sj - j_center + 2
                                
                                # Ensure indices are within bounds (1-3)
                                stencil_i = max(1, min(3, stencil_i))
                                stencil_j = max(1, min(3, stencil_j))
                            end
                            
                            # Now indices are guaranteed to be valid
                            weight = stencil_weights[stencil_i, stencil_j]
                            
                            stencil_volume_change += volume_change * weight
                            stencil_flux += flux * weight
                            
                            # Record diagnostics
                            volume_nonzero[si, sj] = abs(volume_change) > 1e-10
                            flux_nonzero[si, sj] = abs(flux) > 1e-10
                            both_nonzero[si, sj] = volume_nonzero[si, sj] && flux_nonzero[si, sj]
                        end
                    end
                    
                    # F_stencil = ρL * stencil_volume_change - stencil_flux
                    F[eq_idx] = ρL * stencil_volume_change - stencil_flux
                    
                    # Diagnostic pour ce stencil
                    stencil_has_nonzero_vol = abs(stencil_volume_change) > 1e-10
                    stencil_has_nonzero_flux = abs(stencil_flux) > 1e-10
                    
                    if (stencil_has_nonzero_vol && !stencil_has_nonzero_flux) || 
                       (!stencil_has_nonzero_vol && stencil_has_nonzero_flux)
                        mismatches += 1
                        if mismatches <= 5
                            println("Stencil mismatch at $center_idx: volume_change = $stencil_volume_change, flux = $stencil_flux")
                        end
                    end
                end
            else
                # Version standard sans fusion de cellules
                row_indices = Int[]
                col_indices = Int[]
                values = Float64[]
                
                # Number of equations (cells) in our system
                m = length(cells_idx)
                
                # Now build the Jacobian matrix for volume changes
                for (eq_idx, (i, j)) in enumerate(cells_idx)
                    # Handle each marker affecting this cell
                    for (marker_idx, jac_value) in volume_jacobian[(i,j)]
                        if 0 <= marker_idx < n_markers
                            push!(row_indices, eq_idx)
                            push!(col_indices, marker_idx + 1)  # 1-based indexing
                            # Volume Jacobian is multiplied by ρL to match the Stefan condition
                            push!(values, ρL * jac_value)
                        end
                    end
                end
                
                # Create Jacobian matrix J for the system
                J = sparse(row_indices, col_indices, values, m, n_markers)
                
                # Calculate current residual vector F
                F = zeros(m)
                mismatches = 0
                total_cells = 0
                
                # Diagnostic arrays
                flux_nonzero = zeros(Bool, nx, ny)
                volume_nonzero = zeros(Bool, nx, ny)
                both_nonzero = zeros(Bool, nx, ny)
                
                for (eq_idx, (i, j)) in enumerate(cells_idx)
                    # Get the desired volume change based on interface flux
                    volume_change = Vₙ₊₁_matrix[i,j] - Vₙ_matrix[i,j]
                    flux = interface_flux_2d[i,j]
                    
                    # Record diagnostics
                    total_cells += 1
                    volume_nonzero[i,j] = abs(volume_change) > 1e-10
                    flux_nonzero[i,j] = abs(flux) > 1e-10
                    both_nonzero[i,j] = volume_nonzero[i,j] && flux_nonzero[i,j]
                    
                    # Detect mismatches (volume change but no flux or vice versa)
                    if (volume_nonzero[i,j] && !flux_nonzero[i,j]) || (!volume_nonzero[i,j] && flux_nonzero[i,j])
                        mismatches += 1
                        if mismatches <= 5  # Limiter le nombre de messages
                            println("Mismatch at ($i,$j): volume_change = $volume_change, flux = $flux")
                        end
                    end
                    
                    # F_i = ρL * volume_change - interface_flux
                    F[eq_idx] = ρL * volume_change - flux
                end
            end

            results_dir = joinpath(pwd(), "residuals_visualization")
            if !isdir(results_dir)
                mkdir(results_dir)
            end
            
            # Créer la visualisation
            residual_fig = plot_residual_heatmap(mesh, F, cells_idx, 
                                                     enable_stencil_fusion ? stencil_centers : nothing,
                                                     enable_stencil_fusion)

            # Sauvegarder l'image
            timestamp = round(Int, time())
            save(joinpath(results_dir, "residual_timestep$(timestep)_iter$(iter)_$(timestamp).png"), residual_fig)
                
            display(residual_fig)

            
            # Print diagnostic summary
            nonzero_vol = count(volume_nonzero)
            nonzero_flux = count(flux_nonzero)
            nonzero_both = count(both_nonzero)
            println("Cells with nonzero volume change: $nonzero_vol")
            println("Cells with nonzero interface flux: $nonzero_flux")
            println("Cells with both nonzero: $nonzero_both")
            if enable_stencil_fusion
                println("Mismatches at stencil level: $mismatches out of $total_stencils stencils")
            else
                println("Mismatches: $mismatches out of $total_cells cells")
            end
            
            # 4. Implémenter l'algorithme d'optimisation choisi
            JTJ = J' * J
            
            # Diagnostics initiaux
            used_columns = unique(col_indices)
            println("Matrix info: size(J)=$(size(J)), n_markers=$n_markers")
            println("Used marker indices: $(length(used_columns)) of $n_markers")
            
            if uppercase(gmorlm) == "LM"
                println("Using Levenberg-Marquardt with adaptive damping")
                
                # Extraire la diagonale pour la régularisation LM
                diag_JTJ = diag(JTJ)
                
                # Protéger contre les valeurs trop petites
                min_diag = 1e-10 * maximum(diag_JTJ)
                for i in 1:length(diag_JTJ)
                    if diag_JTJ[i] < min_diag
                        diag_JTJ[i] = min_diag
                    end
                end
                
                # Appliquer la régularisation adaptative de Levenberg-Marquardt
                reg_JTJ = JTJ + lambda * Diagonal(diag_JTJ)
                
            else
                println("Using Gauss-Newton algorithm with minimal stabilization")
                
                # Pour GN, juste une petite régularisation pour éviter les matrices singulières
                min_reg = 1e-10 * norm(JTJ, Inf)
                reg_JTJ = JTJ + min_reg * I(size(JTJ, 1))
            end
            
            # Résoudre le système avec méthode robuste
            newton_step = zeros(n_markers)
            try
                newton_step = reg_JTJ \ (J' * F)
            catch e
                println("Matrix solver failed, using SVD as backup")
                # Résolution par SVD comme fallback
                F_svd = svd(Matrix(reg_JTJ))
                svd_tol = eps(Float64) * max(size(reg_JTJ)...) * maximum(F_svd.S)
                S_inv = [s > svd_tol ? 1/s : 0.0 for s in F_svd.S]
                
                # Calculer la solution pseudo-inverse
                JTF = J' * F
                newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
            end
            
            # Calculer la norme de l'incrément de position
            position_increment_norm = α * norm(newton_step)
            push!(position_increment_history, position_increment_norm)
            
            # Pour Levenberg-Marquardt, ajuster lambda en fonction de la convergence
            if uppercase(gmorlm) == "LM" && iter > 1
                residual_norm = norm(F)
                if residual_norm < prev_residual_norm
                    # Amélioration - réduire lambda
                    lambda = max(lambda / lm_lambda_factor, lm_min_lambda)
                    println("Residual improved: decreasing lambda to $lambda")
                else
                    # Dégradation - augmenter lambda
                    lambda = min(lambda * lm_lambda_factor, lm_max_lambda)
                    println("Residual worsened: increasing lambda to $lambda")
                end
                prev_residual_norm = residual_norm
            end
            
            # Appliquer le pas avec facteur d'ajustement
            displacements -= α * newton_step
                
            # For closed curves, match first and last displacement to ensure continuity
            if front.is_closed
                displacements[end] = displacements[1]
            end
            
            # Smooth the displacements for stability
            smooth_displacements!(displacements, markers, front.is_closed, smooth_factor, window_size)
            
            # Print maximum displacement for diagnostics
            max_disp = maximum(abs.(displacements))
            println("Maximum displacement (after smoothing): $max_disp")

            """
            # 8.b Limit maximum displacement to one cell size for stability
            cell_size_x = (mesh.nodes[1][end] - mesh.nodes[1][1]) / nx
            cell_size_y = (mesh.nodes[2][end] - mesh.nodes[2][1]) / ny
            max_allowed_disp = min(cell_size_x, cell_size_y)

            if max_disp > max_allowed_disp
                scaling_factor = max_allowed_disp / max_disp
                displacements .*= scaling_factor
                """
                #println("Limiting displacements (scaling by $(round(scaling_factor, digits=4)))")
                #println("New maximum displacement: $(round(maximum(abs.(displacements)), digits=6))")
            #end
            

            # Calculate residual norm for convergence check
            residual_norm = norm(F)
            push!(residual_norm_history, residual_norm)
            
            # Report progress
            println("Iteration $iter | Residual = $residual_norm")
            
            # Check convergence
            if residual_norm < tol #|| (iter > 1 && abs(residual_norm_history[end] - residual_norm_history[end-1]) < reltol)
                println("Converged after $iter iterations with residual $residual_norm and position increment $position_increment_norm")
                break
            end
            
            # 5. Update marker positions
            new_markers = copy(markers)
            for i in 1:n_markers
                normal = normals[i]
                new_markers[i] = (
                    markers[i][1] + displacements[i] * normal[1],
                    markers[i][2] + displacements[i] * normal[2]
                )
            end
            
            # If interface is closed, update the duplicated last marker
            if front.is_closed
                new_markers[end] = new_markers[1]
            end
            
            # Print mean radius for diagnostic
            if front.is_closed
                # Calculer le centre approximatif
                center_x = sum(m[1] for m in new_markers) / length(new_markers)
                center_y = sum(m[2] for m in new_markers) / length(new_markers)
                
                # Calculer le rayon moyen
                mean_radius = mean([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in new_markers])
                
                # Afficher le rayon moyen
                println("Mean radius: $(round(mean_radius, digits=6))")
                
                # Vérifier la régularité
                std_radius = std([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in new_markers])
                if std_radius / mean_radius > 0.05
                    println("⚠️ Warning: Interface irregularity detected ($(round(100*std_radius/mean_radius, digits=2))% variation)")
                end
            end
            
            # 6. Create updated front tracking object
            updated_front = FrontTracker(new_markers, front.is_closed)
            
            # Create figure
            fig = Figure(size=(800, 700))
            ax = Axis(fig[1, 1], 
                    title="Interface Position Update", 
                    xlabel="x", 
                    ylabel="y",
                    aspect=DataAspect())

            temp_2d = reshape(Tₒ, (nx, ny))
            
            # Extract coordinates from markers and new_markers
            x_old = [m[1] for m in markers]
            y_old = [m[2] for m in markers]
            x_new = [m[1] for m in new_markers]
            y_new = [m[2] for m in new_markers]
            
            hm = heatmap!(ax, mesh.nodes[1], mesh.nodes[2], 
                 temp_2d,
                 colormap=:thermal)
            Colorbar(fig[1, 2], hm, label="Temperature")
    
            # Plot old and new positions
            #lines!(ax, x_old, y_old, color=:blue, linewidth=2, label="Current Position")
            lines!(ax, x_new, y_new, color=:red, linewidth=2)
            
            # Plot individual markers
            scatter!(ax, x_old, y_old, color=:blue, markersize=4)
            scatter!(ax, x_new, y_new, color=:red, markersize=4)
            
            # Display or save the figure
            display(fig)
            
            # Optional: Save the figure
            save("marker_position_update.png", fig)

            # 7. Create space-time level set for capacity calculation
            body = (x, y, t_local, _=0) -> begin
                # Normalized time in [0,1]
                τ = (t_local - tₙ) / Δt

                # Linear interpolation between SDFs
                sdf1 = -sdf(front, x, y)
                sdf2 = -sdf(updated_front, x, y)
                return (1 - τ) * sdf1 + τ * sdf2
            end
            
            # 8. Update space-time mesh and capacity
            STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
            capacity = Capacity(body, STmesh;  method="VOFI", integration_method=:vofijul, compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase_updated = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)
            
            # 9. Rebuild the matrix system
            s.A = A_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, phase_updated.source, 
                                            bc, Tᵢ, Δt, tₙ, scheme)
            
            BC_border_mono!(s.A, s.b, bc_b, mesh; t=tₙ₊₁)
            
            # 10. Update phase for next iteration
            phase = phase_updated

            body_2d = (x, y, _=0) -> body(x, y, tₙ₊₁)
            capacity_2d = Capacity(body_2d, mesh;  method="VOFI", integration_method=:vofijul, compute_centroids=false)
            operator_2d = DiffusionOps(capacity_2d)
            phase_2d = Phase(capacity_2d, operator_2d, phase.source, phase.Diffusion_coeff)
        end
        
        # Store residuals from this time step
        residuals[timestep] = residual_norm_history
        position_increments[timestep] = position_increment_history  # Stocker l'historique des incréments de position

        
        # Update front with new marker positions
        new_markers = copy(markers)
        for i in 1:n_markers
            normal = normals[i]
            new_markers[i] = (
                markers[i][1] + displacements[i] * normal[1],
                markers[i][2] + displacements[i] * normal[2]
            )
        end
        
        # If interface is closed, update the duplicated last marker
        if front.is_closed
            new_markers[end] = new_markers[1]
        end
        
        # Update front with new markers
        set_markers!(front, new_markers)
        
        # Store updated interface position
        xf_log[timestep+1] = new_markers
        
        # Store solution
        push!(s.states, s.x)
        
        println("Time: $(round(t, digits=6))")
        println("Max temperature: $(maximum(abs.(s.x)))")

        # À la fin du pas de temps, après la résolution complète, stocker les déplacements
        last_displacements = copy(displacements)
    
        # Increment timestep counter
        timestep += 1
    end
    
    return s, residuals, xf_log, timestep_history, phase_2d, position_increments
end

"""
Wrapper around `solve_StefanMono2D!` for unclosed interfaces (e.g. planar fronts).
Ensures the provided `FrontTracker` is marked as open and accepts an `adaptive_timestep`
keyword (currently unused) for compatibility with example scripts.
"""
function solve_StefanMono2Dunclosed!(s::Solver, phase::Phase, front::FrontTracker, Δt::Float64, Tₛ::Float64, Tₑ::Float64,
                                     bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions,
                                     mesh::Penguin.Mesh{2}, scheme::String;
                                     adaptive_timestep::Bool=false,
                                     kwargs...)
    # Force the interface to be treated as unclosed and drop duplicated closing marker if present
    markers = get_markers(front)
    if front.is_closed || (length(markers) > 1 && markers[1] == markers[end])
        markers = copy(markers)
        if length(markers) > 1 && markers[1] == markers[end]
            pop!(markers)
        end
        set_markers!(front, markers, false)
        println("solve_StefanMono2Dunclosed!: converted provided front to an open interface with $(length(markers)) markers")
    end

    if adaptive_timestep
        println("solve_StefanMono2Dunclosed!: adaptive_timestep is not implemented; proceeding with fixed Δt = $Δt")
    end

    return solve_StefanMono2D!(s, phase, front, Δt, Tₛ, Tₑ, bc_b, bc, ic, mesh, scheme; kwargs...)
end

function solve_StefanMono2D_geom!(s::Solver, phase::Phase, front::FrontTracker, Δt::Float64, Tₛ::Float64, Tₑ::Float64,
                                  bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions,
                                  mesh::Penguin.Mesh{2}, scheme::String;
                                  method=Base.:\,
                                  Newton_params=(100, 1e-6, 1e-6, 1.0),
                                  smooth_factor=0.5,
                                  window_size=10,
                                  segment_smoothing_iters::Int=0,
                                  algorithm=nothing,
                                  kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving Stefan problem with Front Tracking (geometric update):")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")

    max_iter, tol, reltol, α = Newton_params
    ρL = ic.flux.value

    t = Tₛ
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))
    position_increments = Dict{Int, Vector{Float64}}()

    dims = phase.operator.size
    len_dims = length(dims)
    cap_index = len_dims
    nx, ny, _ = dims

    Tᵢ = s.x
    push!(s.states, s.x)

    markers = get_markers(front)
    xf_log[1] = markers

    timestep = 1
    phase_2d = phase
    last_displacements = nothing

    residual_dir = joinpath(pwd(), "residuals_visualization")
    if !isdir(residual_dir)
        mkdir(residual_dir)
    end

    while t < Tₑ
        markers = get_markers(front)
        normals = compute_marker_normals(front, markers)

        if last_displacements !== nothing && timestep > 1
            extrapolation_factor = 1.0
            println("Initializing interface using previous displacements (geometric extrapolation factor: $extrapolation_factor)")
            n_markers = length(markers) - (front.is_closed ? 1 : 0)
            new_markers = copy(markers)
            for i in 1:n_markers
                normal = normals[i]
                new_markers[i] = (
                    markers[i][1] + extrapolation_factor * last_displacements[i] * normal[1],
                    markers[i][2] + extrapolation_factor * last_displacements[i] * normal[2]
                )
            end
            if front.is_closed && !isempty(new_markers)
                new_markers[end] = new_markers[1]
            end
            set_markers!(front, new_markers)
            markers = get_markers(front)
            normals = compute_marker_normals(front, markers)
        end

        t += Δt
        tₙ = t - Δt
        tₙ₊₁ = t
        time_interval = [tₙ, tₙ₊₁]

        n_markers = length(markers) - (front.is_closed ? 1 : 0)
        if n_markers <= 0
            error("Geometric update requires at least one marker on the front")
        end
        displacements = zeros(n_markers)
        residual_norm_history = Float64[]
        position_increment_history = Float64[]

        for iter in 1:max_iter
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            Tᵢ = s.x

            V_matrices = phase.capacity.A[cap_index]
            Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]
            Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]
            Vₙ₊₁_matrix = reshape(diag(Vₙ₊₁_matrix), (nx, ny))
            Vₙ_matrix = reshape(diag(Vₙ_matrix), (nx, ny))

            W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
            G = phase.operator.G[1:end÷2, 1:end÷2]
            H = phase.operator.H[1:end÷2, 1:end÷2]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:end÷2, 1:end÷2]

            Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
            interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            interface_flux_2d = reshape(interface_flux, (nx, ny))

            markers_unique = markers[1:n_markers]
            normals_unique = normals[1:n_markers]

            segment_displacements, segment_lengths, cells_idx, residual_vector =
                compute_geometric_segment_displacements(front, mesh, Vₙ_matrix, Vₙ₊₁_matrix, interface_flux_2d, ρL, α)

            if segment_smoothing_iters > 0
                smooth_segment_displacements!(segment_displacements, segment_lengths, front.is_closed;
                                              iterations=segment_smoothing_iters)
            end

            energy_residual_norm = isempty(residual_vector) ? 0.0 : norm(residual_vector)

            marker_displacements = segment_to_marker_displacements(segment_displacements, segment_lengths,
                                                                    markers_unique, front.is_closed)

            if front.is_closed && !isempty(marker_displacements)
                marker_displacements[end] = marker_displacements[1]
            end

            # The displacement is already in the correct direction (residual/ρL gives area, divided by length gives displacement)
            # Negative residual (flux > ρL*dV) means interface should grow (positive displacement outward)
            # Positive residual (flux < ρL*dV) means interface should shrink (negative displacement inward)
            # So we need to negate to get the correct motion
            marker_displacements .*= -1.0
            smooth_displacements!(marker_displacements, markers_unique, front.is_closed, smooth_factor, window_size)
            
            # Accumulate displacements for iterative refinement
            displacements .+= marker_displacements

            # Use the displacement increment for convergence check
            displacement_increment_norm = norm(marker_displacements)
            push!(residual_norm_history, displacement_increment_norm)

            position_increment_norm = displacement_increment_norm
            push!(position_increment_history, position_increment_norm)

            max_disp_increment = isempty(marker_displacements) ? 0.0 : maximum(abs.(marker_displacements))
            max_disp_total = isempty(displacements) ? 0.0 : maximum(abs.(displacements))
            println("Maximum displacement increment (geometric): $max_disp_increment")
            println("Maximum total displacement (geometric): $max_disp_total")

            if !isempty(residual_vector)
                residual_fig = plot_residual_heatmap(mesh, residual_vector, cells_idx, nothing, false)
                timestamp = round(Int, time())
                save(joinpath(residual_dir, "residual_timestep$(timestep)_iter$(iter)_$(timestamp).png"), residual_fig)
                display(residual_fig)
            else
                println("No intersected cells for geometric update at iteration $iter")
            end

            println("Residual norm (geometric displacement increment) = $displacement_increment_norm | energy residual = $energy_residual_norm")

            new_markers = copy(markers)
            for i in 1:n_markers
                normal = normals_unique[i]
                new_markers[i] = (
                    markers[i][1] + displacements[i] * normal[1],
                    markers[i][2] + displacements[i] * normal[2]
                )
            end
            if front.is_closed && !isempty(new_markers)
                new_markers[end] = new_markers[1]
            end

            updated_front = FrontTracker(new_markers, front.is_closed)

            if front.is_closed && !isempty(new_markers)
                center_x = sum(m[1] for m in new_markers) / length(new_markers)
                center_y = sum(m[2] for m in new_markers) / length(new_markers)
                mean_radius = mean([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in new_markers])
                println("Mean radius: $(round(mean_radius, digits=6))")
            end

            fig = Figure(size=(800, 700))
            ax = Axis(fig[1, 1],
                      title="Interface Position Update (geometric)",
                      xlabel="x",
                      ylabel="y",
                      aspect=DataAspect())

            temp_2d = reshape(Tₒ, (nx, ny))
            hm = heatmap!(ax, mesh.nodes[1], mesh.nodes[2], temp_2d, colormap=:thermal)
            Colorbar(fig[1, 2], hm, label="Temperature")

            x_new = [m[1] for m in new_markers]
            y_new = [m[2] for m in new_markers]
            x_old = [m[1] for m in markers_unique]
            y_old = [m[2] for m in markers_unique]
            scatter!(ax, x_old, y_old, color=:blue, markersize=4)
            lines!(ax, x_new, y_new, color=:red, linewidth=2)
            scatter!(ax, x_new, y_new, color=:red, markersize=4)
            display(fig)
            save("marker_position_update.png", fig)

            body = (x, y, t_local, _=0) -> begin
                τ = (t_local - tₙ) / Δt
                sdf1 = -sdf(front, x, y)
                sdf2 = -sdf(updated_front, x, y)
                return (1 - τ) * sdf1 + τ * sdf2
            end

            STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
            capacity = Capacity(body, STmesh; compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase_updated = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

            s.A = A_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity,
                                            phase_updated.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity,
                                            phase_updated.Diffusion_coeff, phase_updated.source,
                                            bc, Tᵢ, Δt, tₙ, scheme)

            BC_border_mono!(s.A, s.b, bc_b, mesh; t=tₙ₊₁)

            phase = phase_updated

            body_2d = (x, y, _=0) -> body(x, y, tₙ₊₁)
            capacity_2d = Capacity(body_2d, mesh; compute_centroids=false)
            operator_2d = DiffusionOps(capacity_2d)
            phase_2d = Phase(capacity_2d, operator_2d, phase.source, phase.Diffusion_coeff)

            # Check convergence based on displacement increment
            if displacement_increment_norm < tol || (iter > 1 && abs(residual_norm_history[end] - residual_norm_history[end-1]) < reltol)
                println("Converged after $iter iterations with displacement increment $displacement_increment_norm and position increment $position_increment_norm")
                break
            end
        end

        residuals[timestep] = residual_norm_history
        position_increments[timestep] = position_increment_history

        new_markers = copy(markers)
        for i in 1:n_markers
            normal = normals[i]
            new_markers[i] = (
                markers[i][1] + displacements[i] * normal[1],
                markers[i][2] + displacements[i] * normal[2]
            )
        end
        if front.is_closed && !isempty(new_markers)
            new_markers[end] = new_markers[1]
        end

        set_markers!(front, new_markers)
        xf_log[timestep+1] = new_markers
        push!(s.states, s.x)

        println("Time: $(round(t, digits=6))")
        println("Max temperature: $(maximum(abs.(s.x)))")

        last_displacements = copy(displacements)
        push!(timestep_history, (t, Δt))
        timestep += 1
    end

    return s, residuals, xf_log, timestep_history, phase_2d, position_increments
end



# Define the Stefan diphasic solver
function StefanDiph2D(phase1::Phase, phase2::Phase, bc_b::BorderConditions, 
                     interface_cond::InterfaceConditions, Δt::Float64, 
                     u0::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Stefan problem with front tracking")
    println("- Diphasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Diphasic, Diffusion, nothing, nothing, nothing, [], [])
    
    # Create initial matrix system based on selected scheme
    if scheme == "CN"
        s.A = A_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity,
                                         phase2.capacity, phase1.Diffusion_coeff, 
                                         phase2.Diffusion_coeff, interface_cond, "CN")
        s.b = b_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity,
                                         phase2.capacity, phase1.Diffusion_coeff, 
                                         phase2.Diffusion_coeff, phase1.source, phase2.source, 
                                         interface_cond, u0, Δt, 0.0, "CN")
    else # "BE"
        s.A = A_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, 
                                        phase2.capacity, phase1.Diffusion_coeff, 
                                        phase2.Diffusion_coeff, interface_cond, "BE")
        s.b = b_diph_unstead_diff_moving_stef2(phase1.operator, phase2.operator, phase1.capacity, 
                                        phase2.capacity, phase1.Diffusion_coeff, 
                                        phase2.Diffusion_coeff, phase1.source, phase2.source, 
                                        interface_cond, u0, Δt, 0.0, "BE")
    end
    
    # Apply boundary conditions
    BC_border_diph!(s.A, s.b, bc_b, mesh)
    
    s.x = u0
    return s
end

function solve_StefanDiph2D!(s::Solver, phase1::Phase, phase2::Phase, 
                            front::FrontTracker, Δt::Float64, 
                            Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, 
                            interface_cond::InterfaceConditions, mesh::Penguin.Mesh{2}, 
                            scheme::String;
                            method=Base.:\, 
                            Newton_params=(100, 1e-6, 1e-6, 1.0),
                            jacobian_epsilon=1e-6, smooth_factor=0.5, 
                            window_size=10, 
                            algorithm=nothing,
                            kwargs...)
                            
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving Stefan diphasic problem with Front Tracking:")
    println("- Diphasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Unpack parameters
    max_iter = Newton_params[1]
    tol = Newton_params[2]
    reltol = Newton_params[3]
    α = Newton_params[4]
    
    # Extract interface flux parameter
    ρL = interface_cond.flux.value
    
    # Initialize tracking variables
    t = Tₛ
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))
    position_increments = Dict{Int, Vector{Float64}}()
    
    # Determine how many dimensions
    dims = phase1.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Create the 2D grid dimensions
    nx, ny, _ = dims
    n = nx * ny
    
    # Store initial state
    Tᵢ = s.x
    push!(s.states, s.x)
    
    # Get initial interface markers
    markers = get_markers(front)
    
    # Store initial interface position
    xf_log[1] = markers
    
    # Main time stepping loop
    timestep = 1
    phase1_2d = phase1  # Initialize for later use
    phase2_2d = phase2  # Initialize for later use
    while t < Tₑ
        # Display current time step
        if timestep == 1
            println("\nFirst time step: t = $(round(t, digits=6))")
        else
            println("\nTime step $(timestep), t = $(round(t, digits=6))")
        end

        # Get current markers and calculate normals
        markers = get_markers(front)
        normals = compute_marker_normals(front, markers)
        
        # Update time for this step
        t += Δt
        tₙ = t - Δt
        tₙ₊₁ = t
        time_interval = [tₙ, tₙ₊₁]
        
        # Calculate total number of markers (excluding duplicated closing point)
        n_markers = length(markers) - (front.is_closed ? 1 : 0)
        
        # Initialize displacement vector and residual vector
        displacements = zeros(n_markers)
        residual_norm_history = Float64[]
        position_increment_history = Float64[]

        # Gauss-Newton iterations
        for iter in 1:max_iter
            # 1. Solve temperature field with current interface position
            solve_system!(s; method=method, algorithm=algorithm, kwargs...)
            Tᵢ = s.x
            
            # Separate solution components for each phase
            n_dof = length(Tᵢ) ÷ 4
            T1_bulk = Tᵢ[1:n_dof]                    # Phase 1 bulk
            T1_interface = Tᵢ[n_dof+1:2*n_dof]       # Phase 1 interface
            T2_bulk = Tᵢ[2*n_dof+1:3*n_dof]          # Phase 2 bulk
            T2_interface = Tᵢ[3*n_dof+1:end]         # Phase 2 interface
            
            # Get capacity matrices for Phase 1
            V1_matrices = phase1.capacity.A[cap_index]
            V1ₙ₊₁_matrix = V1_matrices[1:end÷2, 1:end÷2]
            V1ₙ_matrix = V1_matrices[end÷2+1:end, end÷2+1:end]
            V1ₙ₊₁_matrix = diag(V1ₙ₊₁_matrix)
            V1ₙ_matrix = diag(V1ₙ_matrix)
            V1ₙ₊₁_matrix = reshape(V1ₙ₊₁_matrix, (nx, ny))
            V1ₙ_matrix = reshape(V1ₙ_matrix, (nx, ny))
            
            # Get capacity matrices for Phase 2
            V2_matrices = phase2.capacity.A[cap_index]
            V2ₙ₊₁_matrix = V2_matrices[1:end÷2, 1:end÷2]
            V2ₙ_matrix = V2_matrices[end÷2+1:end, end÷2+1:end]
            V2ₙ₊₁_matrix = diag(V2ₙ₊₁_matrix)
            V2ₙ_matrix = diag(V2ₙ_matrix)
            V2ₙ₊₁_matrix = reshape(V2ₙ₊₁_matrix, (nx, ny))
            V2ₙ_matrix = reshape(V2ₙ_matrix, (nx, ny))
            
            # 2. Calculate the interface flux for each phase
            # Phase 1 interface flux
            W1! = phase1.operator.Wꜝ[1:end÷2, 1:end÷2]
            G1 = phase1.operator.G[1:end÷2, 1:end÷2]
            H1 = phase1.operator.H[1:end÷2, 1:end÷2]
            Id1 = build_I_D(phase1.operator, phase1.Diffusion_coeff, phase1.capacity)
            Id1 = Id1[1:end÷2, 1:end÷2]
            
            T1ₒ, T1ᵧ = T1_bulk, T1_interface
            interface_flux1 = Id1 * H1' * W1! * G1 * T1ₒ + Id1 * H1' * W1! * H1 * T1ᵧ
            interface_flux1_2d = reshape(interface_flux1, (nx, ny))
            
            # Phase 2 interface flux
            W2! = phase2.operator.Wꜝ[1:end÷2, 1:end÷2]
            G2 = phase2.operator.G[1:end÷2, 1:end÷2]
            H2 = phase2.operator.H[1:end÷2, 1:end÷2]
            Id2 = build_I_D(phase2.operator, phase2.Diffusion_coeff, phase2.capacity)
            Id2 = Id2[1:end÷2, 1:end÷2]
            
            T2ₒ, T2ᵧ = T2_bulk, T2_interface
            interface_flux2 = Id2 * H2' * W2! * G2 * T2ₒ + Id2 * H2' * W2! * H2 * T2ᵧ
            interface_flux2_2d = reshape(interface_flux2, (nx, ny))
            
            # 3. Compute volume Jacobian for the mesh
            volume_jacobian = compute_volume_jacobian(mesh, front, jacobian_epsilon)
            
            # 4. Build least squares system
            row_indices = Int[]
            col_indices = Int[]
            values = Float64[]
            cells_idx = []
            
            # Precompute affected cells and their indices for residual vector
            for i in 1:nx
                for j in 1:ny
                    if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                        push!(cells_idx, (i, j))
                    end
                end
            end
            
            # Number of equations (cells) in the system
            m = length(cells_idx)
            
            # Build the Jacobian matrix for volume changes
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                # Handle each marker affecting this cell
                for (marker_idx, jac_value) in volume_jacobian[(i,j)]
                    if 0 <= marker_idx < n_markers
                        push!(row_indices, eq_idx)
                        push!(col_indices, marker_idx + 1)  # 1-based indexing
                        # Volume Jacobian is multiplied by ρL to match the Stefan condition
                        push!(values, ρL * jac_value)
                    end
                end
            end
            
            # Create Jacobian matrix J for the system
            J = sparse(row_indices, col_indices, values, m, n_markers)
            
            # 5. Calculate current residual vector F
            F = zeros(m)
            mismatches = 0
            total_cells = 0
            
            # Diagnostic arrays
            flux_nonzero = zeros(Bool, nx, ny)
            volume_nonzero = zeros(Bool, nx, ny)
            
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                # Calculate volume changes for both phases
                volume_change1 = V1ₙ₊₁_matrix[i,j] - V1ₙ_matrix[i,j]
                volume_change2 = V2ₙ₊₁_matrix[i,j] - V2ₙ_matrix[i,j]
                
                # Calculate net flux (take both phases into account)
                # Note: Interface moves from Phase 1 to Phase 2, so invert sign for Phase 1
                flux1 = -interface_flux1_2d[i,j]  # Negative sign for Phase 1 (inside)
                flux2 = interface_flux2_2d[i,j]   # Phase 2 (outside)
                net_flux = flux1 + flux2
                
                # Calculate total volume change
                # Note: For conservation, volume change in Phase 1 should be negated
                net_volume_change = -volume_change1 + volume_change2
                
                # Record diagnostics
                total_cells += 1
                volume_nonzero[i,j] = abs(net_volume_change) > 1e-10
                flux_nonzero[i,j] = abs(net_flux) > 1e-10
                
                # Detect mismatches (volume change without flux or vice versa)
                if (volume_nonzero[i,j] && !flux_nonzero[i,j]) || (!volume_nonzero[i,j] && flux_nonzero[i,j])
                    mismatches += 1
                    if mismatches <= 5
                        println("Mismatch at ($i,$j): net_volume_change = $net_volume_change, net_flux = $net_flux")
                    end
                end
                
                # F_i = ρL * net_volume_change - net_flux
                F[eq_idx] = ρL * net_volume_change - net_flux
            end
            
            # Print diagnostic summary
            nonzero_vol = count(volume_nonzero)
            nonzero_flux = count(flux_nonzero)
            nonzero_both = count(volume_nonzero .& flux_nonzero)
            println("Cells with nonzero volume change: $nonzero_vol")
            println("Cells with nonzero interface flux: $nonzero_flux")
            println("Cells with both nonzero: $nonzero_both")
            println("Mismatches: $mismatches out of $total_cells cells")
            
            # 6. Implement the Gauss-Newton formula with regularization
            JTJ = J' * J
            
            # Diagnose the system
            used_columns = unique(col_indices)
            println("Matrix info: size(J)=$(size(J)), n_markers=$n_markers")
            println("Used marker indices: $(length(used_columns)) of $n_markers")
            
            # Add Tikhonov regularization
            reg_param = 1e-6
            diag_JTJ = diag(JTJ)
            
            # Ensure diagonal elements aren't too small
            min_diag = 1e-10 * maximum(diag_JTJ)
            for i in 1:length(diag_JTJ)
                if diag_JTJ[i] < min_diag
                    diag_JTJ[i] = min_diag
                end
            end
            
            # Apply Levenberg-Marquardt style regularization
            reg_JTJ = JTJ + reg_param * Diagonal(diag_JTJ)
            
            # Solve the system using robust method
            newton_step = zeros(n_markers)
            try
                newton_step = reg_JTJ \ (J' * F)
            catch e
                println("Matrix solver failed, using SVD as backup")
                # SVD-based pseudoinverse for robust solving
                F_svd = svd(Matrix(reg_JTJ))
                svd_tol = eps(Float64) * max(size(reg_JTJ)...) * maximum(F_svd.S)
                S_inv = [s > svd_tol ? 1/s : 0.0 for s in F_svd.S]
                
                # Compute pseudoinverse solution
                JTF = J' * F
                newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
            end
            
            # Calculate position increment norm
            position_increment_norm = α * norm(newton_step)
            push!(position_increment_history, position_increment_norm)
            
            # 7. Apply the step with adjustment factor
            displacements -= α * newton_step
            
            # For closed curves, match first and last displacement to ensure continuity
            if front.is_closed
                displacements[end] = displacements[1]
            end
            
            # 8. Smooth the displacements for stability
            smooth_displacements!(displacements, markers, front.is_closed, smooth_factor, window_size)
            
            # Print maximum displacement for diagnostics
            max_disp = maximum(abs.(displacements))
            println("Maximum displacement (after smoothing): $max_disp")
            
            # 9. Calculate residual norm for convergence check
            residual_norm = norm(F)
            push!(residual_norm_history, residual_norm)
            
            # Report progress
            println("Iteration $iter | Residual = $residual_norm | Position increment = $position_increment_norm")
            
            # 10. Check convergence
            if residual_norm < tol || (iter > 1 && abs(residual_norm_history[end] - residual_norm_history[end-1]) < reltol)
                println("Converged after $iter iterations with residual $residual_norm")
                break
            end
            
            # 11. Update marker positions
            new_markers = copy(markers)
            for i in 1:n_markers
                normal = normals[i]
                new_markers[i] = (
                    markers[i][1] + displacements[i] * normal[1],
                    markers[i][2] + displacements[i] * normal[2]
                )
            end
            
            # If interface is closed, update the duplicated last marker
            if front.is_closed
                new_markers[end] = new_markers[1]
            end
            
            # 12. Create updated front tracking object
            updated_front = FrontTracker(new_markers, front.is_closed)
            
            # 13. Create space-time level set for capacity calculation
            function body1_update(x, y, t_local, _=0)
                # Normalized time in [0,1]
                τ = (t_local - tₙ) / Δt
                
                # Linear interpolation between SDFs
                sdf1 = sdf(front, x, y)
                sdf2 = sdf(updated_front, x, y)
                return (1-τ) * sdf1 + τ * sdf2
            end
            
            function body2_update(x, y, t_local, _=0)
                # Normalized time in [0,1]
                τ = (t_local - tₙ) / Δt
                
                # Linear interpolation between SDFs
                sdf1 = -sdf(front, x, y)
                sdf2 = -sdf(updated_front, x, y)
                return (1-τ) * sdf1 + τ * sdf2
            end
            
            # 14. Update space-time mesh and capacities for both phases
            STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
            
            capacity1_updated = Capacity(body1_update, STmesh; compute_centroids=false)
            operator1_updated = DiffusionOps(capacity1_updated)
            phase1_updated = Phase(capacity1_updated, operator1_updated, phase1.source, phase1.Diffusion_coeff)
            
            capacity2_updated = Capacity(body2_update, STmesh; compute_centroids=false)
            operator2_updated = DiffusionOps(capacity2_updated)
            phase2_updated = Phase(capacity2_updated, operator2_updated, phase2.source, phase2.Diffusion_coeff)
            
            # 15. Rebuild the matrix system
            s.A = A_diph_unstead_diff_moving_stef2(phase1_updated.operator, phase2_updated.operator,
                                            phase1_updated.capacity, phase2_updated.capacity,
                                            phase1_updated.Diffusion_coeff, phase2_updated.Diffusion_coeff, 
                                            interface_cond, scheme)
                                            
            s.b = b_diph_unstead_diff_moving_stef2(phase1_updated.operator, phase2_updated.operator,
                                            phase1_updated.capacity, phase2_updated.capacity,
                                            phase1_updated.Diffusion_coeff, phase2_updated.Diffusion_coeff,
                                            phase1_updated.source, phase2_updated.source,
                                            interface_cond, Tᵢ, Δt, tₙ, scheme)
            
            BC_border_diph!(s.A, s.b, bc_b, mesh)
            
            # 16. Update phases and front for next iteration
            phase1 = phase1_updated
            phase2 = phase2_updated
            front = updated_front

            # Update snapshot phases for visualization
            body1_2d(x,y,_=0) = body1_update(x, y, tₙ₊₁)
            body2_2d(x,y,_=0) = body2_update(x, y, tₙ₊₁)
            
            capacity1_2d = Capacity(body1_2d, mesh; compute_centroids=false)
            operator1_2d = DiffusionOps(capacity1_2d)
            phase1_2d = Phase(capacity1_2d, operator1_2d, phase1.source, phase1.Diffusion_coeff)
            
            capacity2_2d = Capacity(body2_2d, mesh; compute_centroids=false)
            operator2_2d = DiffusionOps(capacity2_2d)
            phase2_2d = Phase(capacity2_2d, operator2_2d, phase2.source, phase2.Diffusion_coeff)
        end
        
        # Store residuals and position increments from this time step
        residuals[timestep] = residual_norm_history
        position_increments[timestep] = position_increment_history
        
        # Store updated interface position
        xf_log[timestep+1] = get_markers(front)
        
        # Store solution
        push!(s.states, s.x)
        
        # Print radius info for a circle
        markers = get_markers(front)
        center_x = sum(m[1] for m in markers) / length(markers)
        center_y = sum(m[2] for m in markers) / length(markers)
        mean_radius = mean([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in markers])
        println("Mean radius: $(round(mean_radius, digits=6))")
        
        println("Time: $(round(t, digits=6))")
        println("Max temperature: $(maximum(abs.(s.x)))")
        
        # Increment timestep counter
        timestep += 1
        
        # Add current time and timestep size to history
        push!(timestep_history, (t, Δt))
    end
    
    return s, residuals, xf_log, timestep_history, phase1_2d, phase2_2d, position_increments
end
