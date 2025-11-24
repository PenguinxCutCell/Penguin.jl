"""
A Julia implementation of front tracking for fluid-solid interfaces.
Replaces the Python implementation to avoid segmentation faults.
"""
mutable struct FrontTracker
    markers::Vector{Tuple{Float64, Float64}}
    is_closed::Bool
    interface::Union{Nothing, LibGEOS.LineString}
    interface_poly::Union{Nothing, LibGEOS.Polygon}
    
    # Constructor for empty front
    function FrontTracker()
        return new([], true, nothing, nothing)
    end
    
    # Constructor with markers
    function FrontTracker(markers::Vector{Tuple{Float64, Float64}}, is_closed::Bool=true)
        ft = new(markers, is_closed, nothing, nothing)
        update_geometry!(ft)
        return ft
    end
end

"""
Updates the interface geometry based on marker positions.
"""
function update_geometry!(ft::FrontTracker)
    if length(ft.markers) < 3
        # Not enough points to create a valid geometry
        ft.interface = nothing
        ft.interface_poly = nothing
        return
    end
    
    # Ensure interface is closed for polygon creation
    markers_to_use = ft.markers
    if ft.is_closed && length(markers_to_use) > 0 && markers_to_use[1] != markers_to_use[end]
        # Add closing point
        push!(markers_to_use, markers_to_use[1])
    end
    
    # For LibGEOS, we need to create the geometry directly from coordinates
    coords = [collect(point) for point in markers_to_use]
    
    # Create LineString
    ft.interface = LibGEOS.LineString(coords)
    
    # Create Polygon (if closed and valid)
    if ft.is_closed && length(ft.markers) >= 3
        ft.interface_poly = LibGEOS.Polygon([coords])
    else
        ft.interface_poly = nothing
    end
end

"""
Gets all markers of the interface.
"""
function get_markers(ft::FrontTracker)
    return ft.markers
end

"""
Adds a marker to the interface.
"""
function add_marker!(ft::FrontTracker, x::Float64, y::Float64)
    push!(ft.markers, (x, y))
    if length(ft.markers) >= 3  # Only update geometry when we have enough points
        update_geometry!(ft)
    end
    return ft
end

"""
Sets all markers of the interface.
"""
function set_markers!(ft::FrontTracker, markers::AbstractVector, is_closed=nothing)
    # Convert to proper type if needed
    typed_markers = convert(Vector{Tuple{Float64, Float64}}, markers)
    ft.markers = typed_markers
    if is_closed !== nothing
        ft.is_closed = is_closed
    end
    update_geometry!(ft)
    return ft
end

"""
Creates a circular interface.
"""
function create_circle!(ft::FrontTracker, center_x::Float64, center_y::Float64, radius::Float64, n_markers::Int=100)
    # Create properly typed vector
    markers = Vector{Tuple{Float64, Float64}}(undef, n_markers)
    
    for i in 0:n_markers-1
        angle = 2.0 * π * (i / n_markers)
        x = center_x + radius * cos(angle)
        y = center_y + radius * sin(angle)
        markers[i+1] = (x, y)
    end
    
    set_markers!(ft, markers, true)
    return ft
end

"""
    create_rectangle!(ft::FrontTracker, min_x::Float64, min_y::Float64, max_x::Float64, max_y::Float64, n_markers::Int=100)

Creates a rectangular interface with markers distributed evenly around the perimeter.
Avoids placing markers at corners for numerical stability.
Starts in the middle of the bottom edge and traces the contour.
- min_x, min_y: Coordinates of bottom-left corner
- max_x, max_y: Coordinates of top-right corner
- n_markers: Number of points to place around the perimeter
"""
function create_rectangle!(ft::FrontTracker, min_x::Float64, min_y::Float64, max_x::Float64, max_y::Float64, n_markers::Int=100)
    # Calculate side lengths
    width = max_x - min_x
    height = max_y - min_y
    
    # Calculate perimeter
    perimeter = 2 * (width + height)
    
    # Calculate markers per side (proportional to side length)
    total_markers = n_markers - 4  # Reserve space to skip corners
    
    # Points per unit length (used for distribution)
    points_per_unit = total_markers / perimeter
    
    # Calculate number of points on each edge (excluding corners)
    n_bottom = max(1, round(Int, width * points_per_unit))
    n_right = max(1, round(Int, height * points_per_unit))
    n_top = max(1, round(Int, width * points_per_unit))
    n_left = max(1, round(Int, height * points_per_unit))
    
    # Adjust total to ensure we use exactly n_markers
    actual_total = n_bottom + n_right + n_top + n_left
    if actual_total < total_markers
        # Add extra points proportionally to the longest sides
        extra_points = total_markers - actual_total
        if width >= height
            extra_bottom = round(Int, extra_points / 2)
            extra_top = extra_points - extra_bottom
            n_bottom += extra_bottom
            n_top += extra_top
        else
            extra_right = round(Int, extra_points / 2)
            extra_left = extra_points - extra_right
            n_right += extra_right
            n_left += extra_left
        end
    elseif actual_total > total_markers
        # Remove points proportionally
        excess_points = actual_total - total_markers
        if width >= height
            excess_bottom = round(Int, excess_points / 2)
            excess_top = excess_points - excess_bottom
            n_bottom = max(1, n_bottom - excess_bottom)
            n_top = max(1, n_top - excess_top)
        else
            excess_right = round(Int, excess_points / 2)
            excess_left = excess_points - excess_right
            n_right = max(1, n_right - excess_right)
            n_left = max(1, n_left - excess_left)
        end
    end
    
    # Create properly typed vector (add +1 for closing point)
    markers = Vector{Tuple{Float64, Float64}}(undef, n_markers+1)
    
    # Start placing markers from the middle of bottom edge
    current_marker = 1
    
    # Bottom edge (left to right)
    for i in 1:n_bottom
        t = i / (n_bottom + 1)  # Skip 0 and 1 to avoid corners
        markers[current_marker] = (min_x + t * width, min_y)
        current_marker += 1
    end
    
    # Right edge (bottom to top)
    for i in 1:n_right
        t = i / (n_right + 1)  # Skip 0 and 1 to avoid corners
        markers[current_marker] = (max_x, min_y + t * height)
        current_marker += 1
    end
    
    # Top edge (right to left)
    for i in 1:n_top
        t = i / (n_top + 1)  # Skip 0 and 1 to avoid corners
        markers[current_marker] = (max_x - t * width, max_y)
        current_marker += 1
    end
    
    # Left edge (top to bottom)
    for i in 1:n_left
        t = i / (n_left + 1)  # Skip 0 and 1 to avoid corners
        markers[current_marker] = (min_x, max_y - t * height)
        current_marker += 1
    end
    
    # Ensure we have the right number of markers
    actual_markers = current_marker - 1
    if actual_markers != n_markers
        println("Warning: Requested $n_markers markers but placed $actual_markers markers")
    end
    
    # Close the curve by duplicating the first marker
    markers[actual_markers+1] = markers[1]
    
    # Set markers in the front tracker
    set_markers!(ft, markers[1:actual_markers+1], true)
    return ft
end

"""
    create_ellipse!(ft::FrontTracker, center_x::Float64, center_y::Float64, radius_x::Float64, radius_y::Float64, n_markers::Int=100)

Creates an elliptical interface.
"""
function create_ellipse!(ft::FrontTracker, center_x::Float64, center_y::Float64, radius_x::Float64, radius_y::Float64, n_markers::Int=100)
    theta = range(0, 2π, length=n_markers+1)[1:end-1] # Avoid duplicating the first point
    markers = [(center_x + radius_x*cos(t), center_y + radius_y*sin(t)) for t in theta]
    set_markers!(ft, markers, true)
    return ft
end

"""
    create_crystal!(ft::FrontTracker, center_x::Float64, center_y::Float64, 
                   base_radius::Float64, n_lobes::Int=6, amplitude::Float64=0.2, 
                   n_markers::Int=100)

Crée une interface en forme de cristal (un cercle avec perturbation angulaire).
- center_x, center_y: coordonnées du centre
- base_radius: rayon de base
- n_lobes: nombre de lobes du cristal (6 pour symétrie hexagonale)
- amplitude: amplitude de la perturbation (0-1)
- n_markers: nombre de points sur l'interface
"""
function create_crystal!(ft::FrontTracker, center_x::Float64, center_y::Float64, 
                        base_radius::Float64, n_lobes::Int=6, amplitude::Float64=0.2, 
                        n_markers::Int=100)
    # Créer un vecteur avec le type approprié
    markers = Vector{Tuple{Float64, Float64}}(undef, n_markers+1)
    
    for i in 1:n_markers
        # Angle pour ce marqueur
        θ = 2.0 * π * (i-1) / n_markers
        
        # Rayon perturbé pour un effet cristallin
        r = base_radius * (1.0 + amplitude * cos(n_lobes * θ))
        
        # Coordonnées cartésiennes
        x = center_x + r * cos(θ)
        y = center_y + r * sin(θ)
        
        markers[i] = (x, y)
    end
    
    # Fermer la courbe en répétant le premier point
    markers[n_markers+1] = markers[1]
    
    # Définir les marqueurs dans le FrontTracker
    set_markers!(ft, markers, true)
    return ft
end

"""
    get_fluid_polygon(ft::FrontTracker)

Returns a polygon representing the fluid domain bounded by the interface.
"""
function get_fluid_polygon(ft::FrontTracker)
    if length(ft.markers) < 3
        return nothing
    end
    
    if ft.is_closed
        # For a closed interface, directly return the polygon
        if !LibGEOS.isValid(ft.interface_poly)
            # Try to fix invalid polygon
            return LibGEOS.buffer(ft.interface_poly, 0.0)
        end
        return ft.interface_poly
    else
        # For an open interface, we need to define domain closure
        # Using convex hull as an approximation
        return LibGEOS.convexhull(ft.interface_poly)
    end
end

"""
    is_point_inside(ft::FrontTracker, x::Float64, y::Float64)

Checks if a point is inside the interface.
"""
function is_point_inside(ft::FrontTracker, x::Float64, y::Float64)
    if isnothing(ft.interface_poly)
        return false
    end
    
    point = LibGEOS.Point(x, y)
    return LibGEOS.contains(ft.interface_poly, point)
end

"""
    get_intersection(ft::FrontTracker, other_geometry)

Calculates the intersection with another geometry.
"""
function get_intersection(ft::FrontTracker, other_geometry)
    if isnothing(ft.interface_poly)
        return nothing
    end
    
    return LibGEOS.intersection(other_geometry, ft.interface_poly)
end


"""
    sdf(ft::FrontTracker, x::Float64, y::Float64)

Calculates the signed distance function for a given point.
"""
function sdf(ft::FrontTracker, x, y)
    if isnothing(ft.interface)
        return Inf
    end
    
    # Create a point using LibGEOS
    point = LibGEOS.Point(x, y)
    
    # Calculate the distance to the interface
    distance = LibGEOS.distance(point, ft.interface)
    
    # Determine the sign (negative inside, positive outside)
    is_inside_val = is_point_inside(ft, x, y)
    
    return is_inside_val ? -distance : distance
end

"""
    compute_marker_normals(ft::FrontTracker, markers=nothing)

Calculates the normal vectors for each marker of the interface.
"""
function compute_marker_normals(ft::FrontTracker, markers=nothing)
    if isnothing(markers)
        markers = ft.markers
    end
    
    if length(markers) < 3
        return [(0.0, 1.0) for _ in markers]
    end
    
    normals = []
    n_markers = length(markers)
    is_closed = ft.is_closed
    
    # For maintaining orientation consistency
    prev_normal = nothing
    
    for i in 1:n_markers
        # Handle indices for previous and next points with boundary conditions
        prev_idx = is_closed ? mod1(i-1, n_markers) : max(1, i-1)
        next_idx = is_closed ? mod1(i+1, n_markers) : min(n_markers, i+1)
        
        # MODIFICATION IMPORTANTE: Utiliser la méthode des tangentes pour tous les points du premier marqueur
        # Pour les contours fermés, traiter le premier marqueur comme un point spécial
        if is_closed && (i == 1 || i == n_markers && markers[1] == markers[end])
            # Utiliser les points adjacents pour calculer les tangentes
            prev_idx = n_markers > 2 ? n_markers - 1 : 1
            next_idx = 2
            
            t1_x = markers[i][1] - markers[prev_idx][1]
            t1_y = markers[i][2] - markers[prev_idx][2]
            
            t2_x = markers[next_idx][1] - markers[i][1]
            t2_y = markers[next_idx][2] - markers[i][2]
            
            # Normaliser les vecteurs tangents
            t1_len = sqrt(t1_x^2 + t1_y^2)
            t2_len = sqrt(t2_x^2 + t2_y^2)
            
            if t1_len > 0 && t2_len > 0
                t1_x, t1_y = t1_x/t1_len, t1_y/t1_len
                t2_x, t2_y = t2_x/t2_len, t2_y/t2_len
                
                # Moyenner les tangentes
                tx = (t1_x + t2_x)
                ty = (t1_y + t2_y)
                
                # Normaliser
                t_len = sqrt(tx^2 + ty^2)
                if t_len > 0
                    tx, ty = tx/t_len, ty/t_len
                    
                    # La normale est perpendiculaire à la tangente
                    n_x, n_y = -ty, tx
                    
                    # Vérifier l'orientation
                    test_x = markers[i][1] + 1e-3 * n_x
                    test_y = markers[i][2] + 1e-3 * n_y
                    if is_point_inside(ft, test_x, test_y)
                        n_x, n_y = -n_x, -n_y
                    end
                    
                    push!(normals, (n_x, n_y))
                    prev_normal = [n_x, n_y]
                    
                    # Si c'est le dernier point qui est dupliqué, assurez-vous de sauter l'itération
                    if i == n_markers && markers[1] == markers[end]
                        continue
                    end
                else
                    push!(normals, (0.0, 1.0))
                    prev_normal = [0.0, 1.0]
                end
                
                continue
            end
        end
        # Special handling for endpoints of open curves
        if !is_closed && (i == 1 || i == n_markers)
            # For first point in open curve, use forward difference
            if i == 1
                # Use the vector from first to second marker as tangent
                t_x = markers[2][1] - markers[1][1]
                t_y = markers[2][2] - markers[1][2]
                
                # Normalize tangent
                t_len = sqrt(t_x^2 + t_y^2)
                if t_len > 0
                    t_x /= t_len
                    t_y /= t_len
                    
                    # Rotate 90° to get normal (outward pointing)
                    n_x, n_y = -t_y, t_x
                else
                    n_x, n_y = 0.0, 1.0
                end
                
                push!(normals, (n_x, n_y))
                prev_normal = [n_x, n_y]
                continue
            end
            
            # For last point in open curve, use backward difference
            if i == n_markers
                # Use the vector from second-to-last to last marker as tangent
                t_x = markers[n_markers][1] - markers[n_markers-1][1]
                t_y = markers[n_markers][2] - markers[n_markers-1][2]
                
                # Normalize tangent
                t_len = sqrt(t_x^2 + t_y^2)
                if t_len > 0
                    t_x /= t_len
                    t_y /= t_len
                    
                    # Rotate 90° to get normal (outward pointing)
                    n_x, n_y = -t_y, t_x
                    
                    # Ensure consistency with previous normal
                    if !isnothing(prev_normal)
                        dot_product = n_x * prev_normal[1] + n_y * prev_normal[2]
                        if dot_product < 0
                            n_x, n_y = -n_x, -n_y
                        end
                    end
                else
                    n_x, n_y = prev_normal[1], prev_normal[2]
                end
                
                push!(normals, (n_x, n_y))
                continue
            end
        end
        
        # Regular points use the osculating circle method
        # Points P1, P2, P3 for calculating the osculating circle
        p1 = collect(markers[prev_idx])
        p2 = collect(markers[i])
        p3 = collect(markers[next_idx])
        
        # Rest of the function remains the same...
        
        try
            # Check if points are distinct
            if (norm(p1 - p2) < 1e-10 || 
                norm(p2 - p3) < 1e-10 || 
                norm(p3 - p1) < 1e-10)
                error("Points too close")
            end
            
            # Equations to find the center of the circle through three points
            # Linear system: Ax = b where x = [center_x, center_y, -r²]
            A = [2*p1[1] 2*p1[2] 1;
                 2*p2[1] 2*p2[2] 1;
                 2*p3[1] 2*p3[2] 1]
            
            b = [p1[1]^2 + p1[2]^2,
                 p2[1]^2 + p2[2]^2,
                 p3[1]^2 + p3[2]^2]
            
            # Solve for the circle center
            x = A \ b
            center = x[1:2]
            
            # Calculate the normal vector (from point to center)
            normal_vector = center - p2
            
            # Normalize the vector
            norm_val = norm(normal_vector)
            if norm_val < 1e-10
                error("Norm too small")
            end
            normal_vector = normal_vector / norm_val
            
            # Check orientation
            if is_closed
                # For closed interface, check if normal points outward
                test_point = p2 + 1e-3 * normal_vector
                if is_point_inside(ft, test_point[1], test_point[2])
                    normal_vector = -normal_vector  # Invert normal
                end
            else
                # For open interface, ensure consistency with previous normal
                if !isnothing(prev_normal) && i > 1
                    # Calculate dot product to check if normals point in similar directions
                    dot_product = normal_vector[1] * prev_normal[1] + normal_vector[2] * prev_normal[2]
                    if dot_product < 0
                        normal_vector = -normal_vector
                    end
                elseif i == 1
                    # For first point, prefer upward-pointing normal
                    if normal_vector[2] < 0
                        normal_vector = -normal_vector
                    end
                end
            end
            
            prev_normal = normal_vector
            push!(normals, (normal_vector[1], normal_vector[2]))
            
        catch e
            # Fallback: use tangent method if osculating circle fails
            if i > 1 || is_closed
                t1_x = markers[i][1] - markers[prev_idx][1]
                t1_y = markers[i][2] - markers[prev_idx][2]
            else
                t1_x = markers[next_idx][1] - markers[i][1]
                t1_y = markers[next_idx][2] - markers[i][2]
            end
            
            if i < n_markers || is_closed
                t2_x = markers[next_idx][1] - markers[i][1]
                t2_y = markers[next_idx][2] - markers[i][2]
            else
                t2_x = markers[i][1] - markers[prev_idx][1]
                t2_y = markers[i][2] - markers[prev_idx][2]
            end
            
            # Normalize tangent vectors
            t1_len = sqrt(t1_x^2 + t1_y^2)
            t2_len = sqrt(t2_x^2 + t2_y^2)
            
            if t1_len > 0 && t2_len > 0
                t1_x, t1_y = t1_x/t1_len, t1_y/t1_len
                t2_x, t2_y = t2_x/t2_len, t2_y/t2_len
                
                # Average tangents
                tx = (t1_x + t2_x)
                ty = (t1_y + t2_y)
                
                # Normalize
                t_len = sqrt(tx^2 + ty^2)
                if t_len > 0
                    tx, ty = tx/t_len, ty/t_len
                    
                    # Normal is perpendicular to tangent
                    nx, ny = -ty, tx
                    
                    # Check orientation
                    if is_closed
                        test_x = markers[i][1] + 1e-3 * nx
                        test_y = markers[i][2] + 1e-3 * ny
                        if is_point_inside(ft, test_x, test_y)
                            nx, ny = -nx, -ny
                        end
                    else
                        # Ensure consistency
                        normal_vector = [nx, ny]
                        if !isnothing(prev_normal) && i > 1
                            dot_product = normal_vector[1] * prev_normal[1] + normal_vector[2] * prev_normal[2]
                            if dot_product < 0
                                nx, ny = -nx, -ny
                            end
                        elseif i == 1
                            if ny < 0
                                nx, ny = -nx, -ny
                            end
                        end
                    end
                    
                    prev_normal = [nx, ny]
                    push!(normals, (nx, ny))
                else
                    # Degenerate case
                    if !isnothing(prev_normal)
                        push!(normals, (prev_normal[1], prev_normal[2]))
                    else
                        push!(normals, (0.0, 1.0))
                    end
                end
            else
                # Degenerate case
                if !isnothing(prev_normal)
                    push!(normals, (prev_normal[1], prev_normal[2]))
                else
                    push!(normals, (0.0, 1.0))
                end
            end
        end
    end
    
    return normals
end

"""
    compute_volume_jacobian(ft::FrontTracker, x_faces::Vector{Float64}, y_faces::Vector{Float64}, epsilon::Float64=1e-6)

Calculates the volume Jacobian matrix for a given mesh and interface.
This is a more reliable Julia implementation that avoids Python interop issues.
"""
function compute_volume_jacobian(ft::FrontTracker, x_faces::AbstractVector{<:Real}, y_faces::AbstractVector{<:Real}, epsilon::Float64=1e-8)
    # Convert ranges to vectors if needed
    x_faces_vec = collect(x_faces)
    y_faces_vec = collect(y_faces)
    
    # Get mesh dimensions
    nx = length(x_faces_vec) - 1
    ny = length(y_faces_vec) - 1
    
    # Get markers and compute their normals
    markers = get_markers(ft)
    normals = compute_marker_normals(ft, markers)
    
    # Calculate original cell volumes
    fluid_poly = get_fluid_polygon(ft)
    original_volumes = Dict{Tuple{Int, Int}, Float64}()
    
    # For each cell in the mesh
    for i in 1:nx
        for j in 1:ny
            # Create cell coordinates properly for LibGEOS
            cell_coords = [
                [x_faces_vec[i], y_faces_vec[j]],
                [x_faces_vec[i+1], y_faces_vec[j]],
                [x_faces_vec[i+1], y_faces_vec[j+1]],
                [x_faces_vec[i], y_faces_vec[j+1]],
                [x_faces_vec[i], y_faces_vec[j]]  # Close the polygon
            ]
            
            # Create cell polygon
            cell_poly = LibGEOS.Polygon([cell_coords])
            
            # Calculate intersection with fluid polygon
            intersection = LibGEOS.intersection(cell_poly, fluid_poly)
            
            # Store the fluid volume for this cell
            if LibGEOS.isEmpty(intersection)
                original_volumes[(i, j)] = 0.0
            else
                original_volumes[(i, j)] = LibGEOS.area(intersection)
            end
        end
    end
    
    # Initialize dictionary for storing the Jacobian
    volume_jacobian = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}()
    for key in keys(original_volumes)
        volume_jacobian[key] = []
    end
    
    # Track which markers have entries
    markers_with_entries = Set{Int}()
    
    # For closed interfaces, only process unique markers
    n_unique_markers = ft.is_closed ? length(markers) - 1 : length(markers)
    
    for marker_idx in 1:n_unique_markers
        # Original marker position
        original_marker = markers[marker_idx]
        
        # Calculate perturbed positions (both positive and negative)
        normal = normals[marker_idx]
        
        # Positive perturbation
        pos_perturbed_marker = (
            original_marker[1] + epsilon * normal[1],
            original_marker[2] + epsilon * normal[2]
        )
        
        # Negative perturbation
        neg_perturbed_marker = (
            original_marker[1] - epsilon * normal[1],
            original_marker[2] - epsilon * normal[2]
        )
        
        # Create copies of markers with positive perturbation
        pos_perturbed_markers = copy(markers)
        pos_perturbed_markers[marker_idx] = pos_perturbed_marker
        
        # Create copies of markers with negative perturbation
        neg_perturbed_markers = copy(markers)
        neg_perturbed_markers[marker_idx] = neg_perturbed_marker
        
        # Update last marker if interface is closed and first marker is perturbed
        if ft.is_closed && marker_idx == 1
            pos_perturbed_markers[end] = pos_perturbed_marker
            neg_perturbed_markers[end] = neg_perturbed_marker
        end
        
        # Create new front trackers with perturbed markers
        pos_perturbed_tracker = FrontTracker(pos_perturbed_markers, ft.is_closed)
        neg_perturbed_tracker = FrontTracker(neg_perturbed_markers, ft.is_closed)
        
        pos_fluid_poly = get_fluid_polygon(pos_perturbed_tracker)
        neg_fluid_poly = get_fluid_polygon(neg_perturbed_tracker)
        
        # Track the max Jacobian value for this marker
        max_jac_value = 0.0
        max_jac_cell = nothing
        
        # Calculate perturbed volumes using central differencing
        for ((i, j), _) in original_volumes
            # Create cell coordinates
            cell_coords = [
                [x_faces_vec[i], y_faces_vec[j]],
                [x_faces_vec[i+1], y_faces_vec[j]],
                [x_faces_vec[i+1], y_faces_vec[j+1]],
                [x_faces_vec[i], y_faces_vec[j+1]],
                [x_faces_vec[i], y_faces_vec[j]]  # Close the polygon
            ]
            
            # Create cell polygon
            cell_poly = LibGEOS.Polygon([cell_coords])
            
            # Calculate intersection with positive perturbed fluid polygon
            pos_intersection = LibGEOS.intersection(cell_poly, pos_fluid_poly)
            pos_volume = LibGEOS.isEmpty(pos_intersection) ? 0.0 : LibGEOS.area(pos_intersection)
            
            # Calculate intersection with negative perturbed fluid polygon
            neg_intersection = LibGEOS.intersection(cell_poly, neg_fluid_poly)
            neg_volume = LibGEOS.isEmpty(neg_intersection) ? 0.0 : LibGEOS.area(neg_intersection)
            
            # Calculate Jacobian value using central differencing
            jacobian_value = (pos_volume - neg_volume) / (2.0 * epsilon)
            
            # Store significant changes and track maximum value
            if abs(jacobian_value) > 1e-10
                push!(volume_jacobian[(i, j)], (marker_idx-1, jacobian_value))
                push!(markers_with_entries, marker_idx)
                
                if abs(jacobian_value) > abs(max_jac_value)
                    max_jac_value = jacobian_value
                    max_jac_cell = (i, j)
                end
            elseif abs(jacobian_value) > abs(max_jac_value)
                max_jac_value = jacobian_value
                max_jac_cell = (i, j)
            end
        end
        
        # If this marker has no entries, add its maximum value entry
        if marker_idx ∉ markers_with_entries && max_jac_cell !== nothing
            push!(volume_jacobian[max_jac_cell], (marker_idx-1, max_jac_value))
            push!(markers_with_entries, marker_idx)
        end
    end

        # For closed interfaces, copy the entries of the first marker to the last marker
    if ft.is_closed && length(markers) > 1
        # Get the index of the last marker
        last_marker_idx = length(markers)
        
        # For each cell with entries
        for ((i, j), entries) in volume_jacobian
            # Check if the first marker has an entry for this cell
            for (marker_idx, jacobian_value) in entries
                if marker_idx == 0
                    # Copy the first marker's entry to the last marker
                    push!(volume_jacobian[(i, j)], (last_marker_idx, jacobian_value))
                    push!(markers_with_entries, last_marker_idx+1)
                    break
                end
            end
        end
    end
    
    return volume_jacobian
end

"""
    fluid_cell_properties(mesh::Mesh{2}, front::FrontTracker)

Calcule pour chaque cellule (i,j):
- la fraction fluide α_{ij}
- la capacité volumique V_{ij}
- les coordonnées du centroïde fluide (X_{i,j}, Y_{i,j})

Retourne (fractions, volumes, centroids_x, centroids_y)
"""
function fluid_cell_properties(mesh::Mesh{2}, front::FrontTracker)
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialisation des matrices de résultats
    fractions = zeros(nx+1, ny+1)  # Fraction fluide
    volumes = zeros(nx+1, ny+1)    # Volume fluide
    centroids_x = zeros(nx+1, ny+1)  # Coordonnées du centroïde fluide en x
    centroids_y = zeros(nx+1, ny+1)  # Coordonnées du centroïde fluide en y
    cell_types = zeros(Int, nx+1, ny+1)  # Type de cellule (0: solide/empty, 1: fluide/full, -1: cut)
    
    # Récupération du domaine fluide
    fluid_poly = get_fluid_polygon(front)
    
    # Parcours de toutes les cellules
    for i in 1:nx
        for j in 1:ny
            # Création du polygone représentant la cellule
            cell_coords = [
                [x_nodes[i], y_nodes[j]],
                [x_nodes[i+1], y_nodes[j]],
                [x_nodes[i+1], y_nodes[j+1]],
                [x_nodes[i], y_nodes[j+1]],
                [x_nodes[i], y_nodes[j]]  # Fermer le polygone
            ]
            cell_poly = LibGEOS.Polygon([cell_coords])
            
            # Calcul de l'aire de la cellule
            cell_area = LibGEOS.area(cell_poly)
            
            # Calcul de l'intersection avec le domaine fluide
            if LibGEOS.intersects(cell_poly, fluid_poly)
                intersection = LibGEOS.intersection(cell_poly, fluid_poly)
                
                if !LibGEOS.isEmpty(intersection)
                    # Calcul de la fraction fluide
                    fluid_area = LibGEOS.area(intersection)
                    fractions[i, j] = fluid_area / cell_area
                    volumes[i, j] = fluid_area
                    
                    # Calcul du centroïde de la partie fluide
                    centroid = LibGEOS.centroid(intersection)
                    centroids_x[i, j] = GeoInterface.x(centroid)
                    centroids_y[i, j] = GeoInterface.y(centroid)
                    
                    # Détermination du type de cellule
                    if isapprox(fractions[i, j], 1.0, atol=1e-10)
                        cell_types[i, j] = 1   # Cellule complètement fluide
                    elseif isapprox(fractions[i, j], 0.0, atol=1e-10)
                        cell_types[i, j] = 0   # Cellule vide
                    else
                        cell_types[i, j] = -1  # Cellule coupée
                    end
                else
                    # Cellule entièrement solide
                    centroids_x[i, j] = (x_nodes[i] + x_nodes[i+1]) / 2
                    centroids_y[i, j] = (y_nodes[j] + y_nodes[j+1]) / 2
                    cell_types[i, j] = 0  # Cellule vide
                end
            else
                # Vérification si le centre est dans le fluide
                center_x = (x_nodes[i] + x_nodes[i+1]) / 2
                center_y = (y_nodes[j] + y_nodes[j+1]) / 2
                
                if is_point_inside(front, center_x, center_y)
                    fractions[i, j] = 1.0
                    volumes[i, j] = cell_area
                    centroids_x[i, j] = center_x
                    centroids_y[i, j] = center_y
                    cell_types[i, j] = 1  # Cellule complètement fluide
                else
                    centroids_x[i, j] = center_x
                    centroids_y[i, j] = center_y
                    cell_types[i, j] = 0  # Cellule vide
                end
            end
        end
    end

    return fractions, volumes, centroids_x, centroids_y, cell_types
end

"""
    compute_surface_capacities(mesh::Mesh{2}, front::FrontTracker)

Calcule les capacités de surface:
- A^x_{i,j}: longueur mouillée des faces verticales
- A^y_{i,j}: longueur mouillée des faces horizontales

Retourne (Ax, Ay)
"""
function compute_surface_capacities(mesh::Mesh{2}, front::FrontTracker)
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialisation des matrices de résultats
    Ax = zeros(nx+1, ny+1)    # Faces verticales
    Ay = zeros(nx+1, ny+1)    # Faces horizontales
    
    # Récupération du domaine fluide
    fluid_poly = get_fluid_polygon(front)
    
    # Calculer les fractions fluides pour toutes les cellules d'abord
    fractions, _, _, _,_ = fluid_cell_properties(mesh, front)
    
    # Calcul pour les faces verticales (Ax)
    for i in 1:nx+1
        for j in 1:ny
            x = x_nodes[i]
            y_min, y_max = y_nodes[j], y_nodes[j+1]
            
            # Création d'une ligne pour la face
            face_line = LibGEOS.LineString([[x, y_min], [x, y_max]])
            
            # Si la face est entre deux cellules (comme dans le code Python)
            if 1 < i <= nx
                # Identifier les cellules gauche et droite
                left_cell_fluid = fractions[i-1, j] > 0
                right_cell_fluid = fractions[i, j] > 0
                
                if left_cell_fluid && right_cell_fluid && fractions[i-1, j] == 1.0 && fractions[i, j] == 1.0
                    # Face entre deux cellules entièrement fluides
                    Ax[i, j] = y_max - y_min
                elseif !left_cell_fluid && !right_cell_fluid
                    # Face entre deux cellules entièrement solides
                    Ax[i, j] = 0.0
                else
                    # Face à la frontière fluide/solide ou impliquant une cut cell
                    if isnothing(front.interface)
                        # Sans interface définie
                        if left_cell_fluid && right_cell_fluid
                            Ax[i, j] = y_max - y_min
                        end
                    else
                        # Vérifier si la face est intersectée par l'interface
                        if LibGEOS.intersects(face_line, fluid_poly)
                            intersection = LibGEOS.intersection(face_line, fluid_poly)
                            
                            # Calculer la longueur correctement selon le type de géométrie
                            if isa(intersection, LibGEOS.LineString)
                                Ax[i, j] = LibGEOS.geomLength(intersection)
                            elseif isa(intersection, LibGEOS.MultiLineString)
                                # Somme des longueurs pour géométries multiples
                                total_length = 0.0
                                for k in 1:LibGEOS.getNumGeometries(intersection)
                                    line = LibGEOS.getGeometry(intersection, k-1)
                                    total_length += LibGEOS.geomLength(line)
                                end
                                Ax[i, j] = total_length
                            elseif isa(intersection, LibGEOS.Point) || isa(intersection, LibGEOS.MultiPoint)
                                # Un point n'a pas de longueur
                                Ax[i, j] = 0.0
                            else
                                # Type de géométrie inconnu - utiliser le test du point milieu
                                mid_y = (y_min + y_max) / 2
                                if is_point_inside(front, x, mid_y)
                                    Ax[i, j] = y_max - y_min
                                end
                            end
                        else
                            # Face non intersectée par l'interface
                            mid_y = (y_min + y_max) / 2
                            if is_point_inside(front, x, mid_y)
                                Ax[i, j] = y_max - y_min
                            else
                                Ax[i, j] = 0.0
                            end
                        end
                    end
                end
            else
                # Faces aux bords du domaine (i=1 ou i=nx+1)
                # Vérifier si la face intersecte le domaine fluide
                if LibGEOS.intersects(face_line, fluid_poly)
                    intersection = LibGEOS.intersection(face_line, fluid_poly)
                    if isa(intersection, LibGEOS.LineString)
                        Ax[i, j] = LibGEOS.geomLength(intersection)
                    elseif isa(intersection, LibGEOS.MultiLineString)
                        total_length = 0.0
                        for k in 1:LibGEOS.getNumGeometries(intersection)
                            line = LibGEOS.getGeometry(intersection, k-1)
                            total_length += LibGEOS.geomLength(line)
                        end
                        Ax[i, j] = total_length
                    end
                else
                    # Vérifier si le milieu est dans le fluide
                    mid_y = (y_min + y_max) / 2
                    if is_point_inside(front, x, mid_y)
                        Ax[i, j] = y_max - y_min
                    end
                end
            end
            
            # Valider la valeur calculée (ne devrait pas dépasser la hauteur de la cellule)
            if Ax[i, j] > (y_max - y_min) * (1 + 1e-10)
                Ax[i, j] = y_max - y_min
            end
        end
    end
    
    # Même logique pour les faces horizontales (Ay)
    for i in 1:nx
        for j in 1:ny+1
            y = y_nodes[j]
            x_min, x_max = x_nodes[i], x_nodes[i+1]
            
            # Création d'une ligne pour la face
            face_line = LibGEOS.LineString([[x_min, y], [x_max, y]])
            
            # Si la face est entre deux cellules
            if 1 < j <= ny
                bottom_cell_fluid = fractions[i, j-1] > 0
                top_cell_fluid = fractions[i, j] > 0
                
                if bottom_cell_fluid && top_cell_fluid && fractions[i, j-1] == 1.0 && fractions[i, j] == 1.0
                    # Face entre deux cellules entièrement fluides
                    Ay[i, j] = x_max - x_min
                elseif !bottom_cell_fluid && !top_cell_fluid
                    # Face entre deux cellules entièrement solides
                    Ay[i, j] = 0.0
                else
                    # Face à la frontière fluide/solide ou impliquant une cut cell
                    if isnothing(front.interface)
                        if bottom_cell_fluid && top_cell_fluid
                            Ay[i, j] = x_max - x_min
                        end
                    else
                        # Vérifier si la face est intersectée par l'interface
                        if LibGEOS.intersects(face_line, fluid_poly)
                            intersection = LibGEOS.intersection(face_line, fluid_poly)
                            
                            # Calculer la longueur selon le type de géométrie
                            if isa(intersection, LibGEOS.LineString)
                                Ay[i, j] = LibGEOS.geomLength(intersection)
                            elseif isa(intersection, LibGEOS.MultiLineString)
                                total_length = 0.0
                                for k in 1:LibGEOS.getNumGeometries(intersection)
                                    line = LibGEOS.getGeometry(intersection, k-1)
                                    total_length += LibGEOS.geomLength(line)
                                end
                                Ay[i, j] = total_length
                            elseif isa(intersection, LibGEOS.Point) || isa(intersection, LibGEOS.MultiPoint)
                                Ay[i, j] = 0.0
                            else
                                # Type de géométrie inconnu - utiliser le test du point milieu
                                mid_x = (x_min + x_max) / 2
                                if is_point_inside(front, mid_x, y)
                                    Ay[i, j] = x_max - x_min
                                end
                            end
                        else
                            # Face non intersectée
                            mid_x = (x_min + x_max) / 2
                            if is_point_inside(front, mid_x, y)
                                Ay[i, j] = x_max - x_min
                            else
                                Ay[i, j] = 0.0
                            end
                        end
                    end
                end
            else
                # Faces aux bords du domaine (j=1 ou j=ny+1)
                if LibGEOS.intersects(face_line, fluid_poly)
                    intersection = LibGEOS.intersection(face_line, fluid_poly)
                    if isa(intersection, LibGEOS.LineString)
                        Ay[i, j] = LibGEOS.geomLength(intersection)
                    elseif isa(intersection, LibGEOS.MultiLineString)
                        total_length = 0.0
                        for k in 1:LibGEOS.getNumGeometries(intersection)
                            line = LibGEOS.getGeometry(intersection, k-1)
                            total_length += LibGEOS.geomLength(line)
                        end
                        Ay[i, j] = total_length
                    end
                else
                    mid_x = (x_min + x_max) / 2
                    if is_point_inside(front, mid_x, y)
                        Ay[i, j] = x_max - x_min
                    end
                end
            end
            
            # Valider la valeur calculée
            if Ay[i, j] > (x_max - x_min) * (1 + 1e-10)
                Ay[i, j] = x_max - x_min
            end
        end
    end
    
    return Ax, Ay
end

"""
    compute_second_type_capacities(mesh::Mesh{2}, front::FrontTracker, centroids_x, centroids_y)

Calcule les capacités de second type:
- W^x: volume fluide entre les centres de masse horizontalement adjacents
- W^y: volume fluide entre les centres de masse verticalement adjacents
- B^x: longueur mouillée des lignes verticales passant par les centroïdes
- B^y: longueur mouillée des lignes horizontales passant par les centroïdes

Retourne (Wx, Wy, Bx, By)
"""
function compute_second_type_capacities(mesh::Mesh{2}, front::FrontTracker, 
                                       centroids_x, centroids_y)
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialisation des matrices de résultats
    Wx = zeros(nx+1, ny+1)    # Capacités horizontales entre centroïdes
    Wy = zeros(nx+1, ny+1)    # Capacités verticales entre centroïdes
    Bx = zeros(nx+1, ny+1)      # Longueur mouillée des lignes verticales
    By = zeros(nx+1, ny+1)      # Longueur mouillée des lignes horizontales
    
    # Récupération du domaine fluide
    fluid_poly = get_fluid_polygon(front)

    # Récupération des volumes fluides pour chaque cellule
    fractions, volumes, _, _,_ = fluid_cell_properties(mesh, front)
    
     # Calcul des capacités de volume horizontales Wx
    for i in 1:nx
        for j in 1:ny
            # Uniquement calculer si au moins une des cellules contient du fluide
            if volumes[i, j] > 0 || volumes[i+1, j] > 0
                # Définition du domaine d'intégration entre les centroïdes
                x_left = centroids_x[i, j]
                x_right = centroids_x[i+1, j]
                y_min, y_max = y_nodes[j], y_nodes[j+1]
                
                # Création du polygone pour le domaine d'intégration
                poly_coords = [
                    [x_left, y_min],
                    [x_right, y_min],
                    [x_right, y_max],
                    [x_left, y_max],
                    [x_left, y_min]
                ]
                poly = LibGEOS.Polygon([poly_coords])
                
                # Calcul de l'intersection avec le domaine fluide
                if LibGEOS.intersects(poly, fluid_poly)
                    intersection = LibGEOS.intersection(poly, fluid_poly)
                    Wx[i+1, j] = LibGEOS.area(intersection)
                else
                    # Vérification si le milieu est dans le fluide
                    mid_x = (x_left + x_right) / 2
                    mid_y = (y_min + y_max) / 2
                    if is_point_inside(front, mid_x, mid_y)
                        Wx[i+1, j] = LibGEOS.area(poly)
                    else
                        # Si complètement dans le solide
                        Wx[i+1, j] = 0.0
                    end
                end
            else
                # Si les deux cellules sont solides, Wx est 0
                Wx[i+1, j] = 0.0
            end
        end
    end
    
    # Calcul des capacités de volume verticales Wy
    for i in 1:nx
        for j in 1:ny
            if volumes[i, j] > 0 || volumes[i, j+1] > 0
                # Définition du domaine d'intégration entre les centroïdes
                y_bottom = centroids_y[i, j]
                y_top = centroids_y[i, j+1]
                x_min, x_max = x_nodes[i], x_nodes[i+1]
                
                # Création du polygone pour le domaine d'intégration
                poly_coords = [
                    [x_min, y_bottom],
                    [x_max, y_bottom],
                    [x_max, y_top],
                    [x_min, y_top],
                    [x_min, y_bottom]
                ]
                poly = LibGEOS.Polygon([poly_coords])
                
                # Calcul de l'intersection avec le domaine fluide
                if LibGEOS.intersects(poly, fluid_poly)
                    intersection = LibGEOS.intersection(poly, fluid_poly)
                    Wy[i, j+1] = LibGEOS.area(intersection)
                else
                    # Vérification si le milieu est dans le fluide
                    mid_x = (x_min + x_max) / 2
                    mid_y = (y_bottom + y_top) / 2
                    if is_point_inside(front, mid_x, mid_y)
                        Wy[i, j+1] = LibGEOS.area(poly)
                    else
                        # Si complètement dans le solide
                        Wy[i, j+1] = 0.0
                    end
                end
            else
                # Si les deux cellules sont solides, Wy est 0
                Wy[i, j+1] = 0.0
            end
        end
    end
    
       # Calcul des longueurs B^x et B^y
    for i in 1:nx
        for j in 1:ny
            # Récupérer le volume et la fraction fluide pour cette cellule
            cell_volume = volumes[i, j]
            cell_fraction = fractions[i, j]
            x_cm = centroids_x[i, j]
            y_cm = centroids_y[i, j]
            
            # Calculer uniquement pour les cellules avec du fluide
            if cell_volume > 0
                # Dimensions de la cellule
                cell_height = y_nodes[j+1] - y_nodes[j]
                cell_width = x_nodes[i+1] - x_nodes[i]
                
                # Cellule complètement fluide vs. cellule coupée
                if cell_fraction == 1.0
                    # Pour une cellule complètement fluide
                    Bx[i, j] = cell_height
                    By[i, j] = cell_width
                else
                    # Pour une cellule coupée - calculer les intersections
                    
                    # Ligne verticale passant par le centroïde fluide
                    vertical_line = LibGEOS.LineString([
                        [x_cm, y_nodes[j]],
                        [x_cm, y_nodes[j+1]]
                    ])
                    
                    # Ligne horizontale passant par le centroïde fluide
                    horizontal_line = LibGEOS.LineString([
                        [x_nodes[i], y_cm],
                        [x_nodes[i+1], y_cm]
                    ])
                    
                    # Calcul de Bx - longueur mouillée verticale
                    if LibGEOS.intersects(vertical_line, fluid_poly)
                        intersection = LibGEOS.intersection(vertical_line, fluid_poly)
                        
                        if isa(intersection, LibGEOS.LineString)
                            Bx[i, j] = LibGEOS.geomLength(intersection)
                        elseif isa(intersection, LibGEOS.MultiLineString)
                            # Somme des longueurs pour géométries multiples
                            total_length = 0.0
                            for k in 1:LibGEOS.getNumGeometries(intersection)
                                line = LibGEOS.getGeometry(intersection, k-1)
                                total_length += LibGEOS.geomLength(line)
                            end
                            Bx[i, j] = total_length
                        else
                            # Point ou autre géométrie - pas de longueur
                            Bx[i, j] = 0.0
                        end
                    else
                        # Si la ligne ne coupe pas l'interface, vérifier si elle est du côté fluide
                        # Utiliser le centre de la cellule comme point de test
                        y_center = (y_nodes[j] + y_nodes[j+1]) / 2
                        if is_point_inside(front, x_cm, y_center)
                            Bx[i, j] = cell_height
                        else
                            Bx[i, j] = 0.0
                        end
                    end
                    
                    # Calcul de By - longueur mouillée horizontale
                    if LibGEOS.intersects(horizontal_line, fluid_poly)
                        intersection = LibGEOS.intersection(horizontal_line, fluid_poly)
                        
                        if isa(intersection, LibGEOS.LineString)
                            By[i, j] = LibGEOS.geomLength(intersection)
                        elseif isa(intersection, LibGEOS.MultiLineString)
                            # Somme des longueurs pour géométries multiples
                            total_length = 0.0
                            for k in 1:LibGEOS.getNumGeometries(intersection)
                                line = LibGEOS.getGeometry(intersection, k-1)
                                total_length += LibGEOS.geomLength(line)
                            end
                            By[i, j] = total_length
                        else
                            # Point ou autre géométrie - pas de longueur
                            By[i, j] = 0.0
                        end
                    else
                        # Si la ligne ne coupe pas l'interface, vérifier si elle est du côté fluide
                        x_center = (x_nodes[i] + x_nodes[i+1]) / 2
                        if is_point_inside(front, x_center, y_cm)
                            By[i, j] = cell_width
                        else
                            By[i, j] = 0.0
                        end
                    end
                end
            else
                # Cellule sans fluide
                Bx[i, j] = 0.0
                By[i, j] = 0.0
            end
        end
    end
    return Wx, Wy, Bx, By
end

"""
    compute_interface_info(mesh::Mesh{2}, front::FrontTracker)

Calcule les longueurs d'interface et les points représentatifs dans chaque cellule coupée.
Retourne (interface_lengths, interface_points)
"""
function compute_interface_info(mesh::Mesh{2}, front::FrontTracker)
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    interface_lengths = Dict{Tuple{Int, Int}, Float64}()
    interface_points = Dict{Tuple{Int, Int}, Tuple{Float64, Float64}}()
    
    # Récupération de l'interface
    interface_line = front.interface
    
    for i in 1:nx
        for j in 1:ny
            # Création du polygone représentant la cellule
            cell_coords = [
                [x_nodes[i], y_nodes[j]],
                [x_nodes[i+1], y_nodes[j]],
                [x_nodes[i+1], y_nodes[j+1]],
                [x_nodes[i], y_nodes[j+1]],
                [x_nodes[i], y_nodes[j]]
            ]
            cell_poly = LibGEOS.Polygon([cell_coords])
            
            # Vérification si la cellule intersecte l'interface
            if LibGEOS.intersects(cell_poly, interface_line)
                intersection = LibGEOS.intersection(cell_poly, interface_line)
                total_length = 0.0
                weighted_x = 0.0
                weighted_y = 0.0
                for line in LibGEOS.getGeometries(intersection)
                    length = LibGEOS.geomLength(line)
                    total_length += length
                    
                    # Point milieu de ce segment
                    mid_point = LibGEOS.interpolate(line, 0.5)
                    
                    # Ajout au barycentre pondéré
                    weighted_x += GeoInterface.x(mid_point) * length
                    weighted_y += GeoInterface.y(mid_point) * length
                end
                # Stockage de la longueur d'interface
                interface_lengths[(i, j)] = total_length

                if total_length > 0
                    interface_points[(i, j)] = (weighted_x / total_length, 
                                               weighted_y / total_length)
                else
                    interface_points[(i, j)] = (0.0, 0.0)  # Cas où il n'y a pas d'interface
                end
                # MODIFICATION IMPORTANTE: Commenté pour éviter les erreurs de type
                       
                """
                if LibGEOS.getGeometryType(intersection) == :LineString
                    # Stockage de la longueur d'interface
                    interface_lengths[(i, j)] = LibGEOS.length(intersection)
                    
                    # Point représentatif (milieu de la ligne)
                    mid_point = LibGEOS.interpolate(intersection, 0.5)
                    interface_points[(i, j)] = (mid_point[1], mid_point[2])
                    
                elseif LibGEOS.getGeometryType(intersection) == :MultiLineString
                    # Calcul de la longueur totale et du barycentre pondéré
                    total_length = 0.0
                    weighted_x = 0.0
                    weighted_y = 0.0
                    
                    for line in LibGEOS.getGeometries(intersection)
                        length = LibGEOS.length(line)
                        total_length += length
                        
                        # Point milieu de ce segment
                        mid_point = LibGEOS.interpolate(line, 0.5)
                       
                        
                        # Ajout au barycentre pondéré
                        weighted_x += mid_point[1] * length
                        weighted_y += mid_point[2] * length
                    end
                    
                    interface_lengths[(i, j)] = total_length
                    
                    if total_length > 0
                        interface_points[(i, j)] = (weighted_x / total_length, 
                                                   weighted_y / total_length)
                    end
                end
                """
            end
        end
    end
    
    return interface_lengths, interface_points
end

"""
    compute_capacities(mesh::Mesh{2}, front::FrontTracker)

Calcule toutes les capacités géométriques pour un maillage et une interface donnés.
Retourne un dictionnaire avec tous les résultats.
"""
function compute_capacities(mesh::Mesh{2}, front::FrontTracker)
    # Calcul des propriétés des cellules
    fractions, volumes, centroids_x, centroids_y, cell_types = fluid_cell_properties(mesh, front)
    
    # Calcul des capacités de surface
    Ax, Ay = compute_surface_capacities(mesh, front)
    
    # Calcul des capacités de second type
    Wx, Wy, Bx, By = compute_second_type_capacities(mesh, front, centroids_x, centroids_y)
    
    # Calcul des informations d'interface
    interface_lengths, interface_points = compute_interface_info(mesh, front)
    
    # Retourne toutes les capacités dans un dictionnaire
    return Dict(
        :fractions => fractions,          # Fractions fluides α_{ij}
        :volumes => volumes,              # Capacités volumiques V_{ij}
        :centroids_x => centroids_x,      # Coordonnées x des centroïdes X_{i,j}
        :centroids_y => centroids_y,      # Coordonnées y des centroïdes Y_{i,j}
        :cell_types => cell_types,        # Types de cellules (0: empty, 1: full, -1: cut)
        :Ax => Ax,                        # Capacités des faces verticales A^x_{i,j}
        :Ay => Ay,                        # Capacités des faces horizontales A^y_{i,j}
        :Wx => Wx,                        # Capacités de volume horizontales W^x_{i+1/2,j}
        :Wy => Wy,                        # Capacités de volume verticales W^y_{i,j+1/2}
        :Bx => Bx,                        # Longueurs mouillées verticales B^x_{i,j}
        :By => By,                        # Longueurs mouillées horizontales B^y_{i,j}
        :interface_lengths => interface_lengths,  # Longueurs d'interface par cellule
        :interface_points => interface_points     # Points représentatifs sur l'interface
    )
end

# Compute Space-Time Capacities
"""
    compute_spacetime_capacities(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)

Calculates the space-time surface capacities using quadrature integration.
"""
function compute_spacetime_capacities(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)
    # Extract mesh information
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialize capacity arrays
    Ax_spacetime = zeros(nx+1, ny+1)
    Ay_spacetime = zeros(nx+1, ny+1)
    V_spacetime = zeros(nx+1, ny+1)  # Space-time volumes for each cell

    # Initialize space-time centroid arrays
    centroid_x_spacetime = zeros(nx+1, ny+1)
    centroid_y_spacetime = zeros(nx+1, ny+1)
    
    # Initialize additional arrays in the main function
    Bx_spacetime = zeros(nx+1, ny+1)
    By_spacetime = zeros(nx+1, ny+1)
    Wx_spacetime = zeros(nx+1, ny+1)
    Wy_spacetime = zeros(nx+1, ny+1)

    # Arrays to store crossing times
    t_crosses_x = zeros(nx+1, ny)
    t_crosses_y = zeros(nx, ny+1)
    t_crosses_v = zeros(nx, ny)  # For volume crossing times

    # Pre-compute fluid cell properties at initial and final times
    _, volumes_n, _, _, _ = fluid_cell_properties(mesh, front_n)
    _, volumes_np1, _, _, _ = fluid_cell_properties(mesh, front_np1)
    
    # 1. Calculate Ax capacities (vertical faces)
    for i in 1:nx+1
        for j in 1:ny
            x_face = x_nodes[i]
            y_min, y_max = y_nodes[j], y_nodes[j+1]
            face_height = y_max - y_min
            
            # Check if vertices are inside fluid at time n and n+1
            bottom_n = is_point_inside(front_n, x_face, y_min)
            bottom_np1 = is_point_inside(front_np1, x_face, y_min)
            top_n = is_point_inside(front_n, x_face, y_max)
            top_np1 = is_point_inside(front_np1, x_face, y_max)
            
            # Find all crossing times
            crossings = Float64[]
            push!(crossings, 0.0)  # Add t_n
            push!(crossings, dt)   # Add t_{n+1}
            
            # Add bottom and top crossing times if the state changes
            if bottom_n != bottom_np1
                t_cross_bottom = find_crossing_time(front_n, front_np1, x_face, y_min, dt)
                push!(crossings, t_cross_bottom)
            end
            if top_n != top_np1
                t_cross_top = find_crossing_time(front_n, front_np1, x_face, y_max, dt)
                push!(crossings, t_cross_top)
            end
            
            # Sort crossing times
            sort!(crossings)
            
            # Integrate capacity over each subinterval using trapezoidal rule
            for k in 1:(length(crossings)-1)
                t_start = crossings[k]
                t_end = crossings[k+1]
                
                # Calculate τ values (normalized time)
                τ_start = t_start / dt
                τ_end = t_end / dt
                
                # Create intermediate fronts at t_start and t_end by linear interpolation
                front_start = interpolate_front(front_n, front_np1, τ_start)
                front_end = interpolate_front(front_n, front_np1, τ_end)
                
                # Calculate face capacity at t_start and t_end
                Ax_start = calculate_face_capacity_x(front_start, x_face, y_min, y_max)
                Ax_end = calculate_face_capacity_x(front_end, x_face, y_min, y_max)
                
                # Trapezoidal rule integration: ∫f(t)dt ≈ (t_end - t_start) * (f(t_start) + f(t_end)) / 2
                subinterval = (t_end - t_start) * (Ax_start + Ax_end) / 2
                
                # Add contribution to total capacity
                Ax_spacetime[i, j] += subinterval
            end
            
            # Store representative crossing time for visualization
            if bottom_n != bottom_np1
                t_crosses_x[i, j] = find_crossing_time(front_n, front_np1, x_face, y_min, dt)
            elseif top_n != top_np1
                t_crosses_x[i, j] = find_crossing_time(front_n, front_np1, x_face, y_max, dt)
            else
                t_crosses_x[i, j] = 0.5 * dt  # Default if no crossing
            end
            
            # Safety bounds check
            Ax_spacetime[i, j] = clamp(Ax_spacetime[i, j], 0.0, face_height * dt)
        end
    end
    
    # 2. Calculate Ay capacities (horizontal faces) - similar approach
    for i in 1:nx
        for j in 1:ny+1
            y_face = y_nodes[j]
            x_min, x_max = x_nodes[i], x_nodes[i+1]
            face_width = x_max - x_min
            
            # Check if vertices are inside fluid at time n and n+1
            left_n = is_point_inside(front_n, x_min, y_face)
            left_np1 = is_point_inside(front_np1, x_min, y_face)
            right_n = is_point_inside(front_n, x_max, y_face)
            right_np1 = is_point_inside(front_np1, x_max, y_face)
            
            # Find all crossing times
            crossings = Float64[]
            push!(crossings, 0.0)  # Add t_n
            push!(crossings, dt)   # Add t_{n+1}
            
            # Add left and right crossing times if the state changes
            if left_n != left_np1
                t_cross_left = find_crossing_time(front_n, front_np1, x_min, y_face, dt)
                push!(crossings, t_cross_left)
            end
            if right_n != right_np1
                t_cross_right = find_crossing_time(front_n, front_np1, x_max, y_face, dt)
                push!(crossings, t_cross_right)
            end
            
            # Sort crossing times
            sort!(crossings)
            
            # Integrate capacity over each subinterval using trapezoidal rule
            for k in 1:(length(crossings)-1)
                t_start = crossings[k]
                t_end = crossings[k+1]
                
                # Calculate τ values
                τ_start = t_start / dt
                τ_end = t_end / dt
                
                # Create intermediate fronts
                front_start = interpolate_front(front_n, front_np1, τ_start)
                front_end = interpolate_front(front_n, front_np1, τ_end)
                
                # Calculate face capacity
                Ay_start = calculate_face_capacity_y(front_start, y_face, x_min, x_max)
                Ay_end = calculate_face_capacity_y(front_end, y_face, x_min, x_max)
                
                # Trapezoidal rule integration
                subinterval = (t_end - t_start) * (Ay_start + Ay_end) / 2
                
                # Add contribution
                Ay_spacetime[i, j] += subinterval
            end
            
            # Store representative crossing time
            if left_n != left_np1
                t_crosses_y[i, j] = find_crossing_time(front_n, front_np1, x_min, y_face, dt)
            elseif right_n != right_np1
                t_crosses_y[i, j] = find_crossing_time(front_n, front_np1, x_max, y_face, dt)
            else
                t_crosses_y[i, j] = 0.5 * dt
            end
            
            # Safety bounds check
            Ay_spacetime[i, j] = clamp(Ay_spacetime[i, j], 0.0, face_width * dt)
        end
    end

    # 3. Calculate V (volume) space-time capacities
    for i in 1:nx
        for j in 1:ny
            # Get cell dimensions
            cell_width = x_nodes[i+1] - x_nodes[i]
            cell_height = y_nodes[j+1] - y_nodes[j]
            cell_area = cell_width * cell_height
            
            # Initial and final volumes
            vol_n = volumes_n[i, j]
            vol_np1 = volumes_np1[i, j]
            
            """
            # If volume didn't change much, use simple trapezoidal rule
            if abs(vol_n - vol_np1) < 1e-10 * cell_area
                V_spacetime[i, j] = 0.5 * (vol_n + vol_np1) * dt
                t_crosses_v[i, j] = 0.5 * dt
                continue
            end
            """

            # Get cell vertices
            vertices = [
                (x_nodes[i], y_nodes[j]),        # Bottom-left
                (x_nodes[i+1], y_nodes[j]),      # Bottom-right
                (x_nodes[i+1], y_nodes[j+1]),    # Top-right
                (x_nodes[i], y_nodes[j+1])       # Top-left
            ]
            
            # Get cell center
            cell_center_x = (x_nodes[i] + x_nodes[i+1]) / 2
            cell_center_y = (y_nodes[j] + y_nodes[j+1]) / 2
            
            # Collect ALL relevant crossing times from the face crossings we already computed
            crossings = Float64[]
            push!(crossings, 0.0)  # Add t_n
            push!(crossings, dt)   # Add t_{n+1}
            
            # Add crossing times from x-faces (left and right face of this cell)
            if t_crosses_x[i, j] > 0
                push!(crossings, t_crosses_x[i, j])
            end
            if t_crosses_x[i+1, j] > 0
                push!(crossings, t_crosses_x[i+1, j])
            end
            
            # Add crossing times from y-faces (bottom and top face of this cell)
            if t_crosses_y[i, j] > 0
                push!(crossings, t_crosses_y[i, j])
            end
            if t_crosses_y[i, j+1] > 0
                push!(crossings, t_crosses_y[i, j+1])
            end
            
            """
            # Check if cell center changes state
            center_n = is_point_inside(front_n, cell_center_x, cell_center_y)
            center_np1 = is_point_inside(front_np1, cell_center_x, cell_center_y)
            
            if center_n != center_np1
                t_cross_center = find_crossing_time(front_n, front_np1, cell_center_x, cell_center_y, dt)
                push!(crossings, t_cross_center)
                t_crosses_v[i, j] = t_cross_center
            else
                t_crosses_v[i, j] = center_n ? 0.0 : dt
            end
            """
            
            # Sort and remove duplicates
            sort!(unique!(crossings))
            
            # For cells with significant volume change, add additional midpoints for better sampling
            if length(crossings) > 2 && abs(vol_np1 - vol_n) > 0.01 * cell_area
                refined_crossings = Float64[]
                for k in 1:(length(crossings)-1)
                    push!(refined_crossings, crossings[k])
                    # Add midpoint between each pair of crossing times
                    midpoint = 0.5 * (crossings[k] + crossings[k+1])
                    push!(refined_crossings, midpoint)
                end
                push!(refined_crossings, crossings[end])
                crossings = sort(refined_crossings)
            end
            
            # Initialize space-time volume for this cell
            V_spacetime[i, j] = 0.0
            weighted_centroid_x = 0.0
            weighted_centroid_y = 0.0
            
            # Perform quadrature integration with improved accuracy
            if length(crossings) >= 3  # Simpson's rule requires at least 3 points
                for k in 1:2:length(crossings)-2
                    t_start = crossings[k]
                    t_mid = crossings[k+1]
                    t_end = crossings[k+2]
                    
                    # Calculate τ values (normalized time)
                    τ_start = t_start / dt
                    τ_mid = t_mid / dt
                    τ_end = t_end / dt
                    
                    # Create intermediate fronts
                    front_start = interpolate_front(front_n, front_np1, τ_start)
                    front_mid = interpolate_front(front_n, front_np1, τ_mid)
                    front_end = interpolate_front(front_n, front_np1, τ_end)
                    
                    # Calculate volumes at these times
                    _, vol_start, cx_start, cy_start, _ = fluid_cell_properties(mesh, front_start)
                    _, vol_mid, cx_mid, cy_mid, _ = fluid_cell_properties(mesh, front_mid)
                    _, vol_end, cx_end, cy_end, _ = fluid_cell_properties(mesh, front_end)
                    
                    # Extract volumes for this cell
                    v_start = vol_start[i, j]
                    v_mid = vol_mid[i, j]
                    v_end = vol_end[i, j]

                    # Weight centroids by volumes
                    wx_start = v_start > 0 ? cx_start[i, j] * v_start : 0.0
                    wy_start = v_start > 0 ? cy_start[i, j] * v_start : 0.0
                    
                    wx_mid = v_mid > 0 ? cx_mid[i, j] * v_mid : 0.0
                    wy_mid = v_mid > 0 ? cy_mid[i, j] * v_mid : 0.0
                    
                    wx_end = v_end > 0 ? cx_end[i, j] * v_end : 0.0
                    wy_end = v_end > 0 ? cy_end[i, j] * v_end : 0.0
                    
                    
                    # Apply Simpson's rule for this interval: (h/6)*(f₁ + 4f₂ + f₃)
                    h = t_end - t_start
                    sub_integral = (h/6) * (v_start + 4*v_mid + v_end)

                    # Weighted centroid integration
                    sub_integral_wx = (h/6) * (wx_start + 4*wx_mid + wx_end)
                    sub_integral_wy = (h/6) * (wy_start + 4*wy_mid + wy_end)
                    
                    # Add contribution to space-time volume
                    V_spacetime[i, j] += sub_integral
                    weighted_centroid_x += sub_integral_wx
                    weighted_centroid_y += sub_integral_wy
                end
                
                # Handle any remaining interval with trapezoidal rule
                if length(crossings) % 2 == 0
                    last_idx = length(crossings)
                    t_start = crossings[last_idx-1]
                    t_end = crossings[last_idx]
                    
                    # Calculate τ values
                    τ_start = t_start / dt
                    τ_end = t_end / dt
                    
                    # Create intermediate fronts
                    front_start = interpolate_front(front_n, front_np1, τ_start)
                    front_end = interpolate_front(front_n, front_np1, τ_end)
                    
                    # Calculate volumes
                    _, vol_start, cx_start, cy_start, _ = fluid_cell_properties(mesh, front_start)
                    _, vol_end,  cx_end, cy_end, _ = fluid_cell_properties(mesh, front_end)
                    
                    # Extract volumes
                    v_start = vol_start[i, j]
                    v_end = vol_end[i, j]

                    # Weight centroids by volumes
                    wx_start = v_start > 0 ? cx_start[i, j] * v_start : 0.0
                    wy_start = v_start > 0 ? cy_start[i, j] * v_start : 0.0
                    
                    wx_end = v_end > 0 ? cx_end[i, j] * v_end : 0.0
                    wy_end = v_end > 0 ? cy_end[i, j] * v_end : 0.0
                    
                     # Apply trapezoidal rule
                    h = t_end - t_start
                    sub_integral_v = h * (v_start + v_end) / 2
                    sub_integral_wx = h * (wx_start + wx_end) / 2
                    sub_integral_wy = h * (wy_start + wy_end) / 2
                    
                    # Add contributions
                    V_spacetime[i, j] += sub_integral_v
                    weighted_centroid_x += sub_integral_wx
                    weighted_centroid_y += sub_integral_wy
                end
            else
                # Use standard trapezoidal rule for simple cases
                for k in 1:(length(crossings)-1)
                    t_start = crossings[k]
                    t_end = crossings[k+1]
                    
                    # Calculate τ values
                    τ_start = t_start / dt
                    τ_end = t_end / dt
                    
                    # Create intermediate fronts
                    front_start = interpolate_front(front_n, front_np1, τ_start)
                    front_end = interpolate_front(front_n, front_np1, τ_end)
                    
                    # Calculate volumes
                    _, vol_start, cx_start, cy_start, _ = fluid_cell_properties(mesh, front_start)
                    _, vol_end, cx_end, cy_end, _ = fluid_cell_properties(mesh, front_end)
                    
                    # Extract volumes
                    v_start = vol_start[i, j]
                    v_end = vol_end[i, j]
                    
                    # Weight centroids by volumes
                    wx_start = v_start > 0 ? cx_start[i, j] * v_start : 0.0
                    wy_start = v_start > 0 ? cy_start[i, j] * v_start : 0.0
                    
                    wx_end = v_end > 0 ? cx_end[i, j] * v_end : 0.0
                    wy_end = v_end > 0 ? cy_end[i, j] * v_end : 0.0
                    
                    # Apply trapezoidal rule
                    h = t_end - t_start
                    sub_integral_v = h * (v_start + v_end) / 2
                    sub_integral_wx = h * (wx_start + wx_end) / 2
                    sub_integral_wy = h * (wy_start + wy_end) / 2
                    
                    # Add contributions
                    V_spacetime[i, j] += sub_integral_v
                    weighted_centroid_x += sub_integral_wx
                    weighted_centroid_y += sub_integral_wy
                end
            end
            
            # Safety bounds check
            V_spacetime[i, j] = clamp(V_spacetime[i, j], 0.0, cell_area * dt)
        end
    end

        # 4. Calculate Bx and By space-time capacities (wet lengths through cell centroids)
    for i in 1:nx
        for j in 1:ny
            # Get cell dimensions
            cell_width = x_nodes[i+1] - x_nodes[i]
            cell_height = y_nodes[j+1] - y_nodes[j]
            
            # Reuse the crossing times calculated for volumes
            crossings = Float64[]
            push!(crossings, 0.0)  # Add t_n
            push!(crossings, dt)   # Add t_{n+1}
            
            # Add crossing times from surrounding faces
            if t_crosses_x[i, j] > 0
                push!(crossings, t_crosses_x[i, j])
            end
            if t_crosses_x[i+1, j] > 0
                push!(crossings, t_crosses_x[i+1, j])
            end
            if t_crosses_y[i, j] > 0
                push!(crossings, t_crosses_y[i, j])
            end
            if t_crosses_y[i, j+1] > 0
                push!(crossings, t_crosses_y[i, j+1])
            end
            
            # Sort and remove duplicates
            sort!(unique!(crossings))
            
            # Initialize space-time capacities
            Bx_spacetime[i, j] = 0.0
            By_spacetime[i, j] = 0.0
            
            # For each time interval, integrate using trapezoidal rule
            for k in 1:(length(crossings)-1)
                t_start = crossings[k]
                t_end = crossings[k+1]
                
                # Calculate τ values
                τ_start = t_start / dt
                τ_end = t_end / dt
                
                # Create intermediate fronts
                front_start = interpolate_front(front_n, front_np1, τ_start)
                front_end = interpolate_front(front_n, front_np1, τ_end)
                
                # Calculate fluid properties at these times
                _, _, cx_start, cy_start, _ = fluid_cell_properties(mesh, front_start)
                _, _, cx_end, cy_end, _ = fluid_cell_properties(mesh, front_end)
                
                # Calculate Bx, By at start and end times
                _, _, Bx_start, By_start = compute_second_type_capacities(
                    mesh, front_start, cx_start, cy_start
                )
                _, _, Bx_end, By_end = compute_second_type_capacities(
                    mesh, front_end, cx_end, cy_end
                )
                
                # Trapezoidal rule integration
                h = t_end - t_start
                Bx_subinterval = h * (Bx_start[i, j] + Bx_end[i, j]) / 2
                By_subinterval = h * (By_start[i, j] + By_end[i, j]) / 2
                
                # Add contributions
                Bx_spacetime[i, j] += Bx_subinterval
                By_spacetime[i, j] += By_subinterval
            end
            
            # Safety bounds check
            Bx_spacetime[i, j] = clamp(Bx_spacetime[i, j], 0.0, cell_height * dt)
            By_spacetime[i, j] = clamp(By_spacetime[i, j], 0.0, cell_width * dt)
        end
    end
    
    # 5. Calculate Wx and Wy space-time capacities (fluid volumes between centroids)
    # Use Simpson's rule integration like for volumes
    for i in 1:nx
        for j in 1:ny
            # Calculate Wx_spacetime at i+1/2, j
            if i < nx
                # Get relevant cell dimensions
                y_min = y_nodes[j]
                y_max = y_nodes[j+1]
                
                # Collect crossing times for this interface
                wx_crossings = Float64[]
                push!(wx_crossings, 0.0)  # Add t_n
                push!(wx_crossings, dt)   # Add t_{n+1}
                
                # Add crossing times from surrounding faces
                if t_crosses_x[i, j] > 0
                    push!(wx_crossings, t_crosses_x[i, j])
                end
                if t_crosses_x[i+1, j] > 0
                    push!(wx_crossings, t_crosses_x[i+1, j])
                end
                if t_crosses_y[i, j] > 0
                    push!(wx_crossings, t_crosses_y[i, j])
                end
                if t_crosses_y[i, j+1] > 0
                    push!(wx_crossings, t_crosses_y[i, j+1])
                end
                if t_crosses_y[i+1, j] > 0
                    push!(wx_crossings, t_crosses_y[i+1, j])
                end
                if t_crosses_y[i+1, j+1] > 0
                    push!(wx_crossings, t_crosses_y[i+1, j+1])
                end
                
                # Sort and remove duplicates
                sort!(unique!(wx_crossings))
                
                # For significant volume changes, add additional midpoints for better sampling
                if length(wx_crossings) > 2
                    refined_crossings = Float64[]
                    for k in 1:(length(wx_crossings)-1)
                        push!(refined_crossings, wx_crossings[k])
                        # Add midpoint between each pair of crossing times
                        midpoint = 0.5 * (wx_crossings[k] + wx_crossings[k+1])
                        push!(refined_crossings, midpoint)
                    end
                    push!(refined_crossings, wx_crossings[end])
                    wx_crossings = sort(refined_crossings)
                end
                
                # Initialize space-time capacity
                Wx_spacetime[i+1, j] = 0.0
                
                # Perform quadrature integration with improved accuracy
                if length(wx_crossings) >= 3  # Simpson's rule requires at least 3 points
                    for k in 1:2:length(wx_crossings)-2
                        t_start = wx_crossings[k]
                        t_mid = wx_crossings[k+1]
                        t_end = wx_crossings[k+2]
                        
                        # Calculate τ values (normalized time)
                        τ_start = t_start / dt
                        τ_mid = t_mid / dt
                        τ_end = t_end / dt
                        
                        # Create intermediate fronts
                        front_start = interpolate_front(front_n, front_np1, τ_start)
                        front_mid = interpolate_front(front_n, front_np1, τ_mid)
                        front_end = interpolate_front(front_n, front_np1, τ_end)
                        
                        # Calculate fluid properties at these times
                        _, _, cx_start, cy_start, _ = fluid_cell_properties(mesh, front_start)
                        _, _, cx_mid, cy_mid, _ = fluid_cell_properties(mesh, front_mid)
                        _, _, cx_end, cy_end, _ = fluid_cell_properties(mesh, front_end)
                        
                        # Calculate Wx at start, mid, and end times
                        Wx_start, _, _, _ = compute_second_type_capacities(
                            mesh, front_start, cx_start, cy_start
                        )
                        Wx_mid, _, _, _ = compute_second_type_capacities(
                            mesh, front_mid, cx_mid, cy_mid
                        )
                        Wx_end, _, _, _ = compute_second_type_capacities(
                            mesh, front_end, cx_end, cy_end
                        )
                        
                        # Apply Simpson's rule: (h/6)*(f₁ + 4*f₂ + f₃)
                        h = t_end - t_start
                        Wx_subinterval = (h/6) * (Wx_start[i+1, j] + 4*Wx_mid[i+1, j] + Wx_end[i+1, j])
                        
                        # Add contribution
                        Wx_spacetime[i+1, j] += Wx_subinterval
                    end
                    
                    # Handle any remaining interval with trapezoidal rule
                    if length(wx_crossings) % 2 == 0
                        last_idx = length(wx_crossings)
                        t_start = wx_crossings[last_idx-1]
                        t_end = wx_crossings[last_idx]
                        
                        # Calculate τ values
                        τ_start = t_start / dt
                        τ_end = t_end / dt
                        
                        # Create intermediate fronts
                        front_start = interpolate_front(front_n, front_np1, τ_start)
                        front_end = interpolate_front(front_n, front_np1, τ_end)
                        
                        # Calculate fluid properties
                        _, _, cx_start, cy_start, _ = fluid_cell_properties(mesh, front_start)
                        _, _, cx_end, cy_end, _ = fluid_cell_properties(mesh, front_end)
                        
                        # Calculate Wx at start and end times
                        Wx_start, _, _, _ = compute_second_type_capacities(
                            mesh, front_start, cx_start, cy_start
                        )
                        Wx_end, _, _, _ = compute_second_type_capacities(
                            mesh, front_end, cx_end, cy_end
                        )
                        
                        # Apply trapezoidal rule
                        h = t_end - t_start
                        Wx_subinterval = h * (Wx_start[i+1, j] + Wx_end[i+1, j]) / 2
                        
                        # Add contribution
                        Wx_spacetime[i+1, j] += Wx_subinterval
                    end
                else
                    # Use standard trapezoidal rule for simple cases
                    for k in 1:(length(wx_crossings)-1)
                        t_start = wx_crossings[k]
                        t_end = wx_crossings[k+1]
                        
                        # Calculate τ values
                        τ_start = t_start / dt
                        τ_end = t_end / dt
                        
                        # Create intermediate fronts
                        front_start = interpolate_front(front_n, front_np1, τ_start)
                        front_end = interpolate_front(front_n, front_np1, τ_end)
                        
                        # Calculate fluid properties
                        _, _, cx_start, cy_start, _ = fluid_cell_properties(mesh, front_start)
                        _, _, cx_end, cy_end, _ = fluid_cell_properties(mesh, front_end)
                        
                        # Calculate Wx at start and end times
                        Wx_start, _, _, _ = compute_second_type_capacities(
                            mesh, front_start, cx_start, cy_start
                        )
                        Wx_end, _, _, _ = compute_second_type_capacities(
                            mesh, front_end, cx_end, cy_end
                        )
                        
                        # Apply trapezoidal rule
                        h = t_end - t_start
                        Wx_subinterval = h * (Wx_start[i+1, j] + Wx_end[i+1, j]) / 2
                        
                        # Add contribution
                        Wx_spacetime[i+1, j] += Wx_subinterval
                    end
                end
            end
            
            # Calculate Wy_spacetime at i, j+1/2 (following the same approach)
            if j < ny
                # Same structure as Wx but for Wy
                # [Similar code structure with Wy calculations]
                # Collect crossing times for this interface
                wy_crossings = Float64[]
                push!(wy_crossings, 0.0)  # Add t_n
                push!(wy_crossings, dt)   # Add t_{n+1}
                
                # Add crossing times from surrounding faces
                if t_crosses_x[i, j] > 0
                    push!(wy_crossings, t_crosses_x[i, j])
                end
                if t_crosses_x[i, j+1] > 0
                    push!(wy_crossings, t_crosses_x[i, j+1])
                end
                if t_crosses_y[i, j] > 0
                    push!(wy_crossings, t_crosses_y[i, j])
                end
                if t_crosses_y[i, j+1] > 0
                    push!(wy_crossings, t_crosses_y[i, j+1])
                end
                if t_crosses_x[i+1, j] > 0
                    push!(wy_crossings, t_crosses_x[i+1, j])
                end
                if t_crosses_x[i+1, j+1] > 0
                    push!(wy_crossings, t_crosses_x[i+1, j+1])
                end
                
                # Sort and remove duplicates
                sort!(unique!(wy_crossings))
                
                # For significant volume changes, add additional midpoints for better sampling
                if length(wy_crossings) > 2
                    refined_crossings = Float64[]
                    for k in 1:(length(wy_crossings)-1)
                        push!(refined_crossings, wy_crossings[k])
                        # Add midpoint between each pair of crossing times
                        midpoint = 0.5 * (wy_crossings[k] + wy_crossings[k+1])
                        push!(refined_crossings, midpoint)
                    end
                    push!(refined_crossings, wy_crossings[end])
                    wy_crossings = sort(refined_crossings)
                end
                
                # Initialize space-time capacity
                Wy_spacetime[i, j+1] = 0.0
                
                # Perform quadrature integration with improved accuracy
                if length(wy_crossings) >= 3  # Simpson's rule requires at least 3 points
                    for k in 1:2:length(wy_crossings)-2
                        t_start = wy_crossings[k]
                        t_mid = wy_crossings[k+1]
                        t_end = wy_crossings[k+2]
                        
                        # Calculate τ values (normalized time)
                        τ_start = t_start / dt
                        τ_mid = t_mid / dt
                        τ_end = t_end / dt
                        
                        # Create intermediate fronts
                        front_start = interpolate_front(front_n, front_np1, τ_start)
                        front_mid = interpolate_front(front_n, front_np1, τ_mid)
                        front_end = interpolate_front(front_n, front_np1, τ_end)
                        
                        # Calculate fluid properties at these times
                        _, _, cx_start, cy_start, _ = fluid_cell_properties(mesh, front_start)
                        _, _, cx_mid, cy_mid, _ = fluid_cell_properties(mesh, front_mid)
                        _, _, cx_end, cy_end, _ = fluid_cell_properties(mesh, front_end)
                        
                        # Calculate Wy at start, mid, and end times
                        _, Wy_start, _, _ = compute_second_type_capacities(
                            mesh, front_start, cx_start, cy_start
                        )
                        _, Wy_mid, _, _ = compute_second_type_capacities(
                            mesh, front_mid, cx_mid, cy_mid
                        )
                        _, Wy_end, _, _ = compute_second_type_capacities(
                            mesh, front_end, cx_end, cy_end
                        )
                        
                        # Apply Simpson's rule: (h/6)*(f₁ + 4*f₂ + f₃)
                        h = t_end - t_start
                        Wy_subinterval = (h/6) * (Wy_start[i, j+1] + 4*Wy_mid[i, j+1] + Wy_end[i, j+1])
                        
                        # Add contribution
                        Wy_spacetime[i, j+1] += Wy_subinterval
                    end
                    
                    # Handle any remaining interval with trapezoidal rule
                    if length(wy_crossings) % 2 == 0
                        last_idx = length(wy_crossings)
                        t_start = wy_crossings[last_idx-1]
                        t_end = wy_crossings[last_idx]
                        
                        # Calculate τ values
                        τ_start = t_start / dt
                        τ_end = t_end / dt
                        
                        # Create intermediate fronts
                        front_start = interpolate_front(front_n, front_np1, τ_start)
                        front_end = interpolate_front(front_n, front_np1, τ_end)
                        
                        # Calculate fluid properties
                        _, _, cx_start, cy_start, _ = fluid_cell_properties(mesh, front_start)
                        _, _, cx_end, cy_end, _ = fluid_cell_properties(mesh, front_end)
                        
                        # Calculate Wy at start and end times
                        _, Wy_start, _, _ = compute_second_type_capacities(
                            mesh, front_start, cx_start, cy_start
                        )
                        _, Wy_end, _, _ = compute_second_type_capacities(
                            mesh, front_end, cx_end, cy_end
                        )
                        
                        # Apply trapezoidal rule
                        h = t_end - t_start
                        Wy_subinterval = h * (Wy_start[i, j+1] + Wy_end[i, j+1]) / 2
                        
                        # Add contribution
                        Wy_spacetime[i, j+1] += Wy_subinterval
                    end
                else
                    # Use standard trapezoidal rule for simple cases
                    for k in 1:(length(wy_crossings)-1)
                        t_start = wy_crossings[k]
                        t_end = wy_crossings[k+1]
                        
                        # Calculate τ values
                        τ_start = t_start / dt
                        τ_end = t_end / dt
                        
                        # Create intermediate fronts
                        front_start = interpolate_front(front_n, front_np1, τ_start)
                        front_end = interpolate_front(front_n, front_np1, τ_end)
                        
                        # Calculate fluid properties
                        _, _, cx_start, cy_start, _ = fluid_cell_properties(mesh, front_start)
                        _, _, cx_end, cy_end, _ = fluid_cell_properties(mesh, front_end)
                        
                        # Calculate Wy at start and end times
                        _, Wy_start, _, _ = compute_second_type_capacities(
                            mesh, front_start, cx_start, cy_start
                        )
                        _, Wy_end, _, _ = compute_second_type_capacities(
                            mesh, front_end, cx_end, cy_end
                        )
                        
                        # Apply trapezoidal rule
                        h = t_end - t_start
                        Wy_subinterval = h * (Wy_start[i, j+1] + Wy_end[i, j+1]) / 2
                        
                        # Add contribution
                        Wy_spacetime[i, j+1] += Wy_subinterval
                    end
                end
            end
        end
    end

    return Dict(
        :Ax_spacetime => Ax_spacetime,
        :Ay_spacetime => Ay_spacetime,
        :V_spacetime => V_spacetime,
        :Bx_spacetime => Bx_spacetime,
        :By_spacetime => By_spacetime,
        :Wx_spacetime => Wx_spacetime,
        :Wy_spacetime => Wy_spacetime,
        :centroid_x_spacetime => centroid_x_spacetime,
        :centroid_y_spacetime => centroid_y_spacetime,
        :t_crosses_x => t_crosses_x,
        :t_crosses_y => t_crosses_y,
        :t_crosses_v => t_crosses_v   
    )
end

"""
    interpolate_front(front_n::FrontTracker, front_np1::FrontTracker, τ::Float64)

Create an intermediate front by linear interpolation between two fronts.
τ should be between 0 and 1 (normalized time).
"""
function interpolate_front(front_n::FrontTracker, front_np1::FrontTracker, τ::Float64)
    # Get markers from both fronts
    markers_n = get_markers(front_n)
    markers_np1 = get_markers(front_np1)
    
    # Ensure we have the same number of markers
    if length(markers_n) != length(markers_np1)
        error("Fronts must have the same number of markers for interpolation")
    end
    
    # Interpolate marker positions
    new_markers = [(
        (1-τ) * markers_n[i][1] + τ * markers_np1[i][1],
        (1-τ) * markers_n[i][2] + τ * markers_np1[i][2]
    ) for i in 1:length(markers_n)]
    
    # Create new front
    return FrontTracker(new_markers, front_n.is_closed)
end

"""
    calculate_face_capacity_x(front::FrontTracker, x::Float64, y_min::Float64, y_max::Float64)

Calculate the capacity (wet length) of a vertical face at position x from y_min to y_max.
"""
function calculate_face_capacity_x(front::FrontTracker, x::Float64, y_min::Float64, y_max::Float64)
    # Create a line representing the face
    face_line = LibGEOS.LineString([[x, y_min], [x, y_max]])
    
    # Get fluid polygon
    fluid_poly = get_fluid_polygon(front)
    
    # Calculate intersection
    if LibGEOS.intersects(face_line, fluid_poly)
        intersection = LibGEOS.intersection(face_line, fluid_poly)
        
        if isa(intersection, LibGEOS.LineString)
            return LibGEOS.geomLength(intersection)
        elseif isa(intersection, LibGEOS.MultiLineString)
            total_length = 0.0
            for k in 1:LibGEOS.getNumGeometries(intersection)
                line = LibGEOS.getGeometry(intersection, k-1)
                total_length += LibGEOS.geomLength(line)
            end
            return total_length
        elseif isa(intersection, LibGEOS.Point) || isa(intersection, LibGEOS.MultiPoint)
            return 0.0
        end
    else
        # Check if midpoint is inside
        mid_y = (y_min + y_max) / 2
        if is_point_inside(front, x, mid_y)
            return y_max - y_min
        else
            return 0.0
        end
    end
    
    return 0.0
end

"""
    calculate_face_capacity_y(front::FrontTracker, y::Float64, x_min::Float64, x_max::Float64)

Calculate the capacity (wet length) of a horizontal face at position y from x_min to x_max.
"""
function calculate_face_capacity_y(front::FrontTracker, y::Float64, x_min::Float64, x_max::Float64)
    # Create a line representing the face
    face_line = LibGEOS.LineString([[x_min, y], [x_max, y]])
    
    # Get fluid polygon
    fluid_poly = get_fluid_polygon(front)
    
    # Calculate intersection
    if LibGEOS.intersects(face_line, fluid_poly)
        intersection = LibGEOS.intersection(face_line, fluid_poly)
        
        if isa(intersection, LibGEOS.LineString)
            return LibGEOS.geomLength(intersection)
        elseif isa(intersection, LibGEOS.MultiLineString)
            total_length = 0.0
            for k in 1:LibGEOS.getNumGeometries(intersection)
                line = LibGEOS.getGeometry(intersection, k-1)
                total_length += LibGEOS.geomLength(line)
            end
            return total_length
        elseif isa(intersection, LibGEOS.Point) || isa(intersection, LibGEOS.MultiPoint)
            return 0.0
        end
    else
        # Check if midpoint is inside
        mid_x = (x_min + x_max) / 2
        if is_point_inside(front, mid_x, y)
            return x_max - x_min
        else
            return 0.0
        end
    end
    
    return 0.0
end

"""
    find_crossing_time(front_n::FrontTracker, front_np1::FrontTracker, x::Float64, y::Float64, dt::Float64)

Estimates the time when the interface crosses a specific point (x,y) between two time steps.
Uses linear interpolation of the signed distance function between time steps.

Parameters:
- front_n: Front tracker at time t_n
- front_np1: Front tracker at time t_n+1
- x, y: Coordinates of the point to check
- dt: Time step size

Returns:
- t_cross: Estimated crossing time within [0, dt] interval
"""
function find_crossing_time(front_n::FrontTracker, front_np1::FrontTracker, x::Float64, y::Float64, dt::Float64)
    # Get the signed distance at both time steps
    sdf_n = sdf(front_n, x, y)
    sdf_np1 = sdf(front_np1, x, y)
    
    # Check if the interface actually crosses this point
    if (sdf_n * sdf_np1 > 0)
        # No crossing (same sign at both time steps)
        if sdf_n < 0
            # Inside fluid at both times
            return dt/2
        else
            # Outside fluid at both times
            return dt/2
        end
    end
    
    # If one SDF is zero, return that time
    if abs(sdf_n) < 1e-10
        return 0.0
    elseif abs(sdf_np1) < 1e-10
        return dt
    end
    
    # Linear interpolation to find crossing time:
    t_cross = dt * abs(sdf_n) / (abs(sdf_n) + abs(sdf_np1))
    
    # Ensure the result is within [0, dt]
    return clamp(t_cross, 0.0, dt)
end







"""
    compute_segment_parameters(ft::FrontTracker)

Calcule les paramètres de chaque segment de l'interface:
- n_I: vecteur normal unitaire du segment
- α_I: intercept du segment (distance signée à l'origine)
- length_I: longueur du segment
- midpoint_I: point milieu du segment

L'équation d'un segment est: n_I ⋅ x = α_I

Retourne (segments, segment_normals, segment_intercepts, segment_lengths, segment_midpoints)
"""
function compute_segment_parameters(ft::FrontTracker)
    markers = ft.markers
    n_markers = length(markers)
    
    if n_markers < 2
        return [], [], [], [], []
    end
    
    # Nombre de segments (n_markers pour un contour fermé, n_markers-1 pour un contour ouvert)
    n_segments = if ft.is_closed
        if n_markers > 0 && markers[1] == markers[end]
            n_markers - 1
        else
            n_markers
        end
    else
        n_markers - 1
    end
    
    # Initialiser les tableaux de résultats
    segments = Vector{Tuple{Int, Int}}(undef, n_segments)
    segment_normals = Vector{Tuple{Float64, Float64}}(undef, n_segments)
    segment_intercepts = Vector{Float64}(undef, n_segments)
    segment_lengths = Vector{Float64}(undef, n_segments)
    segment_midpoints = Vector{Tuple{Float64, Float64}}(undef, n_segments)
    
    # Parcourir tous les segments
    for i in 1:n_segments
        # Indice du marqueur suivant (avec gestion de la boucle pour contour fermé)
        next_i = i < n_markers ? i + 1 : 1
        
        # Points de début et de fin du segment
        p1 = markers[i]
        p2 = markers[next_i]
        
        # Vecteur du segment (de p1 à p2)
        segment_vector = (p2[1] - p1[1], p2[2] - p1[2])
        
        # Longueur du segment
        segment_length = sqrt(segment_vector[1]^2 + segment_vector[2]^2)
        
        if segment_length < 1e-15
            # Éviter la division par zéro pour les segments très courts
            segment_normals[i] = (0.0, 1.0)  # Normal arbitraire
            segment_intercepts[i] = p1[1] * 0.0 + p1[2] * 1.0  # α_I = n_I ⋅ p1
            segment_lengths[i] = 0.0
            segment_midpoints[i] = p1  # Point de milieu = point de début pour segments courts
        else
            # Normale unitaire (rotation de 90° dans le sens trigonométrique)
            normal = (-segment_vector[2] / segment_length, segment_vector[1] / segment_length)
            
            # Vérifier l'orientation de la normale (doit pointer à l'extérieur)
            if ft.is_closed
                # Pour un contour fermé, vérifier si la normale pointe vers l'extérieur
                test_point = (p1[1] + 1e-3 * normal[1], p1[2] + 1e-3 * normal[2])
                if is_point_inside(ft, test_point[1], test_point[2])
                    # Si le point test est à l'intérieur, inverser la normale
                    normal = (-normal[1], -normal[2])
                end
            end
            
            # Calcul de l'intercept α_I = n_I ⋅ p1
            intercept = normal[1] * p1[1] + normal[2] * p1[2]
            
            # Stockage des résultats
            segments[i] = (i, next_i)
            segment_normals[i] = normal
            segment_intercepts[i] = intercept
            segment_lengths[i] = segment_length
            segment_midpoints[i] = ((p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
        end
    end
    
    return segments, segment_normals, segment_intercepts, segment_lengths, segment_midpoints
end

"""
    create_segment_line(ft::FrontTracker, segment_idx::Int)

Crée une LineString représentant un segment de l'interface à partir des marqueurs.
"""
function create_segment_line(ft::FrontTracker, segment_idx::Int)
    markers = ft.markers
    n_markers = length(markers)
    
    if segment_idx < 1 || segment_idx > (ft.is_closed ? n_markers : n_markers - 1)
        error("Indice de segment invalide: $segment_idx")
    end
    
    # Récupérer les indices des marqueurs qui définissent le segment
    next_idx = segment_idx < n_markers ? segment_idx + 1 : 1
    
    # Récupérer les coordonnées des marqueurs
    start_point = markers[segment_idx]
    end_point = markers[next_idx]
    
    # Créer la LineString
    return LibGEOS.LineString([[start_point[1], start_point[2]], [end_point[1], end_point[2]]])
end

"""
    compute_segment_cell_intersections(mesh::Mesh{2}, ft::FrontTracker)

Calcule les intersections entre les segments de l'interface et les cellules du maillage.
Retourne un dictionnaire où les clés sont les indices de cellules (i,j) et les valeurs
sont des listes de tuples (segment_idx, intersection_length).
"""
function compute_segment_cell_intersections(mesh::Mesh{2}, ft::FrontTracker)
    # Calculer les paramètres des segments
    segments, segment_normals, segment_intercepts, segment_lengths, segment_midpoints = 
        compute_segment_parameters(ft)
    
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Dictionnaire pour stocker les intersections segment-cellule
    cell_segment_intersections = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}()
    
    # Initialiser le dictionnaire pour toutes les cellules
    for i in 1:nx
        for j in 1:ny
            cell_segment_intersections[(i,j)] = []
        end
    end
    
    # Pour chaque segment, calculer les intersections avec les cellules
    n_segments = length(segments)
    for segment_idx in 1:n_segments
        # Créer une ligne représentant le segment
        segment_line = create_segment_line(ft, segment_idx)
        
        # Calculer les intersections avec toutes les cellules
        for i in 1:nx
            for j in 1:ny
                # Créer le polygone de la cellule
                cell_coords = [
                    [x_nodes[i], y_nodes[j]],
                    [x_nodes[i+1], y_nodes[j]],
                    [x_nodes[i+1], y_nodes[j+1]],
                    [x_nodes[i], y_nodes[j+1]],
                    [x_nodes[i], y_nodes[j]]
                ]
                cell_poly = LibGEOS.Polygon([cell_coords])
                
                # Vérifier l'intersection
                if LibGEOS.intersects(cell_poly, segment_line)
                    intersection = LibGEOS.intersection(cell_poly, segment_line)
                    
                    # Calculer la longueur d'intersection
                    if isa(intersection, LibGEOS.LineString)
                        intersection_length = LibGEOS.geomLength(intersection)
                        if intersection_length > 1e-10
                            push!(cell_segment_intersections[(i,j)], (segment_idx, intersection_length))
                        end
                    elseif isa(intersection, LibGEOS.MultiLineString)
                        total_length = 0.0
                        for k in 1:LibGEOS.getNumGeometries(intersection)
                            line = LibGEOS.getGeometry(intersection, k-1)
                            total_length += LibGEOS.geomLength(line)
                        end
                        if total_length > 1e-10
                            push!(cell_segment_intersections[(i,j)], (segment_idx, total_length))
                        end
                    end
                end
            end
        end
    end
    
    return cell_segment_intersections, segments, segment_normals, segment_intercepts, segment_lengths
end

"""
    compute_intercept_jacobian(mesh::Mesh{2}, ft::FrontTracker; density::Float64=1.0)

Calcule la jacobienne des volumes par rapport aux déplacements d'intercept.
Pour chaque cellule k=(i,j) et chaque segment I, J[k,I] = ∂V_k/∂δ_I = ρL × A_k,I,
où A_k,I est la longueur d'intersection du segment I avec la cellule k,
et ρL est un facteur physique (densité × latent heat)

Retourne:
- intercept_jacobian: Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}} - la jacobienne
- segments: vecteur des segments (i, j) où i, j sont les indices des marqueurs
- segment_normals: vecteur des normales unitaires pour chaque segment
- segment_intercepts: vecteur des intercepts initiaux pour chaque segment
- segment_lengths: vecteur des longueurs de chaque segment
"""
function compute_intercept_jacobian(mesh::Mesh{2}, ft::FrontTracker; density::Float64=1.0)
    # Calculer les intersections segment-cellule
    cell_segment_intersections, segments, segment_normals, segment_intercepts, segment_lengths = 
        compute_segment_cell_intersections(mesh, ft)
    
    # Récupérer les dimensions du maillage
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Créer un dictionnaire pour stocker la jacobienne
    # Format: Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}
    # Clé: (i,j) = indice de la cellule, Valeur: liste de (segment_idx, jacobian_value)
    intercept_jacobian = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}()
    
    # Pour chaque cellule, calculer sa contribution à la jacobienne
    for i in 1:nx
        for j in 1:ny
            intercept_jacobian[(i,j)] = []
            
            # Parcourir tous les segments qui intersectent cette cellule
            for (segment_idx, intersection_length) in cell_segment_intersections[(i,j)]
                # J[k,I] = ρL × A_k,I
                jacobian_value = density * intersection_length
                
                # Stocker la valeur dans la jacobienne
                push!(intercept_jacobian[(i,j)], (segment_idx, jacobian_value))
            end
        end
    end
    
    return intercept_jacobian, segments, segment_normals, segment_intercepts, segment_lengths
end

"""
    update_front_with_intercept_displacements!(ft::FrontTracker, displacements::AbstractVector{<:Real}, 
                                            segment_normals::Vector{Tuple{Float64, Float64}},
                                            segment_lengths::Vector{Float64})

Met à jour l'interface en déplaçant chaque segment selon ses déplacements d'intercept.
Applique une pondération basée sur la longueur des segments pour distribuer les déplacements
aux marqueurs partagés entre plusieurs segments.

Paramètres:
- ft: l'objet FrontTracker à mettre à jour
- displacements: vecteur des déplacements δ_I pour chaque segment
- segment_normals: vecteur des normales unitaires pour chaque segment
- segment_lengths: vecteur des longueurs de chaque segment

Retourne le FrontTracker mis à jour.
"""
function update_front_with_intercept_displacements!(ft::FrontTracker, displacements::AbstractVector{<:Real}, 
                                                  segment_normals::Vector{Tuple{Float64, Float64}},
                                                  segment_lengths::Vector{Float64})
    markers = copy(ft.markers)
    n_markers = length(markers)
    n_segments = length(displacements)
    
  
    
    # Structure pour stocker les contributions pondérées de chaque segment à chaque marqueur
    segment_contributions = Dict{Int, Vector{Tuple{Float64, Tuple{Float64, Float64}}}}()
    for i in 1:n_markers
        segment_contributions[i] = []
    end
    
    # Pour chaque segment, calculer sa contribution aux marqueurs
    for (s_idx, displacement) in enumerate(displacements)
        # Récupérer les indices des marqueurs aux extrémités du segment
        start_idx = s_idx
        end_idx = s_idx < n_markers ? s_idx + 1 : 1
        
        # Calculer le vecteur de déplacement
        normal = segment_normals[s_idx]
        vector_displacement = (displacement * normal[1], displacement * normal[2])
        
        # Utiliser la longueur du segment comme poids
        segment_weight = max(segment_lengths[s_idx], 1e-10)  # Éviter division par zéro
        
        # Enregistrer la contribution pondérée de ce segment pour chaque marqueur
        push!(segment_contributions[start_idx], (segment_weight, vector_displacement))
        push!(segment_contributions[end_idx], (segment_weight, vector_displacement))
    end
    
    # Calculer et appliquer le déplacement moyen pondéré pour chaque marqueur
    for i in 1:n_markers
        contributions = segment_contributions[i]
        if !isempty(contributions)
            # Calculer la somme des poids
            total_weight = sum(contrib[1] for contrib in contributions)
            
            if total_weight > 0
                # Calculer le déplacement moyen pondéré
                avg_dx = sum(contrib[1] * contrib[2][1] for contrib in contributions) / total_weight
                avg_dy = sum(contrib[1] * contrib[2][2] for contrib in contributions) / total_weight
                
                # Appliquer le déplacement
                markers[i] = (markers[i][1] + avg_dx, markers[i][2] + avg_dy)
            end
        end
    end
    
    # Mettre à jour l'interface avec les nouveaux marqueurs
    set_markers!(ft, markers, ft.is_closed)
    
    return ft
end
