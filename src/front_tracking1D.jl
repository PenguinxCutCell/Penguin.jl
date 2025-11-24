"""
A Julia implementation of 1D front tracking for fluid-solid interfaces.
"""
mutable struct FrontTracker1D
    # In 1D, markers are just x-coordinates of interface points
    markers::Vector{Float64}
    
    # Constructor for empty front
    function FrontTracker1D()
        return new([])
    end
    
    # Constructor with markers
    function FrontTracker1D(markers::Vector{Float64})
        return new(sort(markers))  # Keep markers sorted for easier processing
    end
end

"""
    get_markers(ft::FrontTracker1D)

Gets all markers of the 1D interface.
"""
function get_markers(ft::FrontTracker1D)
    return ft.markers
end

"""
    add_marker!(ft::FrontTracker1D, x::Float64)

Adds a marker to the 1D interface and keeps the collection sorted.
"""
function add_marker!(ft::FrontTracker1D, x::Float64)
    push!(ft.markers, x)
    sort!(ft.markers)
    return ft
end

"""
    set_markers!(ft::FrontTracker1D, markers::AbstractVector{<:Real})

Sets all markers of the 1D interface.
"""
function set_markers!(ft::FrontTracker1D, markers::AbstractVector{<:Real})
    ft.markers = convert(Vector{Float64}, markers)
    sort!(ft.markers)
    return ft
end

"""
    is_point_inside(ft::FrontTracker1D, x::Float64)

Checks if a point is inside the fluid region.
For 1D, we define "inside" as being to the left of an odd-indexed interface point
or to the right of an even-indexed interface point.
"""
function is_point_inside(ft::FrontTracker1D, x::Float64)
    if isempty(ft.markers)
        return false
    end
    
    # Find number of interface points to the left of x
    count = sum(m < x for m in ft.markers)
    
    # If odd number of interface points to the left, point is inside
    return count % 2 == 1
end

"""
    sdf(ft::FrontTracker1D, x::Float64)

Calculates the signed distance function for a given point.
Positive outside fluid, negative inside fluid.
"""
function sdf(ft::FrontTracker1D, x::Float64)
    if isempty(ft.markers)
        return Inf
    end
    
    # Calculate distance to nearest interface point
    distance = minimum(abs.(ft.markers .- x))
    
    # Determine sign (negative inside, positive outside)
    sign_val = is_point_inside(ft, x) ? -1.0 : 1.0
    
    return sign_val * distance
end

"""
    compute_capacities_1d(mesh::Mesh{1}, front::FrontTracker1D)

Calculates all geometric capacities for a 1D mesh and interface.
Returns a dictionary with all results.
"""
function compute_capacities_1d(mesh::Mesh{1}, front::FrontTracker1D)
    # Extract mesh information
    x_nodes = mesh.nodes[1]
    nx = length(x_nodes) - 1
    
    # Initialize capacity arrays
    fractions = zeros(nx+1)    # Fluid fractions
    volumes = zeros(nx+1)      # Fluid volumes (lengths in 1D)
    centroids_x = zeros(nx+1)  # Fluid centroids
    cell_types = zeros(Int, nx+1)  # Cell types (0: solid, 1: fluid, -1: cut)
    Ax = zeros(nx+1)           # Face capacities
    Wx = zeros(nx+1)           # Staggered volumes 
    Bx = zeros(nx+1)           # Center-line capacities
    
    # Interface information
    interface_positions = Dict{Int, Float64}()
    
    # Process each cell
    for i in 1:nx
        x_min, x_max = x_nodes[i], x_nodes[i+1]
        cell_width = x_max - x_min
        
        # Find interface points inside this cell
        interface_points_in_cell = filter(m -> x_min <= m <= x_max, front.markers)
        
        # Check if the cell contains interface points
        if isempty(interface_points_in_cell)
            # Cell is either fully fluid or fully solid
            if is_point_inside(front, (x_min + x_max)/2)
                # Fully fluid cell
                fractions[i] = 1.0
                volumes[i] = cell_width
                centroids_x[i] = (x_min + x_max) / 2
                cell_types[i] = 1
                
                # For fully fluid cells:
                Bx[i] = 1.0  # Full capacity at center
                
                # Wx values will be computed after processing all cells
            else
                # Fully solid cell
                fractions[i] = 0.0
                volumes[i] = 0.0
                centroids_x[i] = (x_min + x_max) / 2
                cell_types[i] = 0
                
                # For fully solid cells:
                Bx[i] = 0.0  # No capacity at center
            end
        else
            # Cut cell
            cell_types[i] = -1
            
            # Store interface positions for later use
            for position in interface_points_in_cell
                interface_positions[i] = position
            end
            
            # Sort interface points within the cell
            sort!(interface_points_in_cell)
            
            # Calculate fluid fraction and volume
            fluid_segments = []
            
            # Start with cell left boundary
            current_x = x_min
            is_fluid = is_point_inside(front, current_x + 1e-10)
            
            # Process each interface point and collect fluid segments
            for point in interface_points_in_cell
                if is_fluid
                    # Add fluid segment from current_x to interface point
                    push!(fluid_segments, (current_x, point))
                end
                
                # Toggle fluid state
                is_fluid = !is_fluid
                current_x = point
            end
            
            # Process the last segment to the right boundary
            if is_fluid
                push!(fluid_segments, (current_x, x_max))
            end
            
            # Calculate total fluid volume and centroid
            total_fluid_volume = 0.0
            weighted_centroid_x = 0.0
            
            for (start_x, end_x) in fluid_segments
                segment_length = end_x - start_x
                total_fluid_volume += segment_length
                
                # Weighted contribution to centroid
                segment_centroid = (start_x + end_x) / 2
                weighted_centroid_x += segment_length * segment_centroid
            end
            
            # Store computed values
            volumes[i] = total_fluid_volume
            fractions[i] = total_fluid_volume / cell_width
            
            if total_fluid_volume > 0
                centroids_x[i] = weighted_centroid_x / total_fluid_volume
            else
                centroids_x[i] = (x_min + x_max) / 2
            end
            
            # Calculate Bx for cut cells (depends on position of centroid relative to interface)
            if fractions[i] > 0
                mid_point = centroids_x[i]
                Bx[i] = is_point_inside(front, mid_point) ? 1.0 : 0.0
            else
                Bx[i] = 0.0
            end
        end
    end
    
    # Calculate face capacities (Ax) at cell boundaries
    for i in 1:nx+1
        x_face = x_nodes[i]
        Ax[i] = is_point_inside(front, x_face) ? 1.0 : 0.0
    end
    
    # Calculate staggered volumes (Wx) between cell centers
    for i in 2:nx
        # Position between cell centroids
        x_left = centroids_x[i-1] 
        x_right = centroids_x[i]
        
        # If either adjacent cell has zero volume, Wx is zero
        if volumes[i-1] == 0.0 && volumes[i] == 0.0
            Wx[i] = 0.0
            continue
        end
        
        # Check for interfaces between the centroids
        interfaces_between = Float64[]
        for marker in front.markers
            if x_left < marker < x_right
                push!(interfaces_between, marker)
            end
        end
        
        if isempty(interfaces_between)
            # No interface between centroids, full connectivity
            Wx[i] = x_right - x_left
        else
            # For cells with interface, calculate connection based on fluid side
            # Find the interface closest to either centroid
            closest_interface = interfaces_between[1]
            min_distance = min(abs(closest_interface - x_left), abs(closest_interface - x_right))
            
            for marker in interfaces_between
                dist_left = abs(marker - x_left)
                dist_right = abs(marker - x_right)
                if min(dist_left, dist_right) < min_distance
                    closest_interface = marker
                    min_distance = min(dist_left, dist_right)
                end
            end
            
            # Determine which side is fluid
            if is_point_inside(front, x_left)
                # Left centroid is in fluid, Wx is distance from left centroid to interface
                Wx[i] = closest_interface - x_left
            elseif is_point_inside(front, x_right)
                # Right centroid is in fluid, Wx is distance from interface to right centroid
                Wx[i] = x_right - closest_interface
            else
                # Neither centroid is in fluid, no connection
                Wx[i] = 0.0
            end
        end
    end
    
    # Return all capacities in a dictionary
    return Dict(
        :fractions => fractions,          # Fluid fractions
        :volumes => volumes,              # Fluid volumes (lengths in 1D)
        :centroids_x => centroids_x,      # Fluid centroids
        :cell_types => cell_types,        # Cell types (0: solid, 1: fluid, -1: cut)
        :Ax => Ax,                        # Face capacities
        :Wx => Wx,                        # Staggered volumes
        :Bx => Bx,                        # Center line capacities
        :interface_positions => interface_positions  # Interface positions
    )
end

"""
    compute_spacetime_capacities_1d(mesh::Mesh{1}, front_n::FrontTracker1D, front_np1::FrontTracker1D, dt::Float64)

Calculates all space-time geometric capacities for a 1D mesh between two time steps.
Now includes Wx time-integrated connection capacities using centroids.
"""
function compute_spacetime_capacities_1d(mesh::Mesh{1}, front_n::FrontTracker1D, front_np1::FrontTracker1D, dt::Float64)
    # Extract mesh information
    x_nodes = mesh.nodes[1]
    nx = length(x_nodes) - 1
    
    # Initialize capacity arrays
    Ax_spacetime = zeros(nx+1)     # Space-time face capacities
    V_spacetime = zeros(nx+1)        # Space-time volumes - Corrigé: nx éléments
    Bx_spacetime = zeros(nx+1)       # Space-time centerline capacities - Corrigé: nx éléments
    Wx_spacetime = zeros(nx+1)     # Space-time connection capacities
    ST_centroids = Vector{Vector{Float64}}(undef, nx)  # Space-time centroid vectors [x,t]
    ms_cases = zeros(Int, nx)      # Marching squares case IDs
    edge_types = Vector{Symbol}(undef, nx+1)  # Edge type classification
    t_crosses = zeros(nx+1)        # Crossing times for interfaces
    center_types = Vector{Symbol}(undef, nx)  # Cell center type classification
    t_crosses_center = zeros(nx)   # Crossing times for cell centers
    
    # 1. Calculate space-time face capacities (Ax) and edge classifications
    for i in 1:nx+1
        x_face = x_nodes[i]
        
        # Check vertex state at both time steps
        is_wet_n = is_point_inside(front_n, x_face)
        is_wet_np1 = is_point_inside(front_np1, x_face)
        
        # Classify edge type and calculate Ax
        if !is_wet_n && !is_wet_np1
            # Case: Empty (dry at both times)
            edge_types[i] = :empty
            Ax_spacetime[i] = 0.0
            t_crosses[i] = dt/2  # Not used, just a placeholder
        
        elseif !is_wet_n && is_wet_np1
            # Case: Fresh (dry -> wet)
            edge_types[i] = :fresh
            # Find crossing time by linear interpolation
            t_cross = find_crossing_time(x_face, front_n, front_np1, dt)
            t_crosses[i] = t_cross
            Ax_spacetime[i] = dt - t_cross  # Δτ = t_n+1 - t_cross
        
        elseif is_wet_n && !is_wet_np1
            # Case: Dead (wet -> dry)
            edge_types[i] = :dead
            # Find crossing time by linear interpolation
            t_cross = find_crossing_time(x_face, front_n, front_np1, dt)
            t_crosses[i] = t_cross
            Ax_spacetime[i] = t_cross  # Δτ = t_cross - t_n
        
        else
            # Case: Full (wet at both times)
            edge_types[i] = :full
            Ax_spacetime[i] = dt
            t_crosses[i] = dt/2  # Not used, just a placeholder
        end
    end
    
    # 2. Calculate space-time volumes and marching squares cases
    for i in 1:nx
        # Cell dimensions
        x_min, x_max = x_nodes[i], x_nodes[i+1]
        dx = x_max - x_min
        
        # Determine cell type using marching squares
        vertices_wet = [
            is_point_inside(front_n, x_min),
            is_point_inside(front_n, x_max),
            is_point_inside(front_np1, x_max),
            is_point_inside(front_np1, x_min)
        ]
        
        # Calculate marching squares case ID
        ms_case = sum(Int(vertices_wet[j]) * 2^(j-1) for j in 1:4)
        ms_cases[i] = ms_case
        
        # Get edge types and crossing times
        left_edge = edge_types[i]
        right_edge = edge_types[i+1]
        t_left = t_crosses[i]
        t_right = t_crosses[i+1]
        
        # Calculate space-time case ID for volume calculation
        case_id = 0
        
        # Left edge contribution
        if left_edge == :dead
            case_id += 1
        elseif left_edge == :fresh
            case_id += 8
        elseif left_edge == :full
            case_id += 9
        end
        
        # Right edge contribution
        if right_edge == :dead
            case_id += 2
        elseif right_edge == :fresh
            case_id += 4
        elseif right_edge == :full
            case_id += 6
        end
        
        # Calculate volume and centroid using LibGEOS
        volume, centroid = calculate_spacetime_geometry_libgeos(case_id, x_min, x_max, dt, t_left, t_right)
        V_spacetime[i] = volume
        ST_centroids[i] = centroid  # Store the centroid vector directly
    end
    
    # 3. Calculate Bx using the centroid positions
    for i in 1:nx
        # If the cell has no fluid volume, Bx is zero
        if V_spacetime[i] <= 0
            Bx_spacetime[i] = 0.0
            center_types[i] = :empty
            t_crosses_center[i] = dt/2  # Placeholder
            continue
        end
        
        # Extract the x-coordinate of the centroid
        centroid_x = ST_centroids[i][1]
        
        # Check centroid state at both time steps
        is_centroid_wet_n = is_point_inside(front_n, centroid_x)
        is_centroid_wet_np1 = is_point_inside(front_np1, centroid_x)
        
        # Classify centroid type and calculate Bx
        if !is_centroid_wet_n && !is_centroid_wet_np1
            # Case: Empty (dry at both times - rare with positive volume)
            center_types[i] = :empty
            Bx_spacetime[i] = 0.0
            t_crosses_center[i] = dt/2  # Not used, just a placeholder
        
        elseif !is_centroid_wet_n && is_centroid_wet_np1
            # Case: Fresh (dry -> wet)
            center_types[i] = :fresh
            # Find crossing time by linear interpolation
            t_cross_center = find_crossing_time(centroid_x, front_n, front_np1, dt)
            t_crosses_center[i] = t_cross_center
            Bx_spacetime[i] = dt - t_cross_center  # Δτ = t_n+1 - t_cross
        
        elseif is_centroid_wet_n && !is_centroid_wet_np1
            # Case: Dead (wet -> dry)
            center_types[i] = :dead
            # Find crossing time by linear interpolation
            t_cross_center = find_crossing_time(centroid_x, front_n, front_np1, dt)
            t_crosses_center[i] = t_cross_center
            Bx_spacetime[i] = t_cross_center  # Δτ = t_cross - t_n
        
        else
            # Case: Full (wet at both times)
            center_types[i] = :full
            Bx_spacetime[i] = dt
            t_crosses_center[i] = dt/2  # Not used, just a placeholder
        end
    end

    # 4. Calculate Wx connections between adjacent centroids
    for i in 2:nx

        # Extract centroid coordinates
        centroid_left = ST_centroids[i-1]
        centroid_right = ST_centroids[i]
        x_left = centroid_left[1]
        x_right = centroid_right[1]
        
        # Use the crossing times for these centroids
        t_cross_left = t_crosses_center[i-1]
        t_cross_right = t_crosses_center[i]
        
        # Calculate Wx using LibGEOS with crossing times
        Wx_spacetime[i] = calculate_spacetime_wx_libgeos(
            centroid_left, 
            centroid_right, 
            front_n, 
            front_np1, 
            dt,
            t_cross_left,
            t_cross_right
        )
    end
    
    # Return all capacities in a dictionary
    return Dict(
        :Ax_spacetime => Ax_spacetime,     # Space-time face capacities
        :V_spacetime => V_spacetime,       # Space-time volumes
        :Bx_spacetime => Bx_spacetime,     # Space-time centerline capacities
        :Wx_spacetime => Wx_spacetime,     # Space-time connection capacities
        :ST_centroids => ST_centroids,     # Space-time centroids as [x,t] vectors
        :ms_cases => ms_cases,             # Marching squares case IDs
        :edge_types => edge_types,         # Edge classifications
        :t_crosses => t_crosses,           # Crossing times for interfaces
        :center_types => center_types,     # Cell center type classifications
        :t_crosses_center => t_crosses_center  # Crossing times for cell centers
    )
end

"""
    calculate_spacetime_wx_libgeos(centroid_left::Vector{Float64}, centroid_right::Vector{Float64}, 
                                  front_n::FrontTracker1D, front_np1::FrontTracker1D, dt::Float64,
                                  t_cross_left::Float64, t_cross_right::Float64)

Calculate the time-integrated connection capacity (Wx) between adjacent centroids using LibGEOS.
Uses crossing times and polygon approach instead of numerical integration.
"""
function calculate_spacetime_wx_libgeos(centroid_left::Vector{Float64}, centroid_right::Vector{Float64}, 
                                      front_n::FrontTracker1D, front_np1::FrontTracker1D, dt::Float64,
                                      t_cross_left::Float64, t_cross_right::Float64)
    # Extract centroid positions
    x_left = centroid_left[1]
    t_left = centroid_left[2]
    x_right = centroid_right[1]
    t_right = centroid_right[2]
    
    # Check if centroids are in fluid at both time steps
    is_left_wet_n = is_point_inside(front_n, x_left)
    is_left_wet_np1 = is_point_inside(front_np1, x_left)
    is_right_wet_n = is_point_inside(front_n, x_right)
    is_right_wet_np1 = is_point_inside(front_np1, x_right)
    
    # Get interface positions at both time steps
    interface_n = isempty(front_n.markers) ? NaN : front_n.markers[1]
    interface_np1 = isempty(front_np1.markers) ? NaN : front_np1.markers[1]
    
    # Create a polygon based on the fluid region between centroids
    coords = Vector{Vector{Float64}}()
    
    # Special cases for quick calculation
    
    # If both centroids are always wet, full connection
    if is_left_wet_n && is_left_wet_np1 && is_right_wet_n && is_right_wet_np1
        # Full connection is the area of the quadrilateral formed by the centroids
        coords = [
            [x_left, 0.0], 
            [x_right, 0.0], 
            [x_right, dt], 
            [x_left, dt],
            [x_left, 0.0]
        ]
    # If both centroids are always dry, no connection
    elseif !is_left_wet_n && !is_left_wet_np1 && !is_right_wet_n && !is_right_wet_np1
        return 0.0
    else
        # Define vertex states (wet/dry) for all four corners
        # Points are ordered: (x_left,0), (x_right,0), (x_right,dt), (x_left,dt)
        vertices_wet = [
            is_left_wet_n,
            is_right_wet_n,
            is_right_wet_np1,
            is_left_wet_np1
        ]
        
        # Calculate case ID (similar to marching squares)
        case_id = sum(Int(vertices_wet[j]) * 2^(j-1) for j in 1:4)
        
        # Build polygon based on case
        if case_id == 1  # Only bottom-left is wet
            # Linear interface from (x_left,0) to intersection with t=t_cross_left
            interface_at_t_cross_left = interface_n + (t_cross_left/dt) * (interface_np1 - interface_n)
            coords = [
                [x_left, 0.0],
                [interface_n, 0.0],
                [interface_at_t_cross_left, t_cross_left],
                [x_left, t_cross_left],
                [x_left, 0.0]
            ]
        elseif case_id == 2  # Only bottom-right is wet
            # Linear interface from (x_right,0) to intersection with t=t_cross_right
            interface_at_t_cross_right = interface_n + (t_cross_right/dt) * (interface_np1 - interface_n)
            coords = [
                [interface_n, 0.0],
                [x_right, 0.0],
                [x_right, t_cross_right],
                [interface_at_t_cross_right, t_cross_right],
                [interface_n, 0.0]
            ]
        elseif case_id == 4  # Only top-right is wet
            # Linear interface from (x_right,dt) to intersection with t=t_cross_right
            interface_at_t_cross_right = interface_n + (t_cross_right/dt) * (interface_np1 - interface_n)
            coords = [
                [interface_at_t_cross_right, t_cross_right],
                [x_right, t_cross_right],
                [x_right, dt],
                [interface_np1, dt],
                [interface_at_t_cross_right, t_cross_right]
            ]
        elseif case_id == 8  # Only top-left is wet
            # Linear interface from (x_left,dt) to intersection with t=t_cross_left
            interface_at_t_cross_left = interface_n + (t_cross_left/dt) * (interface_np1 - interface_n)
            coords = [
                [interface_at_t_cross_left, t_cross_left],
                [x_left, t_cross_left],
                [x_left, dt],
                [interface_np1, dt],
                [interface_at_t_cross_left, t_cross_left]
            ]
        elseif case_id == 3  # Bottom edge is wet (left and right)
            # Both centroids cross from wet to dry
            interface_at_t_cross_left = interface_n + (t_cross_left/dt) * (interface_np1 - interface_n)
            interface_at_t_cross_right = interface_n + (t_cross_right/dt) * (interface_np1 - interface_n)
            coords = [
                [x_left, 0.0],
                [x_right, 0.0],
                [x_right, t_cross_right],
                [interface_at_t_cross_right, t_cross_right],
                [interface_at_t_cross_left, t_cross_left],
                [x_left, t_cross_left],
                [x_left, 0.0]
            ]
        elseif case_id == 12  # Top edge is wet (left and right)
            # Both centroids cross from dry to wet
            interface_at_t_cross_left = interface_n + (t_cross_left/dt) * (interface_np1 - interface_n)
            interface_at_t_cross_right = interface_n + (t_cross_right/dt) * (interface_np1 - interface_n)
            coords = [
                [interface_at_t_cross_left, t_cross_left],
                [interface_at_t_cross_right, t_cross_right],
                [x_right, t_cross_right],
                [x_right, dt],
                [x_left, dt],
                [x_left, t_cross_left],
                [interface_at_t_cross_left, t_cross_left]
            ]
        elseif case_id == 9  # Left edge is wet (top and bottom)
            # Left centroid stays wet, right crosses wet->dry at bottom, dry->wet at top
            interface_at_t_cross_right = interface_n + (t_cross_right/dt) * (interface_np1 - interface_n)
            coords = [
                [x_left, 0.0],
                [interface_n, 0.0],
                [interface_at_t_cross_right, t_cross_right],
                [x_right, t_cross_right],
                [x_right, dt],
                [x_left, dt],
                [x_left, 0.0]
            ]
        elseif case_id == 6  # Right edge is wet (top and bottom)
            # Right centroid stays wet, left crosses wet->dry at bottom, dry->wet at top
            interface_at_t_cross_left = interface_n + (t_cross_left/dt) * (interface_np1 - interface_n)
            coords = [
                [interface_n, 0.0],
                [x_right, 0.0],
                [x_right, dt],
                [interface_np1, dt],
                [interface_at_t_cross_left, t_cross_left],
                [x_left, t_cross_left],
                [x_left, 0.0],
                [interface_n, 0.0]
            ]
        elseif case_id == 5  # Bottom-left and top-right are wet
            # Interface crossing through the window - complex case
            # Calculate t_cross_center where interface passes between centroids
            dx = x_right - x_left
            t_cross_center = (dx/2 + x_left - interface_n) / ((interface_np1 - interface_n) / dt)
            t_cross_center = clamp(t_cross_center, 0.0, dt)
            
            interface_at_t_cross_center = interface_n + (t_cross_center/dt) * (interface_np1 - interface_n)
            
            coords = [
                [x_left, 0.0],
                [interface_n, 0.0],
                [interface_at_t_cross_center, t_cross_center],
                [interface_np1, dt],
                [x_right, dt],
                [x_right, t_cross_right],
                [interface_at_t_cross_right, t_cross_right],
                [x_left, t_cross_left],
                [x_left, 0.0]
            ]
        elseif case_id == 10  # Bottom-right and top-left are wet
            # Interface crossing through the window - complex case
            # Calculate t_cross_center where interface passes between centroids
            dx = x_right - x_left
            t_cross_center = (dx/2 + x_left - interface_n) / ((interface_np1 - interface_n) / dt)
            t_cross_center = clamp(t_cross_center, 0.0, dt)
            
            interface_at_t_cross_center = interface_n + (t_cross_center/dt) * (interface_np1 - interface_n)
            
            coords = [
                [interface_n, 0.0],
                [x_right, 0.0],
                [x_right, t_cross_right],
                [interface_at_t_cross_right, t_cross_right],
                [interface_at_t_cross_center, t_cross_center],
                [interface_at_t_cross_left, t_cross_left],
                [x_left, t_cross_left],
                [x_left, dt],
                [interface_np1, dt],
                [interface_n, 0.0]
            ]
        elseif case_id == 7  # All wet except top-left
            interface_at_t_cross_left = interface_n + (t_cross_left/dt) * (interface_np1 - interface_n)
            coords = [
                [x_left, 0.0],
                [x_right, 0.0],
                [x_right, dt],
                [interface_np1, dt],
                [interface_at_t_cross_left, t_cross_left],
                [x_left, t_cross_left],
                [x_left, 0.0]
            ]
        elseif case_id == 11  # All wet except top-right
            interface_at_t_cross_right = interface_n + (t_cross_right/dt) * (interface_np1 - interface_n)
            coords = [
                [x_left, 0.0],
                [x_right, 0.0],
                [x_right, t_cross_right],
                [interface_at_t_cross_right, t_cross_right],
                [interface_np1, dt],
                [x_left, dt],
                [x_left, 0.0]
            ]
        elseif case_id == 13  # All wet except bottom-right
            interface_at_t_cross_right = interface_n + (t_cross_right/dt) * (interface_np1 - interface_n)
            coords = [
                [x_left, 0.0],
                [interface_n, 0.0],
                [interface_at_t_cross_right, t_cross_right],
                [x_right, t_cross_right],
                [x_right, dt],
                [x_left, dt],
                [x_left, 0.0]
            ]
        elseif case_id == 14  # All wet except bottom-left
            interface_at_t_cross_left = interface_n + (t_cross_left/dt) * (interface_np1 - interface_n)
            coords = [
                [interface_n, 0.0],
                [x_right, 0.0],
                [x_right, dt],
                [x_left, dt],
                [x_left, t_cross_left],
                [interface_at_t_cross_left, t_cross_left],
                [interface_n, 0.0]
            ]
        else
            # Should never reach here with the 16 possible cases covered above
            return 0.0
        end
    end
    
    # Create polygon and calculate area if we have coordinates
    if !isempty(coords)
        polygon = LibGEOS.Polygon([coords])
        return LibGEOS.area(polygon)
    end
    
    return 0.0
end
"""
    calculate_spacetime_geometry_libgeos(case_id::Int, x_min::Float64, x_max::Float64, 
                                       dt::Float64, t_left::Float64, t_right::Float64)

Calculate the space-time volume and centroid using LibGEOS library.
Returns a tuple (volume, [centroid_x, centroid_t]).
"""
function calculate_spacetime_geometry_libgeos(case_id::Int, x_min::Float64, x_max::Float64, 
                                           dt::Float64, t_left::Float64, t_right::Float64)
    # Full cell dimensions
    dx = x_max - x_min
    full_area = dx * dt
    
    # Cases with 0% fluid
    if case_id == 0
        # Default centroid for empty cells is cell center
        return 0.0, [x_min + dx/2, dt/2]
    end
    
    # Cases with 100% fluid
    if case_id == 15
        # For full cells, centroid is cell center
        return full_area, [x_min + dx/2, dt/2]
    end
    
    # Get polygon coordinates for this case
    coords = get_spacetime_polygon_coords(case_id, x_min, x_max, dt, t_left, t_right)
    
    # Create LibGEOS polygon
    polygon = LibGEOS.Polygon([coords])
    
    # Calculate area (volume in space-time)
    volume = LibGEOS.area(polygon)
    
    # Calculate centroid
    centroid_geom = LibGEOS.centroid(polygon)
    centroid_coords = [LibGEOS.getcoord(centroid_geom, i) for i in 1:2]
    
    # Return both results
    return volume, centroid_coords
end

"""
    get_spacetime_polygon_coords(case_id::Int, x_min::Float64, x_max::Float64, 
                               dt::Float64, t_left::Float64, t_right::Float64)

Creates polygon coordinates for each case in space-time.
"""
function get_spacetime_polygon_coords(case_id::Int, x_min::Float64, x_max::Float64, 
                                    dt::Float64, t_left::Float64, t_right::Float64)
    # Case 1: Only bottom-left is wet (left edge dead)
    if case_id == 1
        return [[x_min, 0.0], [x_max, 0.0], [x_min, t_left], [x_min, 0.0]]
    end
    
    # Case 2: Only bottom-right is wet (right edge dead)
    if case_id == 2
        return [[x_min, 0.0], [x_max, 0.0], [x_max, t_right], [x_min, 0.0]]
    end
    
    # Case 4: Only top-right is wet (right edge fresh)
    if case_id == 4
        return [[x_max, t_right], [x_max, dt], [x_min, dt], [x_max, t_right]]
    end
    
    # Case 8: Only top-left is wet (left edge fresh)
    if case_id == 8
        return [[x_min, t_left], [x_min, dt], [x_max, dt], [x_min, t_left]]
    end
    
    # Case 3: Bottom-left and bottom-right are wet
    if case_id == 3
        return [[x_min, 0.0], [x_max, 0.0], [x_max, t_right], [x_min, t_left], [x_min, 0.0]]
    end
    
    # Case 12: Top-left and top-right are wet
    if case_id == 12
        return [[x_min, t_left], [x_max, t_right], [x_max, dt], [x_min, dt], [x_min, t_left]]
    end
    
    # Case 6: Right edge is full (bottom-right and top-right are wet)
    if case_id == 6
        return [[x_min, 0.0], [x_max, 0.0], [x_max, dt], [x_min, dt], [x_min, 0.0]]
    end
    
    # Case 9: Left edge is full (bottom-left and top-left are wet)
    if case_id == 9
        return [[x_min, 0.0], [x_max, 0.0], [x_max, dt], [x_min, dt], [x_min, 0.0]]
    end
    
    # Case 7: All except top-left (left edge dead, right edge full)
    if case_id == 7
        return [[x_min, 0.0], [x_max, 0.0], [x_max, dt], [x_min, t_left], [x_min, 0.0]]
    end
    
    # Case 11: All except top-right (left edge full, right edge dead)
    if case_id == 11
        return [[x_min, 0.0], [x_max, 0.0], [x_max, t_right], [x_min, dt], [x_min, 0.0]]
    end
    
    # Case 13: All except bottom-right (left edge full, right edge fresh)
    if case_id == 13
        return [[x_min, 0.0], [x_max, t_right], [x_max, dt], [x_min, dt], [x_min, 0.0]]
    end
    
    # Case 14: All except bottom-left (left edge fresh, right edge full)
    if case_id == 14
        return [[x_min, t_left], [x_max, 0.0], [x_max, dt], [x_min, dt], [x_min, t_left]]
    end
    
    # Case 10: Left fresh, right dead
    if case_id == 10
        return [[x_min, t_left], [x_max, 0.0], [x_max, t_right], [x_min, dt], [x_min, t_left]]
    end
    
    # Case 5: Left dead, right fresh
    if case_id == 5
        return [[x_min, 0.0], [x_max, t_right], [x_max, dt], [x_min, t_left], [x_min, 0.0]]
    end
    
    # Full cell as fallback for any unhandled cases
    return [[x_min, 0.0], [x_max, 0.0], [x_max, dt], [x_min, dt], [x_min, 0.0]]
end

"""
    calculate_spacetime_volume(case_id::Int, dx::Float64, dt::Float64, t_left::Float64, t_right::Float64)

Helper function that calculates the fluid volume of a space-time cell based on its marching squares case.
"""
function calculate_spacetime_volume(case_id::Int, dx::Float64, dt::Float64, t_left::Float64, t_right::Float64)
    # Full cell area
    full_area = dx * dt
    
    # Cases with 0% fluid
    if case_id == 0
        return 0.0
    end
    
    # Cases with 100% fluid
    if case_id == 15
        return full_area
    end
    
    # Handle specific cases to calculate the fluid area
    
    # Case 1: Only bottom-left is wet (left edge dead)
    if case_id == 1
        # Triangle with base dx and height t_left
        return 0.5 * dx * t_left
    end
    
    # Case 2: Only bottom-right is wet (right edge dead)
    if case_id == 2
        # Triangle with base dx and height t_right
        return 0.5 * dx * t_right
    end
    
    # Case 4: Only top-right is wet (right edge fresh)
    if case_id == 4
        # Triangle with base dx and height (dt-t_right)
        return 0.5 * dx * (dt - t_right)
    end
    
    # Case 8: Only top-left is wet (left edge fresh)
    if case_id == 8
        # Triangle with base dx and height (dt-t_left)
        return 0.5 * dx * (dt - t_left)
    end
    
    # Case 3: Bottom-left and bottom-right are wet
    if case_id == 3
        # Trapezoid with bases t_left and t_right
        return 0.5 * dx * (t_left + t_right)
    end
    
    # Case 6: Bottom-right and top-right are wet (right edge full)
    if case_id == 6
        return dx * dt - 0.5 * dx * (dt - t_right)
    end
    
    # Case 9: Bottom-left and top-left are wet (left edge full)
    if case_id == 9
        return dx * dt - 0.5 * dx * (dt - t_left)
    end
    
    # Case 12: Top-left and top-right are wet
    if case_id == 12
        # Trapezoid with bases (dt-t_left) and (dt-t_right)
        return 0.5 * dx * (2*dt - t_left - t_right)
    end
    
    # Case 7: All except top-left
    if case_id == 7
        return full_area - 0.5 * dx * (dt - t_left)
    end
    
    # Case 11: All except top-right
    if case_id == 11
        return full_area - 0.5 * dx * (dt - t_right)
    end
    
    # Case 13: All except bottom-right
    if case_id == 13
        return full_area - 0.5 * dx * t_right
    end
    
    # Case 14: All except bottom-left
    if case_id == 14
        return full_area - 0.5 * dx * t_left
    end
    
    # Impossible cases (5, 10) and anything unhandled - use average of vertex values
    # This shouldn't happen with proper classification
    println("Warning: Unhandled marching squares case: $case_id")
    return 0.5 * full_area
end

"""
    find_crossing_time(x_face::Float64, front_n::FrontTracker1D, front_np1::FrontTracker1D, dt::Float64)

Estimates the time when the interface crosses the given x_face position.
Uses linear interpolation between time steps.
"""
function find_crossing_time(x_face::Float64, front_n::FrontTracker1D, front_np1::FrontTracker1D, dt::Float64)
    # Pour une interface qui se déplace linéairement, nous pouvons calculer directement
    # Déduire la position de l'interface en fonction du temps
    interface_pos_n = NaN
    interface_pos_np1 = NaN
    
    # Supposons une seule interface pour ce cas test simplifié
    if !isempty(front_n.markers)
        interface_pos_n = front_n.markers[1]
    end
    
    if !isempty(front_np1.markers)
        interface_pos_np1 = front_np1.markers[1]
    end
    
    # Si nous avons les deux positions, nous pouvons interpoler
    if isfinite(interface_pos_n) && isfinite(interface_pos_np1)
        velocity = (interface_pos_np1 - interface_pos_n) / dt
        
        if abs(velocity) < 1e-10
            return dt / 2  # Éviter la division par zéro
        end
        
        # Calcul du temps de traversée par interpolation linéaire
        t_cross = dt * (x_face - interface_pos_n) / (interface_pos_np1 - interface_pos_n)
        
        # Limiter à l'intervalle [0, dt]
        return clamp(t_cross, 0.0, dt)
    end
    
    return dt / 2  # Valeur par défaut
end