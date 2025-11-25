"""
    abstract type AbstractCapacity

Abstract type representing a capacity.
"""
abstract type AbstractCapacity end

"""
    mutable struct Capacity{N} <: AbstractCapacity

The `Capacity` struct represents the capacity of a system in `N` dimensions.

# Fields
- `A`: A capacity represented by `N` sparse matrices (`Ax`, `Ay`).
- `B`: B capacity represented by `N` sparse matrices (`Bx`, `By`).
- `V`: Volume capacity represented by a sparse matrix.
- `W`: Staggered volume capacity represented by `N` sparse matrices.
- `C_ω`: Cell centroid represented by a vector of `N`-dimensional static vectors.
- `C_γ`: Interface centroid represented by a vector of `N`-dimensional static vectors.
- `Γ`: Interface norm represented by a sparse matrix.
- `cell_types`: Cell types.
- `mesh`: Mesh of `N` dimensions.

"""
mutable struct Capacity{N} <: AbstractCapacity
    A :: NTuple{N, SparseMatrixCSC{Float64, Int}}   # A capacity : Ax, Ay
    B :: NTuple{N, SparseMatrixCSC{Float64, Int}}   # B capacity : Bx, By
    V :: SparseMatrixCSC{Float64, Int}              # Volume
    W :: NTuple{N, SparseMatrixCSC{Float64, Int}}   # Staggered Volume
    C_ω :: Vector{SVector{N,Float64}}               # Cell Centroid
    C_γ :: Vector{SVector{N,Float64}}               # Interface Centroid
    Γ :: SparseMatrixCSC{Float64, Int}              # Interface Norm
    cell_types :: Vector{Float64}                   # Cell Types
    mesh :: AbstractMesh                            # Mesh
    body :: Function                                # Body function (signed distance function)
end

"""
    Capacity(body::Function, mesh::CartesianMesh; method::String = "VOFI", integration_method::Symbol = :vofi)

Compute the capacity of a body in a given mesh using a specified method.

# Arguments
- `body::Function`: The body for which to compute the capacity.
- `mesh::CartesianMesh`: The mesh in which the body is located.
- `method::String`: The method to use for computing the capacity. Default is "VOFI".
- `integration_method::Symbol`: Backend for VOFI integration (`:vofi` by default).

# Returns
- `Capacity{N}`: The capacity of the body.
"""
function Capacity(body::Function, mesh::AbstractMesh; method::String = "VOFI", integration_method::Symbol = :vofi, compute_centroids::Bool = true, tol::Float64 = 1e-6)

    if method == "VOFI"
        #println("When using VOFI, the body must be a scalar function.")
        A, B, V, W, C_ω, C_γ, Γ, cell_types = VOFI(body, mesh; compute_centroids=compute_centroids, integration_method=integration_method)
        N = length(A)
        return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, body)
    elseif method == "ImplicitIntegration"
        #println("Computing capacity using geometric moments integration.")
        A, B, V, W, C_ω, C_γ, Γ, cell_types = ImplicitCutIntegration.GeometricMoments(body, mesh.nodes; compute_centroids=compute_centroids, tol=tol)
        N = length(A)
        return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, body)
    end    
end

# VOFI implementation

"""
    VOFI(body::Function, mesh::AbstractMesh; compute_centroids::Bool = true, integration_method::Symbol = :vofi)

Compute capacity quantities based on VOFI for a given body and mesh.

# Arguments
- `body::Function`: The level set function defining the domain
- `mesh::AbstractMesh`: The mesh on which to compute the VOFI quantities
- `compute_centroids::Bool`: Whether to compute interface centroids
- `integration_method::Symbol`: Backend for CartesianGeometry.integrate (e.g. `:vofi` or `:vofijul`)

# Returns
- Tuple of capacity components (A, B, V, W, C_ω, C_γ, Γ, cell_types)
"""
function VOFI(body::Function, mesh::AbstractMesh; compute_centroids::Bool = true, integration_method::Symbol = :vofi)
    N = length(mesh.nodes)
    nc = nC(mesh)
    
    # Only initialize variables we actually need
    local V, A, B, W, C_ω, C_γ, Γ, cell_types
    
    # Get volume capacity, barycenters, interface length and cell types in a single call
    # This avoids redundant computations and memory allocations
    Vs, bary, interface_length, cell_types = CartesianGeometry.integrate(
        Tuple{0}, body, mesh.nodes, Float64, zero; method=integration_method
    )
    
    # Store cell centroids
    C_ω = bary
    
    # Create shared sparse matrices (identical for all dimensions)
    V = spdiagm(0 => Vs)
    Γ = spdiagm(0 => interface_length)
    
    # Do dimension-specific calculations in one shot
    # Compute face capacities, center line capacities, and staggered volumes
    As = CartesianGeometry.integrate(Tuple{1}, body, mesh.nodes, Float64, zero; method=integration_method)
    Ws = CartesianGeometry.integrate(Tuple{0}, body, mesh.nodes, Float64, zero, bary; method=integration_method)
    Bs = CartesianGeometry.integrate(Tuple{1}, body, mesh.nodes, Float64, zero, bary; method=integration_method)
    
    # Create appropriate-sized tuples based on dimension
    # This avoids the dimension-specific if/elseif blocks
    A = ntuple(i -> i <= length(As) ? spdiagm(0 => As[i]) : spzeros(0), N)
    B = ntuple(i -> i <= length(Bs) ? spdiagm(0 => Bs[i]) : spzeros(0), N)
    W = ntuple(i -> i <= length(Ws) ? spdiagm(0 => Ws[i]) : spzeros(0), N)
    
    # Compute interface centroids if requested
    # Use a cache-friendly approach by checking for previously computed values
    if compute_centroids
        C_γ = computeInterfaceCentroids(mesh, body)
    else
        # Create empty vector of appropriate type based on dimension
        C_γ = Vector{SVector{N,Float64}}(undef, 0)
    end
    
    return A, B, V, W, C_ω, C_γ, Γ, cell_types
end

"""
    computeInterfaceCentroids(mesh::Union{Mesh{N}, SpaceTimeMesh{N}}, body) where N

Compute the interface centroids for an N-dimensional mesh and body.

# Arguments
- `mesh::Union{Mesh{N}, SpaceTimeMesh{N}}`: The mesh on which to compute the interface centroids
- `body::Function`: The body level set function

# Returns
- `C_γ::Vector{SVector{N,Float64}}`: Vector of interface centroids
"""
function computeInterfaceCentroids(mesh::Union{Mesh{N}, SpaceTimeMesh{N}}, body) where N
    # Extract node coordinates
    coords = mesh.nodes
    
    # Calculate dimensions of the grid
    dims = mesh.dims
    
    # Create dimension-appropriate level set function
    Φ = if N == 1
        (r) -> body(r[1])
    elseif N == 2
        (r) -> body(r[1], r[2])
    elseif N == 3
        (r) -> body(r[1], r[2], r[3])
    elseif N == 4
        (r) -> body(r[1], r[2], r[3], r[4])
    else
        error("Unsupported dimension: $N")
    end
    
    # Calculate total cells and preallocate result vector with appropriate size
    total_cells = prod(ntuple(i -> dims[i]+1, N))
    C_γ = Vector{SVector{N,Float64}}(undef, total_cells)
    
    # Generate all cell indices
    indices = CartesianIndices(ntuple(i -> 1:dims[i], N))
    
    # Process each cell
    for idx in indices
        # Calculate linear index based on dimension
        if N == 1
            linear_idx = idx[1]
        elseif N == 2
            i, j = idx[1], idx[2]
            linear_idx = (dims[1]+1) * (j-1) + i
        elseif N == 3
            i, j, k = idx[1], idx[2], idx[3]
            linear_idx = (dims[1]+1) * (dims[2]+1) * (k-1) + (dims[1]+1) * (j-1) + i
        end
        
        # Get cell bounds
        a = ntuple(i -> coords[i][idx[i]], N)
        b = ntuple(i -> coords[i][idx[i]+1], N)
        
        # Compute measure
        measure_val = ImplicitIntegration.integrate(_->1, Φ, a, b; surface=true).val
        
        if measure_val > 0
            # Compute centroid coordinates
            centroid_coords = ntuple(N) do d
                ImplicitIntegration.integrate(p->p[d], Φ, a, b; surface=true).val / measure_val
            end
            
            C_γ[linear_idx] = SVector{N,Float64}(centroid_coords)
        else
            C_γ[linear_idx] = SVector{N,Float64}(ntuple(_ -> 0.0, N))
        end
    end
    
    return C_γ
end

# Capacity from Front Tracker

"""
    Capacity(front::FrontTracker, mesh::AbstractMesh; compute_centroids::Bool = true)

Compute the capacity directly from a front tracker without using a level set function.

# Arguments
- `front::FrontTracker`: The front tracker object defining the fluid domain
- `mesh::AbstractMesh`: The mesh on which to compute the capacity 
- `compute_centroids::Bool`: Whether to compute interface centroids

# Returns
- `Capacity{N}`: The capacity of the domain defined by the front tracker
"""
function Capacity(front::FrontTracker, mesh::AbstractMesh; compute_centroids::Bool = true)
    # Convert front tracking capacities to Capacity format
    A, B, V, W, C_ω, C_γ, Γ, cell_types = FrontTrackingToCapacity(front, mesh; compute_centroids=compute_centroids)
    N = 2  # Only 2D is supported
    
    # Create a dummy level set function based on the front tracker's SDF
    dummy_body(x, y, z=0.0) = sdf(front, x, y)
    
    return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, dummy_body)
end

"""
    FrontTrackingToCapacity(front::FrontTracker, mesh::AbstractMesh; compute_centroids::Bool = true)

Convert front tracking capacities to the format expected by the Capacity struct.

# Arguments
- `front::FrontTracker`: The front tracker object defining the fluid domain
- `mesh::AbstractMesh`: The mesh on which to compute the capacity
- `compute_centroids::Bool`: Whether to compute interface centroids

# Returns
- Tuple of capacity components in Capacity struct format
"""
function FrontTrackingToCapacity(front::FrontTracker, mesh::AbstractMesh; compute_centroids::Bool = true)
    if isa(mesh, Mesh{2}) || isa(mesh, SpaceTimeMesh{2})
        # Compute all capacities using front tracking
        ft_capacities = compute_capacities(mesh, front)
        
        # Extract dimensions
        x_nodes = mesh.nodes[1]
        y_nodes = mesh.nodes[2]
        nx = length(x_nodes) 
        ny = length(y_nodes) 
        nc = nx * ny  # Total number of cells
        
        # Convert volumes to the required sparse format
        volumes = ft_capacities[:volumes][1:nx, 1:ny]
        V = spdiagm(0 => reshape(volumes, :))
        
        # Get interface information
        interface_lengths_vec = zeros(nc)
        for ((i, j), length) in ft_capacities[:interface_lengths]
            if 1 <= i <= nx && 1 <= j <= ny
                idx = (j-1)*nx + i
                interface_lengths_vec[idx] = length
            end
        end
        Γ = spdiagm(0 => interface_lengths_vec)
        
        # Extract and convert Ax, Ay to sparse format
        Ax_dense = ft_capacities[:Ax][1:nx, 1:ny]
        Ay_dense = ft_capacities[:Ay][1:nx, 1:ny]
        
        # Create vectors for the sparse matrices
        Ax_vec = reshape(ft_capacities[:Ax][1:nx, 1:ny], :)
        Ay_vec = reshape(ft_capacities[:Ay][1:nx, 1:ny], :)
        
        A = (spdiagm(0 => Ax_vec), spdiagm(0 => Ay_vec))
        
        # Extract and convert Bx, By
        Bx_dense = ft_capacities[:Bx][1:nx, 1:ny]
        By_dense = ft_capacities[:By][1:nx, 1:ny]
        
        Bx_vec = reshape(Bx_dense, :)
        By_vec = reshape(By_dense, :)
        
        B = (spdiagm(0 => Bx_vec), spdiagm(0 => By_vec))
        
        # Extract and convert Wx, Wy
        # Note: Need to adjust indices to match VOFI convention
        Wx_dense = ft_capacities[:Wx][1:nx, 1:ny]  # Wx[i+1,j] in front tracking
        Wy_dense = ft_capacities[:Wy][1:nx, 1:ny]  # Wy[i,j+1] in front tracking
        
        Wx_vec = reshape(Wx_dense, :)
        Wy_vec = reshape(Wy_dense, :)
        
        W = (spdiagm(0 => Wx_vec), spdiagm(0 => Wy_vec))
        
        # Create cell centroids in required format
        centroids_x = ft_capacities[:centroids_x][1:nx, 1:ny]
        centroids_y = ft_capacities[:centroids_y][1:nx, 1:ny]
        
        C_ω = [SVector{2, Float64}(centroids_x[i, j], centroids_y[i, j]) 
               for j in 1:ny for i in 1:nx]
        
        # Get cell fractions (cell types)
        cell_types = ft_capacities[:cell_types][1:nx, 1:ny]
        cell_types = reshape(cell_types, :)
        
        # Create interface centroids if requested
        if compute_centroids
            C_γ = Vector{SVector{2, Float64}}(undef, nc)
            
            # Initialize with zeros
            for i in 1:nc
                C_γ[i] = SVector{2, Float64}(0.0, 0.0)
            end
            
            # Fill in known interface points
            for ((i, j), point) in ft_capacities[:interface_points]
                if 1 <= i <= nx && 1 <= j <= ny
                    idx = (j-1)*nx + i
                    C_γ[idx] = SVector{2, Float64}(point[1], point[2])
                end
            end
        else
            C_γ = Vector{SVector{2, Float64}}(undef, 0)
        end
        
        return A, B, V, W, C_ω, C_γ, Γ, cell_types
    else
        error("Front Tracking capacity computation is only supported for 2D meshes")
    end
end

"""
    FrontTracker1DToCapacity(front::FrontTracker1D, mesh::AbstractMesh; compute_centroids::Bool = true)

Convert 1D front tracking capacities to the format expected by the Capacity struct.
"""
function FrontTracker1DToCapacity(front::FrontTracker1D, mesh::AbstractMesh; compute_centroids::Bool = true)
    if isa(mesh, Mesh{1})
        # Compute all capacities using front tracking
        ft_capacities = compute_capacities_1d(mesh, front)
        
        # Extract dimensions
        x_nodes = mesh.nodes[1]
        nx = length(x_nodes)
        
        # Convert volumes to the required sparse format
        volumes = ft_capacities[:volumes][1:nx]
        V = spdiagm(0 => volumes)
        
        # Create interface information
        interface_lengths = zeros(nx)
        for (i, pos) in ft_capacities[:interface_positions]
            interface_lengths[i] = 1.0  # In 1D, interface "length" is 1
        end
        Γ = spdiagm(0 => interface_lengths)
        
        # Extract face capacities
        Ax_vec = ft_capacities[:Ax][1:nx]
        A = (spdiagm(0 => Ax_vec),)
        
        # Extract center line capacities
        Bx_vec = ft_capacities[:Bx][1:nx]
        B = (spdiagm(0 => Bx_vec),)
        
        # Extract staggered volumes
        Wx_vec = ft_capacities[:Wx][1:nx]
        W = (spdiagm(0 => Wx_vec),)
        
        # Create cell centroids
        centroids_x = ft_capacities[:centroids_x][1:nx]
        C_ω = [SVector{1, Float64}(centroids_x[i]) for i in 1:nx]
        
        # Get cell types
        cell_types = ft_capacities[:cell_types][1:nx]
        
        # Create interface centroids if requested
        if compute_centroids
            C_γ = Vector{SVector{1, Float64}}(undef, nx)
            
            # Initialize with zeros
            for i in 1:nx
                C_γ[i] = SVector{1, Float64}(0.0)
            end
            
            # Fill in known interface points
            for (i, pos) in ft_capacities[:interface_positions]
                if 1 <= i <= nx
                    C_γ[i] = SVector{1, Float64}(pos)
                end
            end
        else
            C_γ = Vector{SVector{1, Float64}}(undef, 0)
        end
        
        return (A, B, V, W, C_ω, C_γ, Γ, cell_types)
    else
        error("1D Front Tracking capacity computation is only supported for 1D meshes")
    end
end

"""
    Capacity(front::FrontTracker1D, mesh::AbstractMesh; compute_centroids::Bool = true)

Compute the capacity directly from a 1D front tracker.
"""
function Capacity(front::FrontTracker1D, mesh::AbstractMesh; compute_centroids::Bool = true)
    # Convert front tracking capacities to Capacity format
    A, B, V, W, C_ω, C_γ, Γ, cell_types = FrontTracker1DToCapacity(front, mesh; compute_centroids=compute_centroids)
    
    # Create a dummy level set function based on the front tracker's SDF
    dummy_body(x, y=0.0, z=0.0) = sdf(front, x)
    
    return Capacity{1}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, dummy_body)
end

# Utilities for capacity cleaning / clamping

"""
    removed = remove_small_volumes!(cap::Capacity{N}, tol::Float64)

Set to zero all capacity entries for cells whose volume `V[i,i] < tol`.
Returns the vector of removed indices.
This is a destructive in-place operation.
"""
function remove_small_volumes!(cap::Capacity{N}, tol::Float64) where N
    nc = size(cap.V, 1)
    removed = Int[]
    # Extract diagonal volumes efficiently
    for i in 1:nc
        Vi = cap.V[i, i]
        if Vi < tol
            push!(removed, i)
            # zero-out primary quantities
            cap.V[i, i] = 0.0
            cap.Γ[i, i] = 0.0
            cap.cell_types[i] = 0.0
            cap.C_ω[i] = SVector{N,Float64}(ntuple(_->0.0, N))
            # zero face / center / staggered capacities
            for A_mat in cap.A
                if size(A_mat,1) ≥ i
                    A_mat[i, i] = 0.0
                end
            end
            for B_mat in cap.B
                if size(B_mat,1) ≥ i
                    B_mat[i, i] = 0.0
                end
            end
            for W_mat in cap.W
                if size(W_mat,1) ≥ i
                    W_mat[i, i] = 0.0
                end
            end
        end
    end
    return removed
end

"""
    mapping = clamp_merge_small_cells!(cap::Capacity{N}; tol=1e-12)

For each cell with volume < tol, find the nearest cell (by Euclidean distance
between cell centroids C_ω) that has volume >= tol and merge the small cell
into that neighbour.

Merging behaviour (in-place):
- volumes, interface measures (Γ) and staggered volumes are summed into the target
- diagonal entries of A and B are summed into the target; source diagonals zeroed
- centroids C_ω are updated by volume-weighted averaging
- source cell entries are zeroed (volume, capacities, Γ, cell_type, centroids)

Returns a Vector{Tuple{Int,Int}} of (source_idx, target_idx) merges performed.

Notes:
- This is a pragmatic heuristic for "clamping" tiny cut cells into nearby
  valid cells. It attempts to preserve total volume and capacity magnitudes.
"""
function clamp_merge_small_cells!(cap::Capacity{N}; tol::Float64=1e-12) where N
    nc = size(cap.V, 1)
    Vs = [cap.V[i,i] for i in 1:nc]
    # indices with small volume
    small_idx = findall(v -> v < tol, Vs)
    # candidates with sufficient volume
    good_idx = findall(v -> v >= tol, Vs)
    merges = Tuple{Int,Int}[]
    if isempty(good_idx)
        # nothing to merge into
        return merges
    end

    # Precompute numeric centroids as vectors for distance computations
    centroids = [collect(cap.C_ω[i]) for i in 1:nc]

    for i in small_idx
        Vi = Vs[i]
        # find nearest good index by Euclidean distance of centroids
        best = nothing
        bestd = Inf
        ci = centroids[i]
        for k in good_idx
            ck = centroids[k]
            # if target has zero volume skip (shouldn't happen)
            if cap.V[k,k] < tol
                continue
            end
            d2 = sum((ci .- ck).^2)
            if d2 < bestd
                bestd = d2
                best = k
            end
        end

        if best === nothing
            # nothing to merge with, skip
            continue
        end
        k = best
        # perform merge: keep previous values
        Vk = cap.V[k,k]
        Γk = cap.Γ[k,k]
        # add volumes and interface measure
        cap.V[k,k] = Vk + cap.V[i,i]
        cap.Γ[k,k] = Γk + cap.Γ[i,i]

        # merge diagonal entries of A, B, W into target and zero source
        for m in 1:length(cap.A)
            A_mat = cap.A[m]
            if size(A_mat,1) ≥ max(i,k)
                Ai = A_mat[i,i]
                Ak = A_mat[k,k]
                A_mat[k,k] = Ak + Ai
                A_mat[i,i] = 0.0
            end
        end
        for m in 1:length(cap.B)
            B_mat = cap.B[m]
            if size(B_mat,1) ≥ max(i,k)
                Bi = B_mat[i,i]
                Bk = B_mat[k,k]
                B_mat[k,k] = Bk + Bi
                B_mat[i,i] = 0.0
            end
        end
        for m in 1:length(cap.W)
            W_mat = cap.W[m]
            if size(W_mat,1) ≥ max(i,k)
                Wi = W_mat[i,i]
                Wk = W_mat[k,k]
                W_mat[k,k] = Wk + Wi
                W_mat[i,i] = 0.0
            end
        end

        # volume-weighted centroid update (avoid div by zero)
        newV = cap.V[k,k]
        if newV > 0
            old_ck = collect(cap.C_ω[k])
            old_ci = collect(cap.C_ω[i])
            # use previous Vk and Vi for weighting (Vk may have been zero)
            wk = Vk
            wi = cap.V[i,i]
            if (wk + wi) > 0
                newc = ((wk .* old_ck) .+ (wi .* old_ci)) ./ (wk + wi)
                cap.C_ω[k] = SVector{N,Float64}(Tuple(newc...))
            end
        end

        # update cell_types heuristically: prefer larger (fluid) type if any
        ct_k = cap.cell_types[k]
        ct_i = cap.cell_types[i]
        cap.cell_types[k] = ifelse(abs(ct_k) >= abs(ct_i), ct_k, ct_i)

        # zero-out source entries
        cap.V[i,i] = 0.0
        cap.Γ[i,i] = 0.0
        cap.cell_types[i] = 0.0
        cap.C_ω[i] = SVector{N,Float64}(ntuple(_->0.0, N))

        push!(merges, (i, k))
    end

    return merges
end
