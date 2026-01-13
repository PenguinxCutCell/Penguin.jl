"""
    MeshTag{N}

A tag structure for a mesh, containing information about border cells.
"""
struct MeshTag{N}
    border_cells::Vector{Tuple{CartesianIndex{N}, NTuple{N, Float64}}}
end

"""
    Dimension{N}

A type parameter representing the spatial dimension N of the grid.
"""
struct Dimension{N} end

"""
    Dimension(N::Int) -> Dimension{N}

Zero-argument constructor for Dimension type.
"""
Dimension(N::Int) = Dimension{N}()
(::Dimension{N})() where {N} = N

abstract type AbstractMesh end

"""
    Mesh{N}(n::NTuple{N, Int}, domain_size::NTuple{N, Float64}, x0::NTuple{N, Float64}=ntuple(_ -> 0.0, N))

Create a mesh object with `N` dimensions, `n` cells in each dimension, and a domain size of `domain_size`.
The mesh uses a cell-centered discretization: |--x--|--x--|--x--| where x represents cell centers, | represents cell boundaries (nodes).

# Arguments
- `n::NTuple{N, Int}`: A tuple of integers specifying the number of cells in each dimension.
- `domain_size::NTuple{N, Float64}`: A tuple of floats specifying the size of the domain in each dimension.
- `x0::NTuple{N, Float64}`: A tuple of floats specifying the origin of the domain in each dimension. Default is the origin.

# Returns
- A `Mesh{N}` object with `N` dimensions, `n` cells in each dimension, and a domain size of `domain_size`.
  - `centers`: Cell center coordinates (at the midpoint of each cell)
  - `nodes`: Cell boundary coordinates (faces between cells)
"""
struct Mesh{N} <: AbstractMesh
    centers::NTuple{N, Vector{Float64}}
    nodes::NTuple{N, Vector{Float64}}
    tag::MeshTag
    dims::NTuple{N, Int}

    function Mesh(n::NTuple{N, Int}, domain_size::NTuple{N, Float64}, x0::NTuple{N, Float64}=ntuple(_ -> 0.0, N)) where N
        # Calculate centers and nodes
        centers_uniform = ntuple(i -> [x0[i] + (j + 0.5) * (domain_size[i] / n[i]) for j in 0:n[i]-1], N)
        nodes_uniform = ntuple(i -> [x0[i] + j * (domain_size[i] / n[i]) for j in 0:(n[i])], N)
        
        # Calculate border cells directly from centers
        dims = ntuple(i -> length(centers_uniform[i]), N)
        border_cells = Vector{Tuple{CartesianIndex{N}, NTuple{N, Float64}}}()
        
        # For each dimension
        for d in 1:N
            # For each face (lower and upper) in dimension d
            for face_val in [1, dims[d]]
                # Create iterators for all other dimensions
                ranges = [1:dims[i] for i in 1:N]
                # Fix the current dimension to the face value
                ranges[d] = face_val:face_val
                
                # Generate all cells on this face
                for idx in Iterators.product(ranges...)
                    pos = ntuple(i -> centers_uniform[i][idx[i]], N)
                    push!(border_cells, (CartesianIndex(idx), pos))
                end
            end
        end
        
        # Remove duplicates (corner cells)
        unique!(border_cells)
        
        # Create the mesh directly with all components
        return new{N}(centers_uniform, nodes_uniform, MeshTag{N}(border_cells), dims)
    end
end

"""
    nC(mesh::AbstractMesh)

Calculate the total number of cells in a mesh.
"""
nC(mesh::AbstractMesh) = prod(mesh.dims)

"""
    size(grid::AbstractMesh{N}) -> NTuple{N, Int}

Return the number of cells in each dimension.
"""
Base.size(grid::AbstractMesh) = grid.dims

"""
    size(grid::AbstractMesh, dim::Int) -> Int

Return the number of cells in dimension `dim`.
"""
Base.size(grid::AbstractMesh, dim::Int) = grid.dims[dim]


"""
    SpaceTimeMesh{M} <: AbstractMesh

A mesh structure that combines a spatial mesh with a temporal dimension.

# Fields
- `nodes::NTuple{M, Vector{Float64}}`: Node coordinates for each dimension
- `centers::NTuple{M, Vector{Float64}}`: Cell center coordinates for each dimension
- `tag::MeshTag`: Mesh tags for identifying regions or boundaries

# Constructor
    SpaceTimeMesh(spaceMesh::Mesh{N}, time::Vector{Float64}; tag::MeshTag=MeshTag{N}([]))

Creates a space-time mesh by extending a spatial `Mesh{N}` with a temporal dimension.

# Arguments
- `spaceMesh::Mesh{N}`: A spatial mesh with N dimensions
- `time::Vector{Float64}`: Vector of time points representing the temporal grid

# Keyword Arguments
- `tag::MeshTag=MeshTag{N}([])`: Optional mesh tag information

# Notes
- The resulting mesh has dimensionality M = N + 1, where the (N+1)th dimension represents time
- Time centers are computed as midpoints between consecutive time nodes
"""
struct SpaceTimeMesh{M} <: AbstractMesh
    nodes::NTuple{M, Vector{Float64}}
    centers::NTuple{M, Vector{Float64}}
    tag::MeshTag
    dims::NTuple{M, Int}

    function SpaceTimeMesh(spaceMesh::Mesh{N}, time::Vector{Float64}; tag::MeshTag=MeshTag{N}([])) where {N}
        local M = N + 1

        centers_time = [(time[i+1] + time[i]) / 2 for i in 1:length(time)-1]
        nodes = ntuple(i -> i<=N ? spaceMesh.nodes[i] : time, M)
        centers = ntuple(i -> i<=N ? spaceMesh.centers[i] : centers_time, M)
        dims = ntuple(i -> length(centers[i]), M)
        return new{M}(nodes, centers, tag, dims)
    end
end
