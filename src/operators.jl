"""
    abstract type AbstractOperators

An abstract type representing a collection of operators.
"""
abstract type AbstractOperators end

# Elementary operators
function ẟ_m(n::Int, periodicity::Bool=false) D = spdiagm(0 => ones(n), -1 => -ones(n-1)); D[n, n] = 0.0; if periodicity; D[1, n-1] = -1.0; D[n, 1] = 1.0; end; D end
function δ_p(n::Int, periodicity::Bool=false) D = spdiagm(0 => -ones(n), 1 => ones(n-1)); D[n, n] = 0.0; if periodicity; D[1, n-1] = -1.0; D[n, 1] = 1.0; end; D end
function Σ_m(n::Int, periodicity::Bool=false) D = 0.5 * spdiagm(0 => ones(n), -1 => ones(n-1)); D[n, n] = 0.0; if periodicity; D[1, n-1] = 0.5; D[n, 1] = 0.5; end; D end
function Σ_p(n::Int, periodicity::Bool=false) D = 0.5 * spdiagm(0 => ones(n), 1 => ones(n-1)); D[n, n] = 0.0; if periodicity; D[1, n-1] = 0.5; D[n, 1] = 0.5; end; D end
function I(n::Int) spdiagm(0 => ones(n)) end

"""
    ∇(operator::AbstractOperators, p::Vector{Float64})

Compute the gradient of a scalar field.
"""
function ∇(operator::AbstractOperators, p::Vector{Float64})
    ∇ = operator.Wꜝ * (operator.G * p[1:div(end,2)] + operator.H * p[div(end,2)+1:end])
    return ∇
end

"""
    ∇₋(operator::AbstractOperators, qω::Vector{Float64}, qγ::Vector{Float64})

Compute the divergence of a vector field.
"""
function ∇₋(operator::AbstractOperators, qω::Vector{Float64}, qγ::Vector{Float64})
    GT = operator.G'
    HT = operator.H'
    return -(GT + HT)*qω + HT * qγ
end

"""
    struct DiffusionOps{N} <: AbstractOperators where N

Struct representing diffusion operators.

# Fields
- `G::SparseMatrixCSC{Float64, Int}`: Matrix representing the diffusion operator G.
- `H::SparseMatrixCSC{Float64, Int}`: Matrix representing the diffusion operator H.
- `Wꜝ::SparseMatrixCSC{Float64, Int}`: Matrix representing the diffusion operator Wꜝ.
- `V::SparseMatrixCSC{Float64, Int}`: Matrix representing the diffusion operator V.
- `size::NTuple{N, Int}`: Tuple representing the size of the diffusion operators.

"""
struct DiffusionOps{N} <: AbstractOperators where N
    G::SparseMatrixCSC{Float64, Int}
    H::SparseMatrixCSC{Float64, Int}
    Wꜝ::SparseMatrixCSC{Float64, Int}
    V::SparseMatrixCSC{Float64, Int}
    size::NTuple{N, Int}
end


"""
    struct ConvectionOps{N} <: AbstractOperators where N

Struct representing a collection of convection operators.

# Fields
- `C`: A tuple of N sparse matrices representing the C operators.
- `K`: A tuple of N sparse matrices representing the K operators.
- `size`: A tuple of N integers representing the size of each operator.

"""
struct ConvectionOps{N} <: AbstractOperators where N
    C :: NTuple{N, SparseMatrixCSC{Float64, Int}}
    K :: NTuple{N, SparseMatrixCSC{Float64, Int}}
    G::SparseMatrixCSC{Float64, Int}
    H::SparseMatrixCSC{Float64, Int}
    Wꜝ::SparseMatrixCSC{Float64, Int}
    V::SparseMatrixCSC{Float64, Int}
    size :: NTuple{N, Int}
end

"""
    build_differential_operator(op_fn, mesh, dim)

Build a dimension-specific differential operator for an N-dimensional mesh.

# Arguments
- `op_fn`: Function to generate the base differential operator (e.g., ẟ_m, δ_p, Σ_m)
- `mesh`: The mesh containing dimensional information
- `dim`: The dimension index to apply the operator to (1 for x, 2 for y, etc.)

# Returns
- A sparse matrix representing the operator in the full N-dimensional space
"""
function build_differential_operator(op_fn, mesh, dim)
    N = length(mesh.nodes)
    
    # Get number of nodes in each dimension
    node_counts = ntuple(i -> length(mesh.nodes[i]), N)
    
    # Special case for 1D
    if N == 1
        return op_fn(node_counts[1])
    end
    
    # For each dimension, use either the operator or identity
    operators = [i == dim ? op_fn(node_counts[i]) : I(node_counts[i]) for i in 1:N]
    
    # Build the kronecker product in reverse order (N to 1)
    result = operators[N]
    for i in (N-1):-1:1
        result = kron(result, operators[i])
    end
    
    return result
end

"""
    compute_base_operators(Capacity)

Compute the common operators used by both diffusion and convection.
Extracts repeated code to avoid redundancy.

# Arguments
- `Capacity`: The capacity object containing mesh and capacity matrices

# Returns
- Tuple of (G, H, Wꜝ, dimensions, D_m, D_p, S_m, S_p) operators
"""
function compute_base_operators(Capacity)
    mesh = Capacity.mesh
    N = length(mesh.nodes)
    
    # Build base differential operators
    D_m = [build_differential_operator(ẟ_m, mesh, d) for d in 1:N]
    D_p = [build_differential_operator(δ_p, mesh, d) for d in 1:N]
    S_m = [build_differential_operator(Σ_m, mesh, d) for d in 1:N]
    S_p = [build_differential_operator(Σ_p, mesh, d) for d in 1:N]
    
    # Compute G and H matrices
    G_parts = [D_m[d] * Capacity.B[d] for d in 1:N]
    G = vcat(G_parts...)
    
    H_parts = [Capacity.A[d]*D_m[d] - D_m[d]*Capacity.B[d] for d in 1:N]
    H = vcat(H_parts...)
    
    # Compute Wꜝ matrix with pre-allocation for better performance
    W_blocks = [Capacity.W[d] for d in 1:N]
    diagW = diag(blockdiag(W_blocks...))
    
    Wꜝ_data = Vector{Float64}(undef, length(diagW))
    for i in 1:length(diagW)
        Wꜝ_data[i] = diagW[i] != 0 ? 1.0 / diagW[i] : 1.0
    end
    Wꜝ = spdiagm(0 => Wꜝ_data)
    
    # Return node counts instead of cell counts (mesh.dims)
    node_counts = ntuple(i -> length(mesh.nodes[i]), N)
    
    return G, H, Wꜝ, node_counts, D_m, D_p, S_m, S_p
end

"""
    DiffusionOps(Capacity::AbstractCapacity)

Compute diffusion operators from a given capacity.
Dimension-agnostic implementation that works for any N.

# Arguments
- `Capacity`: Capacity of the system.

# Returns
- `DiffusionOps`: Diffusion operators for the system.
"""
function DiffusionOps(Capacity::AbstractCapacity)
    # Use the shared base operator computation
    G, H, Wꜝ, dims, _, _, _, _ = compute_base_operators(Capacity)
    N = length(dims)
    
    return DiffusionOps{N}(G, H, Wꜝ, Capacity.V, dims)
end

"""
    ConvectionOps(Capacity::AbstractCapacity, uₒ, uᵧ)

Construct convection operators for a given system.
Dimension-agnostic implementation that works for any N.

# Arguments
- `Capacity`: Capacity of the system.
- `uₒ`: Bulk velocity
- `uᵧ`: Interface velocity

# Returns
- `ConvectionOps`: Convection operators for the system.
"""
function ConvectionOps(Capacity::AbstractCapacity, uₒ, uᵧ)
    # Use the shared base operator computation
    G, H, Wꜝ, dims, D_m, D_p, S_m, S_p = compute_base_operators(Capacity)
    N = length(dims)
    
    # Compute convection matrices for each dimension
    C = ntuple(d -> 
        D_p[d] * spdiagm(0 => (S_m[d] * Capacity.A[d] * uₒ[d])) * S_m[d], 
    N)
    
    # Compute interface velocity matrices
    K = ntuple(d -> 
        spdiagm(0 => S_p[d] * H' * uᵧ), 
    N)
    
    return ConvectionOps{N}(C, K, G, H, Wꜝ, Capacity.V, dims)
end