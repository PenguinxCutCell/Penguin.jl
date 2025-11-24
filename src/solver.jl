@enum TimeType begin
    Steady  # ∂ₜT = 0
    Unsteady # ∂ₜT ≠ 0
end

@enum PhaseType begin
    Monophasic  # Single phase
    Diphasic    # Two phases
end

@enum EquationType begin
    Diffusion           # ∂ₜT = ∇·(∇T) + S
    Advection           # ∂ₜT = -∇·(uT) + S
    DiffusionAdvection  # ∂ₜT = ∇·(D∇T) - ∇·(uT) + S
    Stokes              # Momentum/continuity saddle-point systems
end

"""
    mutable struct Solver{TT<:TimeType, PT<:PhaseType, ET<:EquationType}

The `Solver` struct represents a solver for a specific type of problem.

# Fields
- `time_type::TT`: The type of time used in the solver : `Steady` or `Unsteady`.
- `phase_type::PT`: The type of phase used in the solver : `Monophasic` or `Diphasic`.
- `equation_type::ET`: The type of equation used in the solver : `Diffusion`, `Advection`, `DiffusionAdvection`, or `Stokes`.
- `A::Union{SparseMatrixCSC{Float64, Int}, Nothing}`: The coefficient matrix A of the equation system, if applicable.
- `b::Union{Vector{Float64}, Nothing}`: The right-hand side vector b of the equation system, if applicable.
- `x::Union{Vector{Float64}, Nothing}`: The solution vector x of the equation system, if applicable.
- `states::Vector{Any}`: The states of the system at different times, if applicable.

"""
mutable struct Solver{TT<:TimeType, PT<:PhaseType, ET<:EquationType}
    time_type::TT
    phase_type::PT
    equation_type::ET
    A::Union{SparseMatrixCSC{Float64, Int}, Nothing}
    b::Union{Vector{Float64}, Nothing}
    x::Union{Vector{Float64}, Nothing}
    ch::Vector{Any}  # Convergence history, if applicable
    states::Vector{Any}
end

"""
    remove_zero_rows_cols!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64})

Remove zero rows and columns from the coefficient matrix `A` and the right-hand side vector `b`.

# Arguments
- `A::SparseMatrixCSC{Float64, Int}`: The coefficient matrix A of the equation system.
- `b::Vector{Float64}`: The right-hand side vector b of the equation system.

# Returns
- `A::SparseMatrixCSC{Float64, Int}`: The reduced coefficient matrix A.
- `b::Vector{Float64}`: The reduced right-hand side vector b.
- `rows_idx::Vector{Int}`: The indices of the non-zero rows.
- `cols_idx::Vector{Int}`: The indices of the non-zero columns.
"""
function remove_zero_rows_cols!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64})
    # Compute sums of absolute values along rows and columns
    row_sums = vec(sum(abs.(A), dims=2))
    col_sums = vec(sum(abs.(A), dims=1))

    # Find indices of non-zero rows and columns
    rows_idx = findall(row_sums .!= 0.0)
    cols_idx = findall(col_sums .!= 0.0)
    
    # For square matrices with periodic BCs, we need to ensure we keep the same indices
    # for both rows and columns to maintain the structure of the constraints
    # Before directly : A = A[rows_idx, cols_idx], b = b[rows_idx]
    common_idx = intersect(rows_idx, cols_idx)
    
    # Create new matrix and RHS vector
    A = A[common_idx, common_idx]
    b = b[common_idx]

    return A, b, common_idx, common_idx
end

"""
    remove_zero_rows_cols_separate!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}; atol::Float64=0.0)

Remove zero (or near-zero) rows and columns independently and return separate mappings.

# Arguments
- `A`: Sparse matrix
- `b`: RHS vector (trimmed using row indices)
- `atol`: Absolute tolerance for zero detection (default 0.0)

# Returns
- `Atrim`: Trimmed matrix `A[rows_idx, cols_idx]`
- `btrim`: Trimmed RHS `b[rows_idx]`
- `rows_idx`: Kept row indices (original space)
- `cols_idx`: Kept column indices (original space)
"""
function remove_zero_rows_cols_separate!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}; atol::Float64=0.0)
    row_sums = vec(sum(abs.(A), dims=2))
    col_sums = vec(sum(abs.(A), dims=1))

    rows_idx = findall(row_sums .> atol)
    cols_idx = findall(col_sums .> atol)

    Atrim = A[rows_idx, cols_idx]
    btrim = b[rows_idx]
    return Atrim, btrim, rows_idx, cols_idx
end

"""
    solve_with_linearsolve!(s::Solver, A, b, algorithm; kwargs...)

Solve a linear system using LinearSolve.jl's framework.

# Arguments
- `s::Solver`: The Solver object to store results
- `A`: The coefficient matrix
- `b`: The right-hand side vector
- `algorithm`: The LinearSolve.jl algorithm to use
- `kwargs...`: Additional keyword arguments to pass to LinearSolve.solve

# Returns
- Solution vector
"""
function solve_with_linearsolve!(s::Solver, A, b, algorithm; kwargs...)
    # Create the LinearProblem
    prob = LinearSolve.LinearProblem(A, b)
    
    # Solve with the specified algorithm
    kwargs_nt = (; kwargs...)
    log = get(kwargs_nt, :log, false)
    
    if log
        # Solve with logging enabled
        sol = LinearSolve.solve(prob, algorithm; kwargs...)
        
        # Extract convergence history if available
        if hasfield(typeof(sol), :stats)
            push!(s.ch, sol.stats)
        end
        return sol.u
    else
        # Solve without logging
        sol = LinearSolve.solve(prob, algorithm; kwargs...)
        return sol.u
    end
end

"""
    solve_system!(s::Solver; method=gmres, algorithm=nothing, kwargs...)

Solve the system of equations stored in the `Solver` struct `s`.

# Arguments
- `s::Solver`: The `Solver` struct containing the system of equations to solve.
- `method::Function=gmres`: The method to use for iterative solving (from IterativeSolvers).
- `algorithm=nothing`: The algorithm to use from LinearSolve.jl (if provided, takes precedence over method).
- `kwargs...`: Additional keyword arguments to pass to the solver.
"""
function solve_system!(s::Solver; method::Function=gmres, algorithm=nothing, kwargs...)
    # Compute the problem size
    n = size(s.A, 1)

    # Always remove zero rows and columns to improve conditioning and efficiency
    A_reduced, b_reduced, rows_idx, cols_idx = remove_zero_rows_cols!(s.A, s.b)
    
    # Choose the solution method based on inputs
    if algorithm !== nothing
        # Use LinearSolve.jl if an algorithm is provided
        x_reduced = solve_with_linearsolve!(s, A_reduced, b_reduced, algorithm; kwargs...)
    elseif method === Base.:\
        # Use direct solver
        x_reduced = A_reduced \ b_reduced
    else
        # Use iterative solver from IterativeSolvers
        kwargs_nt = (; kwargs...)
        log = get(kwargs_nt, :log, false)
        if log
            # If logging is enabled, we store the convergence history
            x_reduced, ch = method(A_reduced, b_reduced; kwargs...)
            push!(s.ch, ch)
        else
            x_reduced = method(A_reduced, b_reduced; kwargs...)
        end
    end
    
    # Reconstruct the full solution vector regardless of solver type
    s.x = zeros(n)
    s.x[cols_idx] = x_reduced
end

"""
    build_I_bc(operator::AbstractOperators, bc::AbstractBoundary)

Build the boundary conditions matrices Iₐ and Iᵦ for the given operator and boundary conditions.

# Arguments
- `operator::AbstractOperators`: The operators of the problem.
- `bc::AbstractBoundary`: The boundary conditions of the problem.

# Returns
- `Iₐ::SparseMatrixCSC{Float64, Int}`: The matrix Iₐ for the boundary conditions.
- `Iᵦ::SparseMatrixCSC{Float64, Int}`: The matrix Iᵦ for the boundary conditions.
"""
function build_I_bc(operator::AbstractOperators,bc::AbstractBoundary)
    n = prod(operator.size)
    Iᵦ = spzeros(n, n)
    Iₐ = spzeros(n, n)

    if bc isa Dirichlet
        Iₐ = I(n)
    elseif bc isa GibbsThomson
        Iₐ = I(n)
    elseif bc isa Neumann
        Iᵦ = I(n)
    elseif bc isa Robin
        if bc.α isa Function
            Iₐ = bc.α(I(n))
        else
            Iₐ = bc.α * I(n)
            Iᵦ = bc.β * I(n)
        end 
    end
    return Iₐ, Iᵦ
end

"""
    get_all_coordinates(C_coords)

Efficiently compute all coordinates at once using vectorized operations.
"""
function get_all_coordinates(C_coords)
    n = length(C_coords)
    
    if length(C_coords[1]) == 1
        # 1D case
        return [(C_coords[i][1], 0.0, 0.0) for i in 1:n]
    elseif length(C_coords[1]) == 2
        # 2D case
        return [(C_coords[i][1], C_coords[i][2], 0.0) for i in 1:n]
    elseif length(C_coords[1]) == 3
        # 3D case
        return [(C_coords[i][1], C_coords[i][2], C_coords[i][3]) for i in 1:n]
    elseif length(C_coords[1]) == 4
        # 4D case (e.g., spacetime)
        return [(C_coords[i][1], C_coords[i][2], C_coords[i][3], C_coords[i][4]) for i in 1:n]
    else
        error("Unsupported coordinate dimension: $(length(C_coords[1]))")
    end
end

"""
    build_I_D(operator::AbstractOperators, D::Union{Float64,Function}, capacite::Capacity)

Optimized diffusion matrix construction without loops.
"""
function build_I_D(operator::AbstractOperators, D::Union{Float64,Function}, capacite::Capacity)
    n = prod(operator.size)
    
    if D isa Function
        # Vectorized coordinate computation
        coords = get_all_coordinates(capacite.C_ω)
        diagonal_values = [D(coord...) for coord in coords]
        return spdiagm(0 => diagonal_values)
    else
        return D * I(n)
    end
end

"""
    build_source(operator::AbstractOperators, f::Function, capacite::Capacity)

Optimized source term construction without loops.
"""
function build_source(operator::AbstractOperators, f::Function, capacite::Capacity)
    coords = get_all_coordinates(capacite.C_ω)
    return [f(coord...) for coord in coords]
end

"""
    build_source(operator::AbstractOperators, f::Function, capacite::Capacity, t::Float64)

Optimized time-dependent source term construction.
"""
function build_source(operator::AbstractOperators, f::Function, t::Float64, capacite::Capacity)
    coords = get_all_coordinates(capacite.C_ω)
    return [f(coord..., t) for coord in coords]
end

"""
    build_g_g(operator::AbstractOperators, bc::Union{AbstractBoundary, AbstractInterfaceBC}, capacite::Capacity)

Optimized boundary condition vector construction.
"""
function build_g_g(operator::AbstractOperators, bc::Union{AbstractBoundary, AbstractInterfaceBC}, capacite::Capacity)
    n = prod(operator.size)
    
    if bc.value isa Function
        coords = get_all_coordinates(capacite.C_γ)
        return [bc.value(coord...) for coord in coords]
    else
        return fill(bc.value, n)
    end
end

"""
    build_g_g(operator::AbstractOperators, bc::Union{AbstractBoundary, AbstractInterfaceBC}, capacite::Capacity, t::Float64)

Optimized time-dependent boundary condition vector construction.
"""
function build_g_g(operator::AbstractOperators, bc::Union{AbstractBoundary, AbstractInterfaceBC}, capacite::Capacity, t::Float64)
    n = prod(operator.size)
    
    if bc.value isa Function
        coords = get_all_coordinates(capacite.C_γ)
        # Try time-dependent first, fall back to time-independent
        try
            return [bc.value(coord..., t) for coord in coords]
        catch MethodError
            return [bc.value(coord...) for coord in coords]
        end
    else
        return fill(bc.value, n)
    end
end

function build_g_g(operator::AbstractOperators, bc::GibbsThomson, capacite::Capacity)
    n = prod(operator.size)
    gᵧ = bc.Tm*ones(n) .- bc.ϵᵥ .* bc.vᵞ
    return gᵧ
end

# Helper function to get the opposite boundary
function get_opposite_boundary(key::Symbol)
    if key == :left
        return :right
    elseif key == :right
        return :left
    elseif key == :bottom
        return :top
    elseif key == :top
        return :bottom
    elseif key == :backward
        return :forward
    elseif key == :forward
        return :backward
    else
        error("Unknown boundary key: $key")
    end
end

# Helper function to find the corresponding cell on the opposite boundary
function find_corresponding_cell(cell::CartesianIndex{N}, key::Symbol, opposite_key::Symbol, mesh::AbstractMesh) where {N}
    if key == :left || key == :right
        new_cell = CartesianIndex(key == :left ? length(mesh.centers[1]) : 1, cell[2])
    elseif key == :bottom || key == :top
        new_cell = CartesianIndex(cell[1], key == :bottom ? length(mesh.centers[2]) : 1)
    elseif key == :backward || key == :forward
        new_cell = CartesianIndex(cell[1], cell[2], key == :backward ? length(mesh.centers[3]) : 1)
    end
    return new_cell
end

function cell_to_index(mesh::Union{Mesh{1}, SpaceTimeMesh{1}}, cell::CartesianIndex)
    return LinearIndices((length(mesh.centers[1])+1,))[cell]
end

function cell_to_index(mesh::Union{Mesh{2}, SpaceTimeMesh{2}}, cell::CartesianIndex)
    return LinearIndices((length(mesh.centers[1])+1, length(mesh.centers[2])+1))[cell]
end

function cell_to_index(mesh::Union{Mesh{3}, SpaceTimeMesh{3}}, cell::CartesianIndex)
    return LinearIndices((length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1))[cell]
end

"""
    classify_boundary_cell_fast(ci::CartesianIndex, mesh::AbstractMesh)

Efficiently classify which boundary a cell belongs to in O(1) time.
"""
function classify_boundary_cell_fast(ci::CartesianIndex, mesh::AbstractMesh)
    # Check boundaries based on position indices directly
    ndims = length(mesh.centers)
    
    if ndims >= 2
        # Check left/right boundaries
        if ci[2] == 1
            return :left
        elseif ci[2] == length(mesh.centers[2])
            return :right
        end
    end
    
    # Check bottom/top boundaries
    if ci[1] == 1
        return :bottom
    elseif ci[1] == length(mesh.centers[1])
        return :top
    end
    
    if ndims >= 3
        # Check backward/forward boundaries
        if ci[3] == 1
            return :backward
        elseif ci[3] == length(mesh.centers[3])
            return :forward
        end
    end
    
    error("Cell $ci is not on any boundary")
end

"""
    BC_border_mono_optimized!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, 
                               bc_b::BorderConditions, mesh::AbstractMesh)

Optimized boundary condition application with single pass through boundary cells.
"""
function BC_border_mono!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, 
                                   bc_b::BorderConditions, mesh::AbstractMesh; t=nothing)
    # Single pass through boundary cells - O(n) complexity
    for (ci, pos) in mesh.tag.border_cells
        # Classify boundary in O(1) time
        boundary_key = classify_boundary_cell_fast(ci, mesh)
        
        # Get boundary condition (O(1) dictionary lookup)
        condition = get(bc_b.borders, boundary_key, nothing)
        isnothing(condition) && continue
        
        # Convert to linear index
        li = cell_to_index(mesh, ci)
        
        # Apply boundary condition
        apply_boundary_condition_fast!(A, b, li, pos, condition, boundary_key, bc_b, mesh, t)
    end
end

"""
    apply_boundary_condition_fast!(A, b, li, pos, condition, boundary_key, bc_b, mesh)

Fast application of a single boundary condition.
"""
eval_bc_value(value, pos, t) = value isa Function ? (
    t === nothing ? value(pos...) :
        try
            value(pos..., t)
        catch e
            e isa MethodError ? value(pos...) : rethrow()
        end
) : value

function apply_boundary_condition_fast!(A, b, li, pos, condition, boundary_key, bc_b, mesh, t=nothing; offset::Int=0)
    row_idx = li + offset
    if condition isa Dirichlet
        # Dirichlet: A[li, li] = 1, b[li] = value
        A[row_idx, :] .= 0.0
        A[row_idx, row_idx] = 1.0
        b[row_idx] = eval_bc_value(condition.value, pos, t)
        
    elseif condition isa Periodic
        # Find corresponding cell efficiently
        opposite_key = get_opposite_boundary(boundary_key)
        if haskey(bc_b.borders, opposite_key)
            corresponding_idx = find_corresponding_cell_optimized(li, boundary_key, mesh)
            corresponding_idx += offset
            A[row_idx, :] .= 0.0
            # Apply periodic constraint: x_li - x_corresponding = 0
            A[row_idx, row_idx] += 1.0
            A[row_idx, corresponding_idx] -= 1.0
            b[row_idx] = 0.0
        end
        
    elseif condition isa Neumann
        # Simple 1D Neumann: (u_boundary - u_adjacent) / Δx = g
        # This creates a first-order one-sided derivative constraint.
        if length(mesh.centers) == 1
            # Compute grid spacing from node coordinates
            Δx = minimum(diff(mesh.nodes[1]))
            # Determine adjacent interior index
            dims = (length(mesh.centers[1])+1,)
            if boundary_key == :bottom
                li_adj = min(li+1, dims[1])
            elseif boundary_key == :top
                li_adj = max(li-1, 1)
            else
                # Fallback for unexpected labels in 1D
                li_adj = boundary_key == :left ? min(li+1, dims[1]) : max(li-1, 1)
            end

            gval = eval_bc_value(condition.value, pos, t)

            A[row_idx, :] .= 0.0
            A[row_idx, row_idx] =  1.0/Δx
            A[row_idx, li_adj + offset] += -1.0/Δx
            b[row_idx] = gval
        else
            @warn "Neumann BC currently implemented for 1D only in BC_border_mono!" maxlog=1
        end

    end
end

"""
    find_corresponding_cell_optimized(li::Int, boundary_key::Symbol, mesh::AbstractMesh)

Optimized calculation of corresponding periodic boundary cell.
"""
function find_corresponding_cell_optimized(li::Int, boundary_key::Symbol, mesh::AbstractMesh)
    # Convert linear index to CartesianIndex for more reliable mapping
    dims = [length(center)+1 for center in mesh.centers]
    ci = CartesianIndices(Tuple(dims))[li]
    
    # Create the new CartesianIndex based on the boundary
    new_ci = if boundary_key == :left
        CartesianIndex(ci[1], dims[2])  # Map to rightmost column
    elseif boundary_key == :right
        CartesianIndex(ci[1], 1)  # Map to leftmost column
    elseif boundary_key == :bottom
        CartesianIndex(dims[1], ci[2])  # Map to top row
    elseif boundary_key == :top
        CartesianIndex(1, ci[2])  # Map to bottom row
    elseif boundary_key == :backward && length(mesh.centers) >= 3
        CartesianIndex(ci[1], ci[2], dims[3])  # Map to forward face
    elseif boundary_key == :forward && length(mesh.centers) >= 3
        CartesianIndex(ci[1], ci[2], 1)  # Map to backward face
    else
        error("Unknown boundary key: $boundary_key")
    end
    
    # Convert back to linear index
    return LinearIndices(Tuple(dims))[new_ci]
end

"""
    BC_border_diph!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, 
                    bc_b::BorderConditions, args...; t=nothing)

Apply diphasic boundary conditions either using only the mesh (legacy behavior)
or, when capacities for each phase are provided, applying them only where the
phase exists.
"""
function BC_border_diph!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, 
                         bc_b::BorderConditions, mesh::AbstractMesh; t=nothing)
    _BC_border_diph!(A, b, bc_b, mesh, nothing; t=t)
end

function BC_border_diph!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, 
                         bc_b::BorderConditions, cap1::Capacity, cap2::Capacity; t=nothing)
    mesh = cap1.mesh
    mesh === cap2.mesh || error("Phase capacities must share the same mesh.")
    _BC_border_diph!(A, b, bc_b, mesh, (cap1, cap2); t=t)
end

function _BC_border_diph!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, 
                          bc_b::BorderConditions, mesh::AbstractMesh, capacities; t=nothing)
    nrows = size(A, 1)
    phase_size = nrows ÷ 4
    if phase_size * 4 != nrows
        error("BC_border_diph! expects a 4-block system, got size $(nrows)")
    end

    phase_offsets = (0, 2 * phase_size)

    for (ci, pos) in mesh.tag.border_cells
        boundary_key = classify_boundary_cell_fast(ci, mesh)
        condition = get(bc_b.borders, boundary_key, nothing)
        isnothing(condition) && continue

        li = cell_to_index(mesh, ci)
        if capacities === nothing
            for offset in phase_offsets
                apply_boundary_condition_fast!(A, b, li, pos, condition, boundary_key, bc_b, mesh, t; offset=offset)
            end
        else
            for (offset, cap) in zip(phase_offsets, capacities)
                cell_type = cap.cell_types[li]
                cell_type == 0 && continue
                apply_boundary_condition_fast!(A, b, li, pos, condition, boundary_key, bc_b, mesh, t; offset=offset)
            end
        end
    end
end

function cfl_restriction(mesh::Penguin.Mesh, cfl::Float64, w::Float64)
    # Compute the spatial mesh spacing
    dx = (mesh.nodes[1][end] - mesh.nodes[1][1]) / length(mesh.centers[1])
    δt = cfl * dx / w
    return δt
end


"""
    adapt_timestep(velocity_field, mesh, cfl_target, Δt_current, Δt_min, Δt_max; 
                  growth_factor=1.1, shrink_factor=0.8, safety_factor=0.9)

Adapte le pas de temps en fonction du critère CFL basé sur la vitesse de l'interface.

Paramètres:
- `velocity_field`: Vitesses à l'interface [m/s]
- `mesh`: Maillage de calcul
- `cfl_target`: Nombre CFL cible (typiquement entre 0.1 et 1.0)
- `Δt_current`: Pas de temps actuel [s]
- `Δt_min`: Pas de temps minimum autorisé [s]
- `Δt_max`: Pas de temps maximum autorisé [s]
- `growth_factor`: Facteur maximum d'augmentation du pas de temps (par défaut 1.1)
- `shrink_factor`: Facteur minimum de réduction du pas de temps (par défaut 0.8)
- `safety_factor`: Facteur de sécurité pour le CFL (par défaut 0.9)

Retourne:
- `Δt_new`: Nouveau pas de temps [s]
- `cfl_actual`: Nombre CFL qui sera obtenu avec le nouveau pas de temps
"""
function adapt_timestep(velocity_field, mesh, cfl_target, Δt_current, Δt_min, Δt_max;
                       growth_factor=1.1, shrink_factor=0.8, safety_factor=0.9)
    # 1. Calcul de la vitesse maximale de l'interface
    v_max = maximum(abs.(velocity_field))
    
    # Éviter la division par zéro si l'interface est statique
    if v_max < 1e-10
        # Si la vitesse est très faible, on peut augmenter le pas de temps
        Δt_new = min(Δt_current * growth_factor, Δt_max)
        return Δt_new, 0.0
    end
    
    # 2. Calcul de la taille de maille minimale dans chaque direction
    if length(mesh.nodes) == 1
        Δx = minimum(diff(mesh.nodes[1]))
        Δh_min = Δx
    elseif length(mesh.nodes) == 2
        Δx = minimum(diff(mesh.nodes[1]))
        Δy = minimum(diff(mesh.nodes[2]))
        Δh_min = min(Δx, Δy)
    elseif length(mesh.nodes) == 3
        Δx = minimum(diff(mesh.nodes[1]))
        Δy = minimum(diff(mesh.nodes[2]))
        Δz = minimum(diff(mesh.nodes[3]))
        Δh_min = min(Δx, Δy, Δz)
    else
        error("Unsupported mesh dimension")
    end
    
    # 3. Calcul du CFL actuel
    cfl_current = v_max * Δt_current / Δh_min
    
    # 4. Calcul du pas de temps optimal pour le CFL cible avec un facteur de sécurité
    Δt_optimal = safety_factor * cfl_target * Δh_min / v_max
    
    # 5. Décision d'augmenter ou diminuer le pas de temps
    if Δt_optimal > Δt_current
        # La vitesse a diminué, on peut augmenter le pas de temps (limité par growth_factor)
        Δt_new = min(Δt_optimal, Δt_current * shrink_factor)
    else
        # La vitesse a augmenté, on doit diminuer le pas de temps (mais pas trop brusquement)
        Δt_new = max(Δt_optimal, Δt_current * growth_factor)
    end
    
    # 6. Application des contraintes min/max
    Δt_new = clamp(Δt_new, Δt_min, Δt_max)
    
    # 7. Calcul du CFL effectif avec le nouveau pas de temps
    cfl_actual = v_max * Δt_new / Δh_min
    
    return Δt_new, cfl_actual
end
