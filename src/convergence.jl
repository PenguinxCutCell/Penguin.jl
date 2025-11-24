
# Check Convergence
# Weighted Lp or L∞ norm helper
function lp_norm(errors, indices, pval, capacity)
    if pval == Inf
        return maximum(abs.(errors[indices]), init=0.0)
    else
        part_sum = 0.0
        for i in indices
            Vi = capacity.V[i,i]
            part_sum += (abs(errors[i])^pval) * Vi
        end
        return (part_sum / sum(capacity.V))^(1/pval)
    end
end

# Relative Lp norm helper
function relative_lp_norm(errors, indices, pval, capacity, u_ana)
    if pval == Inf
        return maximum(abs.(errors[indices]/u_ana[indices]))
    else
        part_sum = 0.0
        for i in indices
            Vi = capacity.V[i,i]
            part_sum += (abs(errors[i]/u_ana[i])^pval) * Vi
        end
        return (part_sum / sum(capacity.V))^(1/pval)
    end
end

"""
    check_convergence(u_analytical::Function, solver, capacity::Capacity{D}, p::Real=2, relative::Bool=false) where D

Compute convergence metrics between numerical and analytical solutions.

# Arguments
- `u_analytical::Function`: Analytical solution function
- `solver`: Solver containing numerical solution
- `capacity::Capacity{D}`: Capacity object with dimension D (1, 2, or 3)
- `p::Real=2`: Order of the norm (2=L2 norm)
- `relative::Bool=false`: Whether to compute relative error

# Returns
- Tuple with analytical solution, numerical solution, and error norms
"""
function check_convergence(u_analytical::Function, solver, capacity::Capacity{D}, p::Real=2, relative::Bool=false) where D
    # 1) Compute pointwise error
    cell_centroids = capacity.C_ω
    
    # Apply analytical function based on dimension
    u_ana = if D == 1
        map(c -> u_analytical(c[1]), cell_centroids)
    elseif D == 2
        map(c -> u_analytical(c[1], c[2]), cell_centroids)
    elseif D == 3
        map(c -> u_analytical(c[1], c[2], c[3]), cell_centroids)
    elseif D == 4
        map(c -> u_analytical(c[1], c[2], c[3], c[4]), cell_centroids)
    else
        error("Unsupported dimension: $D")
    end
    
    u_num = solver.x[1:end÷2]
    err = u_ana .- u_num

    # 2) Retrieve cell types and separate full, cut, empty
    cell_types = capacity.cell_types
    idx_all = findall((cell_types .== 1) .| (cell_types .== -1))
    idx_full = findall(cell_types .== 1)
    idx_cut = findall(cell_types .== -1)
    idx_empty = findall(cell_types .== 0)

    # 3) Compute norms (relative or not)
    if relative
        global_err = relative_lp_norm(err, idx_all, p, capacity, u_ana)
        full_err = relative_lp_norm(err, idx_full, p, capacity, u_ana)
        cut_err = relative_lp_norm(err, idx_cut, p, capacity, u_ana)
        empty_err = relative_lp_norm(err, idx_empty, p, capacity, u_ana)
    else
        global_err = lp_norm(err, idx_all, p, capacity)
        full_err = lp_norm(err, idx_full, p, capacity)
        cut_err = lp_norm(err, idx_cut, p, capacity)
        empty_err = lp_norm(err, idx_empty, p, capacity)
    end

    # 4) Print results
    println("All cells L$p norm        = $global_err")
    println("Full cells L$p norm   = $full_err")
    println("Cut cells L$p norm    = $cut_err")
    println("Empty cells L$p norm  = $empty_err")

    return (u_ana, u_num, global_err, full_err, cut_err, empty_err)
end

"""
    check_convergence_diph(u1_analytical::Function, u2_analytical::Function, solver, 
                          capacity1::Capacity{D}, capacity2::Capacity{D}, 
                          p::Real=2, relative::Bool=false) where D

Compute convergence metrics between numerical and analytical solutions for a two-phase system.

# Arguments
- `u1_analytical::Function`: Analytical solution function for phase 1
- `u2_analytical::Function`: Analytical solution function for phase 2
- `solver`: Solver containing numerical solution
- `capacity1::Capacity{D}`: Capacity object for phase 1 with dimension D
- `capacity2::Capacity{D}`: Capacity object for phase 2 with dimension D
- `p::Real=2`: Order of the norm (2=L2 norm)
- `relative::Bool=false`: Whether to compute relative error

# Returns
- Tuple with analytical solutions, numerical solutions, and error norms
"""
function check_convergence_diph(u1_analytical::Function, u2_analytical::Function, solver, 
                               capacity1::Capacity{D}, capacity2::Capacity{D}, 
                               p::Real=2, relative::Bool=false) where D
    # Get mesh size
    n = size(capacity1.V, 1) - 1
    
    # Extract cell centroids for both phases
    cell_centroids1 = capacity1.C_ω
    cell_centroids2 = capacity2.C_ω
    
    # Apply analytical functions based on dimension
    u1_ana = if D == 1
        map(c -> u1_analytical(c[1]), cell_centroids1)
    elseif D == 2
        map(c -> u1_analytical(c[1], c[2]), cell_centroids1)
    elseif D == 3
        map(c -> u1_analytical(c[1], c[2], c[3]), cell_centroids1)
    elseif D == 4
        map(c -> u1_analytical(c[1], c[2], c[3], c[4]), cell_centroids1)
    else
        error("Unsupported dimension: $D")
    end
    
    u2_ana = if D == 1
        map(c -> u2_analytical(c[1]), cell_centroids2)
    elseif D == 2
        map(c -> u2_analytical(c[1], c[2]), cell_centroids2)
    elseif D == 3
        map(c -> u2_analytical(c[1], c[2], c[3]), cell_centroids2)
    elseif D == 4
        map(c -> u2_analytical(c[1], c[2], c[3], c[4]), cell_centroids2)
    else
        error("Unsupported dimension: $D")
    end
    
    # Extract numerical solutions from solver states
    u1_num = solver.states[end][1:n+1]           # Bulk Field - Phase 1
    u2_num = solver.states[end][2*(n+1)+1:3*(n+1)]  # Bulk Field - Phase 2
    
    # Compute errors
    err1 = u1_ana .- u1_num
    err2 = u2_ana .- u2_num
    
    # Get cell types for each phase
    cell_types1 = capacity1.cell_types
    cell_types2 = capacity2.cell_types
    
    # Filter indices by cell type for phase 1
    idx_all1 = findall((cell_types1 .== 1) .| (cell_types1 .== -1))
    idx_full1 = findall(cell_types1 .== 1)
    idx_cut1 = findall(cell_types1 .== -1)
    idx_empty1 = findall(cell_types1 .== 0)
    
    # Filter indices by cell type for phase 2
    idx_all2 = findall((cell_types2 .== 1) .| (cell_types2 .== -1))
    idx_full2 = findall(cell_types2 .== 1)
    idx_cut2 = findall(cell_types2 .== -1)
    idx_empty2 = findall(cell_types2 .== 0)
    
    # Compute norms for Phase 1
    println("\n=== Phase 1 Errors ===")
    if relative
        global_err1 = relative_lp_norm(err1, idx_all1, p, capacity1, u1_ana)
        full_err1 = relative_lp_norm(err1, idx_full1, p, capacity1, u1_ana)
        cut_err1 = relative_lp_norm(err1, idx_cut1, p, capacity1, u1_ana)
        empty_err1 = relative_lp_norm(err1, idx_empty1, p, capacity1, u1_ana)
    else
        global_err1 = lp_norm(err1, idx_all1, p, capacity1)
        full_err1 = lp_norm(err1, idx_full1, p, capacity1)
        cut_err1 = lp_norm(err1, idx_cut1, p, capacity1)
        empty_err1 = lp_norm(err1, idx_empty1, p, capacity1)
    end
    
    println("Phase 1 - All cells L$p norm   = $global_err1")
    println("Phase 1 - Full cells L$p norm  = $full_err1")
    println("Phase 1 - Cut cells L$p norm   = $cut_err1")
    println("Phase 1 - Empty cells L$p norm = $empty_err1")
    
    # Compute norms for Phase 2
    println("\n=== Phase 2 Errors ===")
    if relative
        global_err2 = relative_lp_norm(err2, idx_all2, p, capacity2, u2_ana)
        full_err2 = relative_lp_norm(err2, idx_full2, p, capacity2, u2_ana)
        cut_err2 = relative_lp_norm(err2, idx_cut2, p, capacity2, u2_ana)
        empty_err2 = relative_lp_norm(err2, idx_empty2, p, capacity2, u2_ana)
    else
        global_err2 = lp_norm(err2, idx_all2, p, capacity2)
        full_err2 = lp_norm(err2, idx_full2, p, capacity2)
        cut_err2 = lp_norm(err2, idx_cut2, p, capacity2)
        empty_err2 = lp_norm(err2, idx_empty2, p, capacity2)
    end
    
    println("Phase 2 - All cells L$p norm   = $global_err2")
    println("Phase 2 - Full cells L$p norm  = $full_err2")
    println("Phase 2 - Cut cells L$p norm   = $cut_err2")
    println("Phase 2 - Empty cells L$p norm = $empty_err2")
    
    # Compute combined errors (maximum of both phases)
    global_err = max(global_err1, global_err2)
    full_err = max(full_err1, full_err2)
    cut_err = max(cut_err1, cut_err2)
    empty_err = max(empty_err1, empty_err2)
    
    println("\n=== Combined Errors (maximum of both phases) ===")
    println("Combined - All cells L$p norm   = $global_err")
    println("Combined - Full cells L$p norm  = $full_err")
    println("Combined - Cut cells L$p norm   = $cut_err")
    println("Combined - Empty cells L$p norm = $empty_err")
    
    return (
        # Return analytical solutions
        (u1_ana, u2_ana), 
        # Return numerical solutions
        (u1_num, u2_num), 
        # Return global errors
        (global_err1, global_err2, global_err),
        # Return full cell errors
        (full_err1, full_err2, full_err),
        # Return cut cell errors
        (cut_err1, cut_err2, cut_err),
        # Return empty cell errors
        (empty_err1, empty_err2, empty_err)
    )
end