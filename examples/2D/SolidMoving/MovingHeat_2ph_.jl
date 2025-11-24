using Penguin
using IterativeSolvers, SpecialFunctions
using Roots
using LinearAlgebra

### 2D Test Case: Diphasic Heat Transfer with Moving Interface (Manufactured Solution)
# Define the mesh
nx, ny = 40, 40
lx, ly = 4.0, 4.0
x0, y0 = 0.0, 0.0
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define parameters for the moving circle
ω = 0.0  # Angular frequency for circular motion
circle_center(t) = [2.0 + 0.5*cos(ω*t), 2.0 + 0.5*sin(ω*t)]
circle_radius(t) = 1.0 + 0.2*sin(ω*t)

# Circle velocity and radius derivative
center_velocity(t) = [-0.5*ω*sin(ω*t), 0.5*ω*cos(ω*t)]
radius_derivative(t) = 0.2*ω*cos(ω*t)

# Level set function S(x,t) = ||x - c(t)||^2 - R(t)^2
# body > 0 inside the circle, < 0 outside
function S_func(x, y, t)
    c = circle_center(t)
    R = circle_radius(t)
    return (x - c[1])^2 + (y - c[2])^2 - R^2
end

# Level set functions: negative inside the circle (Ω⁻), positive outside (Ω⁺)
body(x, y, t) = S_func(x, y, t)
body_c(x, y, t) = -S_func(x, y, t)

# Define the Space-Time mesh
Δt = 0.005  # Smaller time step for accuracy
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)
capacity_c = Capacity(body_c, STmesh)

# Define the operators
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the exact solution: Φ(x,t) = e^(-t) * S(x,t)^2
function exact_solution(x, y, t)
    S = S_func(x, y, t)
    return exp(-t) * S^2
end

# Define Dirichlet boundary conditions based on exact solution
function bc_function(x, y, z, t)
    return exact_solution(x, y, t)
end

bc = Dirichlet(bc_function)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}( ))

# Interface conditions (both continuity and flux continuity are automatically satisfied)
ic = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Physical parameters
D_minus = 1.0  # Diffusion coefficient inside circle (Ω⁻)
D_plus = 1.0   # Diffusion coefficient outside circle (Ω⁺)
cp_minus = 1.0 # Heat capacity inside circle
cp_plus = 1.0  # Heat capacity outside circle

# Define the forcing terms
function forcing_term_minus(x, y, z, t)
    c = circle_center(t)
    R = circle_radius(t)
    c_prime = center_velocity(t)
    R_prime = radius_derivative(t)
    
    S = S_func(x, y, t)
    x_minus_c = [x - c[1], y - c[2]]
    x_minus_c_norm_squared = x_minus_c[1]^2 + x_minus_c[2]^2
    x_minus_c_dot_c_prime = x_minus_c[1]*c_prime[1] + x_minus_c[2]*c_prime[2]
    
    # Formula from the manufactured solution
    return cp_minus * exp(-t) * (-S^2 + 2*S*(-x_minus_c_dot_c_prime - R*R_prime)) - 
           D_minus * exp(-t) * (8*x_minus_c_norm_squared + 8*S)
end

function forcing_term_plus(x, y, z, t)
    c = circle_center(t)
    R = circle_radius(t)
    c_prime = center_velocity(t)
    R_prime = radius_derivative(t)
    
    S = S_func(x, y, t)
    x_minus_c = [x - c[1], y - c[2]]
    x_minus_c_norm_squared = x_minus_c[1]^2 + x_minus_c[2]^2
    x_minus_c_dot_c_prime = x_minus_c[1]*c_prime[1] + x_minus_c[2]*c_prime[2]
    
    # Formula from the manufactured solution
    return cp_plus * exp(-t) * (-S^2 + 2*S*(-x_minus_c_dot_c_prime - R*R_prime)) - 
           D_plus * exp(-t) * (8*x_minus_c_norm_squared + 8*S)
end

# Define diffusion coefficients
K_minus = (x, y, z) -> D_minus
K_plus = (x, y, z) -> D_plus

# Define the phases
# Phase inside circle (Ω⁻)
Phase_minus = Phase(capacity, operator, forcing_term_minus, K_minus)
# Phase outside circle (Ω⁺)
Phase_plus = Phase(capacity_c, operator_c, forcing_term_plus, K_plus)

# Initial condition based on the exact solution at t=0
function initialize_solution(mesh)
    u0 = zeros((nx+1)*(ny+1))
    
    for j in 1:ny
        for i in 1:nx
            x = mesh.centers[1][i]
            y = mesh.centers[2][j]
            idx = i + (j-1)*(nx+1)
            u0[idx] = exact_solution(x, y, 0.0)
        end
    end
    
    return u0
end

u0ₒ1 = initialize_solution(mesh)
u0ᵧ1 = initialize_solution(mesh)
u0ₒ2 = initialize_solution(mesh)
u0ᵧ2 = initialize_solution(mesh)
u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Define the solver
solver = MovingDiffusionUnsteadyDiph(Phase_minus, Phase_plus, bc_b, ic, Δt, u0, mesh, "BE")

# Solve the problem
solve_MovingDiffusionUnsteadyDiph!(solver, Phase_minus, Phase_plus, body, body_c, Δt, Tend, bc_b, ic, mesh, "BE"; method=Base.:\)

# Compute error
function compute_error(solver, mesh, body, final_time)
    error_minus = 0.0
    error_plus = 0.0
    count_minus = 0
    count_plus = 0
    
    # Get the size of one phase solution vector
    phase_size = (nx+1)*(ny+1)
    
    # Extract solution vectors for each phase (at final time)
    u_minus = solver.x[1:phase_size]  # Inside circle (Ω⁻)
    u_plus = solver.x[2*phase_size+1:3*phase_size]  # Outside circle (Ω⁺)
    
    for j in 1:ny+1
        for i in 1:nx+1
            x = mesh.nodes[1][i]
            y = mesh.nodes[2][j]
            idx = i + (j-1)*(nx+1)
            
            # Determine which phase this point belongs to
            in_minus = body(x, y, final_time) > 0
            
            # Get exact solution
            exact = exact_solution(x, y, final_time)
            
            # Compute L2 error for the appropriate phase
            if in_minus
                error_minus += (u_minus[idx] - exact)^2
                count_minus += 1
            else
                error_plus += (u_plus[idx] - exact)^2
                count_plus += 1
            end
        end
    end
    
    error_minus = count_minus > 0 ? sqrt(error_minus / count_minus) : 0.0
    error_plus = count_plus > 0 ? sqrt(error_plus / count_plus) : 0.0
    error_total = sqrt((error_minus^2 * count_minus + error_plus^2 * count_plus) / (count_minus + count_plus))
    
    return error_minus, error_plus, error_total
end

# Calculate and print errors
err_minus, err_plus, err_total = compute_error(solver, mesh, body, Tend)
println("\nL2 error at final time:")
println("Inside circle (Ω⁻): ", err_minus)
println("Outside circle (Ω⁺): ", err_plus)
println("Total error: ", err_total)

# Animation
animate_solution(solver, mesh, body)
