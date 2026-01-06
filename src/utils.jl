
## Temperature initialization functions

# Initialize temperature uniformly across the domain
function initialize_temperature_uniform!(T0ₒ::Vector{Float64}, T0ᵧ::Vector{Float64}, value::Float64)
    fill!(T0ₒ, value)
    fill!(T0ᵧ, value)
end

# Initialize temperature within a square region centered at 'center' with half-width 'half_width'
function initialize_temperature_square!(T0ₒ::Vector{Float64}, T0ᵧ::Vector{Float64}, x_coords::Vector{Float64}, y_coords::Vector{Float64}, center::Tuple{Float64, Float64}, half_width::Int, value::Float64, nx::Int, ny::Int)
    center_i = findfirst(x -> x >= center[1], x_coords)
    center_j = findfirst(y -> y >= center[2], y_coords)

    i_min = max(center_i - half_width, 1)
    i_max = min(center_i + half_width, nx + 1)
    j_min = max(center_j - half_width, 1)
    j_max = min(center_j + half_width, ny + 1)
    for j in j_min:j_max
        for i in i_min:i_max
            idx = i + (j - 1) * (nx + 1)
            T0ₒ[idx] = value
            T0ᵧ[idx] = value
        end
    end
end

# Initialize temperature within a circular region centered at 'center' with radius 'radius'
function initialize_temperature_circle!(T0ₒ::Vector{Float64}, T0ᵧ::Vector{Float64}, x_coords::Vector{Float64}, y_coords::Vector{Float64}, center::Tuple{Float64, Float64}, radius::Float64, value::Float64, nx::Int, ny::Int)
    for j in 1:(ny )
        for i in 1:(nx)
            idx = i + (j - 1) * (nx + 1)
            x_i = x_coords[i]
            y_j = y_coords[j]
            distance = sqrt((x_i - center[1])^2 + (y_j - center[2])^2)
            if distance <= radius
                T0ₒ[idx] = value
                T0ᵧ[idx] = value
            end
        end
    end
end

# Initialize temperature using a custom function 'func(x, y)'
function initialize_temperature_function!(T0ₒ::Vector{Float64}, T0ᵧ::Vector{Float64}, x_coords::Vector{Float64}, y_coords::Vector{Float64}, func::Function, nx::Int, ny::Int)
    for j in 1:(ny)
        for i in 1:(nx)
            idx = i + (j - 1) * (nx + 1)
            x = x_coords[i]
            y = y_coords[j]
            T_value = func(x, y)
            T0ₒ[idx] = T_value
            T0ᵧ[idx] = T_value
        end
    end
end


## Velocity field initialization functions

# Initialize the velocity field with a Rotational flow
function initialize_rotating_velocity_field(nx, ny, lx, ly, x0, y0, magnitude)
    # Number of nodes
    N = (nx + 1) * (ny + 1)

    # Initialize velocity components
    uₒx = zeros(N)
    uₒy = zeros(N)

    # Define the center of rotation
    center_x, center_y = lx / 2, ly / 2

    # Calculate rotating velocity components
    for j in 0:ny
        for i in 0:nx
            idx = i + j * (nx + 1) + 1
            x = x0 + i * (lx / nx)
            y = y0 + j * (ly / ny)
            uₒx[idx] = -(y - center_y) * magnitude
            uₒy[idx] = (x - center_x) * magnitude
        end
    end
    return uₒx, uₒy
end


# Initialize the velocity field with a Poiseuille flow
function initialize_poiseuille_velocity_field(nx, ny, lx, ly, x0, y0)
    # Number of nodes
    N = (nx + 1) * (ny + 1)

    # Initialize velocity components
    uₒx = zeros(N)
    uₒy = zeros(N)

    # Calculate Poiseuille velocity components
    for j in 0:ny
        for i in 0:nx
            idx = i + j * (nx + 1) + 1
            y = y0 + j * (ly / ny)
            x = x0 + i * (lx / nx)
            uₒx[idx] =  x * (1 - x) 
            uₒy[idx] =  0.0  # No velocity in the y direction for Poiseuille flow
        end
    end
    return uₒx, uₒy
end

# Initialize the velocity field with a radial flow
function initialize_radial_velocity_field(nx, ny, lx, ly, x0, y0, center, magnitude)
    # Number of nodes
    N = (nx + 1) * (ny + 1)

    # Initialize velocity components
    uₒx = zeros(N)
    uₒy = zeros(N)

    # Calculate radial velocity components
    for j in 0:ny
        for i in 0:nx
            idx = i + j * (nx + 1) + 1
            y = y0 + j * (ly / ny)
            x = x0 + i * (lx / nx)
            r = sqrt((x - center[1])^2 + (y - center[2])^2)
            uₒx[idx] = (x - center[1]) / r * magnitude
            uₒy[idx] = (y - center[2]) / r * magnitude
        end
    end
    return uₒx, uₒy
end


# Volume redefinition
function volume_redefinition!(capacity::Capacity{1}, operator::AbstractOperators)
    pₒ = [capacity.C_ω[i][1] for i in 1:length(capacity.C_ω)]
    pᵧ = [capacity.C_γ[i][1] for i in 1:length(capacity.C_ω)]
    p = vcat(pₒ, pᵧ)
    grad = ∇(operator, p)
    W_new = [grad[i] * capacity.W[1][i,i] for i in eachindex(grad)]
    W_new = spdiagm(0 => W_new)

    pₒ = [(capacity.C_ω[i][1]^2)/2 for i in 1:length(capacity.C_ω)]
    pᵧ = [(capacity.C_γ[i][1]^2)/2 for i in 1:length(capacity.C_ω)]

    p = vcat(pₒ, pᵧ)
    grad = ∇(operator, p)

    qω = vcat(grad)
    qγ = vcat(grad)

    div = ∇₋(operator, qω, qγ)
    V_new = spdiagm(0 => div)

    capacity.W = (W_new,)
    capacity.V = V_new

    println("Don't forget to rebuild the operator")
end

# ---------------------------------------------------------------------------
# Vorticity (2D) via circulation / cut-cell apertures
# ---------------------------------------------------------------------------

"""
    circulation_vorticity(uωx, uωy, cap_p, mesh_ux, mesh_uy)

Compute the cell-centered vorticity ω_z from staggered velocities using the
circulation theorem on a cut Cartesian cell:

    ω = (u_E * A_x,E - u_W * A_x,W - v_N * A_y,N + v_S * A_y,S) / V

where `A_x`/`A_y` are the face apertures (lengths) from the pressure capacity
`cap_p` and `V` is the (possibly cut) cell area. Velocities are assumed to be
stored on the staggered meshes `mesh_ux`/`mesh_uy` with dimensions matching
`uωx`/`uωy`.
"""
function circulation_vorticity(uωx::AbstractVector,
                               uωy::AbstractVector,
                               cap_p::Capacity{2},
                               mesh_ux::AbstractMesh,
                               mesh_uy::AbstractMesh)
    # Infer pressure grid shape from capacity lengths (some capacities use node counts)
    nV = length(diag(cap_p.V))
    nx_nodes = length(cap_p.mesh.nodes[1])
    ny_nodes = length(cap_p.mesh.nodes[2])
    if nx_nodes * ny_nodes == nV
        nx_p, ny_p = nx_nodes, ny_nodes
    elseif (nx_nodes - 1) * (ny_nodes - 1) == nV
        nx_p, ny_p = nx_nodes - 1, ny_nodes - 1
    else
        error("Cannot infer pressure grid shape: V length=$(nV), mesh nodes=($(nx_nodes),$(ny_nodes))")
    end

    Ax = reshape(diag(cap_p.A[1]), (nx_p, ny_p))
    Ay = reshape(diag(cap_p.A[2]), (nx_p, ny_p))
    V  = reshape(diag(cap_p.V),    (nx_p, ny_p))

    # Staggered velocities: Ux on vertical faces, Uy on horizontal faces
    nx_u = length(mesh_ux.nodes[1])
    ny_u = length(mesh_ux.nodes[2])
    nx_v = length(mesh_uy.nodes[1])
    ny_v = length(mesh_uy.nodes[2])

    Ux = reshape(uωx, (nx_u, ny_u))
    Uy = reshape(uωy, (nx_v, ny_v))

    @assert nx_u == nx_p + 1 || nx_u == nx_p "Unexpected Ux size $(nx_u)x$(ny_u) vs pressure $(nx_p)x$(ny_p)"
    @assert ny_u == ny_p || ny_u == ny_p + 1 "Unexpected Ux size $(nx_u)x$(ny_u) vs pressure $(nx_p)x$(ny_p)"
    @assert nx_v == nx_p || nx_v == nx_p + 1 "Unexpected Uy size $(nx_v)x$(ny_v) vs pressure $(nx_p)x$(ny_p)"
    @assert ny_v == ny_p + 1 || ny_v == ny_p "Unexpected Uy size $(nx_v)x$(ny_v) vs pressure $(nx_p)x$(ny_p)"

    omega = zeros(Float64, nx_p, ny_p)
    @inbounds for j in 1:ny_p
        for i in 1:nx_p
            vol = V[i, j]
            if vol <= 0
                omega[i, j] = 0.0
                continue
            end
            ax = Ax[i, j]
            ay = Ay[i, j]
            uE = Ux[min(i + 1, size(Ux, 1)), j]
            uW = Ux[i, j]
            vN = Uy[i, min(j + 1, size(Uy, 2))]
            vS = Uy[i, j]
            omega[i, j] = (uE * ax - uW * ax - vN * ay + vS * ay) / vol
        end
    end
    return omega
end

"""
    circulation_vorticity(fluid::Fluid{2}, state::AbstractVector)

Convenience wrapper that extracts the staggered velocity components from a
Navier–Stokes state vector and calls `circulation_vorticity`.
"""
function circulation_vorticity(fluid::Fluid{2}, state::AbstractVector)
    nu_x = prod(fluid.operator_u[1].size)
    nu_y = prod(fluid.operator_u[2].size)
    uωx = view(state, 1:nu_x)
    uωy = view(state, 2nu_x + 1:2nu_x + nu_y)
    return circulation_vorticity(uωx, uωy, fluid.capacity_p, fluid.mesh_u[1], fluid.mesh_u[2])
end
