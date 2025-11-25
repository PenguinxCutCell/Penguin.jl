using Penguin
using IterativeSolvers, SpecialFunctions
using Roots
using CairoMakie
using Statistics

### 2D Test Case : Monophasic Unsteady Diffusion Equation inside a moving Disk
# Define the mesh
nx, ny = 128, 128
lx, ly = 16.0, 16.0
x0, y0 = -8.0, -8.0
Δx, Δy = lx/(nx), ly/(ny)
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = 1.0, [0.0, 0.0]
c = 1.56
body = (x,y,t)->-(sqrt((x - center[1])^2 + (y - center[2])^2) - (radius + c * sqrt(t)))

# Define F(s) function using the exponential integral E₁
function F(s)
    return expint(s^2/4)  # E₁(s²/4)
end
# Analytical temperature function
function analytical_temperature(r, t)
    # Calculate similarity variable
    s = r / sqrt(t)
    
    if s < c
        return 1.0
    else
        # In liquid region, use the similarity solution
        return F(s)/F(c)
    end
end

# Define the Space-Time mesh
Δt = 1.0*(lx/nx)^2
Tstart = 0.0
Tend = 5.0
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh; method="VOFI", integration_method=:vofijul, compute_centroids=false)

# Define the operators
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(1.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc))

# Define the source term
f = (x,y,z,t)-> 0.0 #sin(x)*cos(10*y)
K = (x,y,z)-> 1.0

Fluide = Phase(capacity, operator, f, K)

# Initial condition
u0ₒ = zeros((nx+1)*(ny+1))
u0ᵧ = ones((nx+1)*(ny+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
solver = MovingDiffusionUnsteadyMono(Fluide, bc_b, bc1, Δt, u0, mesh, "BE")

# Solve the problem
solve_MovingDiffusionUnsteadyMono!(solver, Fluide, body, Δt, Tstart, Tend, bc_b, bc1, mesh, "BE"; method=Base.:\,  geometry_method="VOFI", integration_method=:vofijul, compute_centroids=false)


# Plot the solution
#plot_solution(solver, mesh, body, capacity; state_i=1)


function plot_three_snapshots(solver, mesh, body, Tend; times=[0.0, Tend/2, Tend], filename="moving_heat_snapshots.pdf")
    xi = mesh.centers[1]
    yi = mesh.centers[2]
    npts = (nx+1)*(ny+1)
    center = [0.0, 0.0]
    radius = 1.0
    c = 1.56
    Δx, Δy = lx/nx, ly/ny

    # Find indices closest to requested times
    Δt = solver.states |> length > 1 ? Tend/(length(solver.states)-1) : Tend
    time_indices = [clamp(round(Int, t/Δt)+1, 1, length(solver.states)) for t in times]

    # Temperature limits for consistent color scale
    all_temps = Float64[]
    for Tstate in solver.states
        Tw = Tstate[1:npts]
        Tw_mat = reshape(Tw, (nx+1, ny+1))[1:end-1, 1:end-1]
        push!(all_temps, extrema(Tw_mat)...)
    end
    temp_limits = (minimum(all_temps), maximum(all_temps))

    fig = Figure(size=(1200, 400))
    hm = nothing
    for (i, idx) in enumerate(time_indices)
        ax = Axis(fig[1, i], 
            title="t = $(round(times[i], digits=3))",
            xlabel="x", ylabel="y",
            aspect=DataAspect(),)
        current_time = times[i]
        body_end = (x, y,_=0) -> -(sqrt((x - center[1])^2 + (y - center[2])^2) - (radius + c * sqrt(current_time)))
        capacity_end = Capacity(body_end, mesh; compute_centroids=false)
        Tstate = solver.states[idx]
        Tw = Tstate[1:npts]
        Tw[capacity_end.cell_types .== 0] .= 1.0

        Tmat = reshape(Tw, (nx+1, ny+1))[1:end-1, 1:end-1]
        hm = heatmap!(ax, xi, yi, Tmat, colormap=:viridis, colorrange=temp_limits)

        # Interface
        n_interface_points = 100
        theta = range(0, 2π, length=n_interface_points)
        interface_radius = radius + c * sqrt(current_time)
        adjusted_center_x = center[1] - Δx
        adjusted_center_y = center[2] - Δy
        interface_x = adjusted_center_x .+ interface_radius .* cos.(theta)
        interface_y = adjusted_center_y .+ interface_radius .* sin.(theta)
        lines!(ax, interface_x, interface_y, color=:white, linewidth=2)
    end

    Colorbar(fig[1, end+1], hm, label="Temperature")
    save(filename, fig)
    println("Saved 3-snapshot figure to $filename")
    return fig
end

# Usage:
plot_three_snapshots(solver, mesh, body, Tend)

# Animation
function create_moving_heat_animation(solver, mesh, body, Tend)
    # Extraire les dimensions du maillage
    xi = mesh.centers[1]
    yi = mesh.centers[2]
    nx1, ny1 = length(xi), length(yi)
    npts = (nx+1) * (ny+1)  # Nombre total de points dans les états du solveur
    
    # Extraire les limites de température pour une échelle de couleur cohérente
    all_temps = Float64[]
    for Tstate in solver.states
        Tw = Tstate[1:npts]  # Températures à l'extérieur de l'interface
        # Reshape puis retirer la dernière ligne et colonne pour correspondre aux centres
        Tw_mat = reshape(Tw, (nx+1, ny+1))[1:end-1, 1:end-1]
        push!(all_temps, extrema(Tw_mat)...)
    end
    temp_limits = (minimum(all_temps), maximum(all_temps))
    
    # Obtenir le nombre de pas de temps
    n_timesteps = length(solver.states)
    
    # Créer le répertoire des résultats si nécessaire
    results_dir = joinpath(pwd(),"stefre")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Créer la figure pour l'animation
    fig = Figure(size=(900, 700))
    ax = Axis(fig[1, 1], 
            title="Temperature & Moving Interface", 
            xlabel="x", ylabel="y", 
            aspect=DataAspect(),
            limits=(x0, x0+lx, y0, y0+ly))
    
    # Enregistrer l'animation MP4
    record(fig, joinpath(results_dir, "moving_heat_animation.mp4"), 1:n_timesteps) do i
        empty!(ax)  # Effacer l'axe pour la nouvelle frame
        
        # Calculer le temps actuel
        current_time = (i-1) * Δt
        
        # Mettre à jour le titre
        ax.title = "Temperature & Moving Interface, t=$(round(current_time, digits=4))"
        
        # Extraire et reformater la température pour ce pas de temps
        body_end = (x, y,_=0) -> -(sqrt((x - center[1])^2 + (y - center[2])^2) - (radius + c * sqrt(current_time)))
        capacity_end = Capacity(body_end, mesh; compute_centroids=false)
        Tstate = solver.states[i]
        Tw = Tstate[1:npts]
        Tw[capacity_end.cell_types .== 0] .= 1.0  # Assurer que les cellules vides ont une température de 1.0

        
        # Reshape et retirer la dernière ligne et colonne pour correspondre aux centres
        Tmat = reshape(Tw, (nx+1, ny+1))[1:end-1, 1:end-1]
        
        # Afficher le heatmap de température
        hm = heatmap!(ax, xi, yi, Tmat, colormap=:thermal, colorrange=temp_limits)
        Colorbar(fig[1, 2], hm, label="Temperature")
        
        # Générer des points pour tracer l'interface au temps actuel
        n_interface_points = 100
        theta = range(0, 2π, length=n_interface_points)
        
        # Calcul du rayon à l'instant t
        interface_radius = radius + c * sqrt(current_time)
        
        adjusted_center_x = center[1] - Δx
        adjusted_center_y = center[2] - Δy
        
        interface_x = adjusted_center_x .+ interface_radius .* cos.(theta)
        interface_y = adjusted_center_y .+ interface_radius .* sin.(theta)
        
        # Tracer l'interface
        lines!(ax, interface_x, interface_y, color=:white, linewidth=3)
        
        # Afficher la progression
        println("Frame $i of $n_timesteps")
    end
    
    # Créer également une version GIF
    record(fig, joinpath(results_dir, "moving_heat_animation.gif"), 1:n_timesteps) do i
        empty!(ax)
        
        # Calculer le temps actuel
        current_time = (i-1) * Δt
        
        # Mettre à jour le titre
        ax.title = "Temperature & Moving Interface, t=$(round(current_time, digits=4))"
        
        # Extraire et reformater la température pour ce pas de temps
        body_end = (x, y,_=0) -> -(sqrt((x - center[1])^2 + (y - center[2])^2) - (radius + c * sqrt(current_time)))
        capacity_end = Capacity(body_end, mesh; compute_centroids=false)
        Tstate = solver.states[i]
        Tw = Tstate[1:npts]
        Tw[capacity_end.cell_types .== 0] .= 1.0  # Assurer que les cellules vides ont une température de 1.0

        # Reshape et retirer la dernière ligne et colonne pour correspondre aux centres
        Tmat = reshape(Tw, (nx+1, ny+1))[1:end-1, 1:end-1]

        # Afficher le heatmap de température
        hm = heatmap!(ax, xi, yi, Tmat, colormap=:thermal, colorrange=temp_limits)
        Colorbar(fig[1, 2], hm, label="Temperature")
        
        # Générer des points pour tracer l'interface au temps actuel
        n_interface_points = 100
        theta = range(0, 2π, length=n_interface_points)
        
        # Calcul du rayon à l'instant t
        interface_radius = radius + c * sqrt(current_time)
        
        
        adjusted_center_x = center[1] - Δx
        adjusted_center_y = center[2] - Δy
        
        interface_x = adjusted_center_x .+ interface_radius .* cos.(theta)
        interface_y = adjusted_center_y .+ interface_radius .* sin.(theta)
        
        # Tracer l'interface
        lines!(ax, interface_x, interface_y, color=:white, linewidth=3)
    end
    
    println("\nAnimation saved to:")
    println("- MP4: $(joinpath(results_dir, "moving_heat_animation.mp4"))")
    println("- GIF: $(joinpath(results_dir, "moving_heat_animation.gif"))")
    
    return fig
end

# Appel de la fonction pour créer l'animation
anim = create_moving_heat_animation(solver, mesh, body, Tend)
