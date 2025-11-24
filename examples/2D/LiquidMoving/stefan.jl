using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations
using Colors
using Statistics
using FFTW
using DSP

### 2D Test Case : One-phase Stefan Problem : Growing Planar Interface
# Define the spatial mesh
nx, ny = 32, 32
lx, ly = 1., 1.
x0, y0 = 0., 0.
Δx, Δy = lx/(nx), ly/(ny)
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
sₙ(y) =  0.2*lx + 0.01*cos(2π*y)
body = (x,y,t,_=0)->(x - sₙ(y))

# Define the Space-Time mesh
Δt = 0.001
Tend = 0.01
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)

# Initial Height Vₙ₊₁ and Vₙ
Vₙ₊₁ = capacity.A[3][1:end÷2, 1:end÷2]
Vₙ = capacity.A[3][end÷2+1:end, end÷2+1:end]
Vₙ = diag(Vₙ)
Vₙ₊₁ = diag(Vₙ₊₁)
Vₙ = reshape(Vₙ, (nx+1, ny+1))
Vₙ₊₁ = reshape(Vₙ₊₁, (nx+1, ny+1))

Hₙ⁰ = collect(vec(sum(Vₙ, dims=1)))
Hₙ₊₁⁰ = collect(vec(sum(Vₙ₊₁, dims=1)))

Interface_position = x0 .+ Hₙ⁰./Δy
println("Interface position : $(Interface_position)")
# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(1.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}( :bottom => bc1))
ρ, L = 1.0, 1.0
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))

# Define the source term
f = (x,y,z,t)-> 0.0 #sin(x)*cos(10*y)
K = (x,y,z)-> 1.0

Fluide = Phase(capacity, operator, f, K)

# Initial condition
u0ₒ = zeros((nx+1)*(ny+1))
u0ᵧ = zeros((nx+1)*(ny+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Newton parameters
max_iter = 10000
tol = 1e-6
reltol = 1e-6
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log, reconstruct, timestep_history = solve_MovingLiquidDiffusionUnsteadyMono2D!(solver, Fluide, Interface_position, Hₙ⁰, sₙ, Δt, Tend, bc_b, bc, stef_cond, mesh, "BE"; interpo="quad", Newton_params=Newton_params, adaptive_timestep=false, Δt_min=5e-4, method=Base.:\)


using CairoMakie, Printf

function animate_temperature_3d(solver, mesh, nx, ny, reconstruct; 
                              framerate=15, 
                              resolution=(1000, 800))
    
    println("Creating 3D temperature animation...")
    
    # Create figure
    fig = Figure(resolution=resolution)
    ax = Axis3(fig[1, 1], 
               xlabel="x", ylabel="y", zlabel="Temperature",
               title="Temperature Evolution in Stefan Problem")
    
    # Create x, y coordinates grid for plotting
    x = range(x0, stop=x0 + lx, length=nx+1)
    y = range(x0, stop=x0 + lx, length=ny+1)
    
    # Get the number of time steps
    n_frames = length(solver.states)
    
    # Create the initial surface plot
    state = solver.states[1]
    
    # Extract temperature field (first half of the state vector contains temperature)
    temp = state[1:(nx+1)*(ny+1)]  # First part is temperature
    temperature = reshape(temp, (nx+1, ny+1))
    
    # Create surface plot
    surf = surface!(ax, x, y, temperature, colormap=:viridis)
    
    # Add colorbar
    cbar = Colorbar(fig[1, 2], surf, label="Temperature")
    
    # Set the zlim to the max temperature (typically 1.0 in this problem)
    zlims!(ax, 0, 1)
    
    # Add time label
    time_label = Label(fig[0, :], @sprintf("t = %.4f", 0.0), fontsize=20)
    
    # Create animation
    record(fig, "temperature_3d_evolution.mp4", 1:n_frames; framerate=framerate) do frame
        # Update plot with current time step data
        state = solver.states[frame]
        
        # Extract temperature field
        temp = state[1:(nx+1)*(ny+1)]  # First part is temperature
        temperature = reshape(temp, (nx+1, ny+1))
        
        # Update surface plot
        surf[3] = temperature
        
        # Update time label
        time_t = (frame-1) * Δt
        time_label.text = @sprintf("t = %.4f", time_t)
        
   
        
        # Print progress
        if frame % 10 == 0 || frame == 1 || frame == n_frames
            println("Rendering frame $frame / $n_frames")
        end
    end
    
    println("Animation saved to temperature_3d_evolution.mp4")
    
    return fig
end

# Call the function
fig = animate_temperature_3d(solver, mesh, nx, ny, reconstruct)
display(fig)

readline()

# Plot the timestep:
fig = plot_timestep_history(timestep_history)
display(fig)

# Plot the position of the interface
fig_styles = plot_interface_evolution(xf_log,
                                    time_steps=:all,
                                    n_times=8, 
                                    color_gradient=false,
                                    line_styles=true,
                                    y_resolution=1000)
display(fig_styles)

# Plot the residuals
fig_distinct = plot_newton_residuals(residuals, 
                                   color_mode=:distinct, 
                                   n_times=15)
display(fig_distinct)


# Analyze convergence rates Newton
conv_fig = analyze_convergence_rates_newton(residuals)
display(conv_fig)

# Animation
animate_solution(solver, mesh, body)

# Plot the isotherms
steps = [1, 2, floor(Int, length(solver.states)/2), length(solver.states)]
fig_compare = plot_isotherms(solver, mesh, nx, ny, reconstruct,
                           time_steps=steps,
                           layout=(2,2))
display(fig_compare)

# Analyser le spectre de l'interface
spectral_results = analyze_interface_spectrum(reconstruct, mesh, nothing;
                                           window=DSP.hanning,  # Fenêtre de Hanning pour réduire les fuites
                                           padding_factor=4,    # Plus de padding pour une meilleure résolution
                                           n_points=200,        # Échantillonnage spatial fin
                                           plot_amplitude=true,
                                           plot_spectrum=true,
                                           plot_spectrogram=true)

# Afficher les graphiques
display(spectral_results[:fig_amplitude])
display(spectral_results[:fig_spectrum])
display(spectral_results[:fig_spectrogram])
display(spectral_results[:fig_wavenumber])

# Extraire et afficher les informations clés
println("Résultats de l'analyse spectrale:")
println("--------------------------------")
println("Amplitude initiale: $(spectral_results[:amplitude][1])")
println("Amplitude finale: $(spectral_results[:amplitude][end])")

if haskey(spectral_results, :dominant_frequencies)
    println("Fréquences dominantes: $(round.(spectral_results[:dominant_frequencies][1:min(3,end)], digits=5))")
end

if haskey(spectral_results, :decay_parameters)
    println("Taux de décroissance: $(round(spectral_results[:decay_parameters].gamma, digits=5))")
    println("Temps de demi-vie: $(round(spectral_results[:half_life], digits=5))")
end

if haskey(spectral_results, :dominant_wavelength)
    println("Longueur d'onde dominante: $(round(spectral_results[:dominant_wavelength], digits=5))")
end