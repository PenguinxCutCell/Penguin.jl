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
nx, ny = 64, 64
lx, ly = 1., 1.
x0, y0 = 0., 0.
Δx, Δy = lx/(nx), ly/(ny)
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body : Gaussian bump
ampl = 0.1 * ly
center = 0.5 * ly
sigma = 0.2 * ly
sₙ(y) = 0.1 * ly + ampl * exp(-((y - center)^2) / (2*sigma^2))
body = (x,y,t,_=0)->(x - sₙ(y))

# Define the Space-Time mesh
Δt = 0.001
Tend = 0.020
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
bc = Dirichlet(0.0)
bc1 = Dirichlet(1.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}( :bottom => bc1, :top => bc))
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
solver, residuals, xf_log, reconstruct, timestep_history = solve_MovingLiquidDiffusionUnsteadyMono2D!(solver, Fluide, Interface_position, Hₙ⁰, sₙ, Δt, Tend, bc_b, bc, stef_cond, mesh, "BE"; interpo="quad", Newton_params=Newton_params, adaptive_timestep=true, Δt_min=5e-4, method=Base.:\)

# Animate the solution
animate_solution(solver, mesh, body)

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