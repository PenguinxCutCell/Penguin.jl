using Penguin
using IterativeSolvers
using CairoMakie
### 1D Test Case : Diphasic Unsteady Diffusion Equation 
# Define the mesh
nx = 320
lx = 8.0
x0 = 0.0
domain=((x0,lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
xint = 4.0 #+ 0.1
body = (x, _=0) -> (x - xint)
body_c = (x, _=0) -> -(x - xint)

times = Dict{String,Float64}()

# Define the capacity
times["VOFI"] = @elapsed begin
    capacity  = Capacity(body,  mesh)
    capacity_c = Capacity(body_c, mesh)
end

# Define the operators
times["Operators"] = @elapsed begin
    operator   = DiffusionOps(capacity)
    operator_c = DiffusionOps(capacity_c)
end
  
#volume_redefinition!(capacity, operator)
#volume_redefinition!(capacity_c, operator_c)

operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc1 = Dirichlet(0.0)
bc0 = Dirichlet(1.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc0, :bottom => bc1))

ic = InterfaceConditions(ScalarJump(1.0, 2.0, 0.0), FluxJump(1.0, 1.0, 0.0))
He = 1.0
# Define the source term
f1 = (x,y,z,t)->0.0
f2 = (x,y,z,t)->0.0

D1 = (x,y,z)->1.0
D2 = (x,y,z)->1.0

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, D1)
Fluide_2 = Phase(capacity_c, operator_c, f2, D2)

# Initial condition
u0ₒ1 = zeros(nx+1)
u0ᵧ1 = zeros(nx+1)
u0ₒ2 = ones(nx+1)
u0ᵧ2 = ones(nx+1)

u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Define the solver
Δt = 0.5 * (lx/nx)^2
Tend = 1.0
times["Solver setup"] = @elapsed begin
    # (re-use your u0 definition above)
    solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "CN")
end
# Solve the problem
times["Solving"] = @elapsed begin
    solve_DiffusionUnsteadyDiph!(
      solver, Fluide_1, Fluide_2,
      Δt, Tend, bc_b, ic, "CN";
      method = Base.:\ )
end

using SparseArrays, CairoMakie

A = solver.A
b = solver.b
A, b, rows_idx, cols_idx = remove_zero_rows_cols!(A, b)
row_idx, col_idx, _ = findnz(A)

fig_spy = Figure()
ax_spy = Axis(fig_spy[1, 1];
               xlabel    = "Row",
               ylabel    = "Column",
               title     = "Sparsity of A",
               yreversed=true)    # flip Y so index 1 is at bottom

# swap row/col when scattering
scatter!(ax_spy, col_idx,row_idx;
         markersize = 5,
         color      = :black)

display(fig_spy)


#––– plot the timings as percentages of T_end –––
labels = collect(keys(times))
raw   = collect(values(times))
# use Tend (end time) instead of total runtime
pct   = 100 .* raw ./ Tend   # compute percentage of each step relative to simulation end time

fig_cost = Figure(size = (700,400))
ax_cost = Axis(fig_cost[1,1];
  xticks            = (1:length(labels), labels),
  xticklabelrotation= 45,
  ylabel            = "Time (% of Tend)",
  title             = "Computational cost vs. Tend")

barplot!(ax_cost, 1:length(pct), pct; color = :skyblue)

# optional: annotate bars
for (i, p) in enumerate(pct)
  text!(ax_cost, i, p + 1;
        text = string(round(p, digits=1), "%"),
        align = (:center, :bottom), 
        fontsize = 10)
end


# Write the solution to a VTK file
#write_vtk("solution", mesh, solver)

# Plot the solution
plot_solution(solver, mesh, body, capacity)

# Animation
#animate_solution(solver, mesh, body)

# Plot the solution
state_i = 10
u1ₒ = solver.states[state_i][1:nx+1]
u1ᵧ = solver.states[state_i][nx+2:2*(nx+1)]
u2ₒ = solver.states[state_i][2*(nx+1)+1:3*(nx+1)]
u2ᵧ = solver.states[state_i][3*(nx+1)+1:end]

x = range(x0, stop = lx, length = nx+1)
using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="u", title="Diphasic Unsteady Diffusion Equation")

for (idx, i) in enumerate(1:10:length(solver.states))
    u1ₒ = solver.states[i][1:nx+1]
    u1ᵧ = solver.states[i][nx+2:2*(nx+1)]
    u2ₒ = solver.states[i][2*(nx+1)+1:3*(nx+1)]
    u2ᵧ = solver.states[i][3*(nx+1)+1:end]

    # Display labels only for first iteration
    scatter!(ax, x, u1ₒ, color=:blue,  markersize=3,
        label = (idx == 1 ? "Bulk Field - Phase 1" : nothing))
    scatter!(ax, x, u1ᵧ, color=:red,   markersize=3,
        label = (idx == 1 ? "Interface Field - Phase 1" : nothing))
    scatter!(ax, x, u2ₒ, color=:green, markersize=3,
        label = (idx == 1 ? "Bulk Field - Phase 2" : nothing))
    scatter!(ax, x, u2ᵧ, color=:orange, markersize=3,
        label = (idx == 1 ? "Interface Field - Phase 2" : nothing))
end

axislegend(ax, position=:rb)
display(fig)


# Analytical solution
using SpecialFunctions
He=2.0
D1, D2 = 1.0, 1.0
function T1(x)
    t = Tend
    x = x - xint
    return - He/(1+He*sqrt(D1/D2))*(erfc(x/(2*sqrt(D1*t))) - 2)
end

function T2(x)
    t = Tend
    x = x - xint
    return - He/(1+He*sqrt(D1/D2))*erfc(x/(2*sqrt(D2*t))) + 1
end

u1 = T1.(x)
u2 = T2.(x)

u1[capacity.cell_types .== 0] .= NaN
u2[capacity_c.cell_types .== 0] .= NaN

u1_num = solver.states[end][1:nx+1]
u2_num = solver.states[end][2*(nx+1)+1:3*(nx+1)]

u1_num[capacity.cell_types .== 0] .= NaN
u2_num[capacity_c.cell_types .== 0] .= NaN

fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="u", title="Diphasic Unsteady Diffusion Equation")
scatter!(ax, x, u1, color=:blue, label="Bulk Field - Phase 1")
scatter!(ax, x, u1_num, color=:red, label="Bulk Field - Phase 1 - Numerical")
scatter!(ax, x, u2, color=:green, label="Bulk Field - Phase 2")
scatter!(ax, x, u2_num, color=:orange, label="Bulk Field - Phase 2 - Numerical")
axislegend(ax, position=:rb)
display(fig)

# Now you can use this function to compute the convergence errors:
(ana_sols, num_sols, global_errs, full_errs, cut_errs, empty_errs) = check_convergence_diph(T1, T2, solver, capacity, capacity_c, 2, false)

# You can also add a visualization of the errors:
function visualize_diphasic_errors(x, ana_sols, num_sols, capacity1, capacity2)
    u1_ana, u2_ana = ana_sols
    u1_num, u2_num = num_sols
    
    # Compute pointwise errors
    err1 = abs.(u1_ana .- u1_num)
    err2 = abs.(u2_ana .- u2_num)
    
    # Set errors to NaN for empty cells
    err1[capacity1.cell_types .== 0] .= NaN
    err2[capacity2.cell_types .== 0] .= NaN
    
    # Create figure
    fig = Figure()
    ax = Axis(fig[1, 1], 
              xlabel="x", 
              ylabel="Absolute Error", 
              title="Diphasic Solution Errors",
              yscale=log10)
    
    scatter!(ax, x, err1, color=:blue, label="Phase 1 Error")
    scatter!(ax, x, err2, color=:red, label="Phase 2 Error")
    
    axislegend(ax, position=:rt)
    display(fig)
    
    return fig
end

# Visualize errors
error_fig = visualize_diphasic_errors(x, ana_sols, num_sols, capacity, capacity_c)



# Compute Sherwood number : 1) compute average concentration inside and outside the body : ̅cl = 1/V ∫ cl dV, ̅cg = 1/V ∫ cg dV
# 2) compute the mass transfer rate : k = (cg(n+1) - cg(n))/(Γ * Δt * (He*cg(n+1) - cl(n+1))) with Γ the area of the interface
# 3) compute the Sherwood number : Sh = k * L/D with L the characteristic length and D the diffusion coefficient

function compute_sherwood_all(solver, capacity, capacity_c, Δt, He, L, D)
    nx = size(capacity.V, 1) - 1

    # Precompute volumes
    Vg   = sum(capacity.V)
    Vl   = sum(capacity_c.V)
    Vg_i = [capacity.V[i, i] for i in 1:nx+1]
    Vl_i = [capacity_c.V[i, i] for i in 1:nx+1]
    Γ    = sum(capacity.Γ)

    # Store Sherwood numbers
    Sh_values = Float64[]

    # Loop over consecutive states
    for i in 2:length(solver.states)
        u_nm1 = solver.states[i-1]
        u_n   = solver.states[i]

        cgω_nm1 = u_nm1[1:nx+1]                  # gas in ω at t_{n-1}
        clω_nm1 = u_nm1[2*(nx+1)+1:3*(nx+1)]     # liquid in ω at t_{n-1}

        cgω_n   = u_n[1:nx+1]                    # gas in ω at t_n
        clω_n   = u_n[2*(nx+1)+1:3*(nx+1)]       # liquid in ω at t_n

        # Average concentrations
        cgω̅_nm1 = sum(cgω_nm1 .* Vg_i) / Vg
        clω̅_nm1 = sum(clω_nm1 .* Vl_i) / Vl
        cgω̅_n   = sum(cgω_n   .* Vg_i) / Vg
        clω̅_n   = sum(clω_n   .* Vl_i) / Vl

        # Average concentrations at t_{n+1/2}
        cgω̅_n2 = 0.5 * (cgω̅_n + cgω̅_nm1)
        clω̅_n2 = 0.5 * (clω̅_n + clω̅_nm1)

        # Mass transfer rate
        # (Difference between latest and previous step) / (Γ * Δt * (He * cgω̅_{n+1/2} - clω̅_{n+1/2}))
        numerator   = (cgω̅_n - cgω̅_nm1)
        denominator = Γ * Δt * (He*cgω̅_n2 - clω̅_n2)
        k = numerator / denominator

        # Sherwood
        Sh = k * L / D
        push!(Sh_values, Sh)
    end

    return abs.(Sh_values)
end

# Compute Sherwood number
#Sh_val = compute_sherwood_all(solver, capacity, capacity_c, Δt, He, lx, 1.0)


#––– Compare numerical/analytical profiles for multiple Henry numbers –––
He_values = [1.0, 2.0, 10.0, 100.0]
x_nodes = range(x0, stop = lx, length = nx+1)

function analytic_profile(x, He_val, Tend, xint, D1_val, D2_val)
    shift = x .- xint
    pref = -He_val / (1 + He_val * sqrt(D1_val / D2_val))
    u1 = pref .* (erfc.(shift ./ (2 * sqrt(D1_val * Tend))) .- 2)
    u2 = pref .* erfc.(shift ./ (2 * sqrt(D2_val * Tend))) .+ 1
    return u1, u2
end

function solve_diffusion_for_He(He_val, Δt, Tend)
    ic_local = InterfaceConditions(ScalarJump(1.0, He_val, 0.0), FluxJump(1.0, 1.0, 0.0))
    u0_local = vcat(zeros(nx+1), zeros(nx+1), ones(nx+1), ones(nx+1))
    solver_local = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic_local, Δt, u0_local, "CN")
    solve_DiffusionUnsteadyDiph!(
        solver_local, Fluide_1, Fluide_2,
        Δt, Tend, bc_b, ic_local, "CN";
        method = Base.:\ )
    return solver_local
end

palette = [:blue, :red, :green, :purple]

fig_multi = Figure(resolution = (1000, 600))
ax_multi = Axis(fig_multi[1, 1];
                xlabel = "x",
                ylabel = "u",
                title = "Diphasic diffusion: analytical vs numerical for multiple λ")

for (idx, He_val) in enumerate(He_values)
    color = palette[(idx - 1) % length(palette) + 1]
    solver_he = solve_diffusion_for_He(He_val, Δt, Tend)

    u1_num = solver_he.states[end][1:nx+1]
    u2_num = solver_he.states[end][2*(nx+1)+1:3*(nx+1)]

    u1_num[capacity.cell_types .== 0] .= NaN
    u2_num[capacity_c.cell_types .== 0] .= NaN

    u1_ana, u2_ana = analytic_profile(x_nodes, He_val, Tend, xint, 1.0, 1.0)
    u1_ana[capacity.cell_types .== 0] .= NaN
    u2_ana[capacity_c.cell_types .== 0] .= NaN

    # Plot analytical solution (both phases with same style)
    lines!(ax_multi, x_nodes, u1_ana;
           color = color, linestyle = :solid,
           label = "λ=$(He_val) analytic")
    lines!(ax_multi, x_nodes, u2_ana;
           color = color, linestyle = :solid)
    
    # Plot numerical solution (both phases with same style)
    scatter!(ax_multi, x_nodes, u1_num;
             color = color, markersize = 3, marker = :circle,
             label = "λ=$(He_val) numerical")
    scatter!(ax_multi, x_nodes, u2_num;
             color = color, markersize = 3, marker = :circle)
end

vlines!(ax_multi, [xint]; color = :black, linestyle = :dot, linewidth = 2, label = "Interface")

axislegend(ax_multi, position = :rb, nbanks = 2)
display(fig_multi)
save("diphasic_diffusion_multiple_λ.pdf", fig_multi)
