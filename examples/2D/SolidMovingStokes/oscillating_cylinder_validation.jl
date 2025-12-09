using Penguin
using CairoMakie
using LinearAlgebra
using IterativeSolvers
using Statistics

"""
Laminar flow past a transversely oscillating cylinder (prescribed motion).

The cylinder of diameter D oscillates vertically with:
    y_c(t) = y0 + A * cos(2π f_e t)
where A = 0.2D and f_e is the excitation frequency.

We solve the unsteady Stokes problem with a uniform inflow, compute drag/lift
coefficients from surface tractions, and visualize vorticity and drag history.
This example is intended as a small validation of the cut-cell motion handling.
"""

###########
# Geometry and motion
###########
nx, ny = 64, 64
Lx, Ly = 8.0, 4.0
x0, y0 = -Lx/2, -Ly/2

radius = 0.5
diameter = 2radius
center_x = -1.0
center_y0 = 0.0
A_osc = 0.2 * diameter
f_e = 0.2                  # excitation frequency (Hz)
ω_osc = 2π * f_e

body = (x, y, t) -> begin
    cy = center_y0 + A_osc * cos(ω_osc * t)
    return radius - sqrt((x - center_x)^2 + (y - cy)^2)
end

###########
# Meshes
###########
mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

###########
# Initial capacities/operators (t=0)
###########
capacity_ux = Capacity((x,y,_=0) -> body(x,y,0.0), mesh_ux)
capacity_uy = Capacity((x,y,_=0) -> body(x,y,0.0), mesh_uy)
capacity_p  = Capacity((x,y,_=0) -> body(x,y,0.0), mesh_p)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions: uniform inflow, outflow at right, no-slip walls
###########
U_in = 1.0
ux_left = Dirichlet(U_in)
ux_right = Outflow()
ux_bottom = Dirichlet(0.0)
ux_top = Dirichlet(0.0)

uy_zero = Dirichlet(0.0)

bc_ux = BorderConditions(Dict(
    :left=>ux_left, :right=>ux_right,
    :bottom=>ux_bottom, :top=>ux_top
))
bc_uy = BorderConditions(Dict(
    :left=>uy_zero, :right=>uy_zero,
    :bottom=>uy_zero, :top=>uy_zero
))

pressure_gauge = PinPressureGauge()

# Cut boundary: match rigid body velocity (u_x = 0, u_y = dy/dt)
bc_cut = (
    Dirichlet((x,y,t) -> 0.0),
    Dirichlet((x,y,t) -> 0.0)
)

###########
# Physics
###########
μ = 0.02
ρ = 1.0
println("Re based on D: ", ρ * U_in * diameter / μ)
fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

###########
# Solver setup
###########
nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
Ntot = 2 * (nu_x + nu_y) + np
x0_vec = zeros(Ntot)

scheme = :CN
Δt = 0.01
T_end = 0.1           # several oscillation cycles

solver = MovingStokesUnsteadyMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut;
                                   scheme=scheme, x0=x0_vec)

println("Running oscillating cylinder case:")
println("  Δt=$(Δt), T_end=$(T_end), f_e=$(f_e)Hz, A= $(A_osc)")

times, states = solve_MovingStokesUnsteadyMono!(solver, body, mesh_p,
                                                 Δt, 0.0, T_end,
                                                 (bc_ux, bc_uy), bc_cut;
                                                 scheme=scheme,
                                                 method=IterativeSolvers.gmres,
                                                 geometry_method="VOFI",
                                                 integration_method=:vofijul,
                                                 compute_centroids=true)

println("Completed $(length(times)-1) steps; final time = ", times[end])

###########
# Visualization
###########
function visualize_oscillating_circle(times, states, mesh_ux, 
                                       body, nu_x, nu_y, np;
                                       frames=[1, length(times)])
    xs = mesh_ux.nodes[1]
    ys = mesh_ux.nodes[2]
    
    fig = Figure(size=(800, 400))
    
    for (col, frame_idx) in enumerate(frames)
        t = times[frame_idx]
        state = states[frame_idx]
        
        uωx = state[1:nu_x]
        uωy = state[2*nu_x+1:2*nu_x+nu_y]
        
        Ux = reshape(uωx, (length(xs), length(ys)))
        Uy = reshape(uωy, (length(xs), length(ys)))
        
        speed = sqrt.(Ux.^2 .+ Uy.^2)
        
        ax = Axis(fig[1, col], 
                  xlabel="x", ylabel="y", 
                  title="t = $(round(t, digits=3))",
                  aspect=DataAspect())
        
        hm = heatmap!(ax, xs, ys, speed; colormap=:viridis)
        
        # Draw interface
        θ_circle = range(0, 2π, length=100)
        cy = center_y0 + A_osc * sin(ω_osc * t + φ_osc)
        circle_x = center_x .+ radius .* cos.(θ_circle)
        circle_y = cy .+ radius .* sin.(θ_circle)
        lines!(ax, circle_x, circle_y, color=:white, linewidth=2)
        
        if col == length(frames)
            Colorbar(fig[1, col+1], hm, label="Velocity magnitude")
        end
    end
    
    save("oscillating_circle_stokes.png", fig)
    println("Saved visualization to oscillating_circle_stokes.png")
    return fig
end

# Visualize at start and end
fig = visualize_oscillating_circle(times, states, mesh_ux,
                                    body, nu_x, nu_y, np)
display(fig)

###########
# Utilities
###########
function bilinear_interp(nodes, field, x, y)
    xs, ys = nodes
    nx, ny = length(xs), length(ys)
    ix = clamp(searchsortedlast(xs, x), 1, nx-1)
    iy = clamp(searchsortedlast(ys, y), 1, ny-1)
    x1, x2 = xs[ix], xs[ix+1]; y1, y2 = ys[iy], ys[iy+1]
    tx = (x2 == x1) ? 0.0 : (x - x1)/(x2 - x1)
    ty = (y2 == y1) ? 0.0 : (y - y1)/(y2 - y1)
    f00 = field[ix, iy]; f10 = field[ix+1, iy]
    f01 = field[ix, iy+1]; f11 = field[ix+1, iy+1]
    return (1-ty)*((1-tx)*f00 + tx*f10) + ty*((1-tx)*f01 + tx*f11)
end

function gradients(Ux, Uy, xs, ys, dx, dy)
    nx, ny = size(Ux)
    ∂ux∂x = similar(Ux); ∂ux∂y = similar(Ux)
    ∂uy∂x = similar(Uy); ∂uy∂y = similar(Uy)
    for j in 1:ny
        for i in 1:nx
            im = max(i-1, 1); ip = min(i+1, nx)
            jm = max(j-1, 1); jp = min(j+1, ny)
            ∂ux∂x[i,j] = (Ux[ip,j] - Ux[im,j]) / ((xs[ip]-xs[im]) + eps())
            ∂ux∂y[i,j] = (Ux[i,jp] - Ux[i,jm]) / ((ys[jp]-ys[jm]) + eps())
            ∂uy∂x[i,j] = (Uy[ip,j] - Uy[im,j]) / ((xs[ip]-xs[im]) + eps())
            ∂uy∂y[i,j] = (Uy[i,jp] - Uy[i,jm]) / ((ys[jp]-ys[jm]) + eps())
        end
    end
    return ∂ux∂x, ∂ux∂y, ∂uy∂x, ∂uy∂y
end

function instantaneous_force(Ux, Uy, P, xs, ys, Xp, Yp, μ, ρ, U_ref, D, t;
                             center_x, center_y0, radius, A_osc, ω_osc, n_θ=200)
    cx = center_x
    cy = center_y0 + A_osc * cos(ω_osc * t)
    ∂ux∂x, ∂ux∂y, ∂uy∂x, ∂uy∂y = gradients(Ux, Uy, xs, ys, dx, dy)

    drag = 0.0
    lift = 0.0
    ds = 2π * radius / n_θ
    for k in 1:n_θ
        θ = 2π * (k-1) / n_θ
        nxn = cos(θ); nyn = sin(θ)
        px = cx + radius * nxn
        py = cy + radius * nyn
        p_val = bilinear_interp((Xp, Yp), P, px, py)
        dux_dx = bilinear_interp((xs, ys), ∂ux∂x, px, py)
        dux_dy = bilinear_interp((xs, ys), ∂ux∂y, px, py)
        duy_dx = bilinear_interp((xs, ys), ∂uy∂x, px, py)
        duy_dy = bilinear_interp((xs, ys), ∂uy∂y, px, py)
        τxx = 2μ * dux_dx
        τyy = 2μ * duy_dy
        τxy = μ * (dux_dy + duy_dx)
        fx = -(p_val * nxn) + τxx * nxn + τxy * nyn
        fy = -(p_val * nyn) + τxy * nxn + τyy * nyn
        drag += fx * ds
        lift += fy * ds
    end
    denom = 0.5 * ρ * U_ref^2 * D
    Cd = drag / (denom + eps())
    Cl = lift / (denom + eps())
    return Cd, Cl, drag, lift
end

###########
# Diagnostics over time
###########
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
Xp = mesh_p.nodes[1];  Yp = mesh_p.nodes[2]

Cd_hist = Float64[]; Cl_hist = Float64[]; cy_hist = Float64[]
speed_final = zeros(length(xs), length(ys))

for (idx, state) in enumerate(states)
    uωx = state[1:nu_x]
    uωy = state[2*nu_x+1:2*nu_x+nu_y]
    pω  = state[2*(nu_x + nu_y)+1:end]

    Ux = reshape(uωx, (length(xs), length(ys)))
    Uy = reshape(uωy, (length(xs), length(ys)))
    P  = reshape(pω, (length(Xp), length(Yp)))

    Cd, Cl, _, _ = instantaneous_force(Ux, Uy, P, xs, ys, Xp, Yp, μ, ρ, U_in, diameter, times[idx];
                                       center_x=center_x, center_y0=center_y0, radius=radius, A_osc=A_osc, ω_osc=ω_osc)
    push!(Cd_hist, Cd); push!(Cl_hist, Cl)
    cy = center_y0 + A_osc * cos(ω_osc * times[idx])
    push!(cy_hist, cy)

    if idx == length(states)
        speed_final = sqrt.(Ux.^2 .+ Uy.^2)
    end
end

Cd_mean = mean(Cd_hist)
Cd_rms = sqrt(mean((Cd_hist .- Cd_mean).^2))
Cl_mean = mean(Cl_hist)
Cl_rms = sqrt(mean((Cl_hist .- Cl_mean).^2))

println("Drag coefficient mean=$(Cd_mean), RMS=$(Cd_rms)")
println("Lift coefficient mean=$(Cl_mean), RMS=$(Cl_rms)")

###########
# Plots
###########
fig1 = Figure(size=(1200, 500))
ax_vort = Axis(fig1[1,1], xlabel="x", ylabel="y", title="Speed magnitude at final time")
hm_vort = heatmap!(ax_vort, xs, ys, speed_final; colormap=:plasma)
θ = range(0, 2π, length=200)
cyf = center_y0 + A_osc * cos(ω_osc * times[end])
lines!(ax_vort, center_x .+ radius .* cos.(θ), cyf .+ radius .* sin.(θ), color=:white, linewidth=2)
Colorbar(fig1[1,2], hm_vort, label="|u|")
save("oscillating_cylinder_speed.png", fig1)
println("Saved speed contours to oscillating_cylinder_speed.png")

fig2 = Figure(size=(1200, 500))
ax_drag = Axis(fig2[1,1], xlabel="time", ylabel="C_d", title="Drag/Lift coefficients vs time")
lines!(ax_drag, times, Cd_hist, label="C_d", color=:blue)
lines!(ax_drag, times, Cl_hist, label="C_l", color=:red, linestyle=:dash)
axislegend(ax_drag, position=:rb)

ax_phase = Axis(fig2[1,2], xlabel="c_y(t)", ylabel="C_d", title="Drag vs vertical displacement")
lines!(ax_phase, cy_hist, Cd_hist, color=:purple)
save("oscillating_cylinder_forces.png", fig2)
println("Saved force coefficient plots to oscillating_cylinder_forces.png")
