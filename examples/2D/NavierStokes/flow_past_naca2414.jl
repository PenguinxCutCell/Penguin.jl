using Penguin
using CairoMakie
using LinearAlgebra
using Printf

"""
2D Navier–Stokes flow past a cambered NACA2414 airfoil embedded in a rectangular
domain. The Reynolds number is 10000 based on the chord and freestream speed.
The case is inspired by the Gerris example using an embedded boundary.
"""

###########
# Geometry and NACA definition
###########
Lx, Ly = 16.0, 16.0
x0, y0 = -Lx / 2, -Ly / 2
nx, ny = 128, 96

chord = 1.0
m, p, t = 0.02, 0.4, 0.14 # NACA 2414 parameters
p_theta = 6 * π / 180      # 6° incidence
quarter_chord = (0.25 * chord, 0.0)

function naca2414_levelset(; chord=1.0, m=0.02, p=0.4, t=0.14,
                           incidence=0.0, pivot=quarter_chord, translation=(0.0, 0.0))
    thickness(xi) = begin
        ξ = clamp(xi, 0.0, 1.0)
        5 * t * chord * (
            0.2969 * sqrt(ξ) - 0.1260 * ξ - 0.3516 * ξ^2 + 0.2843 * ξ^3 - 0.1036 * ξ^4
        )
    end

    return (x, y, _=0.0) -> begin
        # Rotate world coordinates back to the airfoil frame (clockwise rotation)
        xr = (x - translation[1] - pivot[1]) * cos(incidence) + (y - translation[2] - pivot[2]) * sin(incidence) + pivot[1]
        yr = -(x - translation[1] - pivot[1]) * sin(incidence) + (y - translation[2] - pivot[2]) * cos(incidence) + pivot[2]

        ξ = xr / chord
        if ξ < 0.0 || ξ > 1.0
            return -1.0 # outside chord: fluid
        end

        if ξ < p
            y_c = m / p^2 * (2p * ξ - ξ^2) * chord
            dyc_dx = 2m / p^2 * (p - ξ)
        else
            y_c = m / (1 - p)^2 * ((1 - 2p) + 2p * ξ - ξ^2) * chord
            dyc_dx = 2m / (1 - p)^2 * (p - ξ)
        end

        y_t = thickness(ξ)
        θ = atan(dyc_dx)
        sθ, cθ = sin(θ), cos(θ)
        x_c = xr

        η = -(xr - x_c) * sθ + (yr - y_c) * cθ # signed normal offset from camber line
        return y_t - abs(η) # positive inside the airfoil, negative in the fluid
    end
end

airfoil_body = naca2414_levelset(; chord=chord, m=m, p=p, t=t, incidence=p_theta, translation=(0.0, 0.0))

###########
# Meshes and capacities
###########
mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

capacity_ux = Capacity(airfoil_body, mesh_ux; compute_centroids=true)
capacity_uy = Capacity(airfoil_body, mesh_uy; compute_centroids=true)
capacity_p  = Capacity(airfoil_body, mesh_p; compute_centroids=true)

operator_ux = DiffusionOps(capacity_ux)
operator_uy = DiffusionOps(capacity_uy)
operator_p  = DiffusionOps(capacity_p)

###########
# Boundary conditions
###########
U_inf = 1.0
ux_left   = Dirichlet((x, y, t=0.0) -> U_inf)
ux_right  = Outflow(0.0)
ux_bottom = Outflow(0.0)
ux_top    = Outflow(0.0)

uy_zero = Dirichlet((x, y, t=0.0) -> 0.0)
uy_outflow = Outflow(0.0)

bc_ux = BorderConditions(Dict(:left=>ux_left, :right=>ux_right, :bottom=>ux_bottom, :top=>ux_top))
bc_uy = BorderConditions(Dict(:left=>uy_zero, :right=>uy_outflow, :bottom=>uy_zero, :top=>uy_zero))

pressure_gauge = PinPressureGauge()
interface_bc = Dirichlet(0.0) # no-slip on the airfoil

###########
# Physics and solver setup
###########
ρ = 1.0
Re = 10_000.0
μ = ρ * U_inf * chord / Re

fᵤ = (x, y, z=0.0, t=0.0) -> 0.0
fₚ = (x, y, z=0.0, t=0.0) -> 0.0

fluid = Fluid((mesh_ux, mesh_uy),
              (capacity_ux, capacity_uy),
              (operator_ux, operator_uy),
              mesh_p,
              capacity_p,
              operator_p,
              μ, ρ, fᵤ, fₚ)

nu_x = prod(operator_ux.size)
nu_y = prod(operator_uy.size)
np = prod(operator_p.size)
fluid_mask_x = diag(capacity_ux.W[1])
fluid_mask_y = diag(capacity_uy.W[2])
uωx0 = U_inf .* fluid_mask_x
uγx0 = copy(uωx0)
uωy0 = zeros(Float64, nu_y)
uγy0 = zeros(Float64, nu_y)
x0_vec = vcat(uωx0, uγx0, uωy0, uγy0, zeros(Float64, np))

solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, interface_bc; x0=x0_vec)

Δt = 0.001
T_end = 4.0

println(@sprintf("Running NACA2414 case at Re = %.1f (μ = %.4g)", Re, μ))
times, states = solve_NavierStokesMono_unsteady!(solver; Δt=Δt, T_end=T_end, scheme=:CN)
println("Stored $(length(states)) states, final time = $(times[end])")

###########
# Forces
###########
force_diag = compute_navierstokes_force_diagnostics(solver)
coeffs = drag_lift_coefficients(force_diag; ρ=ρ, U_ref=U_inf, length_ref=chord, acting_on=:body)
body_force = navierstokes_reaction_force_components(force_diag; acting_on=:body)
println(@sprintf("Cd = %.4f, Cl = %.4f", coeffs.Cd, coeffs.Cl))
println(@sprintf("Fx = %.4f, Fy = %.4f (acting on body)", body_force[1], body_force[2]))

###########
# Visualization of final fields
###########
xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
Xp = mesh_p.nodes[1];  Yp = mesh_p.nodes[2]

uωx = solver.x[1:nu_x]
uωy = solver.x[2nu_x+1:2nu_x+nu_y]
pω  = solver.x[2*(nu_x + nu_y)+1:end]

Ux = reshape(uωx, (length(xs), length(ys)))
Uy = reshape(uωy, (length(xs), length(ys)))
P  = reshape(pω, (length(Xp), length(Yp)))

speed = sqrt.(Ux.^2 .+ Uy.^2)
nearest_index(vec, val) = clamp(argmin(abs.(vec .- val)), 1, length(vec))
velocity_field(x, y) = Point2f(Ux[nearest_index(xs, x), nearest_index(ys, y)],
                               Uy[nearest_index(xs, x), nearest_index(ys, y)])

fig = Figure(resolution=(1200, 700))

ax_speed = Axis(fig[1,1], xlabel="x", ylabel="y", title="Speed magnitude")
hm_speed = heatmap!(ax_speed, xs, ys, speed; colormap=:plasma)
contour!(ax_speed, xs, ys, [airfoil_body(x, y) for x in xs, y in ys]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig[1,2], hm_speed)

ax_pressure = Axis(fig[1,3], xlabel="x", ylabel="y", title="Pressure field")
hm_pressure = heatmap!(ax_pressure, Xp, Yp, P; colormap=:balance)
contour!(ax_pressure, xs, ys, [airfoil_body(x, y) for x in xs, y in ys]; levels=[0.0], color=:white, linewidth=2)
Colorbar(fig[1,4], hm_pressure)

ax_stream = Axis(fig[2,1], xlabel="x", ylabel="y", title="Velocity streamlines")
streamplot!(ax_stream, velocity_field, xs[1]..xs[end], ys[1]..ys[end]; colormap=:thermal)
contour!(ax_stream, xs, ys, [airfoil_body(x, y) for x in xs, y in ys]; levels=[0.0], color=:red, linewidth=2)

ax_contours = Axis(fig[2,3], xlabel="x", ylabel="y", title="Velocity magnitude contours")
contour!(ax_contours, xs, ys, speed; levels=12, color=:navy)
contour!(ax_contours, xs, ys, [airfoil_body(x, y) for x in xs, y in ys]; levels=[0.0], color=:black, linewidth=2)

save("navierstokes2d_naca2414.png", fig)
display(fig)
