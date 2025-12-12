using Penguin
using CairoMakie
using LinearAlgebra
using LinearSolve

"""
Two-layer Couette–Poiseuille flow for the diphasic Stokes solver.

Domain: [0, Lx] × [0, H] with interface at y = d.
Bottom wall at y=0 with velocity U0, top wall at y=H with velocity UH.
Constant pressure gradient in x-direction: dp/dx = -G.
Velocity and shear continuity enforced at the interface.
"""

###########
# Grids
###########
nx, ny = 32, 32
Lx, H = 1.0, 1.0
x0, y0 = 0.0, 0.0
dx, dy = Lx/nx, H/ny

mesh_p  = Penguin.Mesh((nx, ny), (Lx, H), (x0, y0))
mesh_ux = Penguin.Mesh((nx, ny), (Lx, H), (x0 - 0.5*dx, y0))
mesh_uy = Penguin.Mesh((nx, ny), (Lx, H), (x0, y0 - 0.5*dy))

###########
# Capacities and operators
###########
d = H / 2  # interface location
body_lower = (x, y, _=0.0) -> y - d
body_upper = (x, y, _=0.0) -> d - y

cap_ux_a = Capacity(body_lower, mesh_ux; compute_centroids=false)
cap_uy_a = Capacity(body_lower, mesh_uy; compute_centroids=false)
cap_p_a  = Capacity(body_lower, mesh_p;  compute_centroids=false)
cap_ux_b = Capacity(body_upper, mesh_ux; compute_centroids=false)
cap_uy_b = Capacity(body_upper, mesh_uy; compute_centroids=false)
cap_p_b  = Capacity(body_upper, mesh_p;  compute_centroids=false)

op_ux_a = DiffusionOps(cap_ux_a); op_uy_a = DiffusionOps(cap_uy_a); op_p_a = DiffusionOps(cap_p_a)
op_ux_b = DiffusionOps(cap_ux_b); op_uy_b = DiffusionOps(cap_uy_b); op_p_b = DiffusionOps(cap_p_b)

###########
# Physical parameters
###########
U0 = 0.0    # bottom wall velocity  
UH = 1.0    # top wall velocity
G  = 1.0    # pressure gradient magnitude (dp/dx = -G)
μ_a, μ_b = 1.0, 10.0  # viscosities (phase 1 = lower, phase 2 = upper)
ρ_a, ρ_b = 1.0, 1.0

###########
# Boundary conditions
###########
bc_ux = BorderConditions(Dict(
    :bottom => Dirichlet((x, y)->U0),
    :top    => Dirichlet((x, y)->UH),
))

bc_uy = BorderConditions(Dict(
    :bottom => Dirichlet((x, y)->0.0),
    :top    => Dirichlet((x, y)->0.0),
    :left   => Dirichlet((x, y)->0.0),
    :right  => Dirichlet((x, y)->0.0),
))

bc_ux_a, bc_ux_b = bc_ux, bc_ux
bc_uy_a, bc_uy_b = bc_uy, bc_uy

pressure_gauges = (PinPressureGauge(), PinPressureGauge())

###########
# Sources and material
###########
fᵤ = (x, y, z=0.0) -> -G   # body force equivalent to pressure gradient dp/dx = -G
fₚ = (x, y, z=0.0) -> 0.0

fluid_a = Fluid((mesh_ux, mesh_uy),
                (cap_ux_a, cap_uy_a),
                (op_ux_a, op_uy_a),
                mesh_p,
                cap_p_a,
                op_p_a,
                μ_a, ρ_a, fᵤ, fₚ)

fluid_b = Fluid((mesh_ux, mesh_uy),
                (cap_ux_b, cap_uy_b),
                (op_ux_b, op_uy_b),
                mesh_p,
                cap_p_b,
                op_p_b,
                μ_b, ρ_b, fᵤ, fₚ)

###########
# Interface conditions
###########
ic_x = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))  # shear continuity, velocity continuity
ic_y = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0), FluxJump(-1.0, 1.0, 0.0))

###########
# Solve
###########
solver = StokesDiph(fluid_a, fluid_b, (bc_ux_a, bc_uy_a), (bc_ux_b, bc_uy_b), (ic_x, ic_y), pressure_gauges)
solve_StokesDiph!(solver; method=Base.:\)

println("Diphasic Couette–Poiseuille solved. Unknowns = ", length(solver.x))

###########
# Extract fields
###########
nu_x = prod(op_ux_a.size); nu_y = prod(op_uy_a.size)
np_a = prod(op_p_a.size); np_b = prod(op_p_b.size)
sum_nu = nu_x + nu_y
off_p1 = 2 * sum_nu
off_phase2 = off_p1 + np_a

u1ωx = solver.x[1:nu_x];         u1γx = solver.x[nu_x+1:2nu_x]
u1ωy = solver.x[2nu_x+1:2nu_x+nu_y]; u1γy = solver.x[2nu_x+nu_y+1:2nu_x+2nu_y]
p1   = solver.x[off_p1+1:off_p1+np_a]

u2ωx = solver.x[off_phase2+1:off_phase2+nu_x];           u2γx = solver.x[off_phase2+nu_x+1:off_phase2+2nu_x]
u2ωy = solver.x[off_phase2+2nu_x+1:off_phase2+2nu_x+nu_y]; u2γy = solver.x[off_phase2+2nu_x+nu_y+1:off_phase2+2nu_x+2nu_y]
p2   = solver.x[off_phase2+2*(nu_x+nu_y)+1:off_phase2+2*(nu_x+nu_y)+np_b]

xs = mesh_ux.nodes[1]; ys = mesh_ux.nodes[2]
xp = mesh_p.nodes[1];  yp = mesh_p.nodes[2]

LIu = LinearIndices((length(xs), length(ys)))
LIp = LinearIndices((length(xp), length(yp)))

Ux1 = reshape(u1ωx, (length(xs), length(ys)))
Ux2 = reshape(u2ωx, (length(xs), length(ys)))
Uy1 = reshape(u1ωy, (length(xs), length(ys)))
Uy2 = reshape(u2ωy, (length(xs), length(ys)))
P1  = reshape(p1,   (length(xp), length(yp)))
P2  = reshape(p2,   (length(xp), length(yp)))
Ux1g = reshape(u1γx, (length(xs), length(ys)))
Ux2g = reshape(u2γx, (length(xs), length(ys)))
Uy1g = reshape(u1γy, (length(xs), length(ys)))
Uy2g = reshape(u2γy, (length(xs), length(ys)))

###########
# Analytical Couette–Poiseuille solution (corrected)
###########
mu1, mu2 = μ_a, μ_b

function couette_poiseuille_constants(mu1, mu2, G, U0, UH, H, d)
    D = mu1 * (H - d) + mu2 * d
    term_couette = mu2 * (UH - U0) / D
    term_poiseuille = G * (mu1 * (H^2 - d^2) + mu2 * d^2) / (2 * mu1 * D)
    C1 = term_couette - term_poiseuille
    C3 = (mu1 / mu2) * C1
    C4 = UH - G * H^2 / (2 * mu2) - C3 * H
    return C1, C3, C4
end

const_C1, const_C3, const_C4 = couette_poiseuille_constants(mu1, mu2, G, U0, UH, H, d)

function u_exact(y)
    if y <= d
        return (G / (2 * mu1)) * y^2 + const_C1 * y + U0
    else
        return (G / (2 * mu2)) * y^2 + const_C3 * y + const_C4
    end
end

ux_target = similar(ys)
for (j, y) in enumerate(ys)
    ux_target[j] = u_exact(y)
end

###########
# Mid-channel profile
###########
icol = Int(cld(length(xs), 2))
ux_profile = similar(ys)
for (j, y) in enumerate(ys)
    ux_profile[j] = y <= d ? Ux1[icol, j] : Ux2[icol, j]
end

ux_profile = ux_profile[2:end-1]  
ux_target_plot = ux_target[2:end-1]
ys_profile = ys[2:end-1]

# Save the profile to cssv
using CSV
using DataFrames
df = DataFrame(y = ys_profile, u_numerical = ux_profile, u_analytical = ux_target_plot)
CSV.write("stokes_diph_couette_poiseuille_profile_$(nx).csv", df)

fig = Figure(resolution=(800, 600), fontsize=18)
ax1 = Axis(fig[1, 1], 
    xlabel = L"\text{Velocity } u_x",
    ylabel = "Height y", 
    title = "Couette–Poiseuille Flow: Mid-channel Profile",
    xminorticksvisible = true, 
    yminorticksvisible = true,
    xminorgridvisible = true,
    yminorgridvisible = true,
)

# Interface line
hlines!(ax1, [d], color=:gray, linestyle=:dashdot, linewidth=2, label="Interface")

# Analytical
lines!(ax1, ux_target_plot, ys_profile, color=:crimson, linewidth=3, label="Analytical")

# Numerical
scatter!(ax1, ux_profile, ys_profile, color=:dodgerblue, markersize=8, label="Numerical")

axislegend(ax1, position=:rb)

println("Profile Linf error (u_x vs analytical): ",
        maximum(abs, ux_profile .- ux_target_plot))

display(fig)
save("stokes_diph_couette_poiseuille_profile.png", fig)


# Based on the csv data, plot on the same graph the profiles for various grid resolutions
using CSV
using DataFrames
using CairoMakie

function plot_couette_poiseuille_profiles(filenames::Vector{String}, nx_values::Vector{Int})
    fig = Figure(resolution=(800, 600), fontsize=18)
    ax = Axis(fig[1, 1], 
        xlabel = L"\text{Velocity } u_x",
        ylabel = "Height y", 
        title = "Couette–Poiseuille Flow: Mid-channel Profiles (Zoomed)",
        xminorticksvisible = true, 
        yminorticksvisible = true,
        xminorgridvisible = true,
        yminorgridvisible = true,
    )

    # Interface line
    d = 0.5  # interface location
    hlines!(ax, [d], color=:gray, linestyle=:dashdot, linewidth=2, label="Interface")

    # Plot each profile
    for (filename, nx) in zip(filenames, nx_values)
        df = CSV.read(filename, DataFrame)
        ys = df.y
        ux_numerical = df.u_numerical
        scatter!(ax, ux_numerical, ys, label="Numerical (nx=$(nx))")
    end

    # Analytical solution
    function u_exact(y)
        if y <= d
            return (G / (2 * μ_a)) * y^2 + const_C1 * y + U0
        else
            return (G / (2 * μ_b)) * y^2 + const_C3 * y + const_C4
        end
    end

    ys_fine = range(0, H; length=500)
    ux_analytical = [u_exact(y) for y in ys_fine]
    lines!(ax, ux_analytical, ys_fine, color=:black, label="Analytical")

    axislegend(ax, position=:rb)
    # zoom in on the interface region
    xlims!(ax, -0.2, 1.2)
    ylims!(ax, d - 0.1, d + 0.1)
        display(fig)

    save("stokes_diph_couette_poiseuille_profiles_comparison_zoom.png", fig)
end
# Example usage
filenames = [
    "stokes_diph_couette_poiseuille_profile_16.csv",
    "stokes_diph_couette_poiseuille_profile_32.csv",
    "stokes_diph_couette_poiseuille_profile_64.csv",
    "stokes_diph_couette_poiseuille_profile_128.csv",
]
nx_values = [16, 32, 64, 128]
plot_couette_poiseuille_profiles(filenames, nx_values)

# Reonstruct velocity field plots heatmaps + add interface line + add profile plot on the same heatmap
U_global = zeros(length(xs), length(ys))
for i in 1:length(xs)
    for j in 1:length(ys)
        if ys[j] <= d
            U_global[i, j] = Ux1[i, j]
        else
            U_global[i, j] = Ux2[i, j]
        end
    end
end

# replace y=ymax by NaN
for i in 1:length(xs)
    U_global[i, end] = NaN
end

fig2 = Figure(resolution=(900, 500))
ax2 = Axis(fig2[1, 1], xlabel="x", ylabel="y", title="U velocity field")
hm = heatmap!(ax2, xs, ys, U_global; colormap=:viridis)

# Interface line
hlines!(ax2, [d], color=:red, linestyle=:dashdot, linewidth=2, label="Interface")
Colorbar(fig2[1, 2], hm)
ylims!(ax2, 0.0, H)
xlims!(ax2, 0.0, Lx)

# Add arrows 
for j in 1:4:length(ys)
    for i in 1:4:length(xs)
        if ys[j] <= d
            u_val = Ux1[i, j]/10
            v_val = Uy1[i, j]/10
        else
            u_val = Ux2[i, j]/10
            v_val = Uy2[i, j]/10
        end
        arrow_length = sqrt(u_val^2 + v_val^2) 
        if arrow_length > 1e-8
            quiver!(ax2, [xs[i]], [ys[j]], [u_val], [v_val]; color=:white, linewidth=1.5)
        end
    end
end

axislegend(ax2, position=:rt)
display(fig2)
save("stokes_diph_couette_poiseuille_velocity_field.png", fig2)