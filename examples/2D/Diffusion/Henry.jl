using Penguin
using IterativeSolvers
using CairoMakie

### 2D Test Case : Diphasic Unsteady Diffusion Equation with a Disk
# Define the mesh
nx, ny = 32, 32
lx, ly = 8., 8.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/4, (lx/2, ly/2)
circle = (x,y,_=0)->sqrt((x-center[1])^2 + (y-center[2])^2) - radius
circle_c = (x,y,_=0)->-(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# Define the capacity
capacity = Capacity(circle, mesh)
capacity_c = Capacity(circle_c, mesh)

# Define the operators
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}())

He = 1.0  # Henry's law coefficient
Dg,Dl = 1.0, 1.0
ic = InterfaceConditions(ScalarJump(He, 1.0, 0.0), FluxJump(Dg, Dl, 0.0))

# Define the source term
f1 = (x,y,z,t)->0.0
f2 = (x,y,z,t)->0.0

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, (x,y,z)->Dg)
Fluide_2 = Phase(capacity_c, operator_c, f2, (x,y,z)->Dl)

# Initial condition
u0ₒ1 = ones((nx+1)*(ny+1))
u0ᵧ1 = ones((nx+1)*(ny+1))
u0ₒ2 = zeros((nx+1)*(ny+1))
u0ᵧ2 = zeros((nx+1)*(ny+1))
u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Define the solver
Δt = 0.5*min(lx/nx, ly/ny)^2
Tend = 1.0
solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "BE")

# Solve the problem
solve_DiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, Δt, Tend, bc_b, ic, "BE"; method=Base.:\)

# Analytical solution
using QuadGK
using SpecialFunctions

R0 = radius
cg0, cl0 = 1.0, 0.0

D = sqrt(Dg/Dl)

function Phi(u)
    term1 = Dg*sqrt(Dl)*besselj1(u*R0)*bessely0(D*u*R0)
    term2 = He*Dl*sqrt(Dg)*besselj0(u*R0)*bessely1(D*u*R0)
    return term1 - term2
end

function Psi(u)
    term1 = Dg*sqrt(Dl)*besselj1(u*R0)*besselj0(D*u*R0)
    term2 = He*Dl*sqrt(Dg)*besselj0(u*R0)*besselj1(D*u*R0)
    return term1 - term2
end

function cg_integrand(u, x, y, t)
    r = sqrt((x-center[1])^2 + (y-center[2])^2)
    Φu = Phi(u)
    Ψu = Psi(u)
    denom = u^2*(Φu^2 + Ψu^2)
    num   = exp(-Dg*u^2*t)*besselj0(u*r)*besselj1(u*R0)
    return iszero(denom) ? 0.0 : num/denom
end

function cl_integrand(u, x, y, t)
    r = sqrt((x-center[1])^2 + (y-center[2])^2)
    Φu = Phi(u)
    Ψu = Psi(u)
    denom = u*(Φu^2 + Ψu^2)
    term1 = besselj0(D*u*r)*Φu
    term2 = bessely0(D*u*r)*Ψu
    num   = exp(-Dg*u^2*t)*besselj1(u*R0)*(term1 - term2)
    return iszero(denom) ? 0.0 : num/denom
end


cg_prefactor() = (4*cg0*Dg*Dl^2*He)/(π^2*R0)
cl_prefactor() = (2*cg0*Dg*sqrt(Dl)*He)/π

# ---- interfacial concentrations ----
function interfacial_concentrations(t; atol=1e-6, rtol=1e-6, Ufac=5.0)
    @assert t > 0 "Need t>0 (Umax uses 1/sqrt(t))."
    Umax = Ufac / sqrt(Dg*t)

    # cg(R0-,t): substitute r=R0 in gas integrand
    integrand_cg(u) = begin
    Φu = Phi(u); Ψu = Psi(u)
    denom = u^2*(Φu^2 + Ψu^2)
    denom == 0.0 && return 0.0
    exp(-Dg*u^2*t) * besselj0(u*R0) * besselj1(u*R0) / denom
    end

    # cl(R0+,t): substitute r=R0 in liquid integrand
    integrand_cl(u) = begin
    Φu = Phi(u); Ψu = Psi(u)
    denom = u*(Φu^2 + Ψu^2)
    denom == 0.0 && return 0.0
    contrib = besselj0(D*u*R0)*Φu - bessely0(D*u*R0)*Ψu
    exp(-Dg*u^2*t) * besselj1(u*R0) * contrib / denom
    end

    Ig, _ = quadgk(integrand_cg, 0.0, Umax; atol=atol, rtol=rtol)
    Il, _ = quadgk(integrand_cl, 0.0, Umax; atol=atol, rtol=rtol)

    cgR = cg_prefactor() * Ig
    clR = cl_prefactor() * Il

    return (cgR=cgR, clR=clR, mismatch=cgR - He*clR, Umax=Umax)
end

function compute_cg(x_values, y_values, t_values)
    prefactor = (4*cg0*Dg*Dl*Dl*He)/(π^2*R0)
    cg_results = Array{Float64}(undef, length(t_values), length(y_values), length(x_values))
    for (i, t) in pairs(t_values)
        Umax = 5.0/sqrt(Dg*t)
        for (j, y) in pairs(y_values)
            for (k, x) in pairs(x_values)
                val, _ = quadgk(u->cg_integrand(u, x, y, t), 0, Umax; atol=1e-6, rtol=1e-6)
                cg_results[i, j, k] = prefactor*val
            end
        end
    end
    return cg_results
end

function compute_cl(x_values, y_values, t_values)
    prefactor = (2*cg0*Dg*sqrt(Dl)*He)/π
    cl_results = Array{Float64}(undef, length(t_values), length(y_values), length(x_values))
    for (i, t) in pairs(t_values)
        Umax = 5.0/sqrt(Dg*t)
        for (j, y) in pairs(y_values)
            for (k, x) in pairs(x_values)
                val, _ = quadgk(u->cl_integrand(u, x, y, t), 0, Umax; atol=1e-6, rtol=1e-6)
                cl_results[i, j, k] = prefactor*val
            end
        end
    end
    return cl_results
end

function interfacial_flux_gas(; t=Tend)
    R  = radius
    Ag = (4 * cg0 * Dg * Dl^2 * He) / (π^2 * R)

    Umax = 5.0 / sqrt(Dg * t)   # same spirit as your solution cutoff

    integrand(u) = begin
        Φu = Phi(u)
        Ψu = Psi(u)
        denom = u * (Φu^2 + Ψu^2)
        denom == 0.0 && return 0.0
        exp(-Dg * u^2 * t) * besselj1(u*R)^2 / denom
    end

    I, _ = quadgk(integrand, 0, Umax; atol=1e-6, rtol=1e-6)
    return Dg * Ag * I   # jΓ(t) = -Dg ∂r cg(R,t)
end

# optional:
interfacial_total_rate(; t=Tend) =
    2π * radius * interfacial_flux_gas(; t=t)


n = prod(operator.size)
Tω = @view solver.states[end][1:n]
Tγ = @view solver.states[end][n+1:2n]
Tω_c = @view solver.states[end][2n+1:3n]
Tγ_c = @view solver.states[end][3n+1:4n]

# Discrete volume interface contribution : volume integrated normal heat flux on the interface : ∫_{Γ} q_n ds
Q = operator.H' * operator.Wꜝ * (operator.G * Tω + operator.H * Tγ)
Q_c = operator_c.H' * operator_c.Wꜝ * (operator_c.G * Tω_c + operator_c.H * Tγ_c)

# Extract interface lengths
Γ = diag(capacity.Γ)
Γ_c = diag(capacity_c.Γ)

# Compute mean interfacial fluxes : ∫_{Γ} q_n ds. It is already integrated over the interface length
q_nds = Q 
q_nds_c = Q_c

# Divide by total interface length to get mean flux
q_n_mean = sum(q_nds[.!isnan.(q_nds)])
q_n_mean_c = sum(q_nds_c[.!isnan.(q_nds_c)])
q_n_mean /= sum(Γ)
q_n_mean_c /= sum(Γ_c)

# Multiply by diffusivity to get flux
q_n_mean *= -Dg
q_n_mean_c *= -Dl

# compute analytical mean interfacial flux
q_n_mean_analytical = interfacial_flux_gas()
q_n_mean_analytical_c = -q_n_mean_analytical  # by flux continuity

println("Mean interfacial flux in inner (gas) phase: $q_n_mean")
println("Mean interfacial flux in outer (liquid) phase: $q_n_mean_c")
println("Analytical interfacial flux in gas phase: $q_n_mean_analytical")
println("Analytical interfacial flux in liquid phase: $q_n_mean_analytical_c")

# Compute mean concentration at the interface
c_interface = 0.0
c_interface_c = 0.0
for i in 1:length(Tγ)
    if !isnan(Γ[i]) && Γ[i] > 0.0
        global c_interface += Tγ[i] * Γ[i]
    end
end
for i in 1:length(Tγ_c)
    if !isnan(Γ_c[i]) && Γ_c[i] > 0.0
        global c_interface_c += Tγ_c[i] * Γ_c[i]
    end
end
c_interface /= sum(Γ[.!isnan.(Γ)])
c_interface_c /= sum(Γ_c[.!isnan.(Γ_c)])

# compute analytical mean interfacial concentration
res = interfacial_concentrations(Tend)

println("Mean interfacial concentration in inner (gas) phase: $c_interface")
println("Mean interfacial concentration in outer (liquid) phase: $c_interface_c")
println("Analytical interfacial concentration in gas phase: $(res.cgR)")
println("Analytical interfacial concentration in liquid phase: $(res.clR)")

# compute mass transfer coefficient
c_ref = 0.0  # reference concentration in liquid far from interface
h = q_n_mean / (c_interface - c_ref)
h_c = q_n_mean_c / (c_interface_c - c_ref)

# compute analytical mass transfer coefficient
h_analytical = q_n_mean_analytical / (res.cgR - c_ref)
h_c_analytical = q_n_mean_analytical_c / (res.clR - c_ref)

println("Mass transfer coefficient in inner (gas) phase: $h")
println("Mass transfer coefficient in outer (liquid) phase: $h_c")
println("Analytical mass transfer coefficient in inner (gas) phase: $h_analytical")
println("Analytical mass transfer coefficient in outer (liquid) phase: $h_c_analytical")

# Characteristic length
L = 2 * radius

# Compute Sherwood number
Sh = h * L / Dg
Sh_c = h_c * L / Dl

# compute analytical Sherwood number
Sh_analytical = h_analytical * L / Dg
Sh_c_analytical = h_c_analytical * L / Dl

println("Sherwood number in inner (gas) phase: $Sh")
println("Sherwood number in outer (liquid) phase: $Sh_c")
println("Analytical Sherwood number in inner (gas) phase: $Sh_analytical")
println("Analytical Sherwood number in outer (liquid) phase: $Sh_c_analytical")
