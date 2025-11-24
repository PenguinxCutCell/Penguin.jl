# Navier–Stokes (incompressible)

This section documents the prototype incompressible Navier–Stokes solver built on the same staggered velocity / collocated pressure layout as the Stokes solvers. The current implementation supports 1D and 2D problems and exposes both unsteady and steady solves driven either by a Picard iteration or by a full Newton update.

## Equations & Unknown Ordering

- State layout uses face-based velocities and cell-centered pressure:
  - `u = (u₁, u₂)` lives on component-aligned face grids (`mesh_ux`, `mesh_uy`).
  - `p` resides on the cell-centered pressure grid (`mesh_p`).
- Global unknown ordering reused across assembly and solves:
  `x = [uω₁, uγ₁, uω₂, uγ₂, pω]`, where `ω` denotes bulk (staggered face) DoFs and `γ` the interface/cut DoFs.
- Governing equations in conservative form:
  - Face-based momentum: `∂ₜ u + ∇·(u ⊗ u) + ∇p = ν Δu + f`.
  - Cell-centered continuity: `∇·u = 0`.

## Discretization Details

### Operators

- Base operators come from the capacity build: centered differences/averages `D_{·}^{±}`, `S_{·}^{±}`, face-area operators `A`, interface coupling `Hᵀ`, and staggered weights `Wᵝ`.
- Pressure gradient and divergence reuse the Stokes operators (`G`, `H`) applied to the staggered layout.

### Flux-form Convection

- For each velocity component α, the advection operator `C_α(uω)` collects fluxes using symmetric averages and centered differences:
  - Diagonal flux weights: `S⁻ A uω`.
  - Directional differences: `D⁺` along α and cross directions.
- Interface contribution `K_α(·)` is assembled from `S⁺ Hᵀ` and added symmetrically: the resulting discrete convection is skew-symmetric on the bulk subspace and yields zero on constant fields (free-stream preservation).

### Viscous & Pressure Blocks

- Viscous terms reuse Stokes operators: `-(I(μ⁻¹) Gᵀ Wᵝ G)` on bulk faces and `-(I(μ⁻¹) Gᵀ Wᵝ H)` across interface ties.
- Pressure gradient rows are split per component to match the staggered layout; continuity rows assemble `-(Gᵀ + Hᵀ)` against bulk/interface velocities and `Hᵀ` for the interface tie variables.

## Time Integration

- Implicit θ-scheme for viscous and pressure terms (`θ = 1` gives backward Euler, `θ = 1/2` Crank–Nicolson).
- Convection advanced explicitly with Adams–Bashforth 2: `1.5·conv(uⁿ) - 0.5·conv(uⁿ⁻¹)`.
- Practitioners should observe the usual CFL restrictions; for high-Re steady states, prefer the steady solvers.

## Steady Nonlinear Solvers

- Picard: lag the advecting velocity in `C(u)`; supports relaxation to improve robustness. Suitable as a first stage.
- Newton: assembles the full Jacobian of the residual and solves for the increment; converges faster near the solution but benefits from a good initial guess (typically the Picard result).

## Boundary Conditions

- **Dirichlet:** enforced strongly on both momentum and tie rows (and respected inside the Newton residual/Jacobian).
- **Neumann:** implemented as one-sided derivative constraints on the staggered faces.
- **Periodic:** pairs opposite faces for both bulk and interface unknowns.
- **Pressure gauge:** no pressure Dirichlet values are imposed. Choose either a pin gauge (fix one cell) or a mean gauge to enforce `∫p dV = 0`. Provide the gauge when constructing the solver.

### Outflow Boundaries

- Current implementation treats velocity outlets as traction-free (the natural “do-nothing” boundary from the weak form).
- `Outflow()` keeps the velocity natural; the global pressure reference is handled solely by the chosen gauge (pin or mean).
- Roadmap: extend the outlet handling to support arbitrary traction `σ·n = g`, including non-zero normal/tangential components; this will subsume the pressure reference and allow proper stress specification.

## Usage Snippets

Construct a fluid/solver exactly as in the Stokes documentation (see `docs/src/blocks/stokes.md`). For 1D, pass a single velocity mesh/capacity/operator tuple; for 2D, pass the pair `(mesh_ux, mesh_uy)` etc.

```julia
using Penguin

# ... build meshes, capacities, operators, construct Fluid and boundary conditions ...

pressure_gauge = MeanPressureGauge()  # or PinPressureGauge()
solver = NavierStokesMono(fluid, (bc_ux, bc_uy), pressure_gauge, bc_cut; x0=zeros(Ntot))

# Unsteady solve
times, histories = solve_NavierStokesMono_unsteady!(solver; Δt=Δt, T_end=Tend, scheme=:CN)

# Steady Picard (robust)
x_picard, it_pic, res_pic = solve_NavierStokesMono_steady!(solver; tol=1e-8, maxiter=40, relaxation=0.7, nlsolve_method=:picard)

# Newton polish
x_newton, it_new, res_new = solve_NavierStokesMono_steady!(solver; tol=1e-10, maxiter=20, nlsolve_method=:newton)
```

## Parameters & Notes

- μ (mu) and ρ (rho) are passed via the `Fluid` struct and used consistently in viscous and convective terms.
- `relaxation` controls Picard under-relaxation (0–1).
- `tol`, `maxiter` control nonlinear stopping criteria; monitor both the velocity increment and mass-residual to ensure convergence.

## Validation References

- **Poiseuille channel (1D/2D):** check parabolic velocity profile under a uniform body-force driving term; pressure is determined up to the gauge.
- **Lid-driven cavity:** steady benchmark for Picard/Newton iterations.
- **Taylor–Green vortex:** unsteady regression for time integration accuracy and kinetic energy decay.

## Visualization & Examples

- 1D steady: `examples/1D/NavierStokes/poiseuille_1d.jl`.
- 2D steady: `examples/2D/NavierStokes/lid_driven_cavity_steady.jl` (Picard and Newton).
- 2D unsteady: `examples/2D/NavierStokes/flow_around_circle_2d.jl` (goes with the animation below).

![Streamlines around a circular obstacle](./assets/navierstokes2d_streamlines.gif)

## Limitations & Future Work

- Currently limited to 1D and 2D; 3D support is on the roadmap.
- Newton path lacks line search/preconditioning; larger problems may require custom solvers via `LinearSolve`.
- General traction outflow (`σ·n = g`) and variable-density/viscosity models remain to be implemented.
