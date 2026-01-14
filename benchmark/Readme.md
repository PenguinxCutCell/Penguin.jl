# 📌 README — MMS scaling and effective domain (IMPORTANT)

## Context

In the current implementation of **Penguin**, the numerical solution and error norms are evaluated on
`capacity.C_ω` (cell centroids / effective DOFs).

In practice, for a Cartesian mesh with `nx` cells over a domain of length `L`:

```julia
Δx = L / nx
maximum(mesh.centers[1]) ≈ L - Δx
```

This means that **the solver effectively operates on a truncated domain**:

$
\Omega_{\text{eff}} = [x_0,; x_0 + L - \Delta x]
$

(and similarly in higher dimensions).

This is **not a bug**, but a consequence of where the unknowns and boundary conditions are applied.

---

## Consequence for MMS (Method of Manufactured Solutions)

When performing MMS-based convergence studies:

> ❗ **The analytical solution must be constructed on the effective domain**
> ($[x_0,; x_0 + L - \Delta x]$),
> **not** on the nominal domain ($[x_0,; x_0 + L]$).

Otherwise, a geometric mismatch of order (O(h)) appears and dominates the error, resulting in **first-order convergence**, even if the scheme itself is second-order.

---

## Correct MMS scaling (1D)

### Effective length

```julia
L_eff = L - Δx
```

### Homogeneous Dirichlet MMS (works, order 2)

```julia
u_exact(x) = sin(π * (x - Δx) / L_eff)
λ = (π / L_eff)^2
f(x) = λ * u_exact(x)
```

### Non-homogeneous Dirichlet example

```julia
u_exact(x) = u_left + (u_right - u_left) * (x - Δx) / L_eff
```

---

## Correct MMS scaling (2D)

### Effective lengths

```julia
Lx_eff = Lx - Δx
Ly_eff = Ly - Δy
```

### Homogeneous MMS (reference case, order 2)

```julia
u_exact(x,y) =
    sin(π * (x - Δx) / Lx_eff) *
    sin(π * (y - Δy) / Ly_eff)

f(x,y) =
    ( (π/Lx_eff)^2 + (π/Ly_eff)^2 ) * u_exact(x,y)
```

This MMS has:

* homogeneous Dirichlet BCs,
* no corner incompatibilities,
* clean second-order convergence.

---

## What was observed

* Using a MMS defined on ([0,L]) leads to **order 1 convergence**.
* Rebuilding the MMS on ([0,L-\Delta]) (with the appropriate shift) restores **order 2**.
* The diffusion operator itself is second-order; the issue was purely **geometric/scaling-related**.

---

## Takeaway

* The **effective computational domain** is defined by `capacity.C_ω`, not by the nominal mesh bounds.
* MMS **must be aligned with the effective domain**.
* If a standard MMS on ([0,L]) gives order 1, **check the scaling before debugging the scheme**.

---

Until then, **use the `L - Δ` scaling for all MMS and validation tests**.

