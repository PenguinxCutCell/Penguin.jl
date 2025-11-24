# Taylor–Green Vortex (Unsteady Stokes) – Convergence

This benchmark exercises the prototype unsteady Stokes solver (`StokesMono`) on the periodic square domain $[0,2\pi]^2$ using the manufactured Taylor–Green vortex velocity field
$$
 u(x,y,t) =  \sin(kx)\cos(ky) e^{-2 \nu k^2 t}, \qquad
 v(x,y,t) = -\cos(kx)\sin(ky) e^{-2 \nu k^2 t},
$$
with $k=1$, density $\rho$, kinematic viscosity $\nu=\mu/\rho$, and final time `t_end = 0.1`.

Because the current solver omits the nonlinear advection term $(u\cdot\nabla)u$, the consistent Stokes pressure for this manufactured velocity is constant (set to 0). The script therefore reports:

- Volume–integrated weighted L2 errors for `u` and `v` (second-order decay expected)
- Mean-removed pressure fluctuation error (should diminish with refinement but does not represent physical Taylor–Green pressure)

## Running the benchmark

Activate the project (for exact dependencies) and run:

```julia
julia --project benchmark/TaylorGreen.jl
```

## Output artifacts

| File | Description |
| ---- | ----------- |
| `taylor_green_convergence.csv` | Tabulated mesh size `h` vs errors (`error_u`, `error_v`, `error_p`). |
| `taylor_green_convergence.png` | Log–log velocity error convergence with reference $O(h^2)$ slope. |
| `taylor_green_highest_resolution_fields.png` | Heatmaps of $u_x$, $u_y$, and pressure at the finest grid. |


## Notes

To compare against the classical Navier–Stokes Taylor–Green pressure
\[ p_{TG}(x,y,t) = -\tfrac{\rho}{4}\big( \cos(2kx) + \cos(2ky) \big) e^{-4\nu k^2 t}, \]
you must either (a) add the convective term to the solver or (b) add a manufactured body force $f = -\rho (u\cdot\nabla)u$ so that $(u, p_{TG})$ also solves the *forced* Stokes equations. The present script leaves `f ≡ 0` and uses the constant pressure reference.

