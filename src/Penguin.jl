module Penguin

using SparseArrays
using StaticArrays
using CartesianGeometry
using ImplicitIntegration
using ImplicitCutIntegration
using LinearAlgebra
using IterativeSolvers
using CairoMakie
using LsqFit
using SpecialFunctions
using Roots
using Interpolations
using FFTW
using DSP
using Colors
using Statistics
using LinearSolve
using FrontCutTracking
using VTKOutputs

# Write your package code here.

include("mesh.jl")
export Dimension, Mesh, nC, SpaceTimeMesh

include("capacity.jl")
export Capacity

include("operators.jl")
export AbstractOperators, ẟ_m, δ_p, Σ_m, Σ_p, I
export ∇, ∇₋
export DiffusionOps, ConvectionOps

include("boundary.jl")
export AbstractBoundary, Dirichlet, Neumann, Robin, Periodic, Symmetry, Outflow, Traction
export AbstractInterfaceBC, ScalarJump, FluxJump, BorderConditions, InterfaceConditions
export GibbsThomson

include("phase.jl")
export Phase, Fluid

include("utils.jl")
export initialize_temperature_uniform!, initialize_temperature_square!, initialize_temperature_circle!, initialize_temperature_function!
export initialize_rotating_velocity_field, initialize_radial_velocity_field, initialize_poiseuille_velocity_field
export nusselt_profile, volume_redefinition!

include("interpolation.jl")
export lin_interpol, quad_interpol, cubic_interpol

include("solver.jl")
export TimeType, PhaseType, EquationType
export Solver, solve_system!
export build_I_bc, build_I_D, build_I_D_harmonic, build_I_D_arithmetic, build_source, build_g_g
export BC_border_mono!, BC_border_diph!
export cfl_restriction
export remove_zero_rows_cols!, remove_zero_rows_cols_separate!

include("solver/diffusion.jl")
export DiffusionSteadyMono, solve_DiffusionSteadyMono!
export DiffusionSteadyMonoVariable
export DiffusionSteadyDiph, solve_DiffusionSteadyDiph!
export DiffusionUnsteadyMono, solve_DiffusionUnsteadyMono!
export DiffusionUnsteadyDiph, solve_DiffusionUnsteadyDiph!

include("solver/advectiondiffusion.jl")
export AdvectionDiffusionSteadyMono, solve_AdvectionDiffusionSteadyMono!
export AdvectionDiffusionSteadyDiph, solve_AdvectionDiffusionSteadyDiph!
export AdvectionDiffusionUnsteadyMono, solve_AdvectionDiffusionUnsteadyMono!
export AdvectionDiffusionUnsteadyDiph, solve_AdvectionDiffusionUnsteadyDiph!

include("solver/darcy.jl")
export DarcyFlow, solve_DarcyFlow!, solve_darcy_velocity
export DarcyFlowUnsteady, solve_DarcyFlowUnsteady!

include("prescribedmotionsolver/diffusion.jl")
export MovingDiffusionUnsteadyMono, solve_MovingDiffusionUnsteadyMono!, solve_MovingDiffusionUnsteadyMono_cfl!
export MovingDiffusionUnsteadyDiph, solve_MovingDiffusionUnsteadyDiph!
export A_mono_unstead_diff_moving, b_mono_unstead_diff_moving

include("prescribedmotionsolver/advectiondiffusion.jl")
export MovingAdvDiffusionUnsteadyMono, solve_MovingAdvDiffusionUnsteadyMono!
export MovingAdvDiffusionUnsteadyDiph, solve_MovingAdvDiffusionUnsteadyDiph!

include("liquidmotionsolver/height_tracking.jl")

include("liquidmotionsolver/diffusion.jl")
export MovingLiquidDiffusionUnsteadyMono, solve_MovingLiquidDiffusionUnsteadyMono!
export solve_MovingLiquidDiffusionUnsteadyMono_coupledNewton!
export MovingLiquidDiffusionUnsteadyDiph, solve_MovingLiquidDiffusionUnsteadyDiph!

include("liquidmotionsolver/diffusion_coupled.jl")
export solve_MovingLiquidDiffusionUnsteadyMono_coupledNewton!

include("liquidmotionsolver/diffusion2d.jl")
export solve_MovingLiquidDiffusionUnsteadyMono2D!
export MovingLiquidDiffusionUnsteadyDiph2D, solve_MovingLiquidDiffusionUnsteadyDiph2D!

include("liquidmotionsolver/diffusion3d.jl")
export solve_MovingLiquidDiffusionUnsteadyMono3D!
export bilinear_interpolation_3d, ensure_periodic_3d!

include("liquidmotionsolver/stefan.jl")
export StefanMono2D, solve_StefanMono2D!
export StefanDiph2D, solve_StefanDiph2D!
export compute_volume_jacobian
export solve_StefanMono2Dunclosed!
export solve_StefanMono2D_geom!

include("liquidmotionsolver/stefan3D.jl")
export StefanMono3D, solve_StefanMono3D!
export compute_volume_jacobian_3D
export smooth_displacements_3D!

include("concentrationsolver/species.jl")
export DiffusionUnsteadyConcentration, solve_DiffusionUnsteadyConcentration!

include("binarysolver/binary.jl")
export DiffusionUnsteadyBinary, solve_DiffusionUnsteadyBinary!

include("vizualize.jl")
export plot_solution, animate_solution
export plot_isotherms

include("vizualize_mov.jl")
export analyze_convergence_rates_newton, plot_timestep_history, plot_interface_evolution, plot_newton_residuals, analyze_interface_spectrum

include("convergence.jl")
export check_convergence, check_convergence_diph

include("solver/stokes.jl")
export AbstractPressureGauge, PinPressureGauge, MeanPressureGauge
export StokesMono, solve_StokesMono!, solve_StokesMono_unsteady!

include("prescribedmotionsolver/stokes.jl")
export MovingStokesUnsteadyMono, solve_MovingStokesUnsteadyMono!

include("prescribedmotionsolver/stokes_fsi.jl")
export MovingStokesFSI2D, solve_MovingStokesFSI2D!

include("solver/stokes_diph.jl")
export StokesDiph, solve_StokesDiph!, solve_StokesDiph_unsteady!

include("solver/navierstokes.jl")
export NavierStokesMono, solve_NavierStokesMono_unsteady!, solve_NavierStokesMono_unsteady_picard!,
       solve_NavierStokesMono_unsteady_newton!, solve_NavierStokesMono_steady!, build_convection_operators
export compute_navierstokes_force_diagnostics, navierstokes_reaction_force_components
export drag_lift_coefficients, pressure_trace_on_cut

include("solver/navier_stokes_proj.jl")
export NavierStokesProjectionMono, solve_NavierStokesProjection_unsteady!

include("prescribedmotionsolver/navierstokes.jl")
export MovingNavierStokesUnsteadyMono, solve_MovingNavierStokesUnsteadyMono!, compute_moving_navierstokes_force_2D
export MovingNavierStokesFSI2D, solve_MovingNavierStokesFSI2D!

include("solver/navierstokes_scalar_coupling.jl")
export CouplingStrategy, PassiveCoupling, PicardCoupling, MonolithicCoupling
export NavierStokesScalarCoupler, step!, solve_NavierStokesScalarCoupling!, solve_NavierStokesScalarCoupling_steady!,
       solve_NavierStokesScalarCoupling_steady_monolithic!

include("solver/streamfunction_vorticity.jl")
export StreamVorticity, solve_StreamVorticity!, step_StreamVorticity!, run_StreamVorticity!, run_until_StreamVorticity!


include("liquidmotionsolver/rayleigh_benard_melting.jl")
export RayleighBenardMeltingSetup, solve_MovingRayleighBenardMelting2D!

end
