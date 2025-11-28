using Penguin
using Test

@testset "Mesh Test" begin
    # Write your tests here.
    include("mesh_test.jl")
end

@testset "Capacity Test" begin
    # Write your tests here.
    include("capacity_test.jl")
end

@testset "Operators Test" begin
    # Write your tests here.
    include("operators_test.jl")
end

@testset "Boundary Test" begin
    # Write your tests here.
    include("boundary_test.jl")
end

@testset "Phase Test" begin
    # Write your tests here.
    include("phase_test.jl")
end

@testset "Solver Test" begin
    # Write your tests here.
    include("solver_test.jl")
    include("solver/darcy_test.jl")
    include("solver/diffusion_test.jl")
    include("solver/stokes_test.jl")
    include("solver/stream_vorticity_test.jl")
    include("solver/navierstokes_scalar_coupling_test.jl")
    #include("solver/stokes_diph.jl")
    #include("solver/stefan_test.jl")
end

@testset "Convergence Test" begin
    # Write your tests here.
    include("convergence_test.jl")
end

@testset "Utils Test" begin
    # Write your tests here.
    include("utils_test.jl")
end

@testset "Interpolation Test" begin
    # Write your tests here.
    include("interpolation_test.jl")
end

@testset "Front Tracking Test" begin
    # Write your tests here.
    include("front_tracking_test.jl")
end

@testset "Height Tracking Test" begin
    # Write your tests here.
    include("height_tracking_test.jl")
end
