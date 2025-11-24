using Penguin
using Test

@testset "Boundary Test" begin
    bc = Dirichlet(1.0)
    @test bc.value == 1.0

    bc = Dirichlet(x -> sin(x))
    @test bc.value(0.0) ≈ 0.0

    bc = Dirichlet((x, t) -> sin(x) * cos(t))
    @test bc.value(0.0, 0.0) ≈ 0.0

    bc = Neumann(1.0)
    @test bc.value == 1.0

    bc = Neumann(x -> sin(x))
    @test bc.value(0.0) ≈ 0.0

    bc = Robin(1.0, 1.0, 1.0)
    @test bc.α == 1.0
    @test bc.β == 1.0
    @test bc.value == 1.0

    bc = Robin(x -> sin(x), 1.0, 1.0)
    @test bc.α(0.0) ≈ 0.0
    @test bc.β == 1.0
    @test bc.value == 1.0

    bc = ScalarJump(0.0, 1.0, 0.0)
    @test bc.α₁ == 0.0
    @test bc.α₂ == 1.0
    @test bc.value == 0.0

    bc = FluxJump(x -> sin(x), 1.0, 0.0)
    @test bc.β₁(0.0) ≈ 0.0
    @test bc.β₂ == 1.0
    @test bc.value == 0.0
end