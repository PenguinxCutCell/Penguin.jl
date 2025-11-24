using Documenter, Penguin

makedocs(sitename="Penguin.jl", remotes=nothing, modules = [Penguin],
        pages = [
            "index.md",
            "Simulation key blocks" => [
                "blocks/mesh.md",
                "blocks/body.md",
                "blocks/capacity.md",
                "blocks/operators.md",
                "blocks/boundary.md",
                "blocks/phase.md",
                "blocks/solver.md",
                "blocks/solverlist.md",
                "blocks/stokes.md",
                "blocks/navierstokes.md",
                "blocks/vizualize.md",
            ],
            "Examples" => [
                "tests/operators.md",
                "tests/poisson.md",
                "tests/poisson_2ph.md",
                "tests/heat.md",
                "tests/heat_2ph.md",
                "tests/darcy.md",
                "tests/solidmoving.md",
                "tests/liquidmoving1D.md",
                ],
            "Benchmark" => [
                "benchmark/poisson.md",
                "benchmark/heat.md",
                "benchmark/heat_2ph.md",
                "benchmark/taylor_green.md",
            ],
        ])