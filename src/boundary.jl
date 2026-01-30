abstract type AbstractBoundary end

"""
    Dirichlet(value::Union{Function, Float64})

Structure to define Dirichlet boundary conditions. The value can be a constant or a function of the space variable.
`T=g`
- g = Dirichlet(0.0) # Constant Dirichlet boundary condition
- g = Dirichlet(x -> sin(x)) # Dirichlet boundary condition that depends on the space variable
- g = Dirichlet((x, t) -> sin(x) * cos(t)) # Dirichlet boundary condition that depends on the space and time variables
"""
struct Dirichlet <: AbstractBoundary
    value::Union{Function, Float64}
end

"""
    Neumann(value::Union{Function, Float64})

Structure to define Neumann boundary conditions. The value can be a constant or a function of the space variable.
`∇T.n = g`
- g = Neumann(0.0) # Constant Neumann boundary condition
- g = Neumann(x -> sin(x)) # Neumann boundary condition that depends on the space variable
- g = Neumann((x, t) -> sin(x) * cos(t)) # Neumann boundary condition that depends on the space and time variables
"""
struct Neumann <: AbstractBoundary
    value::Union{Function, Float64}
end

"""
    Robin(alpha::Union{Function, Float64}, beta::Union{Function, Float64}, value::Union{Function, Float64})

Structure to define Robin boundary conditions. The value can be a constant or a function of the space variable.
`αT + β∇T.n = g`
- g = Robin(0.0, 1.0, 0.0) # Constant Robin boundary condition
- g = Robin(x -> sin(x), 1.0, 0.0) # Robin boundary condition that depends on the space variable
- g = Robin((x, t) -> sin(x) * cos(t), 1.0, 0.0) # Robin boundary condition that depends on the space and time variables
"""
struct Robin <: AbstractBoundary
    α::Union{Function, Float64}
    β::Union{Function, Float64}
    value::Union{Function, Float64}
end

"""
    Periodic()

Structure to define periodic boundary conditions.
"""
struct Periodic <: AbstractBoundary
end

"""
    Symmetry()

Mirror-symmetry boundary condition. For velocity fields it corresponds to zero
normal component and zero normal gradient of tangential components.
"""
struct Symmetry <: AbstractBoundary
end

"""
    Outflow(; pressure=nothing)

Outflow boundary condition prescribing a zero normal gradient for velocity and
an optional reference pressure. If `pressure` is provided (constant or
function), it is imposed on the pressure field at the boundary; otherwise the
pressure remains free and a gauge condition is still required elsewhere.
"""
struct Outflow <: AbstractBoundary
    pressure::Union{Nothing,Float64,Function}
    function Outflow(p::Union{Nothing,Float64,Function}=nothing)
        new(p)
    end
end

"""
    Traction(value)
Interface traction boundary condition prescribing the total stress vector on a cut surface.
`value` can be a scalar (1D) or a tuple/vector providing one entry per velocity component.
Functions may depend on space (and optionally time) coordinates.
"""
struct Traction <: AbstractBoundary
    value::Union{Function,Float64,Tuple,AbstractVector}
end

abstract type AbstractInterfaceBC end

"""
    ScalarJump(α₁::Union{Function,Float64}, α₂::Union{Function,Float64}, value::Union{Function,Float64})

Structure to define scalar jump conditions. The value can be a constant or a function of the space variable.
`[[αT]] = α₂T2 - α₁T1 = g``
- g = ScalarJump(0.0, 1.0, 0.0) # Constant scalar jump condition
- g = ScalarJump(x -> sin(x), 1.0, 0.0) # Scalar jump condition that depends on the space variable
- g = ScalarJump((x, t) -> sin(x) * cos(t), 1.0, 0.0) # Scalar jump condition that depends on the space and time variables
"""
struct ScalarJump <: AbstractInterfaceBC
    α₁::Union{Function,Float64}
    α₂::Union{Function,Float64}
    value::Union{Function,Float64}
end

"""
    FluxJump(β₁::Union{Function,Float64}, β₂::Union{Function,Float64}, value::Union{Function,Float64})

Structure to define flux jump conditions. The value can be a constant or a function of the space variable.
`[[β∇T.n]] = β₂∇T2.n - β₁∇T1.n = g`
- g = FluxJump(0.0, 1.0, 0.0) # Constant flux jump condition
- g = FluxJump(x -> sin(x), 1.0, 0.0) # Flux jump condition that depends on the space variable
- g = FluxJump((x, t) -> sin(x) * cos(t), 1.0, 0.0) # Flux jump condition that depends on the space and time variables
"""
struct FluxJump <: AbstractInterfaceBC
    β₁::Union{Function,Float64}
    β₂::Union{Function,Float64}
    value::Union{Function,Float64}
end


"""
    RobinJump(α::Union{Function,Float64}, β::Union{Function,Float64}, value::Union{Function,Float64})
Structure to define Robin jump conditions. The value can be a constant or a function of the space variable.
`α[[T]] + β∇T.n = g`
- g = RobinJump(1.0, 0.0, 0.0) # Temperature continuity condition
- g = RobinJump(1.0, x -> sin(x), 0.0) # Robin jump condition that depends on the space variable
- g = RobinJump(1.0, (x, t) -> sin(x) * cos(t), 0.0) # Robin jump condition that depends on the space and time variables
"""
struct RobinJump <: AbstractInterfaceBC
    α::Union{Function,Float64}
    β::Union{Function,Float64}
    value::Union{Function,Float64}
end

"""
    BorderConditions(borders::Dict{Symbol, AbstractBoundary})

Structure to define border conditions. The keys are :left, :right, :top, :bottom, :forward, :backward.
Important if the problem is diphasic or monophasic (solved outside the geometry) to set the conditions for the domain borders.
"""
struct BorderConditions
    borders::Dict{Symbol, AbstractBoundary}       # Keys: :left, :right, :top, :bottom, :forward, :backward
end

"""
    InterfaceConditions(interfaces::Dict{Symbol, AbstractInterfaceBC})

Structure to define interface conditions. The keys are :scalar, :flux.
Important if the problem is diphasic to set the conditions for the interface between the two phases.
"""
struct InterfaceConditions
    scalar::Union{Nothing, AbstractInterfaceBC}
    flux::Union{Nothing,AbstractInterfaceBC}
end

"""
    GibbsThomson(Tm, ϵₖ, ϵᵥ)

A Gibbs-Thomson boundary condition for the interface
- Tm: Melting temperature
- ϵₖ: Capillarity coefficient
- ϵᵥ: Kinetic coefficient
"""
mutable struct GibbsThomson <: AbstractBoundary
    Tm::Float64
    ϵₖ::Float64
    ϵᵥ::Float64
    vᵞ::Vector{Float64}
    value::Float64

    function GibbsThomson(Tm::Float64, ϵₖ::Float64, ϵᵥ::Float64, operator::AbstractOperators)
        # Initialize the vector by zeros 
        new(Tm, ϵₖ, ϵᵥ, zeros(prod(operator.size)), Tm)
    end
end
