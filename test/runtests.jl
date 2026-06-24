# Run from the repo root with:
#   julia --project=. test/runtests.jl
#
# The core model is duplicated here (rather than imported) because the project is
# environment-only: the canonical code lives in notebooks/markov_asset.ipynb and is
# kept in sync with the definitions below. The reference numbers asserted at the end
# match the notebook's "Verification" cell.

using Test
using LinearAlgebra

# ---------------------------------------------------------------------------
# Core model (mirror of notebooks/markov_asset.ipynb)
# ---------------------------------------------------------------------------
struct MarkovAsset
    P::Matrix{Float64}
    M::Int
    function MarkovAsset(P::AbstractMatrix; atol = 1e-10)
        P = Float64.(P)
        @assert size(P, 1) == size(P, 2) "P must be square"
        @assert all(P .>= -atol) "P must have non-negative entries"
        @assert all(isapprox.(sum(P; dims = 2), 1.0; atol = atol)) "rows must sum to 1"
        new(P, size(P, 1))
    end
end

function stationary_distribution(a::MarkovAsset)
    F = eigen(permutedims(a.P))
    idx = argmin(abs.(F.values .- 1.0))
    pivec = real.(F.vectors[:, idx])
    pivec ./= sum(pivec)
    @assert all(pivec .> 0) "chain must be irreducible (pi > 0)"
    return pivec
end

slem(a::MarkovAsset) = sort(abs.(eigvals(a.P)); rev = true)[2]

struct DynamicStoppingSolver
    asset::MarkovAsset
    R::Vector{Float64}
    c::Float64
    T::Int
    function DynamicStoppingSolver(asset::MarkovAsset, R, c, T)
        R = Float64.(R)
        @assert length(R) == asset.M "R must give one payoff per state"
        @assert c >= 0 && T >= 1
        new(asset, R, Float64(c), Int(T))
    end
end

function solve(s::DynamicStoppingSolver)
    M, P, R, c, T = s.asset.M, s.asset.P, s.R, s.c, s.T
    V = zeros(T + 1, M)
    policy = ones(Int, T + 1, M)
    V[T + 1, :] .= R
    for t in (T - 1):-1:0
        continuation = -c .+ P * V[t + 2, :]
        V[t + 1, :] .= max.(R, continuation)
        policy[t + 1, :] .= Int.(R .>= continuation)
    end
    return V, policy
end

# ---------------------------------------------------------------------------
# Fixture: the experiment from the notebook
# ---------------------------------------------------------------------------
const P = [
    0.50 0.40 0.10 0.00 0.00;
    0.20 0.45 0.30 0.05 0.00;
    0.05 0.25 0.45 0.20 0.05;
    0.00 0.10 0.30 0.45 0.15;
    0.00 0.05 0.15 0.35 0.45
]
const R = [20.0, 40.0, 60.0, 80.0, 100.0]
const c, T = 2.0, 12
const asset = MarkovAsset(P)
const V, policy = solve(DynamicStoppingSolver(asset, R, c, T))

@testset "Markov environment" begin
    pivec = stationary_distribution(asset)
    @test isapprox(sum(pivec), 1.0; atol = 1e-12)         # pi is a probability vector
    @test all(pivec .> 0)
    @test isapprox(P' * pivec, pivec; atol = 1e-10)       # stationarity: pi P = pi
    @test isapprox(pivec, [0.1445, 0.2849, 0.3051, 0.1869, 0.0787]; atol = 1e-3)
    @test isapprox(slem(asset), 0.6795; atol = 1e-3)      # second-largest eigenvalue modulus
    @test isapprox(dot(pivec, R), 55.41; atol = 1e-2)     # long-run average reward
    @test_throws AssertionError MarkovAsset([0.5 0.4; 0.2 0.2])  # rows must sum to 1
end

@testset "Optimal stopping engine" begin
    @test V[T + 1, :] == R                              # terminal condition V_T = R
    @test all(policy[T + 1, :] .== 1)                   # forced sale at the horizon
    @test isapprox(round.(V[1, :]; digits = 2), [53.35, 57.25, 64.68, 80.0, 100.0]; atol = 1e-2)

    uniform = fill(1 / asset.M, asset.M)
    optimal = dot(uniform, V[1, :])
    naive = dot(uniform, R)
    @test isapprox(optimal, 71.06; atol = 1e-2)
    @test isapprox(naive, 60.00; atol = 1e-2)
    @test optimal > naive                               # timing adds value

    # The optimal rule is a monotone state threshold: HOLD (0) for low s, SELL (1) for
    # high s, so every policy row is sorted ascending across states.
    for t in 1:(T + 1)
        @test issorted(policy[t, :])
    end

    cutoff = [findfirst(==(1), policy[t + 1, :]) - 1 for t in 0:T]  # smallest SELL state
    @test cutoff == [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 0]
end
