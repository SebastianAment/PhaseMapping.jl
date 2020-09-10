module TestPhase
using Test
using LinearAlgebra
using PhaseMapping
using PhaseMapping: PeakProfile, Lorentz, Gauss, PseudoVoigt
using PhaseMapping: Phase, optimize!, representativ
# using PhaseMapping: Phase, optimize!, representative, optimize_a!, optimize_α!,
#                     optimize_σ!, optimize_dc!, optimize_aασ!

using Plots
plotly()
@testset "Phase" begin

    Data, Sticks = PhaseMapping.load("AlLiFe")
    PhaseMapping.removeduplicates(Sticks)

    nsticks = length(Sticks)
    profile = Gauss()
    phases = Vector{typeof(Phase(Sticks[1], profile = profile))}(undef, nsticks)
    k = 8
    for i in 1:nsticks
        phases[i] = Phase(Sticks[i], width_init = 1e-1, profile = profile)
    end
    composition = Data.composition[:, 1:k]

    x = collect(0:.1:90)
    P = phases[1]
    println(P)
    p = representative(P, x)
    @test p isa AbstractVector
    @test length(p) == length(x)

    tol = 1e-1
    # stochastic optimization
    P = Phase(P, a = rand())
    y = P.(x)
    println(P.a)
    println(P.α)
    println(P.σ)

    P2 = Phase(P, a = 0.)
    yc = copy(y)
    P2 = optimize!([P2], x, y, maxiter = 16)
    P2 = P2[1]
    @test isapprox(P2.a, P.a, atol = tol)
    println(P2.α)
    println(P2.σ)
    plot(x, y)
    plot!(x, P2.(x))
    gui()


    # optimization of a
    # a = rand(k)
    # P.a .= a
    # y = P(x)
    # P.a .= 0
    # optimize_a!(P, x, y)
    # @test isapprox(P.a, a, atol = tol)
    #
    # # optimization of α
    # α = 1 .+ .01 * randn(k)
    # P.α .= α
    # y = P(x)
    # P.α .= 1
    # optimize_α!(P, x, y)
    # @test isapprox(P.α, α, atol = tol)
    #
    # # optimization of σ
    # σ = exp.(.1 * randn(k))
    # P.σ .= σ
    # y = P(x)
    # P.σ .= 1
    # optimize_σ!(P, x, y)
    # @test isapprox(P.σ, σ, atol = tol)
    #
    # # optimization of dc
    # dc = rand(length(P.dc))
    # @. dc = .1 #sign(dc) * min(P.c, abs(dc))
    # P.dc .= dc
    # y = P(x)
    # P.dc .= 0
    # optimize_dc!(P, x, y)
    # @test isapprox(P.dc, dc, atol = tol)

    # # optimization of Phase
    # a = rand(k)
    # α = 1 .+ .01 * randn(k)
    # σ = exp.(.1 * randn(k))
    # P.a .= a
    # P.α .= α
    # P.σ .= σ
    # y = P(x)
    # P.a .= 0
    # P.α .= 1
    # P.σ .= 1
    # optimize!(P, x, y)
    # @test isapprox(P.a, a, atol = tol)
    # @test isapprox(P.α, α, atol = tol)
    # @test isapprox(P.σ, σ, atol = tol)

end

end # TestPhase
