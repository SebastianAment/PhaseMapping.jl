using LinearAlgebra
using PhaseMapping
using PhaseMapping: Phase, optimize!, Lorentz, Gauss, Library, pmp!, SmoothPattern, ternary, get_prior
using Test
using Plots
plotly()

################################# loading data #################################
Data, Sticks = PhaseMapping.load("AlLiFe")
PhaseMapping.removeduplicates(Sticks)
x = Data.Q
Y = Data.I
C = Data.composition
nq, nc = size(Y)

# subsampling
spectroind = round.(Int, range(1, nc, length = 8)) # 1:nc #
nc = length(spectroind)
Y = Y[:, spectroind]
C = C[:, spectroind]

# normalizing
Y ./= maximum(Y, dims = 1) # sum(abs2, Y, dims = 1)

########################### initializing phases ################################
width_init = 2e-1
phases = Phase.(Sticks, nc, width_init = width_init, profile = Gauss())

# prior over log(a)
using Kernel
using NormalDistributions
using NormalDistributions: sample, Normal, cov, mean
import WoodburyIdentity: Woodbury

using LinearAlgebraExtensions
using LinearAlgebraExtensions: LowRank

# prior distributions
var_a = 1.
var_α = .01^2
var_σ = .1^2
l = .5
Ga = get_prior(C, -2., var_a, l, 1e-8)
Gα = get_prior(C, 0., var_α, l, 1e-8)
Gσ = get_prior(C, log(width_init), var_σ, l, 1e-8)
G = (Ga, Gα, Gσ)

# sampling
log_a = sample(Ga)
log_α = sample(Gα)
log_σ = sample(Gσ)

doplot = false
doplot && ternary(C, exp.(log_a))

# creating synthetic data
indices = (1:nc)
P = Phase(phases[1], a = exp.(log_a), α = exp.(log_α), σ = exp.(log_σ))
Y = P.(x, indices') # P.(x) #

# test optimization
using PhaseMapping: optimize_a!, optimize_α!, optimize_σ!
Q = Phase(P, a = fill(1e-3, size(P.a)))
σ = 1e-4
Q = optimize_a!(Q, x, Y, σ, Ga)
tol = 1e-2
# println(norm(Q.(x, indices') - P.(x, indices')))
# println(norm(Q.a - P.a))
@test isapprox(Q.a, P.a, atol = tol)

################# α
Q = Phase(P, α = ones(size(P.α))) #exp.(.001randn(size(P.α))))
Q = optimize_α!(Q, x, Y, σ, Gα)
@test isapprox(Q.α, P.α, atol = tol)

################## σ
Q = Phase(P, σ = fill(width_init, size(P.σ)))
Q = optimize_σ!(Q, x, Y, σ, Gσ)
# println(norm(Q.σ - P.σ))
# println(norm(Q.(x, indices') - P.(x, indices')))
@test isapprox(Q.σ, P.σ, atol = tol)

################### optimize all via block-coordinate descent
Q = Phase(P, a = ones(size(P.a)), α = ones(size(P.a)), σ = fill(width_init, size(P.σ)))
optimize!(Q, x, Y, σ, G)
@test isapprox(Q.a, P.a, atol = tol)
@test isapprox(Q.α, P.α, atol = tol)
@test isapprox(Q.σ, P.σ, atol = tol)

# creating synthetic phase mixture
# indices = (1:nc)
# P1 = Phase(phases[1], a = exp.(log_a), α = exp.(log_α), σ = exp.(log_σ))
# P2 = Phase(phases[2], a = exp.(log_a), α = exp.(log_α), σ = exp.(log_σ))
# @. Y = P1(x, indices') + P2(x, indices') # phase mixture
#
# phases = Phase.(phases[1:2], a = ones(size(P1.a)), α = ones(size(P1.a)), σ = fill(width_init, size(P1.a)))
# L = Library(phases)
# optimize!(L, x, Y, σ, G)

# doplot = false
# if doplot
#     plot(x, Q.(x, (1:nc)'))
#     gui()
# end

# (y-A*x)'*Σ*(y-A*x) / 2
# -A'*Σ*(y-A*x)
# A'*Σ*A # hessian
