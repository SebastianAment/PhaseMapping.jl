using LinearAlgebra
using PhaseMapping
using PhaseMapping: Phase, optimize!, Gauss, substrate_background!
using Plots
plotly()

Data, Sticks = PhaseMapping.load("BiCuV")

x = Data.Q
Y = Data.I
Y ./= maximum(Y, dims = 1) # sum(abs2, Y, dims = 1)
ind = 1:16
Y = Y[:, ind] # subsample for testing
c = Data.composition[:, ind]
nsticks = length(Sticks)
subid = 72
substrate = Phase(Sticks[subid], size(Y, 2), width_init = .1, profile = Gauss()) # substrate

using BackgroundSubtraction

l_x = 4
l_c = .05
nsigma = 1.
maxiter = 32
minres = 1e-4
@time kb = kronecker_mcbl(Y, x, l_x, c, l_c,
                        nsigma = nsigma, maxiter = maxiter, minres = minres)
k = 8
@time b = mcbl(Y, k, x, l_x,
            nsigma = nsigma, maxiter = maxiter, minres = minres)

i = 1
y = Y[:, i]
plot(x, y)
sticks!(sub.μ, sub.c, label = "substrate")
plot!(x, kb[:, i], label = "kronecker_mcbl")
plot!(x, b[:, i], label = "mcbl")
gui()

substrate_background!(Y-kb, substrate, x, nsigma = nsigma, minres = minres)
println(substrate.a)
println(substrate.α)
println(substrate.σ)

s = substrate.(x, i)
plot!(x, s + b[:,i], label = "substrate")
