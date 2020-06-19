using LinearAlgebra
using PhaseMapping
using PhaseMapping: Phase, optimize

Data, Sticks = PhaseMapping.load("AlLiFe")
PhaseMapping.removeduplicates(Sticks)


nsticks = length(Sticks)
phases = Vector{typeof(Phase(Sticks[1], 2))}(undef, nsticks)
for i in 1:nsticks
    Q0 = Phase(Sticks[i], 1., 1., .5)
    phases[i] = Q0
end

# make library
# L = Library(phases[end-6:end])
# L = Library(phases[150:end])

using Plots
plotly()
x = Data.Q
Y = Data.I
Y ./= maximum(Y, dims = 1) #sum(abs2, Y, dims = 1)
ind = 1 # 128, 4, 150
y = Y[:, ind]

# likelihood
Σ = 1e-4*I(length(x))
# Σ = sqrt.(y) .+ 1.
# Σ = Diagonal(Σ)
# y += 1e-2*randn(length(y))
like(r) = nld(Normal(zero(x), Σ), r)

println(SpectroscopicAnalysis.getparams(L))

doplot = true
if doplot
    # plot(x, y)
    # gui()
    a, R, A = fit(L, x, y, Σ)
    plot(x, y)
    plot!(x, R)
    plot!(x, A.*a')
    gui()
end
maxiter = 512
# prior = x->0
println(SpectroscopicAnalysis.getparams(L))
@time L = optimize(L, x, y, Σ, prior; maxiter = maxiter)#, prior; maxiter = maxiter)
println(SpectroscopicAnalysis.getparams(L))

if doplot
    a, R, A = fit(L, x, y, Σ)
    plot(x, y)
    plot!(x, R)
    plot!(x, A.*a')
    gui()
end
