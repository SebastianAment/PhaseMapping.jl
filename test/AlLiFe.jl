using LinearAlgebra
using PhaseMapping
using PhaseMapping: Phase, optimize!, Lorentz, Gauss, Library, pmp!, SmoothPattern, ternary, pmp_path!
using Plots
plotly()

# k = 6
# display(cumbabel(patterns, k))
# display(cumbabel(precondition!.(patterns, 1-1e-6), k))
# P! = preconditioner(patterns)
# display(cumbabel(P!.(patterns), k))

Data, Sticks = PhaseMapping.load("AlLiFe")
PhaseMapping.removeduplicates(Sticks)
nq, nc = size(Data.I)

# spectroind = 1:64 # 126:128 # 1:8 # 1:32
spectroind = 2 # round.(Int, range(1, nc, length = nc))
nc = length(spectroind)
composition = Data.composition[:, spectroind]

x = Data.Q
Y = Data.I
Y ./= maximum(Y, dims = 1) #sum(abs2, Y, dims = 1)
y = Y[:, spectroind]

width_init = 2e-1
if y isa AbstractVector
    all_phases = Phase.(Sticks, width_init = width_init, profile = Gauss())
else
    all_phases = Phase.(Sticks, nc, width_init = width_init, profile = Gauss())
end

nsticks = length(Sticks)
stickind = nsticks-5:nsticks
# stickind = nsticks-16:nsticks
# stickind = nsticks-64:nsticks
# stickind = 1:length(phases)
phases = all_phases[stickind]

# α = [1.]
# marginal_patterns = [MarginalPattern(p, x, α) for p in phases]
# println(sum(abs2, y))
# pat = [p.p for p in marginal_patterns]
# println(sum.(abs2, pat))
# plot(x, pat)
# gui()

verbose = false
if verbose
    println("before")
    println("activations")
    display([p.a for p in L.phases])
    println("shifts")
    display([p.α for p in L.phases])
    println("widths")
    display([p.σ for p in L.phases])
    println("ids")
    display([p.id for p in L.phases])
end

# creating full pmp path with corresponding residuals
libraries, residuals = pmp_path!(phases, x, y, k)
# diff(residuals) look at difference in residual norms

# @time optimize!(phases, x, y, maxiter = 128)
# @time phases = pmp(phases, x, y, 3)
# @time phases = pmp(phases, x, reshape(y, :, 1), 3)
k = 3
phases = pmp!(phases, x, y, k)

L = Library(phases)
verbose = true
if verbose
    println("after")
    println([p.a for p in L.phases])
    println([p.α for p in L.phases])
    println([p.σ for p in L.phases])
end
println([p.id for p in L.phases])
resnorm = [norm(L.(x, i) - y[:, i]) for i in 1:nc]
display(resnorm)
# phaseind = [L.phases[i].id for i in [p.a for p in L.phases] > 1e-2]
# println(phaseind)

# i = spectroind # 128, 4, 150
plotind = findall(>(.1), resnorm)
println(plotind)
plotind = spectroind

# plotind = 1:4:maximum(spectroind)
doplot = true
if nc > 8
    doplot = false
end
if doplot
    for (i, ind) in enumerate(plotind)
        y = Y[:, ind]
        plot(x, y, label = "observation")
        plot!(x, L.(x, i), label = "library")
        plot!(x, [p.(x, i) for p in L.phases], label = "phases")
        # plot!(x, P.(x, i))
        gui()
    end
end
