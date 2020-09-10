using DelimitedFiles
using HDF5

# this should be in the sara module
function load_stripe(file, stripe_id)
    load_stripe(file[stripe_id])
end

function load_stripe(stripe)
    ns = length(stripe)
    nq = size(read(stripe, "0/integrated_1d"), 1)
    X = zeros(nq, ns)
    Y = similar(X)
    for i in 1:ns
        xy = read(stripe, "$(i-1)/integrated_1d")
        X[:, i] = xy[:, 1]
        Y[:, i] = xy[:, 2]
    end
    X, Y
end

datapath = "/Users/sebastianament/Documents/SEA/XRD Analysis/SARA/Bi2O3_19F44_01/"
datafile = h5open(datapath * "Bi2O3_19F44_01_all_oned.h5")
datafile = datafile["exp"]

# get stick patterns
using PhaseMapping: readsticks
Sticks = readsticks(datapath * "sticks/sticks.txt")

# names(datafile)
# stripe_id = "tau_10000_T_1000" # weird data
# stripe_id = "tau_800_T_800" # also weird
# stripe_id = "tau_700_T_800"
stripe_id = "tau_700_T_880" # 90 - 100 are good
# stripe_id = "tau_600_T_880" # 90 - 100 are good

X, Y = load_stripe(datafile, stripe_id)
Y ./= maximum(Y)
i = 90:100 # these seem interesting
x = X[:, 1]
y = Y[:, i]

out = readdlm(datapath * "sticks/names.txt", '=', String, '\n')
sticknames = out[:, 2]

############################# Phase Identification #############################
using LinearAlgebra
x = X[:, 1]

# 1. Background subtraction
using BackgroundSubtraction
l_x = 4 # lengthscale of background component in x (q)
indices = range(1., size(Y, 2), step = 1)
l_i = 8
B = mcbl(Y, x, l_x, indices, l_i, minres = 1e-3, nsigma = 1.8, maxiter = 64)

using Plots
plotly()
doplot = true
if doplot
    surface(Y)
    gui()
    surface(Y-B)
    gui()
end

@. Y -= B # subtract background

# 2. Phase identification
using PhaseMapping
using PhaseMapping: optimize_a!, optimize_α!, optimize_σ!, Gauss, Lorentz, pmp!, Phase, Library, get_prior
# y = Y[:, 105] # 87, 98, #90-98
nc = size(Y, 2)
width_init = .2
phases = Phase.(Sticks, nc, width_init = width_init, profile = Lorentz())
# phases = Phase.(Sticks, width_init = width_init, profile = Lorentz())

σ = 1e-2 # likelihood

# prior distributions
var_a = 1.
var_α = .01^2
var_σ = .1^2
l = 3
C = indices
Ga = get_prior(C, -2., var_a, l, 1e-8)
Gα = get_prior(C, 0., var_α, l, 1e-8)
Gσ = get_prior(C, log(width_init), var_σ, l, 1e-8)
G = (Ga, Gα, Gσ)

using Profile
k = 2
ind = 90
y = Y[:, ind]
@profile pmp!(phases, x, Y, σ, G, k)
println(phases)
L = Library(phases)

# plot(x, y)
# # plot!(x, L.(x))
# plot!(x, [p.(x, ind) for p in phases])
# gui()

# plot([p.a for p in phases], label = permutedims(sticknames), title = stripe_id)
# gui()

# function identify_phases(x, Y, σ, Sticks, ε = 1e-3)
#     σ_init = 2e-1
#     phases = Phase.(Sticks, nc, width_init = width_init, profile = Lorentz())
#
#     # prior distributions
#     var_a = 1.
#     var_α = .01^2
#     var_σ = .1^2
#     l = .5
#     Ga = get_prior(C, -2., var_a, l, 1e-8)
#     Gα = get_prior(C, 0., var_α, l, 1e-8)
#     Gσ = get_prior(C, log(width_init), var_σ, l, 1e-8)
#     G = (Ga, Gα, Gσ)
#
#     Q = optimize_a!(Q, x, Y, σ, Ga)
# end

#
# # Profile.print(format = :flat, sortedby=:count)
# ε = 1e-2 # activation variance threshold
# @time L = identify_phases(x, y, W, Sticks, ε)
# println("after optimization")
# summarize(L)
#
# doplot = true
# if doplot
#     pp = SpectroscopicAnalysis.plot_posterior(L, x, y, Σ, B, sticknames[getid(L)])
#     plotlyjs()
#     gui()
# end
#
# # filename = "phaseid.h5"
# # savefile = h5open(datapath * filename, "w")
#
# using SpectroscopicAnalysis: h5write_model
# # TODO: loop over stripes to complete analysis
# # ε = 1e-2
# # namesf = names(datafile)
# # for stripe_id in namesf[1:2]
# function analyze_stripe(datafile, stripe_id, savefile)
#     # likelihood and background model
#     σ_like = 2e-3 # 2e-3 # standard deviation of noise
#     a_back = 1. #1e-1 # activation of background component
#     l_back = 3. # 10 # lengthscale of background component
#     ε = 1e-2 # activation variance threshold
#
#     X, Y = load_stripe(datafile, stripe_id)
#     # g_create(savefile, stripe_id)
#     # stripefile = savefile[stripe_id]
#     Y ./= maximum(Y)
#     for i in 86:90 #1:size(X, 2)
#         x, y = @views X[:,i], Y[:, i]
#         y ./= maximum(y)
#
#         # background model has regularizing effect
#         Σ, B = initialize_background(x, y, σ_like, a_back, l_back)
#         W = Woodbury(Σ, B)
#         @time L = identify_phases(x, y, W, Sticks, ε)
#         # g_create(stripefile, "$i")
#         # h5write_model(stripefile["$i"], L, x, y, Σ, B, sticknames[getid(L)])
#         SpectroscopicAnalysis.plot_posterior(L, x, y, Σ, B, sticknames[getid(L)])
#     end
# end
#
# # analyze_stripe(datafile, stripe_id, savefile)
# # names(savefile)

# for i in 1:16:size(Y, 2)
#     plot(x, Y[:, i])
#     # bi = mcbl(Y[:, i], x, l_x) #, minres = 1e-3, nsigma = 1.1)
#     # plot!(x, bi)
#     # plot!(x, Y[:, i]-bi)
#     plot!(x, B[:, i])
#     plot!(x, Y[:, i]-B[:,i])
#     gui()
# end
