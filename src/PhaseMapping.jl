module PhaseMapping
# base
using Base.Threads
# tools
using LinearAlgebra
using LinearAlgebraExtensions: AbstractMatOrFac, AbstractMatOrUni
using Kernel
using StatsBase: mean, std, sample
# file io
using DelimitedFiles
using HDF5

# optimization
# using JuMP
# using JuMP: optimizer_with_attributes
# using Ipopt
using ForwardDiff
const FD = ForwardDiff
using Optimization
using Optimization: LevenbergMarquart, LevenbergMarquartSettings, GaussNewton
using Optimization: SaddleFreeNewton, DecreasingStep, fixedpoint!, StoppingCriterion, CustomDirection
using NormalDistributions

include("datastructures.jl")
include("peakprofile.jl")
include("rkhs.jl")
include("phase.jl")
include("library.jl")
include("pattern.jl")
include("matchingpursuit.jl")
include("globalphasepursuit.jl")
include("plots.jl")
include("nmf.jl")

function substrate_projection(substrate::Phase, x)
    return function projection!(background, measurement)
        optimize!(substrate, x, measurement, maxiter = 1, opt_c = true)
        background .= substrate(x)
    end
end
using BackgroundSubtraction: projected_background!
function substrate_background!(A::AbstractMatrix, substrate::Phase, x::AbstractVector;
                                            minres::Real = 1e-2, nsigma::Real = 2,
                                            maxiter::Int = 4, minnpeak::Int = 1)
    measurement = copy(A)
    background = similar(measurement)
    projection! = substrate_projection(substrate, x)
    projected_background!(background, measurement, projection!,
                                        minres = minres, nsigma = nsigma,
                                        maxiter = maxiter, minnpeak = minnpeak)
end

end # module
