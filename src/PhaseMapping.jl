module PhaseMapping
# base
using Base.Threads
# tools
using LinearAlgebra
# using LinearAlgebraExtensions: , AbstractMatOrUni
const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
# using Kernel
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
using OptimizationAlgorithms
using OptimizationAlgorithms: LevenbergMarquart, LevenbergMarquartSettings, GaussNewton
using OptimizationAlgorithms: SaddleFreeNewton, DecreasingStep, fixedpoint!, StoppingCriterion, CustomDirection

colnorms(A) = [norm(a) for a in eachcol(A)]

include("datastructures.jl")
include("peakprofile.jl")
include("phase.jl")
include("library.jl")
include("pattern.jl")
include("optimization.jl")
include("matchingpursuit.jl")
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
