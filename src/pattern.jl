################################################################################
# smoothed stick pattern, encompassing shifted versions
struct SmoothPattern{T, AT<:AbstractVector, PAT<:AbstractMatrix{T}, NAT}
	α::AT
	A::PAT
	normA::NAT
	id::Int
end

# TODO: Gauss-Hermite quadrature for log α
# using FastGaussQuadrature
function SmoothPattern(P::Phase, x::AbstractVector, α::AbstractVector)
	A = zeros(eltype(x), length(x), length(α))
	n = length(P.a)
	for (i, α_i) in enumerate(α)
		A[:, i] .= Phase(P, a = 1., α = α_i, σ = P.σ[1]).(x)
	end
	normA = sqrt.(sum(abs2, A, dims = 1))
	normA = vec(normA)
	A ./= normA'
	SmoothPattern(α, A, normA, P.id)
end

function SmoothPattern(P::Phase, x::AbstractVector)
	SmoothPattern(P, x, range(0.95, 1.05, length = 35))
end

function findmaxinner(S::SmoothPattern, x::AbstractVector, xPa::AbstractVector = zeros(size(S.A, 2)))
	mul!(xPa, S.A', x) # xPa = mp.A' * x
	@. xPa = max(0, xPa)
	m, k = findmax(xPa)
end
# function findmaxinner(S::SmoothPattern, x::AbstractVector, xPa::AbstractVector = zeros(size(S.A, 2)))
# 	# precondition!(S.A, 1e-6)
# 	mul!(xPa, S.A', x) # xPa = mp.A' * x
# 	@. xPa = max(0, xPa)
# 	m, k = findmax(xPa)
# end
maxinner(S::SmoothPattern, x) = findmaxinner(S, x)[1]
maxinner(S::SmoothPattern, x, xPa) = findmaxinner(S, x, xPa)[1]
argmaxinner(S::SmoothPattern, x) = findmaxinner(S, x)[2]
argmaxinner(S::SmoothPattern, x, xPa) = findmaxinner(S, x, xPa)[2]

function maxinnerparameters(S::SmoothPattern, x::AbstractVecOrMat)
	m, i = findmaxinner(S, x)
	m./S.normA[i], S.α[i]
end

# squared euclidean norm of positive part of x
posnorm2(x) = sum(y->max(0, y)^2, x)

# maximal positive inner product between X and SmoothPattern
# norm of maximal projected gradient
# could decide on norm
function maxgrad(S::SmoothPattern, X::AbstractVecOrMat, xA::AbstractVecOrMat = zeros(size(S.A, 2), size(X, 2)))
	mul!(xA, S.A', X) # xA = S.A' * X
	inner = zero(eltype(xA))
	for c in eachcol(xA)
		inner = max(inner, maximum(c)) # maximum of non-negative part of xPa
	end
	return inner
end

function maxinner(S::SmoothPattern, X::AbstractMatrix, xA::AbstractMatrix = zeros(size(S.A, 2), size(X, 2)))
	mul!(xA, S.A', X) # xA = S.A' * X
	inner = zero(eltype(X))
	for c in eachcol(xA)
		inner += max(0, maximum(c))^2 # sum of squares of non-negative part of xPa
	end
	return inner
end

function findmaxinner(S::SmoothPattern, X::AbstractMatrix, xA = zeros(size(S.A, 2), size(X, 2)))
	mul!(xA, S.A', X) # xA = S.A' * X
	@. xA = max(0, xA)
	inner, indices = zeros(eltype(X), size(X, 2)), zeros(Int, size(X, 2))
	for (i, c) in enumerate(eachcol(xA))
		inner[i], indices[i] = findmax(c)
	end
	return inner, indices
end

# calculates phase pattern without shifting and activation
function representative(P::Phase, x::AbstractVector)
    representative!(similar(x), P, x)
end
function representative!(y::AbstractVector, P::Phase, x::AbstractVector)
    y .= 0
    σ = P.σ[1]
    for j in eachindex(x)
        @simd for i in eachindex(P.c)
            @inbounds y[j] += P.c[i] * P.profile((x[j]-P.μ[i])/σ)
        end
    end
    return y
end

############################ dictionary preconditioning ########################
using CompressedSensing: preconditioner, precondition!, normalize!
function mean_preconditioner(p, ε::Real = 1e-6)
	f(A::AbstractVecOrMat) = preconditioner(ε)(A)
	function f(p::SmoothPattern)
		f(p.A)
		return p
	end
	return f
end

function renormalize!(p::SmoothPattern)
	normA = sqrt.(sum(abs2, p.A, dims = 1)) # renormalize
	p.A ./= normA
	p.normA .*= vec(normA)
	return p
end

import CompressedSensing: cumbabel
function cumbabel(patterns::AbstractVector{<:SmoothPattern}, k::Int)
	A = matricify(patterns)
	return cumbabel(A, k)
end

# accumulate all shifted patterns in columns of a single matrix
function matricify(patterns::AbstractVector{<:SmoothPattern})
	nrow, ncol = size(patterns[1].A)
	# for p in patterns
	# 	ncol += length(p.α)
	# end
	A = zeros(nrow, ncol, length(patterns))
	for (i, p) in enumerate(patterns)
		A[:, :, i] .= p.A
	end
	A = reshape(A, nrow, :)
end

# svd preconditioner
function svd_preconditioner(patterns::AbstractVector{<:SmoothPattern}, min_σ = 1e-3)
	A = matricify(patterns)
	P! = preconditioner(A, min_σ)
	f(A::AbstractVecOrMat) = P!(A)
	function f(p::SmoothPattern)
		f(p.A)
		return p
	end
	return f
end


# should we integrate over or choose maximum? -> maximum
# function marginal_inner(P::Phase, x::AbstractVector, y::AbstractVector)
# 	δ = 5e-2
# 	α = range(1-δ, stop = 1+δ, length = 7)
# 	w = 1/length(α)
# end
