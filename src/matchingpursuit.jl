# IDEA: run outlier detection?
# IDEA: GP Prior
# IDEA: Background model
# IDEA: tree search?
# unify matrix and vector Y?
# l2 vs max norm of gradient?
struct PhaseMatchingPursuit{T, PH<:AbstractVector{<:Phase{T}},
			PT<:AbstractVector{<:SmoothPattern{T}}, AS<:AbstractVector{Bool},
			XT<:AbstractVecOrMat, RT<:AbstractVecOrMat, AT<:AbstractVecOrMat,
			# P1, P2, P3,
			PR}
	phases::PH
	patterns::PT
	isactive::AS

	x::XT
	r::RT
	rA::AT

	# preconditioner
	precondition!::PR

	# prior normal distributions
	# prior_a::P1
	# prior_α::P2
	# prior_σ::P3

	# background model
end

const PMP = PhaseMatchingPursuit
function PMP(phases::AbstractVector{<:Phase}, x::AbstractVector, y::AbstractVecOrMat,
										isactive = fill(false, length(phases)),
										preconditioner = mean_preconditioner; # svd_preconditioner; #
										max_shift::Real = .1, nshift::Int = 35)
	α = range(1-max_shift, stop = 1+max_shift, length = nshift)
	patterns = [SmoothPattern(p, x, α) for p in phases]
	precondition! = preconditioner(patterns)
	@. renormalize!(precondition!(patterns))
	PMP(phases, patterns, x, y, isactive, precondition!)
end

function PMP(phases::AbstractVector{<:Phase}, patterns::AbstractVector{<:SmoothPattern},
									x::AbstractVector, y::AbstractVecOrMat,
									isactive = fill(false, length(phases)),
									precondition! = identity)
	r = similar(y)
	# WARNING, need to allocate one rA for each thread!
	nshift = length(patterns[1].α)
	rA = y isa AbstractVector ? zeros(nshift) : zeros(nshift, size(y, 2))
	PMP(phases, patterns, isactive, x, r, rA, precondition!)
end

function Base.getproperty(P::PMP, s::Symbol)
	if s == :active_phases
		@view P.phases[P.isactive]
	elseif s == :active_patterns
		@view P.patterns[P.isactive]
	elseif s == :passive_phases
		@view P.phases[.!P.isactive]
	elseif s == :passive_patterns
		@view P.phases[.!P.isactive]
	elseif s == :nactive
		sum(==(true), P.isactive)
	elseif s == :npassive
		sum(==(false), P.isactive)
	else
		getfield(P, s)
	end
end

################################### pmp ########################################
function pmp!(phases::Vector{<:Phase}, x, y, k::Int; tol = 1e-3)
	P = PMP(phases, x, y)
	pmp!(P, x, y, k, tol = tol)
	P.active_phases
end
function pmp!(P::PMP, x, y, k::Int; tol = 1e-3)
	for _ in 1:k
		update!(P, x, y, tol = tol) || break
	end
	return P
end
# local phase matching pursuit (for ith spectrogram)
# here, k is maximal sparsity level per spectrogram
# this could be modeled as a subspace pursuit algorithm
function pmp!(P::PMP, x::AbstractVector, y::AbstractVector, i::Int, k::Int;
													maxiter = 256, tol = 1e-3)
	local_phases = Phase.(P.active_phases, a = 0., α = 1., σ = .2) # active set local to ith pattern
	local_PMP = PMP(local_phases, P.active_patterns, x, y,
					fill(false, length(local_phases)), P.precondition!)
	pmp!(local_PMP, x, y, k, tol = tol)
	for (p, lp) in zip(P.active_phases, local_phases)
		p.a[i], p.α[i], p.σ[i] = lp.a, lp.α, lp.σ
	end
end

function pmp_index!(P::PMP, x::AbstractVector, Y::AbstractVecOrMat)
    inner = zeros(length(P.patterns))
	r = residual!(P, x, Y)
	r = P.precondition!(r)
    for i in eachindex(P.patterns)
		if !P.isactive[i]
			inner[i] = maxinner(P.patterns[i], P.r, P.rA) # maxgrad(P.patterns[i], r, P.rA) #
		end
    end
	return findmax(inner)
end

function update!(P::PMP, x::AbstractVector, y::AbstractVector; tol::Real = 1e-3)
    m, i = pmp_index!(P, x, y)
    if m > tol
		a, α = maxinnerparameters(P.patterns[i], P.r) # initialize with best fitting activation and shift
		P.phases[i] = Phase(P.phases[i], a = a, α = α)
		P.isactive[i] = true
		optimize!(P.active_phases, x, y)
		return true
	else
		return false
	end
end

function residual!(P::PMP, x::AbstractVector, Y::AbstractVecOrMat)
	L = Library(P.active_phases)
	if Y isa AbstractVector
		@. P.r = Y - L(x)
	elseif Y isa AbstractMatrix
		indices = 1:size(Y, 2)
		@. P.r = Y - L(x, indices')
	end
	P.r
end

# function preallocate_rA(A::AbstractMatrix, Y::AbstractVecOrMat)
# 	nshift = size(A, 2)
# 	return if Y isa AbstractVector
# 		zeros(nshift)
# 	elseif Y isa AbstractMatrix
# 		zeros(nshift, size(Y, 2))
# 	end
# end

# phase subspace pursuit
# function update!(P::PMP, x::AbstractVector, Y::AbstractMatrix, psp::Val{true}; tol::Real = 1e-3)
#     m, i = psp_index!(P, x, Y)
# 	println(P.phases[i].id)
# 	println(m)
#     if m > tol
# 		P.isactive[i] = true
# 		k_gibbs = 4
# 		for (j, y) in enumerate(eachcol(Y))
# 			pmp!(P, x, y, j, k_gibbs)
# 		end
# 		return true
# 	else
# 	    return false
# 	end
# end

# could return true / false
# function update!(P::PMP, x::AbstractVector, Y::AbstractMatrix; tol::Real = 1e-3)
#     m, i = pmp_index!(P, x, Y)
# 	println(P.phases[i].id)
# 	println(m)
#     if m > tol
# 		P.isactive[i] = true
# 		k_gibbs = 6
# 		for (j, y) in enumerate(eachcol(Y))
# 			pmp!(P, x, y, j, k_gibbs)
# 		end
# 		return true
# 	else
# 	    return false
# 	end
# end
