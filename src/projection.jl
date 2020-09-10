######################### projects phase parameters ############################
struct PhaseProjection{T<:Real}
	δ_α::T
	min_σ::T
	max_σ::T
	function PhaseProjection(δ_α::T, min_σ::T, max_σ::T) where {T<:Real}
		δ_α > 0 || throw("d_α ≯ 0")
		min_σ > 0 || throw("min_σ ≯ 0")
		max_σ > min_σ || throw("max_σ ≯ min_σ")
		new{T}(δ_α, min_σ, max_σ)
	end
end
function PhaseProjection(δ_α::Real, min_σ::Real, max_σ::Real)
	δ_α, min_σ, max_σ = promote(δ_α, min_σ, max_σ)
	PhaseProjection{eltype(δ_α)}(δ_α, min_σ, max_σ)
end
function PhaseProjection(;delta_shift = 1e-2, min_width = 5e-2, max_width = 3e-1)
	PhaseProjection(delta_shift, min_width, max_width)
end

function (P::PhaseProjection)(θ::AbstractVector)
	a, α, σ = view_aασ(θ)
	@. a = max(a, 0)
	@. α = boxbound(1-P.δ_α, 1+P.δ_α, α)
	@. σ = boxbound(P.min_σ, P.max_σ, σ)
	return θ
end
# returns projection of x into [a, b]
boxbound(a::Real, b::Real, x::Real) = min(max(x, a), b)
boxbound(a::Real, b::Real) = x::Real -> boxbound(a, b, x)

# applies same projection to each column of θ for library optimization
function (P::PhaseProjection)(θ::AbstractMatrix)
	for c in eachcol(θ)
		P(c)
	end
	return θ
end
