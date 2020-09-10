struct Library{T, P<:AbstractVector{<:Phase{T}}}
	phases::P
end

function (L::Library)(x::Real, j::Int)
	isempty(L.phases) ? zero(x) : sum(p->p(x, j), L.phases)
end
function (L::Library)(x::Real)
	isempty(L.phases) ? zero(x) : sum(p->p(x), L.phases)
end

# function (L::Library)(x::Real, j::Int)
# 	y = zero(x)
# 	for p in L.phases
# 		y += p(x, j)
# 	end
# 	return y
# end

# to evaluate all spectrograms
# L.((x,), (1:length(P.a))') for scalar x
# L.(x, (1:length(P.a))') for vector x

function get_parameters!(θ::AbstractMatrix, L::Library)
	get_parameters!(θ, L.phases)
end
get_parameters(L::Library) = get_parameters(L.phases)
function set_parameters!(L::Library, θ::AbstractMatrix)
	set_parameters!(phases, θ)
end

function get_parameters!(θ::AbstractMatrix, phases::AbstractVector{<:Phase})
	n = check_div3(size(θ, 1))
	for (i, P) in enumerate(phases)
		@. θ[1:n, i] = P.a
		@. θ[n+1:2n, i] = P.α
		@. θ[2n+1:3n, i] = P.σ
	end
	return θ
end

function get_parameters(phases::AbstractVector{<:Phase})
	n = length(phases[1].a)
	θ = zeros(3n, length(phases))
	get_parameters!(θ, phases)
end
function get_parameters(phases::AbstractVector{<:Phase}, i::Int)
	θ = zeros(3, length(phases))
	for k in 1:length(phases)
		θ[:, k] .= get_parameters(phases[k], i)
	end
	return θ
end
function set_parameters!(phases::AbstractVector{<:Phase}, j::Int, θ::AbstractMatrix)
	for (i, P) in enumerate(phases)
		θ_i = @view θ[:, i]
		set_parameters!(P, j, θ_i)
	end
	return phases
end

function set_parameters!(phases::AbstractVector{<:Phase}, θ::AbstractMatrix)
	for (i, P) in enumerate(phases)
		θ_i = @view θ[:, i]
		set_parameters!(P, θ_i)
	end
	return phases
end

function scale_gradient!(θ::AbstractMatrix, a_scale = 1., α_scale = 1e-2, σ_scale = 1e-2)
	n = check_div3(size(θ, 1))
	a = @view θ[1:n, :]
	α = @view θ[n+1:2n, :]
	σ = @view θ[2n+1:3n, :]
	a .*= a_scale
	α .*= α_scale
	σ .*= σ_scale
	θ
end
