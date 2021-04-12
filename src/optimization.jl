# returns the phase which leads to the minimal l2 error in fitting y
function fit_phases(phases::Vector{<:Phase}, x::AbstractVector, y::AbstractVector,
					std_noise::Real = .01, mean_θ::AbstractVector = [1., 1., .2],
					std_θ::AbstractVector = [3., .01, 1.];
					maxiter = 32, regularization::Bool = true)
	optimized_phases = similar(phases)
	residuals = zeros(length(x), length(phases))
	@threads for i in eachindex(phases) # optimize phases individually
		P = phases[i]
		P = optimize(P, x, y, std_noise, mean_θ, std_θ,
							maxiter = maxiter, regularization = regularization)
		optimized_phases[i] = P
		@. residuals[:, i] = y - P(x)
	end
	return optimized_phases, residuals
end

function optimize(phase::Phase, x::AbstractVector, y::AbstractVector,
									std_noise::Real,
									mean_θ::AbstractVector,
									std_θ::AbstractVector;
									maxiter = 32, regularization::Bool = true)
	optimize!([phase], x, y, std_noise, mean_θ, std_θ, maxiter = maxiter, regularization = regularization)[]
end

# calls optimize! below and assigns new phase objects based on result
function optimize!(phases::AbstractVector{<:Phase}, x::AbstractVector, y::AbstractVector,
									std_noise::Real,
									mean_θ::AbstractVector,
									std_θ::AbstractVector;
									maxiter = 32, regularization::Bool = true)
	θ = get_parameters(phases)
	optimize!(θ, phases, x, y, std_noise, mean_θ, std_θ,
			maxiter = maxiter, regularization = regularization)
	for (k, p) in enumerate(phases)
		a, α, σ = θ[:, k]
		phases[k] = Phase(p, a = a, α = α, σ = σ)
	end
	return phases
end

# initializes activation parameter based on maximum inner product
function initialize_activation(P::Phase, x::AbstractVector, y::AbstractVector)
	p = representative(P, x)
	return dot(p, y) / sum(abs2, p)
end

# TODO: needs noise standard deviation
# LevenbergMarquart optimization
# returns θ and residual of optimal solution r (including residual corresponding to prior term)
function optimize!(θ::AbstractMatrix, phases::AbstractVector{<:Phase},
					x::AbstractVector, y::AbstractVector,
					std_noise::Real, mean_θ::AbstractVector, std_θ::AbstractVector;
					maxiter = 32, regularization::Bool = true)
	function residual!(r::AbstractVector, θ::AbstractMatrix)
		@. r = y
		for k in 1:length(phases) # TODO: this could be a Library constructor + eval
			a, α, σ = exp.(θ[1:3, k])
			P = Phase(phases[k], a = a, α = α, σ = σ)
			@. r -= P(x)
		end
		r ./= sqrt(2) * std_noise # from Gaussian observation likelihood
		return r
	end
	function residual!(r::AbstractVector, θ::AbstractVector)
		check_div3(length(θ))
		residual!(r, reshape(θ, 3, :))
	end
	# l2 regularization in "residual form"
	function prior!(p::AbstractMatrix, θ::AbstractMatrix)
		μ =  log.(mean_θ) # [0, 0, 0]
		σ² = std_θ.^2 # var_a, var_α, var_σ
		@. p = (θ - μ) / 2σ²
	end
	function prior!(p::AbstractVector, θ::AbstractVector)
		prior!(reshape(p, 3, :), reshape(θ, 3, :))
	end
	function f(rp::AbstractVector, θ::AbstractVector)
		# length(rp) == length(θ) || throw(DimensionMismatch(""))
		r = @view rp[1:length(y)] # residual term
		residual!(r, θ)
		p = @view rp[length(y)+1:end] # prior term
		prior!(p, θ)
		return rp
	end
	# if necessary, initialize activations
	if any(!>(0), @view θ[1,:])
		@. θ[1, :] = initialize_activation(phases, (x,), (y,))
		@. θ[1, :] = max(θ[1, :], 1e-3) # make certain activations are above positive threshold
	end
	@. θ = log(θ) # transform to log-space
	(any(isnan, θ) || any(isinf, θ)) && throw("any(isinf, θ) = $(any(isinf, θ)), any(isnan, θ) = $(any(isnan, θ))")
	θ = vec(θ)
	if regularization
		r = zeros(eltype(θ), length(y) + length(θ)) # residual + prior terms
		LM = LevenbergMarquart(f, θ, r) # pre-allocate?
	else
		r = zeros(eltype(θ), size(y)) # residual
		LM = LevenbergMarquart(residual!, θ, r)
	end
	stn = LevenbergMarquartSettings(min_resnorm = 1e-2, min_res = 1e-3,
						min_decrease = 1e-8, max_iter = maxiter,
						decrease_factor = 7, increase_factor = 10, max_step = 1.0)
	λ = 1e-6
	OptimizationAlgorithms.optimize!(LM, θ, copy(r), stn, λ, Val(false))
	θ = reshape(θ, 3, :)
	@. θ = exp(θ) # transform back to real space
	return θ
end
