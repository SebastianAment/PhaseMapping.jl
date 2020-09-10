using NormalDistributions: Normal, nld, mean, cov
using DiffResults
using DiffResults: DiffResult
using ForwardDiff: derivative!, derivative #, gradient
using LinearAlgebraExtensions: difference, LowRank
using LazyInverse: inverse
using WoodburyIdentity

using Optimization: TrustedDirection

function pmp!(phases::AbstractVector{<:Phase}, x, y, σ::Real, G::NTuple{3, Normal},
	 															k::Int; tol = 1e-3)
	P = PMP(phases, x, y)
	pmp!(P, x, y, σ, G, k, tol = tol)
	P.active_phases
end
function pmp!(P::PMP, x, y, σ::Real, G::NTuple{3, Normal}, k::Int; tol = 1e-3)
	for _ in 1:k
		update!(P, x, y, σ, G, tol = tol) || break
	end
	return P
end
function update!(P::PMP, x::AbstractVector, Y::AbstractMatrix, σ::Real,
										G::NTuple{3, Normal}; tol::Real = 1e-3)
    m, i = pmp_index!(P, x, Y)
    if m > tol
		P.isactive[i] = true
		optimize!(P.active_phases, x, Y, σ, G)
		return true
	else
	    return false
	end
end

# coordinate ascent for library
# G is tuple of a, α, σ priors, assumed to be the same for each phase
# σ is the noise standard deviation
function optimize!(phases::AbstractVector{<:Phase}, x::AbstractVector,
								Y::AbstractMatrix, σ::Real, G::NTuple{3, Normal})
	for (i, p) in enumerate(phases)
		optimize!(p, x, Y, σ, G)
	end
	return phases
end

# coordinate ascent for phase
function optimize!(P::Phase, x::AbstractVector, Y::AbstractMatrix, σ::Real,
						G::NTuple{3, Normal}; max_iter = 4, min_change = 1e-4)
	# TODO: initialization of a, α
	old = similar(P.a)
	for i in 1:max_iter
		δ = 0.

		@. old = P.a
		@time optimize_a!(P, x, Y, σ, G[1])
		δ = max(δ, maximum(abs, old-P.a))

		# @. old = P.α
		# optimize_α!(P, x, Y, σ, G[2])
		# δ = max(δ, maximum(abs, old-P.α))

		@. old = P.σ
		@time optimize_σ!(P, x, Y, σ, G[3])
		δ = max(δ, maximum(abs, old-P.σ))

		println(i)
		δ > min_change || break
	end
	return P
end

# optimize activations of phase
function optimize_a!(P::Phase, x::AbstractVector, Y::AbstractMatrix, σ::Real, G::Normal)
    indices = 1:size(Y, 2)
    log_a = log.(P.a)
	# display(log_a)
	log_a[isinf.(log_a)] .= -2
	# display(log_a)
    any(isnan, log_a) && throw("NaN in log_a")
    any(isinf, log_a) && throw("Inf in log_a")
    PA = P.(x, indices')

    function likelihood(log_a::AbstractVector)
        sum(abs2, Y - PA .* exp.(log_a)') / (2σ^2)
    end
    function likelihood(log_a_j::Real, i::Int, j::Int)
        (Y[i,j] - PA[i,j] * exp(log_a_j))^2 / (2σ^2)
    end
    function objective(log_a::AbstractVector)
        likelihood(log_a) + nld(G, log_a)
    end
    function valdir(log_a::AbstractVector)
        function d(i::Int, j::Int)
            log_a_j = log_a[j]
            r = DiffResult(log_a_j, (one(log_a_j),))
            g(z) = FD.derivative(x->likelihood(x, i, j), z)
            r = derivative!(r, g, log_a_j) # need return value since r is immutable
            ∇, Δ = DiffResults.value(r), DiffResults.derivative(r)
        end

        # gradient and laplacian of likelihood
        ∇_l, Δ_l = zero(log_a), zero(log_a)
        for j in eachindex(log_a)
            for i in eachindex(x)
                ∇, Δ = d(i,j)# d(Y[i,j], PA[i,j], log_a[j])
                ∇_l[j] += ∇
                Δ_l[j] += Δ
            end
        end
        H_l = Diagonal(abs.(Δ_l))

        # gradient and laplacian of GP prior
        Σ = factorize(G.Σ)
        ∇_gp = Σ \ (log_a - G.μ)
        H_gp = inverse(Σ)
        # combining everything
        ∇ = ∇_l + ∇_gp # gradient
        if H_gp isa Woodbury
            H = Woodbury(H_l + H_gp.A, H_gp.U, H_gp.C, H_gp.V, H_gp.α)
        else
            H = H_l + Matrix(H_gp)
        end
        H = factorize(H)
        step = -(H\∇)
        return objective(log_a), step # Newton step
    end
    D = CustomDirection(objective, valdir, log_a)
    D = DecreasingStep(D, log_a)
	fixedpoint!(D, log_a, StoppingCriterion(log_a, dx = 1e-6, maxiter = 64))
    @. P.a = exp(log_a)
    return P
end


function optimize_α!(P::Phase, x::AbstractVector, Y::AbstractMatrix, σ::Real, G::Normal)
    indices = 1:size(Y, 2)
    log_α = log.(P.α)
    any(isnan, log_α) && throw("NaN in log_α")
    any(isinf, log_α) && throw("Inf in log_α")
    @. P.α = 1 # TODO: initialization
    PA = P.(x, indices')
    function likelihood(log_α::AbstractVector)
        @. P.α = exp.(log_α)
        sum(abs2, Y - P.(x, indices')) / (2σ^2)
        # sum(abs2, Y - Phase(P, α = exp.(log_α)).(x, indices')) / (2σ^2)
    end
    function likelihood(log_α_j::Real, i::Int, j::Int)
        Pj = Phase(P, a = P.a[j], α = exp(log_α_j), σ = P.σ[j])
        (Y[i,j] - Pj(x[i]))^2 / (2σ^2)
    end
    function objective(log_α::AbstractVector)
        likelihood(log_α) + nld(G, log_α)
    end
    function valdir(log_α::AbstractVector)
        function d(i::Int, j::Int)
            log_α_j = log_α[j]
            r = DiffResult(log_α_j, (one(log_α_j),))
            g(z) = FD.derivative(x->likelihood(x, i, j), z)
            r = derivative!(r, g, log_α_j) # need return value since r is immutable
            ∇, Δ = DiffResults.value(r), DiffResults.derivative(r)
        end

        # gradient and laplacian of likelihood
        ∇_l, Δ_l = zero(log_α), zero(log_α)
        for j in eachindex(log_α)
            for i in eachindex(x)
                ∇, Δ = d(i,j)
                ∇_l[j] += ∇
                Δ_l[j] += Δ
            end
        end
        H_l = Diagonal(abs.(Δ_l))

        # gradient and laplacian of GP prior
        Σ = factorize(G.Σ)
        ∇_gp = Σ \ (log_α - G.μ)
        H_gp = inverse(Σ)

        # combining everything
        ∇ = ∇_l + ∇_gp # gradient
        if H_gp isa Woodbury
            H = Woodbury(H_l + H_gp.A, H_gp.U, H_gp.C, H_gp.V, H_gp.α)
        else
            H = H_l + Matrix(H_gp)
        end
        H = factorize(H)
        step = -(H\∇)
        return objective(log_α), step # Newton step
    end

    D = CustomDirection(objective, valdir, log_α)
	# safeguard against taking too large steps (trust region)
	D = TrustedDirection(D, .1, .01)
    D = DecreasingStep(D, log_α)
	fixedpoint!(D, log_α, StoppingCriterion(log_α, dx = 1e-6, maxiter = 128))
    @. P.α = exp(log_α)
    return P
end


function optimize_σ!(P::Phase, x::AbstractVector, Y::AbstractMatrix, σ::Real, G::Normal)
    indices = 1:size(Y, 2)
    log_σ = log.(P.σ)
    any(isnan, log_σ) && throw("NaN in log_σ")
    any(isinf, log_σ) && throw("Inf in log_σ")
    @. P.σ = 1 # TODO: initialization
    PA = P.(x, indices')
    function likelihood(log_σ::AbstractVector)
        @. P.σ = exp.(log_σ)
        sum(abs2, Y - P.(x, indices')) / (2σ^2)
        # sum(abs2, Y - Phase(P, σ = exp.(log_σ)).(x, indices')) / (2σ^2)
    end
    function likelihood(log_σ_j::Real, i::Int, j::Int)
        Pj = Phase(P, a = P.a[j], α = P.α[j], σ = exp(log_σ_j))
        (Y[i,j] - Pj(x[i]))^2 / (2σ^2)
    end
    function objective(log_σ::AbstractVector)
        likelihood(log_σ) + nld(G, log_σ)
    end
    function valdir(log_σ::AbstractVector)
        function d(i::Int, j::Int)
            log_σ_j = log_σ[j]
            r = DiffResult(log_σ_j, (one(log_σ_j),))
            g(z) = FD.derivative(x->likelihood(x, i, j), z)
            r = derivative!(r, g, log_σ_j) # need return value since r is immutable
            ∇, Δ = DiffResults.value(r), DiffResults.derivative(r)
        end

        # gradient and laplacian of likelihood
        ∇_l, Δ_l = zero(log_σ), zero(log_σ)
        for j in eachindex(log_σ)
            for i in eachindex(x)
                ∇, Δ = d(i, j)
                ∇_l[j] += ∇
                Δ_l[j] += Δ
            end
        end
        H_l = Diagonal(abs.(Δ_l))

        # gradient and laplacian of GP prior
        Σ = factorize(G.Σ)
        ∇_gp = Σ \ (log_σ - G.μ)
        H_gp = inverse(Σ)

        # combining everything
        ∇ = ∇_l + ∇_gp # gradient
        if H_gp isa Woodbury
            H = Woodbury(H_l + H_gp.A, H_gp.U, H_gp.C, H_gp.V, H_gp.α)
        else
            H = H_l + Matrix(H_gp)
        end
        H = factorize(H)
        step = -(H\∇)
        return objective(log_σ), step # Newton step
    end
    D = CustomDirection(objective, valdir, log_σ)
    D = DecreasingStep(D, log_σ)
	fixedpoint!(D, log_σ, StoppingCriterion(log_σ, dx = 1e-6, maxiter = 64))
    @. P.σ = exp(log_σ)
    return P
end

# struct DirectionA{T, PAT} <: Direction
#     PA::PAT
# end

# gp prior over X
function get_prior(X::AbstractVecOrMat, μ::Real, a::Real, l::Real, σ²::Real = 1e-8;
				   k = Kernel.EQ())
   nx = X isa AbstractMatrix ? size(X, 2) : length(X)
   μ = fill(μ, nx)
   k = a*Kernel.Lengthscale(k, l) # smooth in composition .1*Kernel.Dot()
   Σ = Kernel.gramian(k, X)
   F = factorize(Σ, tol = 1e-6)
   if F isa Union{LowRank, CholeskyPivoted}
	   Σ = Woodbury(σ²*I(nx), F)
   else
	   Σ = factorize(σ²*I(nx)+Σ, tol = 1e-6)
   end
   return Normal(μ, Σ)
end
