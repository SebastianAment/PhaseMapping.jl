# PhaseMatchingPursuit
struct PhaseMappingPursuit{T, P<:Vector{<:Phase}}
    active_phases::P
    passive_phases::P
end
const PMAP = PhaseMappingPursuit

function PMAP(phases::Vector{<:Phase})
    active_phases = Vector{eltype(phases)}(undef, 0)
    PMAP(active_phases, phases)
end

# initializing active phases corresponding to phases with indices
function PMAP(phases::Vector{<:Phase}, indices)
    !isnothing(indices) || return PMAP(phases)
    active_phases = phases[indices]
    deleteat!(phases, indices)
    PMAP(active_phases, phases)
end

# returns index of maximal sum of non-negative inner products with residual
function pmp_index(P::PMAP, x::AbstractVector, Y::AbstractMatrix)
    absdot = zeros(length(P.passive_phases))
    R = Y - sum(p->p(x), P.active_phases) # residual
    for (i, p) in enumerate(P.passive_phases)
        pi = representative(p, x)
        absdot[i] = sum(x->max(x, 0), R'*pi)
    end
    return argmax(absdot)
end

function update!(P::PMAP, x, Y)
    i = pmp_index!(P, x, y)
    push!(P.active_phases, passive_phases[i])
    deleteat!(P.passive_phases, i)
    optimize!(P.active_phases, x, Y)
    return P
end

function pmp(phases::Vector{<:Phase}, indices)
    P = PMAP(phases, indices)
end

struct LibraryDirection{T, PH<:Vector{<:Phase{T}}, RA, RAL, RS} <: Optimization.Direction{T}
	phases::PH # phase object
	reg_a::RA
	reg_α::RAL
	reg_σ::RS
	# x::X # data input
	# value::V
	# gradient::D
	# b::BT # termporary storage for phase array
	# J::JT # phase jacobian
	# scaling::S
end

function LibraryDirection(phases::Vector{<:Phase}, c::AbstractMatrix)
	l = .1
	k = Kernel.EQ()
	var_a = 2e0^2
	var_α = 5e-2^2
	var_σ = 5e-1^2
	ra = a_regularizer(c, λ = 1/var_a, l = l, k = k)
	rα = α_regularizer(c, λ = 1/var_α, l = l, k = k)
	rσ = σ_regularizer(c, λ = 1/var_σ, l = l, k = k)
	LibraryDirection(phases, ra, rα, rσ)
end

function optimize!(phases::Vector{<:Phase}, x::AbstractVector, Y::AbstractMatrix,
	 								c::AbstractMatrix; maxiter = 256, tol = 1e-4)
	optimize!(LibraryDirection(phases, c), x, Y, maxiter = maxiter, tol = tol)
end

function kl(x::Real, y::Real, ε::Real = 1e-8)
	x * log((x+ε)/(y+ε)) - x + y
end

function optimize!(D::LibraryDirection, x::AbstractVector, Y::AbstractMatrix;
													maxiter = 256, tol = 1e-4)

	phases = D.phases
	n_phases = length(phases)
	n = size(Y, 2)
	function objective(θ::AbstractMatrix, indices = 1:n)
		Yi = @view Y[:, indices]
		A = zeros(eltype(θ), size(Yi))
		prior = zero(eltype(θ))
		for k in 1:n_phases
			θ_k = @view θ[:, k]
			P = phases[k]
			a, α, σ = view_aασ(θ_k)
	        A .+= Phase(P, a = a, α = α, σ = σ).(x, indices')
			prior += D.reg_a(a) + D.reg_α(α) #+ D.reg_σ(σ)
		end
		bias = n / length(indices) # bias correction
        # bias * residual_abs2(Yi, A) + prior
		bias * sum(kl.(Yi, A)) #+ prior
     end
	function gradient(θ::AbstractMatrix, indices = 1:n) # w.r.t. patterns chosen by indices
		f(θ) = objective(θ, indices)
		g = FD.gradient(f, θ)
	end
	function stochastic_direction(batch_size::Int = 4)
		batch_size = min(batch_size, n)
		function d(θ)
			indices = sample(1:n, batch_size, replace = false)
			-gradient(θ, indices)
		end
	end
	delta_shift = 1e-1
	min_width = 1e-1
	max_width = 4e-1
	project! = PhaseProjection(delta_shift, min_width, max_width)
	# could have variable dependent learning rate
	θ = get_parameters(phases)
	batch_size = 2
	∇ = stochastic_direction(batch_size)
	α = 1e-2 # learning rate
	β₁ = .99 # gradient momentum
	β₂ = .999 # 2nd moment momentum
	update! = AMSGRAD(∇, θ, α, β₁, β₂)
	# update! = PolyakMomentum(∇, copy(θ), α, β₁)
	for t in 1:maxiter
		update!(θ, t)
		project!(θ)
		# println(θ)
		println(t)
	end
	set_parameters!(phases, θ)
	return phases
end
