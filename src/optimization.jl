using Optimization
using Optimization: AMSGRAD, PolyakMomentum

function residual_abs2(X::AbstractArray, Y::AbstractArray)
    size(X) == size(Y) || throw(DimensionMismatch())
    r = zero(promote_type(eltype(X), eltype(Y)))
    @simd for i in eachindex(X)
        @inbounds r += abs2(X[i]-Y[i])
    end
    r
end

struct PhaseDirection{T, PH<:Phase{T}, RA, RAL, RS}
	phase::PH # phase object
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

function PhaseDirection(P::Phase, c::AbstractMatrix)
	l = .1
	k = Kernel.EQ()
	var_a = 10e0^2
	var_α = 2e-2^2
	var_σ = 1e0^2
	ra = a_regularizer(c, λ = 1/var_a, l = l, k = k)
	rα = α_regularizer(c, λ = 1/var_α, l = l, k = k)
	rσ = σ_regularizer(c, λ = 1/var_σ, l = l, k = k)
	PhaseDirection(P, ra, rα, rσ)
end

function optimize!(D::PhaseDirection, x::AbstractVector, Y::AbstractMatrix;
													maxiter = 256, tol = 1e-4)
	P = D.phase
	n = size(Y, 2)
	function objective(θ, indices = 1:n)
		a, α, σ = view_aασ(θ)
        A = Phase(P, a = a, α = α, σ = σ).(x, indices')
		bias = n / length(indices) # bias correction
		Yi = @view Y[:, indices]
        bias * residual_abs2(Yi, A) #+ D.reg_a(a) + D.reg_α(α) #+ D.reg_σ(σ)
     end
	function gradient(θ, indices = 1:n) # w.r.t. patterns chosen by indices
		f(θ) = objective(θ, indices)
		g = FD.gradient(f, θ)
		# h = FD.hessian(f, θ)
		# g ./ diag(h)
	end
	function stochastic_direction(batch_size::Int = 4)
		batch_size = min(batch_size, n)
		function d(θ)
			indices = sample(1:n, batch_size, replace = false)
			-gradient(θ, indices)
 		end
	end

	delta_shift = 1e-2
	min_width = 5e-2
	max_width = 1e-1
	project! = PhaseProjection(delta_shift, min_width, max_width)
	# could have variable dependent learning rate
	θ = get_parameters(P)
	batch_size = 8
	∇ = stochastic_direction(batch_size)
	α = 1e-2 # learning rate
	β₁ = .99 # gradient momentum
	β₂ = .999 # 2nd moment momentum
	update! = AMSGRAD(∇, θ, α, β₁, β₂)
	for t in 1:maxiter
		update!(θ, t)
		project!(θ)
		# println(θ)
		# println(t)
	end
	set_parameters!(P, θ)
	return P
end

function optimize!(P::Phase, x::AbstractVector, Y::AbstractMatrix, c::AbstractMatrix;
													maxiter = 256, tol = 1e-4)
	optimize!(PhaseDirection(P, c), x, Y, maxiter = maxiter, tol = tol)
end

# a optimization is quadratic program
function optimize_a!(P::PhaseDirection, x::AbstractVector, Y::AbstractMatrix;
													maxiter = 128, tol = 1e-4)
    project!(x) = (@. x = nonnegative(x))
	function objective(a, indices = 1:length(P.a))
        A = Phase(P, a = a).(x, indices')
        residual_abs2(Y, A) + P.reg_a(a)
     end
	function direction(a, indices) # w.r.t. patterns chosen by indices
		f(a) = objective(a, indices)
		FD.gradient(f, a)
	end

	a = P.a
	γ = 5e-2 # learning rate
	β = .5 # momentum variable
	for i in 1:maxiter

	end
end

############################ old optimization ##################################
# function optimize!(P::Phase, x::AbstractVector, y::AbstractVecOrMat;
#                             tol = 1e-4, maxiter::Int = 16, opt_c::Bool = false)
#     for i in 1:maxiter
#         optimize_a!(P, x, y, tol = tol)
#         opt_c && optimize_dc!(P, x, y, tol = tol)
#         optimize_α!(P, x, y, tol = tol)
#         optimize_σ!(P, x, y, tol = tol)
#     end
#     return P
# end
#
# function get_optimizer(;tol = 1e-4, print_level = 0, max_iter = 32)
#     Model(optimizer_with_attributes(Ipopt.Optimizer, "tol" => tol,
#                                                 "print_level" => print_level,
#                                                 "max_iter" => max_iter))
# end
#
# # a optimization is quadratic program
# function optimize_a!(P::Phase, x::AbstractVector, Y::AbstractMatrix; tol = 1e-4)
#     model = get_optimizer(tol = tol)
#     n = length(P.a)
#     @variable(model, 0 ≤ a[i = 1:n], start = P.a[i])
#     function objective_a(a...)
#         A = Phase(P, a = [a...])(x)
#         residual_abs2(Y, A) # TODO: plus prior on a
#     end
#     register(model, :objective_a, n, objective_a, autodiff = true)
#     @NLobjective(model, Min, objective_a(a...))
#     JuMP.optimize!(model)
#     @. P.a = value(a)
# end
#
# function optimize_α!(P::Phase, x::AbstractVector, Y::AbstractMatrix;
#                                                     tol = 1e-4, dα = 5e-2)
#     model = get_optimizer(tol = tol)
#     n = length(P.a)
#     @variable(model, 1-dα ≤ α[i = 1:n] ≤ 1+dα, start = P.α[i])
#     function objective_α(α...)
#         A = Phase(P, α = [α...])(x)
#         residual_abs2(Y, A) # TODO: plus prior on α
#     end
#     register(model, :objective_α, n, objective_α, autodiff = true)
#     @NLobjective(model, Min, objective_α(α...))
#     JuMP.optimize!(model)
#     @. P.α = value(α)
# end
#
# function optimize_σ!(P::Phase, x::AbstractVector, Y::AbstractMatrix;
#                                         tol = 1e-4, min_σ = 1e-1, max_σ = 2e-1)
#     model = get_optimizer(tol = tol)
#     n = length(P.a)
#     @variable(model, min_σ ≤ σ[i = 1:n] ≤ max_σ, start = P.σ[i])
#     function objective_σ(σ...)
#         A = Phase(P, σ = [σ...])(x)
#         residual_abs2(Y, A) # TODO: plus prior on σ
#     end
#     register(model, :objective_σ, n, objective_σ, autodiff = true)
#     @NLobjective(model, Min, objective_σ(σ...))
#     JuMP.optimize!(model)
#     @. P.σ = value(σ)
# end
#
# # dc optimization is quadratic program
# function optimize_dc!(P::Phase, x::AbstractVector, Y::AbstractMatrix; tol = 1e-4)
#     model = get_optimizer(tol = tol)
#     n = length(P.c)
#     @variable(model, 0 ≤ c[i = 1:n], start = P.c[i] + P.dc[i])
#     function objective_dc(c...)
#         dc = @. c - P.c
#         A = Phase(P, dc = dc)(x)
#         residual_abs2(Y, A) # TODO: plus prior on c
#     end
#     register(model, :objective_dc, n, objective_dc, autodiff = true)
#     @NLobjective(model, Min, objective_dc(c...))
#     JuMP.optimize!(model)
#     @. P.dc = value(c) - P.c
# end
#
# function optimize_aασ!(P::Phase, x::AbstractVector, Y::AbstractMatrix;
#                                         dα = 5e-2, min_σ = 1e-1, max_σ = 3)
#     model = get_optimizer()
#     n = length(P.a)
#     @variable(model, 0 ≤ a[i = 1:n], start = P.a[i])
#     @variable(model, 1-dα ≤ α[i = 1:n] ≤ 1+dα, start = P.α[i])
#     @variable(model, min_σ ≤ σ[i = 1:n] ≤ max_σ, start = P.σ[i])
#     function objective_θ(θ...)
#         a = [ai for ai in θ[1:n]]
#         α = [ai for ai in θ[n+1:2n]]
#         σ = [ai for ai in θ[2n+1:3n]]
#         A = Phase(P, a = a, α = α, σ = σ)(x)
#         residual_abs2(Y, A) # TODO: plus prior on θ
#     end
#     register(model, :objective_θ, 3n, objective_θ, autodiff = true)
#     @NLobjective(model, Min, objective_θ(a..., α..., σ...))
#     JuMP.optimize!(model)
#     @. P.σ = value(σ)
# end
