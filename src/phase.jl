struct Phase{T, V<:AbstractVector{T}, P}
    # per peak parameters
    c::V # peak intensity
    μ::V # peak location
    id::Int

    dc::V # peak intensity variation

    # per spectrogram parameters
    a::V # activation
    α::V # multiplicative shift
    σ::V # peak width

    # peak profile
    profile::P
end

# n is the number of spectrograms in the data
function Phase(c, μ, id::Int, n::Int; profile = Lorentz(), width_init::Real = 1.)
    length(c) == length(μ) || throw(DimensionMismatch())
    c, μ = promote(c, μ)
    dc = similar(c)
    T = eltype(c)
    a, α, σ = zeros(T, n), ones(T, n), fill(convert(T, width_init), n)
    Phase(c, μ, id, dc, a, α, σ, profile)
end

function Phase(S::StickPattern, n::Int; profile = Lorentz(), width_init::Real = 1.)
    Phase(S.c, S.μ, S.id, n, profile = profile, width_init = width_init)
end

# allows copying Phase object and only changing certain parameters
function Phase(P::Phase; dc = P.dc, a = P.a, α = P.α, σ = P.σ, profile = P.profile)
    length(P.a) == length(P.α) == length(P.σ) || throw(DimensionMismatch())
    c, μ, dc, a, α, σ = promote(P.c, P.μ, dc, a, α, σ)
    Phase(c, μ, P.id, dc, a, α, σ, profile)
end

# not counting c and μ which are assumed fixed
nparameters(P::Phase) = 3length(P.a) + length(P.dc)

# evaluate phase at x at jth spectrogram
function (P::Phase)(x::Real, j::Int)
    y = zero(x)
    α, σ = P.α[j], P.σ[j]
    @simd for i in eachindex(P.c)
        @inbounds begin
            c = P.c[i] + P.dc[i]
            μ = α * P.μ[i]
            y += c * P.profile((x-μ)/σ)
        end
    end
    P.a[j] * y
end

# evaluate all spectrograms
(P::Phase)(x::Real) = P.((x,), (1:length(P.a))')
(P::Phase)(x::AbstractVector) = P.(x, (1:length(P.a))')

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

function residual_abs2(X::AbstractArray, Y::AbstractArray)
    size(X) == size(Y) || throw(DimensionMismatch())
    r = zero(promote_type(eltype(X), eltype(Y)))
    @simd for i in eachindex(X)
        @inbounds r += abs2(X[i]-Y[i])
    end
    r
end

function optimize!(P::Phase, x::AbstractVector, y::AbstractVecOrMat;
                            tol = 1e-4, maxiter::Int = 16, opt_c::Bool = false)
    for i in 1:maxiter
        optimize_a!(P, x, y, tol = tol)
        opt_c && optimize_dc!(P, x, y, tol = tol)
        optimize_α!(P, x, y, tol = tol)
        optimize_σ!(P, x, y, tol = tol)
    end
    return P
end

function get_optimizer(;tol = 1e-4, print_level = 0, max_iter = 32)
    Model(optimizer_with_attributes(Ipopt.Optimizer, "tol" => tol,
                                                "print_level" => print_level,
                                                "max_iter" => max_iter))
end
# a optimization is quadratic program
function optimize_a!(P::Phase, x::AbstractVector, Y::AbstractMatrix; tol = 1e-4)
    model = get_optimizer(tol = tol)
    n = length(P.a)
    @variable(model, 0 ≤ a[i = 1:n], start = P.a[i])
    function objective_a(a...)
        A = Phase(P, a = [a...])(x)
        residual_abs2(Y, A) # TODO: plus prior on a
    end
    register(model, :objective_a, n, objective_a, autodiff = true)
    @NLobjective(model, Min, objective_a(a...))
    JuMP.optimize!(model)
    @. P.a = value(a)
end

function optimize_α!(P::Phase, x::AbstractVector, Y::AbstractMatrix;
                                                    tol = 1e-4, dα = 5e-2)
    model = get_optimizer(tol = tol)
    n = length(P.a)
    @variable(model, 1-dα ≤ α[i = 1:n] ≤ 1+dα, start = P.α[i])
    function objective_α(α...)
        A = Phase(P, α = [α...])(x)
        residual_abs2(Y, A) # TODO: plus prior on α
    end
    register(model, :objective_α, n, objective_α, autodiff = true)
    @NLobjective(model, Min, objective_α(α...))
    JuMP.optimize!(model)
    @. P.α = value(α)
end

function optimize_σ!(P::Phase, x::AbstractVector, Y::AbstractMatrix;
                                        tol = 1e-4, min_σ = 1e-1, max_σ = 2e-1)
    model = get_optimizer(tol = tol)
    n = length(P.a)
    @variable(model, min_σ ≤ σ[i = 1:n] ≤ max_σ, start = P.σ[i])
    function objective_σ(σ...)
        A = Phase(P, σ = [σ...])(x)
        residual_abs2(Y, A) # TODO: plus prior on σ
    end
    register(model, :objective_σ, n, objective_σ, autodiff = true)
    @NLobjective(model, Min, objective_σ(σ...))
    JuMP.optimize!(model)
    @. P.σ = value(σ)
end

# dc optimization is quadratic program
function optimize_dc!(P::Phase, x::AbstractVector, Y::AbstractMatrix; tol = 1e-4)
    model = get_optimizer(tol = tol)
    n = length(P.c)
    @variable(model, 0 ≤ c[i = 1:n], start = P.c[i] + P.dc[i])
    function objective_dc(c...)
        dc = @. c - P.c
        A = Phase(P, dc = dc)(x)
        residual_abs2(Y, A) # TODO: plus prior on c
    end
    register(model, :objective_dc, n, objective_dc, autodiff = true)
    @NLobjective(model, Min, objective_dc(c...))
    JuMP.optimize!(model)
    @. P.dc = value(c) - P.c
end

function optimize_aασ!(P::Phase, x::AbstractVector, Y::AbstractMatrix;
                                        dα = 5e-2, min_σ = 1e-1, max_σ = 3)
    model = get_optimizer()
    n = length(P.a)
    @variable(model, 0 ≤ a[i = 1:n], start = P.a[i])
    @variable(model, 1-dα ≤ α[i = 1:n] ≤ 1+dα, start = P.α[i])
    @variable(model, min_σ ≤ σ[i = 1:n] ≤ max_σ, start = P.σ[i])
    function objective_θ(θ...)
        a = [ai for ai in θ[1:n]]
        α = [ai for ai in θ[n+1:2n]]
        σ = [ai for ai in θ[2n+1:3n]]
        A = Phase(P, a = a, α = α, σ = σ)(x)
        residual_abs2(Y, A) # TODO: plus prior on θ
    end
    register(model, :objective_θ, 3n, objective_θ, autodiff = true)
    @NLobjective(model, Min, objective_θ(a..., α..., σ...))
    JuMP.optimize!(model)
    @. P.σ = value(σ)
end

#### old code
# temporary per peak parameters
# ct::V
# μt::V
# ct, μt = similar.((c, μ))
# calculate temporaries for jth spectrogram
# function temporaries(P::Phase, j::Int)
#     @. P.ct = P.c[i] + P.dc[i]
#     @. P.μt = P.α[j] * P.μ[i]
# end

# function (P::Phase)(x::AbstractVector)
#     c, μ, dc = reshape.((P.c, P.μ, P.dc), 1, :)
#     a, α, σ = reshape.((P.a, P.α, P.σ), 1, :)
#
#     @. a * (c + dc) * P.profile( (x - α * μ) / σ )
# end

# # prior distribution over phase parameters
# function prior(θ::AbstractMatrix)
#     a, α, σ = θ[1,:], θ[2,:], θ[3,:]
#     # prior_a = nld(Normal(0., 1.)) # prior on activation variance
#     prior_a(a) = 10a
#     prior_α = nld(Normal(log(1.), 2e-3^2)) # controls prior of shift, prior variance between 1e-2 to 1e-3 squared
#     prior_σ = nld(Normal(log(.5), 1e-1^2)) # prior of peak width
#     p = 0
#     p += sum(prior_a, exp.(a))
#     p += sum(prior_α, α)
#     p += sum(prior_σ, σ)
# end
