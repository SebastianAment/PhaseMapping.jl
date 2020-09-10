using Kernel: MercerKernel
using LinearAlgebraExtensions: lowrank
using WoodburyIdentity: Woodbury

struct ReproducingKernelHilbertSpace{T, K, KT<:AbstractMatOrFac{T}}
    K⁻¹::KT
end
const RKHS = ReproducingKernelHilbertSpace
function RKHS(k::MercerKernel, x::AbstractVecOrMat)
    K = Kernel.gramian(k, x)
    K = lowrank(cholesky, K)
    K⁻¹ = pinverse(factorize(K)) # TODO: need pinverse for low rank factorization
    RKHS(K⁻¹)
end

# using LazyInverse
# using LinearAlgebraExtensions: LowRank
# function LazyInverse.pinverse(F::LowRank)
#     qrU = pinverse(qr(F.U))
#     qrV = F.U ≡ F.V' ? qrU : pinverse(qr(F.V'))''
#     LowRank(qrV, qrU)
# end

function RKHS(k::MercerKernel, x::AbstractVecOrMat, σ::Real)
    K = Kernel.gramian(k, x)
    K = lowrank(cholesky, K)
    K = Woodbury(σ^2 * I())
    K⁻¹ = inverse(factorize(K))
    RKHS(K⁻¹)
end

LinearAlgebra.norm(H::RKHS, x::AbstractVector) = sqrt(H(x, x))
function (H::RKHS)(x::AbstractVector, y::AbstractVector = x)
    dot(x, H.K⁻¹, y)
end

# for phase regularization
# activation regularization, c is composition
function a_regularizer(c::AbstractMatrix; a::Real = 1., l::Real = .05, k = Kernel.EQ())
    k = a * Kernel.Lengthscale(k, l)
    return RKHS(k, c)
end

# shift regularization, c is composition
# needs to be called like H(α-1) to penalize deviation from 1
function α_regularizer(c::AbstractMatrix; a::Real = 1e-2^2, l::Real = .05, k = Kernel.EQ())
    k = a * Kernel.Lengthscale(k, l)
    H = RKHS(k, c)
    return α -> H(α.-1)
end

# activation regularization, c is composition
function σ_regularizer(c::AbstractMatrix; a::Real = 2e-1^2, l::Real = .05, k = Kernel.EQ())
    k = a * Kernel.Lengthscale(k, l)
    H = RKHS(k, c)
    return σ -> H(σ.-mean(σ)) # penalize deviation from mean width
end
