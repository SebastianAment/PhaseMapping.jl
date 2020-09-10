module TestNMF
using PhaseMapping: xray
using Test
using LinearAlgebra

n, r, m = 16, 3, 32

tol = 1e-3
W, H = rand(n, r), rand(r, m)
H ./= 2sum(H, dims = 1)
H[1:r, 1:r] = I(3)
A = W*H
@test A[:, 1:r] ≈ W
x = rand(m)
b = A*x

# testing nnls
# xn = nnls(A, b)
# @test all(≥(0), xn)
# @test norm(A*xn-b) < tol

WX, HX, K = xray(A, 2r, tol)

using BenchmarkTools
@btime xray($A, 2r, $tol)

@test sort(K) == 1:r # recovers correct anchor words
@test norm(WX*HX-A) < tol

end
