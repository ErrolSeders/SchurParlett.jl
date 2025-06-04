module SchurParlett

include("blocking.jl")
include("unionfind.jl")

using TaylorSeries
using UnicodePlots
using LinearAlgebra
using DelaunayTriangulation

#!TODO!:
# 1:    Add condition number calculation
# 2:    Try and trim Delaunay blockings allocations size down.
#       Doing so should make it faster than the other algorithm
#       for sufficiently large matrices.
# 3:    Investigate more accurate and fast ways to do
#       Taylor Series and Sylvester solving.

function parlett_recurrance(f, T::UpperTriangular)
    n = size(T)[1]
    F = zeros(ComplexF64, (n, n))
    for i ∈ 1:n
        F[i, i] = f(T[i, i])
    end
    for j ∈ 2:n
        for i in j-1:-1:1
            denom = T[i, i] - T[j, j]
            S = (F[i, i] - F[j, j]) / denom
            for k ∈ i+1:j-1
                S += (F[i, k] * T[k, j] - T[i, k] * F[k, j]) / denom
            end
            F[i, j] = T[i, j] * S
        end
    end
    return F
end

function block_parlett_recurrance(f, T, blocks)
    F = zeros(ComplexF64, size(T))
    for (i, block) in enumerate(blocks)
        Tii = T[block, block]
        n = size(Tii)[1]
        if n == 1
            F[block, block] = f.(Tii)
        else
            σ = tr(Tii) / n
            M = Tii - σ * I(n)
            taylor = taylor_expand(f, σ, order=10)
            F[block, block] = taylor(M)
        end
        n = length(blocks)
        for i ∈ 1:n
            for j ∈ i+1:n
                bi, bj = blocks[i], blocks[j]
                C = F[bi, bi] * T[bi, bj] - T[bi, bj] * F[bj, bj]
                for k ∈ i+1:j-1
                    bk = blocks[k]
                    C += F[bi, bk] * T[bk, bj] - T[bi, bk] * F[bk, bj]
                end
                A = T[bi, bi]
                B = T[bj, bj]
                # Is there a faster way to calculate this?
                # Sylvester does schur decompositions on A and B!
                # This can also fail for some matrices with a very cryptic
                # LAPACKException(1)
                F[bi, bj] .= sylvester(A, -B, -C)
            end
        end
    end
    return F
end

"""
  `schur_parlett(f, A [, ε=0.1])`

  For given *complex valued* function `f: C -> C` and complex matrix `A`
  compute the matrix function `f(A)` using the Schur-Parlett algorithm

  `ε > 0` is a blocking parameter. This method is highly sensitive to the
  clustering and distribution of eigenvalues, *tuning this value is important*.

"""
function schur_parlett(f, A::Matrix{<:Complex}, δ=0.1)::AbstractMatrix
    @assert δ > 0 "Blocking parameter `ε` cannot be negative!"
    Q = schur(A) #! Consider writing a mutating version that uses schur!(A) for potential 2x speedup.
    T, Z, Λ = Q
    if isdiag(T)
        F = f.(T)
        return Z * F * Z'
    else
        T = T |> UpperTriangular
    end

    if all(z -> norm(z.im) <= Float64 |> eps |> sqrt, Λ) # All imaginary components ≈ 0
        L = map(z -> z.re, Λ)
        q = real_eigenvalue_blocking(δ, L)
    else
        q = block_pattern(δ, Λ)
    end

    perm = collect(1:length(q))
    for c ∈ reverse(1:length(q))
        select = (q .== c)

        selected = findall(select)
        remaining = findall(.!select)
        reorder = [selected; remaining]
        ordschur!(Q, select)
        perm = perm[reorder]
    end
    T, Z, Λ = Q
    q_reorder = q[perm]
    blocks = block_indices(q_reorder)
    F = block_parlett_recurrance(f, T, blocks)
    Z * F * Z'
end


end
