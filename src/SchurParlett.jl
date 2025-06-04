module SchurParlett


using TaylorSeries, LinearAlgebra, DelaunayTriangulation

include("blocking.jl")
include("parlett.jl")



#!TODO!:
# 1:    Add condition number calculation
# 2:    Try and trim Delaunay blockings allocations size down.
#       Doing so should make it faster than the other algorithm
#       for sufficiently large matrices.
# 3:    Investigate more accurate and fast ways to do
#       Taylor Series and Sylvester solving.

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

export schur_parlett

end
