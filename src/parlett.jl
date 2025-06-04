using SchurParlett

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
            taylor = taylor_expand(f, σ, order=n)
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
                # AX + XB + C = 0 → AX - XB = C
            end
        end
    end
    return F
end
