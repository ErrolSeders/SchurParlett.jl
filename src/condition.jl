using SchurParlett

"""
    Compute the absolute condition number for a normal matrix.
"""
function abs_cond_func(f, A::Matrix{<:Complex})::Float64
    # TODO: Assert normal?

    Λ = eigvals(A)
    max = 0.0
    for λ ∈ Λ # Only half
        for μ ∈ Λ # Only half
            div_diff = abs(divided_difference(f, λ, μ)) # Only if not equal
            if div_diff > max
                max = div_diff
            end
        end
    end
    return max
end

"""
    The divided_difference between λ and μ for f.
"""
function divided_difference(f, λ, μ)::ComplexF64
    return (f(λ) - f(μ)) / (λ - μ)
end
