using SchurParlett

"""
    Generates a random normal matrix of size n
"""
function random_normal_matrix(n::Int)
    # Create random eigenvalues
    D = diagm(complex.(randn(n), randn(n)))
    
    # Create a random unitary matrix for the similarity transform
    Q = qr(randn(n)).Q
    
    # Return the normal matrix
    return Q * D * Q'
end
