using SchurParlett

struct UnionFind
    parent::Vector{Int}
    size::Vector{Int}
end

function UnionFind(n)
    UnionFind(collect(1:n), ones(Int, n))
end

function find(uf::UnionFind, x::Int)
    while uf.parent[x] != x
        uf.parent[x] = uf.parent[uf.parent[x]] # Path compression
        x = uf.parent[x]
    end
    return x
end

function union!(uf::UnionFind, x::Int, y::Int)
    xroot = find(uf, x)
    yroot = find(uf, y)
    if xroot == yroot
        return
    end
    # Union by size
    if uf.size[xroot] < uf.size[yroot]
        uf.parent[xroot] = yroot
        uf.size[yroot] += uf.size[xroot]
    else
        uf.parent[yroot] = xroot
        uf.size[xroot] += uf.size[yroot]
    end
end

function labels_from_uf(uf::UnionFind)
    n = length(uf.parent)
    labels = zeros(Int, n)
    label = 0
    root_to_label = Dict{Int,Int}()
    for i in 1:n
        root = find(uf, i)
        if !haskey(root_to_label, root)
            label += 1
            root_to_label[root] = label
        end
        labels[i] = root_to_label[root]
    end
    return labels
end

function real_eigenvalue_blocking(δ::AbstractFloat, Λ::Vector{<:Real})
    sorted, perm = sort(Λ), sortperm(Λ)
    gaps = [Inf; diff(sorted)]
    cluster_ids = cumsum(gaps .> δ)

    invperm = invpermute!([1:(sorted|>length)...], perm)
    return cluster_ids[invperm]
end

# This function works as expected, but is ≈5x slower than block_pattern for matrices of a reasonable size sadly.
function delaunay_graph_blocking(δ::AbstractFloat, Λ::Vector)
    λp = map(z -> (z.re, z.im), Λ)
    n = λp |> length
    uf = UnionFind(n)
    for (i, j) ∈ (triangulate(λp) |> get_graph).edges
        if i > 0 && j > 0 && abs(Λ[i] - Λ[j]) <= δ
            union!(uf, i, j)
        end
    end
    return labels_from_uf(uf)
end

"""
  `block_pattern(ε::AbstractFloat, Λ::Vector{Complex})`

  Reorder and block the upper triangular schur factor `T` so that
  each diagonal block contains closely clustered eigenvalues from `T`.
  Eigenvalues within diagonal blocks should also be well distributed
  locally in order to preserve numerical stability.

"""
function block_pattern(δ::AbstractFloat, Λ::Vector{<:Complex})
    n = length(Λ)
    labels = zeros(Int, n)
    nextlabel = 0
    for i in 1:n
        if labels[i] == 0
            nextlabel += 1
            labels[i] = nextlabel
        end
        for j in i+1:n
            if abs(Λ[i] - Λ[j]) ≤ δ
                if labels[j] == 0
                    labels[j] = labels[i]
                else
                    # unify clusters labels[i] and labels[j]
                    old = labels[j]
                    labels[labels.==old] .= labels[i]
                end
            end
        end
    end
    labels
end

function block_indices(q::Vector{Int})
    unique_labels = unique(q)
    blocks = Vector{UnitRange{Int}}()
    for label in unique_labels
        idxs = findall(q .== label)
        push!(blocks, minimum(idxs):maximum(idxs))
    end
    blocks
end
