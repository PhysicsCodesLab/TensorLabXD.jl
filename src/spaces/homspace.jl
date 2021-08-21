"""
    struct HomSpace{S<:ElementarySpace, P1<:CompositeSpace{S}, P2<:CompositeSpace{S}}
        codomain::P1
        domain::P2
    end

Represents the linear space of morphisms with codomain of type `P1` and domain of type `P2`.
Note that HomSpace is not a subtype of VectorSpace, i.e. we restrict the latter to denote
certain categories and their objects, and keep HomSpace distinct.
"""
struct HomSpace{S<:ElementarySpace, P1<:CompositeSpace{S}, P2<:CompositeSpace{S}}
    codomain::P1
    domain::P2
end

"""
    TensorSpace{S<:ElementarySpace}

A constant which is the union type of both elementary and product vector space and defined
as `HomSpace{S, ProductSpace{S, N₁}, ProductSpace{S, N₂}}`.
"""
const TensorSpace{S<:ElementarySpace} = Union{S, ProductSpace{S}}

"""
    TensorMapSpace{S<:ElementarySpace, N₁, N₂}

A constant which is an alias name of the HomSpace and defined as
`HomSpace{S, ProductSpace{S, N₁}, ProductSpace{S, N₂}}`.
"""
const TensorMapSpace{S<:ElementarySpace, N₁, N₂} =
    HomSpace{S, ProductSpace{S, N₁}, ProductSpace{S, N₂}}

"""
    →(dom::TensorSpace{S}, codom::TensorSpace{S}) where {S<:ElementarySpace}
    ←(codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:ElementarySpace}

Convenient constructor of HomSpace instance.
"""
→(dom::TensorSpace{S}, codom::TensorSpace{S}) where {S<:ElementarySpace} =
    HomSpace(ProductSpace(codom), ProductSpace(dom))

←(codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:ElementarySpace} =
    HomSpace(ProductSpace(codom), ProductSpace(dom))

# The index of a HomSpace is defined as the following: first, index spaces for codomain,
# then, index spaces for domain; dual is taken for each index space of domain, but the order
# are not reversed.
Base.getindex(W::TensorMapSpace{<:IndexSpace, N₁, N₂}, i) where {N₁, N₂} =
    i <= N₁ ? codomain(W)[i] : dual(domain(W)[i-N₁])

"""
    codomain(W::HomSpace)

Return the codomain of a HomSpace.
"""
codomain(W::HomSpace) = W.codomain

"""
    domain(W::HomSpace)

Return the domain of a HomSpace.
"""
domain(W::HomSpace) = W.domain

"""
    spacetype(W::HomSpace) -> Type{ElementarySpace}
    spacetype(::Type{<:HomSpace{S}})

Return the ElementarySpace type associated to a HomSpace instance or type.
"""
spacetype(W::HomSpace) = spacetype(typeof(W))
spacetype(::Type{<:HomSpace{S}}) where S = S

"""
    field(W::HomSpace) -> Field
    field(L::Type{<:HomSpace})

Return the field type over which a HomSpace instance or type is defined.
"""
field(W::HomSpace) = field(typeof(W))
field(L::Type{<:HomSpace}) = field(spacetype(L))

"""
    sectortype(W::HomSpace) -> Type{<:Sector}
    sectortype(L::Type{<:HomSpace)

Return the type of sector over which the HomSpace `W` is defined.
"""
sectortype(W::HomSpace) = sectortype(typeof(W))
sectortype(L::Type{<:HomSpace}) = sectortype(spacetype(L))

"""
    blocksectors(W::HomSpace)

Return an iterator over the different unique coupled sector labels, i.e. the intersection
of the different fusion outputs that can be obtained by fusing the sectors present in the
domain, as well as from the codomain.
"""
function blocksectors(W::HomSpace)
    sectortype(W) === Trivial &&
        return TrivialOrEmptyIterator(dim(domain(W)) == 0 || dim(codomain(W)) == 0)
    return intersect(blocksectors(codomain(W)), blocksectors(domain(W)))
end

"""
    dim(W::HomSpace)

Return the total dimension of a `HomSpace`, i.e. the number of linearly independent
morphisms that can be constructed within this space.
"""
function dim(W::HomSpace)
    d = 0
    for c in blocksectors(W)
        d += blockdim(codomain(W), c) * blockdim(domain(W), c)
    end
    return d
end

"""
    dual(W::HomSpace)

Return the dual of a HomSpace which contains the dual of morphisms in this space.
It corresponds to 180 degree rotation in the graphical representation.
"""
dual(W::HomSpace) = HomSpace(dual(W.domain), dual(W.codomain))

"""
    adjoint(W::HomSpace{<:EuclideanSpace})

Return the adjoint of a HomSpace which contains the dagger of morphisms in this space.
It corresponds to mirror operation and then reversing all arrows in the graphical
representation.
"""
Base.adjoint(W::HomSpace{<:EuclideanSpace}) = HomSpace(W.domain, W.codomain)

Base.hash(W::HomSpace, h::UInt) = hash(domain(W), hash(codomain(W), h))
Base.:(==)(W1::HomSpace, W2::HomSpace) =
    (W1.codomain == W2.codomain) && (W1.domain == W2.domain)

function Base.show(io::IO, W::HomSpace)
    if length(W.codomain) == 1
        print(io, W.codomain[1])
    else
        print(io, W.codomain)
    end
    print(io, " ← ")
    if length(W.domain) == 1
        print(io, W.domain[1])
    else
        print(io, W.domain)
    end
end
