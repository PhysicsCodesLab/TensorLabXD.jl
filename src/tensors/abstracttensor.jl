"""
    abstract type AbstractTensorMap{S<:IndexSpace, N₁, N₂} end

Abstract supertype of all tensor maps, i.e. linear maps between tensor products
of vector spaces of type `S<:IndexSpace`. An `AbstractTensorMap` maps from
an input space of type `ProductSpace{S, N₂}` to an output space of type
`ProductSpace{S, N₁}`.
"""
abstract type AbstractTensorMap{S<:IndexSpace, N₁, N₂} end

"""
    const AbstractTensor{S<:IndexSpace, N} = AbstractTensorMap{S, N, 0}

Abstract supertype of all tensors, i.e. elements in the tensor product space
of type `ProductSpace{S, N}`, built from elementary spaces of type `S<:IndexSpace`.

An `AbstractTensor{S, N}` is actually a special case `AbstractTensorMap{S, N, 0}`,
i.e. a tensor map with only non-trivial output space.
"""
const AbstractTensor{S<:IndexSpace, N} = AbstractTensorMap{S, N, 0}

"""
    storagetype(t::AbstractTensorMap)

Return the storage type of the data for the tensor map instance, for example,
Matrix{Float64}. Now we only have this type as DenseMatrix.
"""
storagetype(t::AbstractTensorMap) = storagetype(typeof(t))

similarstoragetype(TT::Type{<:AbstractTensorMap}, ::Type{T}) where {T} =
    Core.Compiler.return_type(similar, Tuple{storagetype(TT), Type{T}})
similarstoragetype(t::AbstractTensorMap, T) = similarstoragetype(typeof(t), T)

"""
    Base.eltype(T::Type{<:AbstractTensorMap})
    Base.eltype(t::AbstractTensorMap)

Element type of the data for the tensormap in storage. Return, for example, `Float64`.
"""
Base.eltype(T::Type{<:AbstractTensorMap}) = eltype(storagetype(T))
Base.eltype(t::AbstractTensorMap) = eltype(typeof(t))

"""
    field(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace}
    field(t::AbstractTensorMap)

Return the field type over which a vector space instance or type is defined.
"""
field(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = field(S)
field(t::AbstractTensorMap) = field(typeof(t))

"""
    spacetype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace}
    spacetype(t::AbstractTensorMap)

Return the ElementarySpace type associated to a AbstractTensorMap instance or type.
"""
spacetype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = S
spacetype(t::AbstractTensorMap) = spacetype(typeof(t))

"""
    sectortype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace}
    sectortype(t::AbstractTensorMap) -> Type{<:Sector}

Return the type of sector over which the AbstractTensorMap `t` or type is defined.
"""
sectortype(::Type{<:AbstractTensorMap{S}}) where {S<:IndexSpace} = sectortype(S)
sectortype(t::AbstractTensorMap) = sectortype(typeof(t))

"""
    numout(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂}
    numout(t::AbstractTensorMap) -> Int

Return the number of the index spaces in the codomain of a AbstractTensorMap instance or
type.
"""
numout(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂} = N₁
numout(t::AbstractTensorMap) = numout(typeof(t))
"""
    numin(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂}
    numin(t::AbstractTensorMap) -> Int

Return the number of the index spaces in the domain of a AbstractTensorMap instance or type.
"""
numin(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂} = N₂
numin(t::AbstractTensorMap) = numin(typeof(t))

"""
    numind(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂}
    numind(t::AbstractTensorMap) -> Int

Return the total number of the index spaces of a AbstractTensorMap instance or type.
`order` is the alias of `numind`.
"""
numind(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂} = N₁ + N₂
numind(t::AbstractTensorMap) = numind(typeof(t))
const order = numind

"""
    codomainind(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂}
    codomainind(t::AbstractTensorMap) -> Tuple

Return the indices of the codomain of a tensor map instance or type as a tuple of Int.
"""
codomainind(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂} = ntuple(n->n, N₁)
codomainind(t::AbstractTensorMap) = codomainind(typeof(t))

"""
    domainind(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂}
    domainind(t::AbstractTensorMap) -> Tuple{Int}

Return the indices of the domain of a tensor map instance or type as a tuple of Int.
"""
domainind(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂} = ntuple(n-> N₁+n, N₂)
domainind(t::AbstractTensorMap) = domainind(typeof(t))

"""
    allind(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂}
    allind(t::AbstractTensorMap) -> Tuple{Int}

Return all indices of a tensor map instance or type as a tuple of Int.
"""
allind(::Type{<:AbstractTensorMap{<:IndexSpace, N₁, N₂}}) where {N₁, N₂} = ntuple(n->n, N₁+N₂)
allind(t::AbstractTensorMap) = allind(typeof(t))

"""
    codomian(t::AbstractTensorMap)

Return the codomain of the tensor map `t`.

    codomain(t::AbstractTensorMap, i)

Return the `i`th index space of the codomain of the tensor map `t`.
"""
codomain(t::AbstractTensorMap, i) = codomain(t)[i]

"""
    domain(t::AbstractTensorMap)

Return the domain of the tensor map `t`.

    domain(t::AbstractTensorMap, i)

Return the `i`th index space of the domain of the tensor map `t`.
"""
domain(t::AbstractTensorMap, i) = domain(t)[i]

"""
    target(t::AbstractTensorMap)

Return the codomain of the tensor map `t`.
"""
target(t::AbstractTensorMap) = codomain(t) # categorical terminology

"""
    source(t::AbstractTensorMap)

Return the domain of the tensor map `t`.
"""
source(t::AbstractTensorMap) = domain(t) # categorical terminology

"""
    space(t::AbstractTensorMap) -> HomSpace

Return the HomSpace corresponding to the tensor map `t` instance.
"""
space(t::AbstractTensorMap) = HomSpace(codomain(t), domain(t))

"""
    space(t::AbstractTensorMap, i::Int)

Return the `i`th index space of the HomSpace corresponding to the tensor map `t`.
"""
space(t::AbstractTensorMap, i::Int) = space(t)[i]

"""
    dim(t::AbstractTensorMap)

Return the total dimension of the HomSpace corresponding to the tensor map `t`.
"""
dim(t::AbstractTensorMap) = dim(space(t))

"""
    adjointtensorindex(t::AbstractTensorMap{<:IndexSpace, N₁, N₂}, i) where {N₁, N₂} -> Int

Return the index of the `i`th index of the tensor map `t` in its adjoint.
"""
adjointtensorindex(t::AbstractTensorMap{<:IndexSpace, N₁, N₂}, i) where {N₁, N₂} =
    ifelse(i<=N₁, N₂+i, i-N₁)

"""
    adjointtensorindices(t::AbstractTensorMap, indices::IndexTuple)

Return the indices of the `indices` of the tensor map `t` in its adjoint.
"""
adjointtensorindices(t::AbstractTensorMap, indices::IndexTuple) =
    map(i->adjointtensorindex(t, i), indices)


function Base.:(==)(t1::AbstractTensorMap, t2::AbstractTensorMap)
    (codomain(t1) == codomain(t2) && domain(t1) == domain(t2)) || return false
    for c in blocksectors(t1)
        block(t1, c) == block(t2, c) || return false
    end
    return true
end
function Base.hash(t::AbstractTensorMap, h::UInt)
    h = hash(codomain(t), h)
    h = hash(domain(t), h)
    for (c, b) in blocks(t)
        h = hash(c, hash(b, h))
    end
    return h
end

function Base.isapprox(t1::AbstractTensorMap, t2::AbstractTensorMap;
                atol::Real=0, rtol::Real=Base.rtoldefault(eltype(t1), eltype(t2), atol))
    d = norm(t1 - t2)
    if isfinite(d)
        return d <= max(atol, rtol*max(norm(t1), norm(t2)))
    else
        return false
    end
end

# Conversion to Array:
# probably not optimized for speed, only for checking purposes
function Base.convert(::Type{Array}, t::AbstractTensorMap{S, N₁, N₂}) where {S, N₁, N₂}
    I = sectortype(t)
    if I === Trivial
        convert(Array, t[])
    else
        cod = codomain(t)
        dom = domain(t)
        local A
        for (f1, f2) in fusiontrees(t)
            F1 = convert(Array, f1)
            F2 = convert(Array, f2)
            sz1 = size(F1)
            sz2 = size(F2)
            d1 = TupleLabXD.front(sz1)
            d2 = TupleLabXD.front(sz2)
            F = reshape(reshape(F1, TupleLabXD.prod(d1), sz1[end])*reshape(F2, TupleLabXD.prod(d2), sz2[end])', (d1..., d2...))
            if !(@isdefined A)
                if eltype(F) <: Complex
                    T = complex(float(eltype(t)))
                elseif eltype(F) <: Integer
                    T = eltype(t)
                else
                    T = float(eltype(t))
                end
                A = fill(zero(T), (dims(cod)..., dims(dom)...))
            end
            Aslice = StridedView(A)[axes(cod, f1.uncoupled)..., axes(dom, f2.uncoupled)...]
            axpy!(1, StridedView(_kron(convert(Array, t[f1, f2]), F)), Aslice)
        end
        return A
    end
end
