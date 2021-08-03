# FIELDS:
"""
    abstract type Field end

Abstract type at the top of the type hierarchy for denoting fields over which vector spaces
(or more generally, linear categories) can be defined. Two common fields are `ℝ` and `ℂ`,
representing the field of real or complex numbers respectively.
"""
abstract type Field end

struct RealNumbers <: Field end
struct ComplexNumbers <: Field end

"""
    const ℝ = RealNumbers()

Convenient representation of `RealNumbers` instance.
"""
const ℝ = RealNumbers()

"""
    const ℂ = ComplexNumbers()

Convenient representation of `ComplexNumbers` instance.
"""
const ℂ = ComplexNumbers()

Base.in(::Any, ::Field) = false
Base.in(::Real, ::RealNumbers) = true
Base.in(::Number, ::ComplexNumbers) = true

Base.issubset(::Type, ::Field) = false
Base.issubset(::Type{<:Real}, ::RealNumbers) = true
Base.issubset(::Type{<:Number}, ::ComplexNumbers) = true
Base.issubset(::RealNumbers, ::RealNumbers) = true
Base.issubset(::RealNumbers, ::ComplexNumbers) = true
Base.issubset(::ComplexNumbers, ::RealNumbers) = false
Base.issubset(::ComplexNumbers, ::ComplexNumbers) = true

Base.show(io::IO, ::RealNumbers) = print(io, "ℝ")
Base.show(io::IO, ::ComplexNumbers) = print(io, "ℂ")

# VECTOR SPACES:
"""
    abstract type VectorSpace end

Abstract type at the top of the type hierarchy for denoting vector spaces, or, more
accurately, the objects of 𝕜-linear monoidal categories.
"""
abstract type VectorSpace end

"""
    abstract type ElementarySpace{𝕜} <: VectorSpace end

ElementarySpace over a field `𝕜` is a super type for all vector spaces (objects) that can
be associated with the individual indices of a tensor, as hinted to by its alias IndexSpace.

Every elementary vector space should has the methods [`conj`](@ref) and [`dual`](@ref),
returning the complex conjugate space and the dual space respectively. The complex conjugate
of the dual space is obtained as `dual(conj(V)) === conj(dual(V))`.

These different spaces should be of the same type, so that a tensor can be defined as an
element of a homogeneous tensor product of these spaces.
"""
abstract type ElementarySpace{𝕜} <: VectorSpace end

"""
    const IndexSpace = ElementarySpace

Alias for the abstact type ElementarySpace.
"""
const IndexSpace = ElementarySpace

"""
    abstract type InnerProductSpace{𝕜} <: ElementarySpace{𝕜} end

Abstract type for denoting vector spaces with an inner product, thus a canonical mapping
from `dual(V)` to `conj(V)`. This mapping is provided by the metric, but no further support
for working with metrics is currently implemented.
"""
abstract type InnerProductSpace{𝕜} <: ElementarySpace{𝕜} end

"""
    abstract type EuclideanSpace{𝕜} <: InnerProductSpace{𝕜} end

Abstract type for denoting real or complex spaces with a standard Euclidean inner product
(i.e. orthonormal basis, and the metric is identity), such that the dual space is naturally
isomorphic to the conjugate space `dual(V) == conj(V)`, also known as the category of
finite-dimensional Hilbert spaces ``FdHilb``. In the language of categories, this subtype
represents dagger or unitary categories, and support an adjoint operation.
"""
abstract type EuclideanSpace{𝕜} <: InnerProductSpace{𝕜} end # 𝕜 should be ℝ or ℂ

# Specific realizations of ElementarySpace types
# spaces without internal structure
include("cartesianspace.jl")
include("complexspace.jl")
include("generalspace.jl") # never used at other parts of the package

# space with internal structure corresponding to the irreducible representations
# of a group, or more generally, the simple objects of a fusion category.
include("gradedspace.jl")

# Composite vector spaces
"""
    abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end

Abstract type for composite spaces that are defined in terms of a number of elementary
vector spaces of a homogeneous type `S<:ElementarySpace{𝕜}`.
"""
abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end

# Specific realizations of CompositeSpace types
# a tensor product of N elementary spaces of the same type S
include("productspace.jl")
# deligne tensor product
include("deligne.jl")

# HomSpace: space of morphisms
include("homspace.jl")

# general VectorSpace methods
"""
    field(::VectorSpace) -> Field
    field(::Type{<:VectorSpace})

Return the `Field` over which a vector space instance or type is defined.
"""
function field end
field(V::VectorSpace) = field(typeof(V))
field(::Type{<:ElementarySpace{𝕜}}) where {𝕜} = 𝕜
field(P::Type{<:CompositeSpace}) = field(spacetype(P))

"""
    spacetype(::VectorSpace) -> Type{ElementarySpace}
    spacetype(::Type{<:VectorSpace})

Return the `ElementarySpace` associated to a VectorSpace instance or type.
"""
function spacetype end
spacetype(V::ElementarySpace) = typeof(V)
spacetype(S::Type{<:ElementarySpace}) = S
spacetype(V::CompositeSpace) = spacetype(typeof(V))
spacetype(::Type{<:CompositeSpace{S}}) where S = S

"""
    oneunit(V::S) where {S<:ElementarySpace} -> S
    oneunit(::Type{<:ElementarySpace})

Extend Base.oneunit. Return the corresponding vector space of type `S` that represents the
trivial one-dimensional space, i.e. the space that is isomorphic to the corresponding field.

Note that this is different from `one(V::S)`, which returns the empty product space
`ProductSpace{S,0}(())`.
"""
Base.oneunit(V::ElementarySpace) = oneunit(typeof(V))

"""
    sectortype(V::VectorSpace) -> Type{<:Sector}
    sectortype(::Type{<:VectorSpace})

Return the `Sector` over which the vector space `V` or type is defined.
"""
function sectortype end
sectortype(V::VectorSpace) = sectortype(typeof(V))
sectortype(::Type{<:ElementarySpace}) = Trivial
sectortype(P::Type{<:CompositeSpace}) = sectortype(spacetype(P))

"""
    struct TrivialOrEmptyIterator
        isempty::Bool
    end

An iterator which returns `nothing` if `isempty = true`; returns `Trivial()` if
`isempty = false`.
"""
struct TrivialOrEmptyIterator
    isempty::Bool
end
Base.IteratorSize(::TrivialOrEmptyIterator) = Base.HasLength()
Base.IteratorEltype(::TrivialOrEmptyIterator) = Base.HasEltype()
Base.isempty(V::TrivialOrEmptyIterator) = V.isempty
Base.length(V::TrivialOrEmptyIterator) = isempty(V) ? 0 : 1
Base.eltype(::TrivialOrEmptyIterator) = Trivial
function Base.iterate(V::TrivialOrEmptyIterator, state = true)
    return isempty(V) == state ? nothing : (Trivial(), false)
end

"""
    sectors(V::ElementarySpace) -> Iterator

Return an iterator over different sectors in an elementary space `V`.
"""
function sectors end
sectors(V::ElementarySpace) = TrivialOrEmptyIterator(dim(V) == 0)

"""
    blocksectors(V::ElementarySpace) = sectors(V)

Return an iterator over different sectors of an elementary space `V`.

Make ElementarySpace instances behave similar to ProductSpace instances.
"""
blocksectors(V::ElementarySpace) = sectors(V)

"""
    dim(V::VectorSpace) -> Int

Return the total dimension of the vector space `V` as an Int.
"""
function dim end

"""
    dim(V::ElementarySpace, a::Sector) -> Int

Return the degeneracy or multiplicity of sector `a` that appear in the elementary space `V`.
"""
dim(V::ElementarySpace, ::Trivial) =
    sectortype(V) == Trivial ? dim(V) : throw(SectorMismatch())

"""
    blockdim(V::ElementarySpace, a::Sector) = dim(V, a)

Return the degeneracy or multiplicity of sector `a` that appear in the elementary space `V`.

Make ElementarySpace instances behave similar to ProductSpace instances.
"""
blockdim(V::ElementarySpace, c::Sector) = dim(V, c)

"""
    hassector(V::ElementarySpace, a::Sector) -> Bool

Return whether a elementary space `V` has a subspace corresponding to sector `a` with
non-zero dimension, i.e. `dim(V, a) > 0`.
"""
function hassector end
hassector(V::ElementarySpace, ::Trivial) = dim(V) != 0

"""
    axes(V::ElementarySpace) -> UnitRange

Extend `Base.axes`. Return the axes of an elementary space as `1:dim(V)`.

    axes(V::ElementarySpace, a::Sector) -> UnitRange

Return axes corresponding to the sector `a` in an elementary space as a UnitRange.
Note that the sectors in a GradedSpace is sorted, thus the axes of a sector is well defined.
"""
Base.axes(V::ElementarySpace, ::Trivial) = axes(V)

"""
    conj(V::S) where {S<:ElementarySpace} -> S

Return the conjugate space of `V`. This should satisfy `conj(conj(V)) == V`.

For `field(V)==ℝ`, `conj(V) == V`. It is assumed that `typeof(V) == typeof(conj(V))`.

`conj(V)` is implimented by change the field `conj` of a general elementary space, or the
field `dual` of a Euclidean space instance.
"""
Base.conj(V::ElementarySpace{ℝ}) = V

"""
    dual(V::EuclideanSpace) -> EuclideanSpace

Return the dual space of the EuclideanSpace `V` which is equal to `conj(V)`,
which extend `Base.conj()`.
"""
function dual end
dual(V::EuclideanSpace) = conj(V)

"""
    adjoint(V::VectorSpace) = dual(V)

Extend Base.adjoint(). Return the dual space of a VectorSpace instance `V`.
Also obtained via `V'`.
"""
Base.adjoint(V::VectorSpace) = dual(V)

"""
    isdual(V::ElementarySpace) -> Bool

Return wether an ElementarySpace `V` is normal or rather a dual space. Always returns
`false` for spaces where `V == dual(V)`.
"""
function isdual end
isdual(V::EuclideanSpace{ℝ}) = false

"""
    flip(V::S) where {S<:ElementarySpace} -> S

Return a single vector space of type `S` that has the same value of [`isdual`](@ref) as
`dual(V)`, but yet is isomorphic to `V`.
"""
function flip end

"""
    ⊕(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the direct sum of the
spaces `V1`, `V2`, ... Note that all the individual spaces should have the same value for
[`isdual`](@ref), as otherwise the direct sum is not defined.
"""
function ⊕ end
⊕(V1, V2, V3, V4...) = ⊕(⊕(V1, V2), V3, V4...)

"""
    ⊗(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S

Create a [`ProductSpace{S}(V1, V2, V3...)`](@ref) representing the tensor product of several
elementary vector spaces. For convience, Julia's regular multiplication operator `*` applied
to vector spaces has the same effect.

The tensor product structure is preserved, see [`fuse`](@ref) for returning a single
elementary space of type `S` that is isomorphic to this tensor product.
"""
function ⊗ end
⊗(V1, V2, V3, V4...) = ⊗(⊗(V1, V2), V3, V4...)

# convenience definitions:
Base.:*(V1::VectorSpace, V2::VectorSpace) = ⊗(V1, V2)

"""
    fuse(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} -> S
    fuse(P::ProductSpace{S}) where {S<:ElementarySpace} -> S

Return a single vector space of type `S` that is isomorphic to the fusion product of the
individual spaces `V1`, `V2`, ..., or the spaces contained in `P`.
"""
function fuse end
fuse(V::ElementarySpace) = V
fuse(V1::VectorSpace, V2::VectorSpace, V3::VectorSpace...) =
    fuse(fuse(fuse(V1), fuse(V2)), V3...)
# calling fuse on V1 and V2 will allow these to be `ProductSpace`

# Partial order for vector spaces
"""
    ismonomorphic(V1::VectorSpace, V2::VectorSpace)
    V1 ≾ V2

Return whether there exist monomorphisms from `V1` to `V2`, i.e. 'injective' morphisms with
left inverses.
"""
function ismonomorphic(V1::VectorSpace, V2::VectorSpace)
    spacetype(V1) == spacetype(V2) || return false
    for c in blocksectors(V1)
        if blockdim(V1, c) > blockdim(V2, c)
            return false
        end
    end
    return true
end

"""
    isepimorphic(V1::VectorSpace, V2::VectorSpace)
    V1 ≿ V2

Return whether there exist epimorphisms from `V1` to `V2`, i.e. 'surjective' morphisms with
right inverses.
"""
function isepimorphic(V1::VectorSpace, V2::VectorSpace)
    spacetype(V1) == spacetype(V2) || return false
    for c in blocksectors(V2)
        if blockdim(V1, c) < blockdim(V2, c)
            return false
        end
    end
    return true
end

"""
    isisomorphic(V1::VectorSpace, V2::VectorSpace)
    V1 ≅ V2

Return if `V1` and `V2` are isomorphic, meaning that there exists isomorphisms from `V1` to
`V2`, i.e. morphisms with left and right inverses.
"""
function isisomorphic(V1::VectorSpace, V2::VectorSpace)
    spacetype(V1) == spacetype(V2) || return false
    for c in union(blocksectors(V1), blocksectors(V2))
        if blockdim(V1, c) != blockdim(V2, c)
            return false
        end
    end
    return true
end

# unicode alternatives
const ≾ = ismonomorphic
const ≿ = isepimorphic
const ≅ = isisomorphic

≺(V1::VectorSpace, V2::VectorSpace) = V1 ≾ V2 && !(V1 ≿ V2)
≻(V1::VectorSpace, V2::VectorSpace) = V1 ≿ V2 && !(V1 ≾ V2)

"""
    infimum(V1::ElementarySpace, V2::ElementarySpace, V3::ElementarySpace...)

Return the infimum of a number of elementary spaces, i.e. an instance `V::ElementarySpace`
such that `V ≾ V1`, `V ≾ V2`, ... and no other `W ≻ V` has this property. This requires
that all arguments have the same value of `isdual( )`, and also the return value `V` will
have the same value.
"""
infimum(V1::ElementarySpace, V2::ElementarySpace, V3::ElementarySpace...) =
    infimum(infimum(V1, V2), V3...)

"""
    supremum(V1::ElementarySpace, V2::ElementarySpace, V3::ElementarySpace...)

Return the supremum of a number of elementary spaces, i.e. an instance `V::ElementarySpace`
such that `V ≿ V1`, `V ≿ V2`, ... and no other `W ≺ V` has this property. This requires
that all arguments have the same value of `isdual( )`, and also the return value `V` will
have the same value.
"""
supremum(V1::ElementarySpace, V2::ElementarySpace, V3::ElementarySpace...) =
    supremum(supremum(V1, V2), V3...)
