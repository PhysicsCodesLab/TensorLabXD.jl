# [Vector spaces](@id s_spaces)

```@setup TensorLabXD
using TensorLabXD
```
### Types
```julia
# Field
abstract type Field end
struct RealNumbers <: Field end
struct ComplexNumbers <: Field end
const ℝ = RealNumbers()
const ℂ = ComplexNumbers()

# Vector Space
abstract type VectorSpace end

## Elementary Space
abstract type ElementarySpace{𝕜} <: VectorSpace end
const IndexSpace = ElementarySpace
struct GeneralSpace{𝕜} <: ElementarySpace{𝕜}
    d::Int
    dual::Bool
    conj::Bool
end

abstract type InnerProductSpace{𝕜} <: ElementarySpace{𝕜} end
abstract type EuclideanSpace{𝕜} <: InnerProductSpace{𝕜} end
struct CartesianSpace <: EuclideanSpace{ℝ}
    d::Int
end
struct ComplexSpace <: EuclideanSpace{ℂ}
  d::Int
  dual::Bool
end
struct GradedSpace{I<:Sector, D} <: EuclideanSpace{ℂ}
    dims::D
    dual::Bool
end

# Composite Space
abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end
struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}
    spaces::NTuple{N, S}
end
const TensorSpace{S<:ElementarySpace} = Union{S, ProductSpace{S}}

# Space of Morphisms
struct HomSpace{S<:ElementarySpace, P1<:CompositeSpace{S}, P2<:CompositeSpace{S}}
    codomain::P1
    domain::P2
end
const TensorMapSpace{S<:ElementarySpace, N₁, N₂} = HomSpace{S, ProductSpace{S, N₁}, ProductSpace{S, N₂}}
```
### Properties
On both `VectorSpace` instances and types:
```julia
spacetype # type of ElementarySpace associated with a composite space or a tensor or a HomSpace
field # field of a vector space or a tensor map or a HomSpace
Base.oneunit # the corresponding vector space that represents the trivial 1D space isomorphic to the corresponding field
sectortype # sector type of a space or a tensor or a HomSpace
one(::S) where {S<:ElementarySpace} -> ProductSpace{S, 0}
one(::ProductSpace{S}) where {S<:ElementarySpace} -> ProductSpace{S, 0}  # Return a tensor product of zero spaces of type `S`, i.e. this is the unit object under the tensor product operation, such that `V ⊗ one(V) == V`.
```
On `VectorSpace` instances:
```julia
sectors # an iterator over the different sectors of an ElementarySpace
sectors(P::ProductSpace{S, N}) # Return an iterator over all possible combinations of sectors (represented as an `NTuple{N, sectortype(S)}`) that can appear within the tensor product space `P`.
blocksectors(V::ElementarySpace) = sectors(V) # make ElementarySpace instances behave similar to ProductSpace instances
blocksectors(P::ProductSpace) # Return an iterator over the different unique coupled sector labels
blocksectors(W::HomSpace) # Return an iterator over the different unique coupled sector labels, i.e. the intersection of the different fusion outputs that can be obtained by fusing the sectors present in the domain, as well as from the codomain.
blocksectors(t::TensorMap) # Return an iterator over the different unique coupled sector labels
dim # total dimension of a vector space or a product space
dim(V::ElementarySpace, ::Trivial) # return dim(V)
dim(V::GradedSpace, c::I) # the degeneracy or multiplicity of sector c in a Graded Space
dim(W::HomSpace) # Return the total dimension of a `HomSpace`, i.e. the number of linearly independent morphisms that can be constructed within this space.
dim(P::ProductSpace, n::Int) # dim for the `n`th vector space of the product space
dim(P::ProductSpace{S, N}, s::NTuple{N, sectortype(S)}) # Return the total degeneracy dimension corresponding to a tuple of sectors for each of the spaces in the tensor product, obtained as `prod(dims(P, s))``.
dim(t::AbstractTensorMap) # total dim for corresponding HomSpace
dim(t::TensorMap) # Return the total dimension of the tensor map, i.e., the number of elements in all DenseMatrix of the data.
dims(P::ProductSpace) # Return the dimensions of the spaces in the tensor product space as a tuple of integers.
dims(P::ProductSpace{S, N}, s::NTuple{N, sectortype(S)}) # Return the degeneracy dimensions corresponding to a tuple of sectors `s` for each of the spaces in the tensor product `P`.
blockdim(V::ElementarySpace, c::Sector) = dim(V, c) # make ElementarySpace instances behave similar to ProductSpace instances
blockdim(P::ProductSpace, c::Sector) # Return the total dimension of a coupled sector `c` in the product space
hassector(V::ElementarySpace, a::Sector) # whether a vector space `V` has a subspace corresponding to sector `a` with non-zero multiplicity
hassector(P::ProductSpace{S, N}, s::NTuple{N, sectortype(S)}) # Query whether `P` has a non-zero degeneracy of sector `s`, representing a combination of sectors on the individual tensor indices.
Base.axes(V::ElementarySpace) # the axes of an elementary space as `1:dim(V)`
Base.axes(V::ElementarySpace, a::Sector) # axes corresponding to the sector `a` in an elementary space as a UnitRange.
Base.axes(P::ProductSpace) # the axes for all index spaces in product space `P`
Base.axes(P::ProductSpace, n::Int) # the axes for `n`th index space of product space `P`
Base.axes(P::ProductSpace{<:ElementarySpace, N}, sectors::NTuple{N, <:Sector}) where {N} # the axes of `sectors[i]` in `P.spaces[i]` for all `i ∈ 1:N`.
Base.conj(V::ElementarySpace) # returns the complex conjugate space (conj(V)==V̅)
dual(V::EuclideanSpace) = conj(V) # returns the dual space (dual(V)==V^*); for product space the sequence of the vector spaces are reversed.
dual(P::ProductSpace) # Return a new product space with reversed order of index spaces of the input product space and take the dual of each index space.
dual(W::HomSpace) # Return the dual of a HomSpace which contains the dual of morphisms in this space. It corresponds to 180 degree rotation in the graphical representation.
Base.adjoint(V::VectorSpace) = dual(V) # make V' as the dual of V
Base.adjoint(W::HomSpace{<:EuclideanSpace}) # Return the adjoint of a HomSpace which contains the dagger of morphisms in this space. It corresponds to mirror operation and then reversing all arrows in the graphical
representation.
isdual(V::ElementarySpace) # wether an ElementarySpace `V` is normal or rather a dual space
flip(V::ElementarySpace) # flip(V)==V̅^*
⊕ # direct sum of the elementary spaces `V1`, `V2`, ...
⊗ # representing the tensor product of several elementary vector spaces
Base.:*(V1::VectorSpace, V2::VectorSpace) = ⊗(V1, V2)
fuse(V1::S, V2::S, V3::S...) where {S<:ElementarySpace} # returns a single vector space that is isomorphic to the fusion product of the individual spaces
fuse(P::ProductSpace{S}) where {S<:ElementarySpace}
ismonomorphic # Return whether there exist monomorphisms from `V1` to `V2`, i.e. 'injective' morphisms with left inverses.
isepimorphic # Return whether there exist epimorphisms from `V1` to `V2`, i.e. 'surjective' morphisms with right inverses.
isisomorphic # Return if `V1` and `V2` are isomorphic, meaning that there exists isomorphisms from `V1` to `V2`, i.e. morphisms with left and right inverses.
const ≾ = ismonomorphic
const ≿ = isepimorphic
const ≅ = isisomorphic
≺(V1::VectorSpace, V2::VectorSpace) = V1 ≾ V2 && !(V1 ≿ V2)
≻(V1::VectorSpace, V2::VectorSpace) = V1 ≿ V2 && !(V1 ≾ V2)
infimum # Return the infimum of a number of elementary spaces
supremum # Return the supremum of a number of elementary spaces
Base.:(==)
Base.hash
Base.length(P::ProductSpace) # number of vector spaces
Base.iterate
Base.indexed_iterate
Base.eltype
Base.IteratorEltype
Base.IteratorSize
Base.convert
codomain(W::HomSpace) # codomain of a HomSpace.
domain(W::HomSpace) # domain of a HomSpace.

```

### Constructors
```julia
# Cartesian and Complex Space
CartesianSpace(d::Integer = 0; dual = false) # Constructed by an integer number which is the dim of the space
ComplexSpace(d::Integer = 0; dual = false)
CartesianSpace(dim::Pair; dual = false) # Constructed by (Trivial(),d)
ComplexSpace(dim::Pair; dual = false)
CartesianSpace(dims::AbstractDict; kwargs...) # Constructed by (Trivial() => d)
ComplexSpace(dims::AbstractDict; kwargs...)
Base.getindex(::RealNumbers) = CartesianSpace # Make ℝ[] a synonyms for CartesianSpace.
Base.getindex(::ComplexNumbers) = ComplexSpace # make ℂ[] a synonyms for ComplexSpace
Base.:^(::RealNumbers, d::Int) # Return a CartesianSpace with dimension `d`.
Base.:^(::ComplexNumbers, d::Int) # Return a ComplexSpace with dimension `d`

# Graded Space
GradedSpace{I, NTuple{N, Int}}(dims; dual::Bool = false) where {I, N} # dims = (c=>dc,...)
GradedSpace{I, NTuple{N, Int}}(dims::Pair; dual::Bool = false) where {I, N}
GradedSpace{I, SectorDict{I, Int}}(dims; dual::Bool = false) where {I<:Sector}
GradedSpace{I, SectorDict{I, Int}}(dims::Pair; dual::Bool = false) where {I<:Sector}
GradedSpace{I,D}(; kwargs...) where {I<:Sector,D}
GradedSpace{I,D}(d1::Pair, d2::Pair, dims::Vararg{Pair}; kwargs...) where {I<:Sector,D}
GradedSpace{I}(args...; kwargs...) where {I<:Sector}
GradedSpace(dims::Tuple{Vararg{Pair{I, <:Integer}}}; dual::Bool = false) where {I<:Sector}
GradedSpace(dims::Vararg{Pair{I, <:Integer}}; dual::Bool = false) where {I<:Sector}
GradedSpace(dims::AbstractDict{I, <:Integer}; dual::Bool = false) where {I<:Sector}

struct SpaceTable end
const Vect = SpaceTable()
Base.getindex(::SpaceTable) = ComplexSpace # Vect[] = ComplexSpace
Base.getindex(::SpaceTable, ::Type{Trivial}) = ComplexSpace
Base.getindex(::SpaceTable, I::Type{<:Sector}) # # Vect[I]; Return `GradedSpace{I, NTuple{N, Int}}` if `HasLength`; return `GradedSpace{I, SectorDict{I, Int}}` if `IsInfinite`.
Base.getindex(::ComplexNumbers, I::Type{<:Sector}) = Vect[I] # Make ℂ[I] = Vect[I]

struct RepTable end
const Rep = RepTable()
Base.getindex(::RepTable, G::Type{<:Group}) # Rep[G] = Vect[Irrep[G]]
const ZNSpace{N} = GradedSpace{ZNIrrep{N}, NTuple{N,Int}}
const Z2Space = ZNSpace{2}
const Z3Space = ZNSpace{3}
const Z4Space = ZNSpace{4}
const U1Space = Rep[U₁]
const CU1Space = Rep[CU₁]
const SU2Space = Rep[SU₂]
const ℤ₂Space = Z2Space
const ℤ₃Space = Z3Space
const ℤ₄Space = Z4Space
const U₁Space = U1Space
const CU₁Space = CU1Space
const SU₂Space = SU2Space

# Product Space
ProductSpace(spaces::Vararg{S, N}) where {S<:ElementarySpace, N}
ProductSpace{S, N}(spaces::Vararg{S, N}) where {S<:ElementarySpace, N}
ProductSpace{S}(spaces) where {S<:ElementarySpace}
ProductSpace(P::ProductSpace)
⊗(V1::S, V2::S) where {S<:ElementarySpace}= ProductSpace((V1, V2))
⊗(P1::ProductSpace{S}, V2::S) where {S<:ElementarySpace}
⊗(V1::S, P2::ProductSpace{S}) where {S<:ElementarySpace}
⊗(P1::ProductSpace{S}, P2::ProductSpace{S}) where {S<:ElementarySpace}
⊗(P::ProductSpace{S, 0}, ::ProductSpace{S, 0}) where {S<:ElementarySpace} = P
⊗(P::ProductSpace{S}, ::ProductSpace{S, 0}) where {S<:ElementarySpace} = P
⊗(::ProductSpace{S, 0}, P::ProductSpace{S}) where {S<:ElementarySpace} = P
⊗(V::ElementarySpace) = ProductSpace((V,))
⊗(P::ProductSpace) = P
Base.:^(V::ElementarySpace, N::Int) = ProductSpace{typeof(V), N}(ntuple(n->V, N))
Base.:^(V::ProductSpace, N::Int) = ⊗(ntuple(n->V, N)...)
Base.literal_pow(::typeof(^), V::ElementarySpace, p::Val{N}) where N =
    ProductSpace{typeof(V), N}(ntuple(n->V, p))
insertunit(P::ProductSpace, i::Int = length(P)+1; dual = false, conj = false) # For `P::ProductSpace{S,N}`, this adds an extra tensor product factor at position `1 <= i <= N+1` (last position by default) which is just a the `S`-equivalent of the underlying field of scalars, i.e. `oneunit(S)`.

# HomSpace
→(dom::TensorSpace{S}, codom::TensorSpace{S}) where {S<:ElementarySpace} =
    HomSpace(ProductSpace(codom), ProductSpace(dom))
←(codom::TensorSpace{S}, dom::TensorSpace{S}) where {S<:ElementarySpace} =
    HomSpace(ProductSpace(codom), ProductSpace(dom))    
```

### Others structures
```julia
struct TrivialOrEmptyIterator
    isempty::Bool
end # returns nothing is isempty = true, otherwise returns Trivial()
```

### Details about `dual`, `conj`, `flip`
In `vectorspaces.jl`:
```julia
function dual end
dual(V::EuclideanSpace) = conj(V)
Base.adjoint(V::VectorSpace) = dual(V)
function flip end
```

In `generalspace.jl`:
```julia
dual(V::GeneralSpace{𝕜}) where {𝕜} =
    GeneralSpace{𝕜}(dim(V), !isdual(V), isconj(V))
Base.conj(V::GeneralSpace{𝕜}) where {𝕜} =
    GeneralSpace{𝕜}(dim(V), isdual(V), !isconj(V))
```

In `cartesianspace.jl`:
```julia
flip(V::CartesianSpace) = V
```

In `complexspace.jl`:
```julia
Base.conj(V::ComplexSpace) = ComplexSpace(dim(V), !isdual(V))
flip(V::ComplexSpace) = dual(V)
```

In `sectors.jl`:

`Base.conj(a::I)`: ``\overline{a}``, conjugate or dual label of ``a``.
```julia
dual(a::Sector) = conj(a)
```

In `trivial.jl`:
```julia
Base.conj(::Trivial) = Trivial()
```

In `anyons.jl`:
```julia
Base.conj(s::IsingAnyon) = s
Base.conj(s::FibonacciAnyon) = s
```

In `irreps.jl`:
```julia
Base.conj(c::ZNIrrep{N}) where {N} = ZNIrrep{N}(-c.n)
Base.conj(c::U1Irrep) = U1Irrep(-c.charge)
Base.conj(s::SU2Irrep) = s
Base.conj(c::CU1Irrep) = c
```

In `gradedspace.jl`:

In fact, `GradedSpace` is the reason `flip` exists, cause
in this case it is different than `dual`. The existence of flip originates from the
non-trivial isomorphism between ``R_{\overline{a}}`` and ``R_{a}^*``, i.e. the
representation space of the dual ``\overline{a}`` of sector ``a`` and the dual of the
representation space of sector ``a``. In order for `flip(V)` to be isomorphic to `V`, it is
such that, if `V = GradedSpace(a=>n_a,...)` then
`flip(V) = dual(GradedSpace(dual(a)=>n_a,....))`.

In the structure of `TensorLabXD.jl`, we only keep the simple objects. It means
that we don't have objects correspond to ``a^*`` in the language of category.
Therefore, `dual(a) = conj(a)` both correspond to ``\overline{a}``. The dual space
of a space is denoted in the field named as `dual` in the type definitions. If
`dual = true`, it means that we represent the space ``R_a^*`` which is isomorphic to
``R_{\overline{a}}``, and in the methods like `sectors` and `dim` we get the sectors
and corresponding dims in the corresponding ``R_{\overline{a}}``.
```julia
sectors(V::GradedSpace{I,<:AbstractDict}) where {I<:Sector} =
    SectorSet{I}(s->isdual(V) ? dual(s) : s, keys(V.dims))
sectors(V::GradedSpace{I,NTuple{N,Int}}) where {I<:Sector, N} =
    SectorSet{I}(Iterators.filter(n->V.dims[n]!=0, 1:N)) do n
        isdual(V) ? dual(values(I)[n]) : values(I)[n]
dim(V::GradedSpace{I,<:AbstractDict}, c::I) where {I<:Sector} =
    get(V.dims, isdual(V) ? dual(c) : c, 0)
dim(V::GradedSpace{I,<:Tuple}, c::I) where {I<:Sector} =
    V.dims[findindex(values(I), isdual(V) ? dual(c) : c)]    
Base.conj(V::GradedSpace) = typeof(V)(V.dims, !V.dual)
function flip(V::GradedSpace{I}) where {I<:Sector}
    if isdual(V)
        typeof(V)(c=>dim(V, c) for c in sectors(V))
    else
        typeof(V)(dual(c)=>dim(V, c) for c in sectors(V))'
    end
end        
```

In `productspace.jl`:

The order of the spaces are reversed before taking the dual of each elementray
vecor space:
```julia
dual(P::ProductSpace{<:ElementarySpace, 0}) = P
dual(P::ProductSpace) = ProductSpace(map(dual, reverse(P.spaces)))
```

In  `homespace.jl`:

For a morphism the dual of the morphism is different with the adjoint of it.
In the tensor category language, the dual of a morphism is called the transpose
of the morphism, while the adjoint of a morphism is called the dagger of the morphism.
```julia
dual(W::HomSpace) = HomSpace(dual(W.domain), dual(W.codomain))
Base.adjoint(W::HomSpace{<:EuclideanSpace}) = HomSpace(W.domain, W.codomain)
```
The sequence of the elementary spaces in a TensorMapSpace is defined as ``1:N_1``
for codomain vectors, and ``N_1+1:N_1+N_2`` for domain dual vectors. Note that
the sequence of the domain vectors are not reversed, and the dual is taken
individually for each elementary space.
```julia
Base.getindex(W::TensorMapSpace{<:IndexSpace, N₁, N₂}, i) where {N₁, N₂} =
    i <= N₁ ? codomain(W)[i] : dual(domain(W)[i-N₁])
```

## [VectorSpace type](@id ss_vectorspace_type)

From the [Introduction](@ref s_intro), it should be clear that an important aspect in the
definition of a tensor (map) is specifying the vector spaces and their structure in the
domain and codomain of the map. The starting point is an abstract type `VectorSpace`
```julia
abstract type VectorSpace end
```
which is actually a too restricted name. Subtypes of `VectorSpace` will in general represent
``𝕜``-linear tensor categories, which can go beyond ``\mathbf{Vect}`` and
``\mathbf{SVect}``. The instances of it represent the objects of the category.

In order not to make the remaining discussion too abstract or complicated, we will simply
refer to subtypes of `VectorSpace` instead of specific categories, and to spaces instead of
objects from these categories.

We define two abstract subtypes
```julia
abstract type ElementarySpace{𝕜} <: VectorSpace end
const IndexSpace = ElementarySpace

abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end
```
The `ElementarySpace` is a super type for all categories that can be associated with the
individual indices of a tensor, as hinted to by its alias `IndexSpace`.
The parameter `𝕜` here could represent the field of vector spaces of Vect, or the field of
the morphism space of `𝕜`-linear tensor categories.

The `CompositeSpace{S}` where `S<:ElementarySpace` is a super type for all vector spaces
that are composed of a number of elementary spaces of type `S`. One concrete subtype of it
is the `ProductSpace{S,N}` which represents a homogeneous tensor product of `N` vector
spaces of type `S`.

Throughout TensorLabXD.jl, the function `spacetype` returns the type of `ElementarySpace`
associated with e.g. a composite space or a tensor. It works both on instances and type.

## [Fields](@id ss_fields)

Vector spaces (linear categories) are defined over a field of scalars ``𝕜``. We define a
type hierarchy to specify the scalar field, but so far only support real and complex
numbers, via
```julia
abstract type Field end

struct RealNumbers <: Field end
struct ComplexNumbers <: Field end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()
```
Note that `ℝ` and `ℂ` can be typed as `\bbR`+TAB and `\bbC`+TAB. One reason for defining
this new type hierarchy instead of recycling the types from Julia's `Number` hierarchy is
to introduce some syntactic sugar without committing type piracy.

Some examples:
```@repl TensorLabXD
3 ∈ ℝ
5.0 ∈ ℂ
5.0+1.0*im ∈ ℝ
Float64 ⊆ ℝ
ComplexF64 ⊆ ℂ
ℝ ⊆ ℂ
ℂ ⊆ ℝ
```
The field of a vector space or tensor `a` can be obtained with `field(a)`.

## [Elementary spaces](@id ss_elementaryspaces)

Vector spaces that are associated with the individual indices of a tensor should be
implemented as subtypes of `ElementarySpace`. As the domain and codomain of a tensor map
will be the tensor product of such objects which all have the same type, it is important
that related vector spaces, e.g. the dual space, are objects of the same concrete type.

Every `ElementarySpace` should implement the following methods

*   `dim(::ElementarySpace) -> ::Int` returns the dimension of the space as an `Int`

*   `dual(::ElementarySpace)` returns the dual space, using an instance of
    the same concrete type. The dual of a space `V` can also be obtained as `V'`.

*   `conj(::ElementarySpace)` returns the complex conjugate space, using an instance of
    the same concrete type

The `GeneralSpace` is one of the concrete type of the `ElementarySpace`. It is completely
characterized by its field `𝕜`, its dimension and whether its the dual and/or complex
conjugate of ``𝕜^d``.
```julia
struct GeneralSpace{𝕜} <: ElementarySpace{𝕜}
    d::Int
    dual::Bool
    conj::Bool
end
```

The abstract type
```julia
abstract type InnerProductSpace{𝕜} <: ElementarySpace{𝕜} end
```
is defined to contain all vector spaces `V` which have an inner product and thus a canonical
mapping from `dual(V)` to `conj(V)`. This mapping is provided by the metric, but no further
support for working with metrics is currently implemented.

The abstract type
```julia
abstract type EuclideanSpace{𝕜} <: InnerProductSpace{𝕜} end
```
is defined to contain all spaces `V` with a standard Euclidean inner product. The canonical
mapping from `dual(V)` to `conj(V)` is identity. This subtype represents dagger categories.

We have two concrete types
```julia
struct CartesianSpace <: EuclideanSpace{ℝ}
    d::Int
end
struct ComplexSpace <: EuclideanSpace{ℂ}
  d::Int
  dual::Bool
end
```
to represent the Euclidean spaces ``ℝ^d`` and ``ℂ^d``. They can be created using the syntax
`CartesianSpace(d) == ℝ^d == ℝ[](d)` and `ComplexSpace(d) == ℂ^d == ℂ[](d)`. The dual space
of ``\mathbb{C}^d`` can be created by
`ComplexSpace(d, true) == ComplexSpace(d; dual = true) == (ℂ^d)' == ℂ[](d)'`. Note that the
brackets are required because of the precedence rules, since `d' == d` for `d::Integer`.

Some examples:
```@repl TensorLabXD
dim(ℝ^10)
(ℝ^10)' == ℝ^10 == ℝ[](10)
isdual((ℂ^5))
isdual((ℂ^5)')
isdual((ℝ^5)')
dual(ℂ^5) == (ℂ^5)' == conj(ℂ^5) == ComplexSpace(5; dual = true)
typeof(ℝ^3)
spacetype(ℝ^3)
spacetype(ℝ[])
field(ℂ^5)
```
Note that `ℝ[]` and `ℂ[]` are synonyms for `CartesianSpace` and `ComplexSpace` respectively.
This is not very useful in itself, and is motivated by its generalization to `GradedSpace`.
We refer to the subsection on
[graded spaces](@ref s_rep) on the [next page](@ref s_sectorsrepfusion) for further
information about `GradedSpace`, which is another subtype of `EuclideanSpace{ℂ}`
with an inner structure corresponding to the irreducible representations of a group, or more
generally, the simple objects of a fusion category.

!!! note
    For `ℂ^n` the dual space is equal (or naturally isomorphic) to the conjugate space, but
    not to the space itself. This means that even for `ℂ^n`, arrows matter in the
    diagrammatic notation for categories or for tensors, and in particular that a
    contraction between two tensor indices will check that one is living in the space and
    the other in the dual space. This is in contrast with several other software packages,
    especially in the context of tensor networks, where arrows are only introduced when
    discussing symmetries. We believe that our more purist approach can be useful to detect
    errors (e.g. unintended contractions). Only with `ℝ^n` will their be no distinction
    between a space and its dual. When creating tensors with indices in `ℝ^n` that have
    complex data, a one-time warning will be printed, but most operations should continue
    to work nonetheless.

## [Composite spaces](@id ss_compositespaces)

Composite spaces are vector spaces that are built up out of individual elementary vector
spaces of the same type. The most prominent and currently only subtype is a tensor
product of `N` elementary spaces of the same type `S`:
```julia
struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}
    spaces::NTuple{N, S}
end
```
Given some `V1::S`, `V2::S`, `V3::S` of the same type `S<:ElementarySpace`, we can easily
construct `ProductSpace{S,3}((V1,V2,V3))` as `ProductSpace(V1,V2,V3)` or using
`V1 ⊗ V2 ⊗ V3`, where `⊗` is simply obtained by typing `\otimes`+TAB. For
convenience, the regular multiplication operator `*` also acts as tensor product between
vector spaces, and as a consequence so does raising a vector space to a positive integer
power, i.e.
```@repl TensorLabXD
V1 = ℂ^2
V2 = ℂ^3
V1 ⊗ V2 ⊗ V1' == V1 * V2 * V1' == ProductSpace(V1,V2,V1') == ProductSpace(V1,V2) ⊗ V1'
V1^3
dim(V1 ⊗ V2)
dims(V1 ⊗ V2)
dual(V1 ⊗ V2)
spacetype(V1 ⊗ V2)
spacetype(ProductSpace{ComplexSpace,3})
```
Here, the new function `dims` gives the dimension of the individual spaces in a
`ProductSpace` as a tuple.

The function `one` applied to a `ProductSpace{S,N}` or an instance `V` of
`S::ElementarySpace` returns the multiplicative identity, which is `ProductSpace{S,0}(())`.
Note that `V ⊗ one(V)` or `⊗(V)` will yield a `ProductSpace{S,1}(V)` and not `V` itself.

In the future, other `CompositeSpace` types could be added. For example, the wave function
of an `N`-particle quantum system in first quantization would require the introduction of a
`SymmetricSpace{S,N}` or a `AntiSymmetricSpace{S,N}` for bosons or fermions respectively,
which correspond to the symmetric (permutation invariant) or antisymmetric subspace of
`V^N`, where `V::S` represents the Hilbert space of the single particle system. Other
domains, like general relativity, might also benefit from tensors living in a subspace with
certain symmetries under specific index permutations.

## [Space of morphisms](@id ss_homspaces)
In a ``𝕜``-linear category ``C``, the set of morphisms ``\mathrm{Hom}(W,V)`` for ``V,W ∈ C``
form a vector space with field ``𝕜``, irrespective of whether or not ``C`` is a subcategory
of ``\mathbf{(S)Vect}``, and we define tensor maps as the morphisms.

The space of morphisms is represented by the type
```julia
struct HomSpace{S<:ElementarySpace, P1<:CompositeSpace{S}, P2<:CompositeSpace{S}}
    codomain::P1
    domain::P2
end
```
It can be created by `domain → codomain` or `codomain ← domain` (where the arrows
are obtained as `\to+TAB` or `\leftarrow+TAB`, and as `\rightarrow+TAB` respectively).

Note that `HomSpace` is not a subtype of `VectorSpace`.

Some properties of `HomSpace`:
```@repl TensorLabXD
W = ℂ^2 ⊗ ℂ^3 → ℂ^3 ⊗ dual(ℂ^4)
field(W)
dual(W)
adjoint(W)
spacetype(W)
spacetype(typeof(W))
W[1]
W[2]
W[3]
W[4]
dim(W)
```
The indexing `W` yields first the spaces in the codomain, followed by the dual of the
individual spaces in the domain. This convention is useful in combination with the
instances of type [`TensorMap`](@ref), which represent morphisms living in such a
`HomSpace`. The `dim(::HomSpace)` represent the number of linearly independent morphisms
in this space.

## Partial order among vector spaces

Vector spaces of the same `spacetype` can be given a partial order, based on whether there
exist injective morphisms (a.k.a *monomorphisms*) or surjective morphisms (a.k.a.
*epimorphisms*) between them.

We define `ismonomorphic(V1, V2)`, with Unicode synonym `V1 ≾ V2` (obtained as
`\precsim+TAB`), to express whether there exist injective morphisms in `V1 → V2`.

We define `isepimorphic(V1, V2)`, with Unicode synonym `V1 ≿ V2` (obtained as
`\succsim+TAB`), to express whether there exist surjective morphisms in `V1 → V2`.

We define `isisomorphic(V1, V2)`, with Unicode alternative `V1 ≅ V2` (obtained as
`\cong+TAB`), to express whether there exist isomorphism in `V1 → V2`. `V1 ≅ V2` if and
only if `V1 ≾ V2 && V1 ≿ V2`.

The strict comparison operators `≺` and `≻` (`\prec+TAB` and `\succ+TAB`) are defined by
```julia
≺(V1::VectorSpace, V2::VectorSpace) = V1 ≾ V2 && !(V1 ≿ V2)
≻(V1::VectorSpace, V2::VectorSpace) = V1 ≿ V2 && !(V1 ≾ V2)
```
However, as we expect these to be less commonly used, no ASCII alternative is provided.

In the context of `spacetype(V) <: EuclideanSpace`, `V1 ≾ V2` implies that there exists
isometries ``W:V1 → V2`` such that ``W^† ∘ W = \mathrm{id}_{V1}``, while `V1 ≅ V2` implies
that there exist unitaries ``U:V1 → V2`` such that ``U^† ∘ U = \mathrm{id}_{V1}`` and
``U ∘ U^† = \mathrm{id}_{V2}``.

Note that spaces that are isomorphic are not necessarily equal. One can be a dual space,
and the other a normal space, or one can be an instance of `ProductSpace`, while the other
is an `ElementarySpace`. There will exist (infinitely) many isomorphisms between the
corresponding spaces, but in general none of those will be canonical.

There are a number of convenience functions to create isomorphic spaces. The function
`fuse(V1, V2, ...)` or `fuse(V1 ⊗ V2 ⊗ ...)` returns an elementary space that is isomorphic
to `V1 ⊗ V2 ⊗ ...`. The function `flip(V::ElementarySpace)` returns a space that is
isomorphic to `V` but has `isdual(flip(V)) == isdual(V')`, i.e. if `V` is a normal space
then `flip(V)` is a dual space. `flip(V)` is different from `dual(V)` in the case of
[`GradedSpace`](@ref). It is useful to flip a tensor index from a ket to a bra (or
vice versa), by contracting that index with a unitary map from `V` to `flip(V)`.
(**In the language of category, the we have `flip(a)==` ``\overline{a}^*`` .**)

Some examples:
```@repl TensorLabXD
ℝ^3 ≾ ℝ^5
ℂ^3 ≾ (ℂ^5)'
(ℂ^5) ≅ (ℂ^5)'
fuse(ℝ^5, ℝ^3)
fuse(ℂ^3, (ℂ^5)' ⊗ ℂ^2)
fuse(ℂ^3, (ℂ^5)') ⊗ ℂ^2 ≅ fuse(ℂ^3, (ℂ^5)', ℂ^2) ≅ ℂ^3 ⊗ (ℂ^5)' ⊗ ℂ^2
flip(ℂ^4)
flip(ℂ^4) ≅ ℂ^4
flip(ℂ^4) == ℂ^4
```

We define the direct sum `V1` and `V2` as `V1 ⊕ V2`, where `⊕` is obtained by typing
`\oplus`+TAB. This is possible only if `isdual(V1) == isdual(V2)`.

Applying `oneunit` to an elementary space returns the one-dimensional space, which is
isomorphic to the scalar field of the space itself.

Some examples:
```@repl TensorLabXD
ℝ^5 ⊕ ℝ^3
ℂ^5 ⊕ ℂ^3
ℂ^5 ⊕ (ℂ^3)'
oneunit(ℝ^3)
ℂ^5 ⊕ oneunit(ComplexSpace)
oneunit((ℂ^3)')
(ℂ^5) ⊕ oneunit((ℂ^5))
(ℂ^5)' ⊕ oneunit((ℂ^5)')
```

If `V1` and `V2` are two `ElementarySpace` instances with `isdual(V1) == isdual(V2)`, we
can define a unique infimum `V::ElementarySpace` with the same value of `isdual` that
satisfies `V ≾ V1` and `V ≾ V2`, as well as a unique supremum `W::ElementarySpace` that
satisfies `W ≿ V1` and `W ≿ V2`. For `CartesianSpace` and `ComplexSpace`, this simply
amounts to the space with minimal or maximal dimension, i.e.
```@repl TensorLabXD
infimum(ℝ^5, ℝ^3)
supremum(ℂ^5, ℂ^3)
supremum(ℂ^5, (ℂ^3)')
```
The names `infimum` and `supremum` are especially suited in the case of
[`GradedSpace`](@ref), as the infimum of two spaces might be different from either
of those two spaces, and similar for the supremum.
