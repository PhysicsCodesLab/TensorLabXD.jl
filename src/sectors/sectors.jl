"""
    abstract type Sector end

Abstract type representing unitary and pivotal (pre-)fusion categories.

The instances of a concrete subtype are simple objects of this concrete type.

Every concret subtype `I<:Sector` should implement the following methods:
*   `Base.one(::Type{I})`: unit object of `I`
*   `Base.conj(a::I)`: give ``a̅``, the conjugate label of `a`
*   `Base.isreal(::Type{I})`: whether the Fsymbol and Rsymbol of the sector is real
*   `Base.isless(a::I, b::I)`: give a partial order between simple objects
*   `FusionStyle(::Type{I})`: `UniqueFusion()`, `SimpleFusion()` or `GenericFusion()`
*   `⊗(a::I, b::I)`: iterable with unique fusion outputs of ``a ⊗ b``
    (i.e. don't repeat in case of multiplicities)
*   `Nsymbol(a::I, b::I, c::I)`: number of times `c` appears in `a ⊗ b`; return a Bool in
    the MultiplicityFreeFusion case.
*   `Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I)`: scalar (`MultiplicityFreeFusion`)
    or rank-4 Array (`GenericFusion`)
*   (optionally)`vertex_ind2label(i::Int, a::I, b::I, c::I)`: a custom label for the `i`th
    copy of `c` appearing in `a ⊗ b`
*   (optioanlly)`vertex_labeltype(::Type{I})`: the type of labels for the fusion vertices
*   (optionally)`dim(a::I)`: quantum dimension of simple object `a`; sqrtdim(), isqrtdim()
*   (optionally)`frobeniusschur(a::I)`: Frobenius-Schur indicator of `a`
*   (optionally)`Bsymbol(a::I, b::I, c::I)`: B-symbol: scalar (`MultiplicityFreeFusion`)
    or matrix (`GenericFusion`)
*   `BraidingStyle(::Type{I})`: `Bosonic()`, `Fermionic()` or `Anyonic()`
*   `Rsymbol(a::I, b::I, c::I)`: R-symbol: scalar (`MultiplicityFreeFusion`) or matrix
    (`GenericFusion`)
*   (optionally)`twist(a::I)`: twist of simple object `a`

Furthermore, an iterator over all simple objects of the sector `I` is implemented by the
singleton type [`SectorValues{I}`](@ref).
"""
abstract type Sector end

"""
    struct SectorValues{I<:Sector}

Singleton type to represent an iterator over all simple objects of sector `I`, whose
instance is obtained by `values(I)`.

For a new `I::Sector`, the following should be defined:
*   `Base.iterate(::SectorValues{I}[, state])`: iterate over the simple objects
*   `Base.IteratorSize(::Type{SectorValues{I}})`: `HasLenght()` or `IsInfinite()`

If `IteratorSize(I) == HasLength()`, also the following must be implemented:
*   `Base.length(::SectorValues{I})`: the number of different simple objects
*   `Base.getindex(::SectorValues{I}, i::Int)`: map from an index `i` to a simple object
*   `findindex(::SectorValues{I}, a::I)`: map from a simple object `a::I` to an index
    `i::Integer ∈ 1:length(values(I))`
"""
struct SectorValues{I<:Sector} end
Base.IteratorEltype(::Type{<:SectorValues}) = HasEltype()
Base.eltype(::Type{SectorValues{I}}) where {I<:Sector} = I
Base.values(::Type{I}) where {I<:Sector} = SectorValues{I}()

# basic properties of the simple objects of a sector
"""
    one(::Sector) -> Sector
    one(::Type{<:Sector}) -> Sector

Return the unit object of the sector.
"""
Base.one(a::Sector) = one(typeof(a))

"""
    dual(a::Sector) -> Sector

Return the conjugate label `conj(a)`.

In the language of category, `conj(a)` is `a̅`, which is in the set of simple objects and
is isomorphic to the dual of `a`, i.e. `a*`. Since we only represent the simple objects in
the codes, we don't have an instance that refer to `a*`, and we identify `dual(a)` and
`conj(a)`.

See also: [`conj`](@ref)
"""
dual(a::Sector) = conj(a)

"""
    conj(a::Sector) -> Sector

Extend `Base.conj`. Return `a̅`, which is the conjugate or dual label of `a` which is in the
set of simple objects.
"""
Base.conj(a::Sector) = conj(a)

"""
    isless(a::I,b::I) where {I<:Sector} -> Bool

Extend `Base.isless()`. Give a partial order between simple objects in Sector `I`.
"""
Base.isless(a::I, b::I) where {I<:Sector} = isless(a,b)

# Fusion
#===============================================================================#
abstract type FusionStyle end
struct UniqueFusion <: FusionStyle end # unique fusion output when fusion two sectors

abstract type MultipleFusion <: FusionStyle end
struct SimpleFusion <: MultipleFusion end # multiple fusion but multiplicity free
struct GenericFusion <: MultipleFusion end # multiple fusion with multiplicities

const MultiplicityFreeFusion = Union{UniqueFusion, SimpleFusion}

Base.:&(f::F, ::F) where {F<:FusionStyle} = f
Base.:&(f1::FusionStyle, f2::FusionStyle) = f2 & f1

Base.:&(::SimpleFusion, ::UniqueFusion) = SimpleFusion()
Base.:&(::GenericFusion, ::UniqueFusion) = GenericFusion()
Base.:&(::GenericFusion, ::SimpleFusion) = GenericFusion()

"""
    FusionStyle(a::Sector) -> ::FusionStyle
    FusionStyle(I::Type{<:Sector}) -> ::FusionStyle

Return the type of fusion behavior of sector `I`, which can be either
*   `UniqueFusion()`: single fusion output when fusing any two simple objects;
*   `SimpleFusion()`: multiple outputs, but every output occurs at most one,
    also known as multiplicity-free (e.g. irreps of ``SU(2)``);
*   `GenericFusion()`: multiple outputs that can occur more than once (e.g. irreps
    of ``SU(3)``).
There is an abstract supertype `MultipleFusion` of which both `SimpleFusion` and
`GenericFusion` are subtypes.
"""
FusionStyle(a::Sector) = FusionStyle(typeof(a))

"""
    fusiontensor(a::I, b::I, c::I) where {I<:Sector}

Return the fusiontensor ``X^{ab}_{c,μ}: c → a ⊗ b`` as a rank-4 tensor with size
`(dim(a),dim(b),dim(c),Int(Nsymbol(a,b,c)))`.
"""
function fusiontensor(a::I, b::I, c::I) where {I<:Sector}
    I <: AbstractIrrep{G<:Group} || error("fusiontensor does not exist!")
end

"""
    ⊗(a::I, b::I) where {I<:Sector}
    ⊗(a::I, b::I, c::I, rest::Vararg{I}) where {I<:Sector}

Return an iterable of elements of `c::I` that appear in the fusion product `a ⊗ b`.

Note that every element `c` should appear at most once, fusion degeneracies (if
`FusionStyle(I) == GenericFusion()`) should be accessed via `Nsymbol(a, b, c)`.

The arguments can be more than two.
"""
@inline function ⊗(a::I, b::I, c::I, rest::Vararg{I}) where {I<:Sector}
    if FusionStyle(I) isa UniqueFusion
        return a ⊗ first(⊗(b, c, rest...))
    else
        s = Set{I}()
        for d in ⊗(b, c, rest...)
            for e in a ⊗ d
                push!(s, e)
            end
        end
        return s
    end
end

"""
    Nsymbol(a::I, b::I, c::I) where {I<:Sector} -> Bool or Int

Return a `Bool` if `FusionStyle(I) == UniqueFusion()` or `SimpleFusion()`.

Return an `Int` representing the number of times `c` appears in the fusion product `a ⊗ b`
if `FusionStyle(I) == GenericFusion()`.
"""
function Nsymbol end

"""
    Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I<:Sector}

The F-symbol ``F^{abc}_d`` that associates the two different fusion orders of simple objects
`a`, `b` and `c` into an ouput simple object `d`, using either an intermediate object
``a ⊗ b → e`` or ``b ⊗ c → f``:
```
a-<-μ-<-e-<-ν-<-d                                     a-<-λ-<-d
    ∨       ∨       -> Fsymbol(a,b,c,d,e,f)[μ,ν,κ,λ]      ∨
    b       c                                             f
                                                          v
                                                      b-<-κ
                                                          ∨
                                                          c
```
Return a number, if `FusionStyle(I) == UniqueFusion()` or `SimpleFusion()`.

Return a rank-4 array of size
`(Nsymbol(a, b, e), Nsymbol(e, c, d), Nsymbol(b, c, f), Nsymbol(a, f, d))` if
`FusionStyle(I) == GenericFusion()`.
"""
function Fsymbol end

"""
    vertex_ind2label(k::Int, a::I, b::I, c::I) where {I<:Sector}

Convert the index `k` of the fusion vertex ``a ⊗ b → c`` into a label.

For `FusionStyle(I) == UniqueFusion()` or `FusionStyle( MultipleFusion()`, where every
fusion the default is to suppress vertex labels by
setting them equal to `nothing`.

For `FusionStyle(I) == GenericFusion()`, the default is to just use `k`, unless a
specialized method is provided.
"""
vertex_ind2label(k::Int, a::I, b::I, c::I) where {I<:Sector}=
    _ind2label(FusionStyle(I), k::Int, a::I, b::I, c::I)
_ind2label(::UniqueFusion, k, a, b, c) = nothing
_ind2label(::SimpleFusion, k, a, b, c) = nothing
_ind2label(::GenericFusion, k, a, b, c) = k

"""
    vertex_labeltype(I::Type{<:Sector}) -> Type

Return the type of labels for the fusion vertices of sectors of type `I`.
"""
vertex_labeltype(I::Type{<:Sector}) = typeof(vertex_ind2label(1, one(I), one(I), one(I)))

# properties that can be determined by Fsymbol and Nsymbol
"""
    dim(a::Sector)

Return the (quantum) dimension of the sector `a`.
"""
function dim(a::Sector)
    if FusionStyle(a) isa UniqueFusion
        1
    elseif FusionStyle(a) isa SimpleFusion
        abs(1/Fsymbol(a, conj(a), a, a, one(a), one(a)))
    else
        abs(1/Fsymbol(a, conj(a), a, a, one(a), one(a))[1])
    end
end

"""
    sqrtdim(a::Sector)

Return the square root of the quantum dimension of the sector `a`.
"""
sqrtdim(a::Sector) = (FusionStyle(a) isa UniqueFusion) ? 1 : sqrt(dim(a))

"""
    isqrtdim(a::Sector)

Return the inverse of the square root of the quantum dimension of the sector `a`.
"""
isqrtdim(a::Sector) = (FusionStyle(a) isa UniqueFusion) ? 1 : inv(sqrt(dim(a)))

"""
    frobeniusschur(a::Sector)

Return the Frobenius-Schur indicator of a sector `a`.
"""
function frobeniusschur(a::Sector)
    if FusionStyle(a) isa UniqueFusion || FusionStyle(a) isa SimpleFusion
        sign(Fsymbol(a, conj(a), a, a, one(a), one(a)))
    else
        sign(Fsymbol(a, conj(a), a, a, one(a), one(a))[1])
    end
end

"""
    Bsymbol(a::I, b::I, c::I) where {I<:Sector}

Return the value of ``B^{ab}_c`` which appears in transforming a splitting vertex
into a fusion vertex using the transformation
```
a -<-μ-<- c                                                    a -<-ν-<- c
     ∨          -> √(dim(c)/dim(a)) * Bsymbol(a,b,c)[μ,ν]           ∧
     b                                                            dual(b)
```
If `FusionStyle(I)` is `UniqueFusion()` or `SimpleFusion()`, the B-symbol is a
number. Otherwise it is a square matrix with row and column size
`Nsymbol(a, b, c) == Nsymbol(c, dual(b), a)`.
"""
function Bsymbol(a::I, b::I, c::I) where {I<:Sector}
    if FusionStyle(I) isa UniqueFusion || FusionStyle(I) isa SimpleFusion
        (sqrtdim(a)*sqrtdim(b)*isqrtdim(c))*Fsymbol(a, b, dual(b), a, c, one(a))
    else
        reshape((sqrtdim(a)*sqrtdim(b)*isqrtdim(c))*Fsymbol(a, b, dual(b), a, c, one(a)),
            (Nsymbol(a, b, c), Nsymbol(c, dual(b), a)))
    end
end

# Not necessary
function Asymbol(a::I, b::I, c::I) where {I<:Sector}
    if FusionStyle(I) isa UniqueFusion || FusionStyle(I) isa SimpleFusion
        (sqrtdim(a)*sqrtdim(b)*isqrtdim(c))*
            conj(frobeniusschur(a)*Fsymbol(dual(a), a, b, b, one(a), c))
    else
        reshape((sqrtdim(a)*sqrtdim(b)*isqrtdim(c))*
                    conj(frobeniusschur(a)*Fsymbol(dual(a), a, b, b, one(a), c)),
                (Nsymbol(a, b, c), Nsymbol(dual(a), c, b)))
    end
end

# Braiding
#===============================================================================#
abstract type BraidingStyle end # generic braiding
abstract type HasBraiding <: BraidingStyle end
struct NoBraiding <: BraidingStyle end

abstract type SymmetricBraiding <: HasBraiding end
struct Bosonic <: SymmetricBraiding end # all twists are one
struct Fermionic <: SymmetricBraiding end # twists one and minus one

struct Anyonic <: HasBraiding end

Base.:&(b::B, ::B) where {B<:BraidingStyle} = b
Base.:&(B1::BraidingStyle, B2::BraidingStyle) = B2 & B1
Base.:&(::Bosonic, ::Fermionic) = Fermionic()
Base.:&(::Bosonic, ::Anyonic) = Anyonic()
Base.:&(::Fermionic, ::Anyonic) = Anyonic()
Base.:&(::Bosonic, ::NoBraiding) = NoBraiding()
Base.:&(::Fermionic, ::NoBraiding) = NoBraiding()
Base.:&(::Anyonic, ::NoBraiding) = NoBraiding()

"""
    BraidingStyle(::Sector) -> ::BraidingStyle
    BraidingStyle(I::Type{<:Sector}) -> ::BraidingStyle

Return the type of braiding and twist behavior of simple objects of type `I`, which can be
*   `Bosonic()`: symmetric braiding with trivial twist (i.e. identity)
*   `Fermionic()`: symmetric braiding with non-trivial twist (squares to identity)
*   `Anyonic()`: general ``R_(a,b)^c`` phase or matrix (depending on `SimpleFusion` or
    `GenericFusion` fusion) and arbitrary twists

Note that `Bosonic` and `Fermionic` are subtypes of `SymmetricBraiding`, which means that
braids are in fact equivalent to crossings (i.e. braiding twice is an identity:
`isone(Rsymbol(b,a,c)*Rsymbol(a,b,c)) == true`) and permutations are uniquely defined.
"""
BraidingStyle(a::Sector) = BraidingStyle(typeof(a))

"""
    Rsymbol(a::I, b::I, c::I) where {I<:Sector}

The R-symbol ``R^{ab}_c`` that maps from ``c → a ⊗ b`` to ``c → b ⊗ a`` as in
```
a -<-μ-<- c                                 b -<-ν-<- c
     ∨          -> Rsymbol(a,b,c)[μ,ν]           v
     b                                           a
```
Return a number if `FusionStyle(I) == UniqueFusion()` or `SimpleFusion()`.

Return a square matrix with row and column size `Nsymbol(a,b,c) == Nsymbol(b,a,c)` if
`FusionStyle(I) == GenericFusion()`.
"""
function Rsymbol end

# properties that can be determined by Rsymbol and Fsymbol
"""
    twist(a::Sector)

Return the twist of a simple object `a`.
"""
twist(a::Sector) = sum(dim(b)/dim(a)*tr(Rsymbol(a,a,b)) for b in a ⊗ a)

"""
    isreal(::Type{<:Sector}) -> Bool

Return whether the topological data (Fsymbol, Rsymbol) of the sector is real or not (in
which case it is complex).
"""
function Base.isreal(I::Type{<:Sector})
    u = one(I)
    if BraidingStyle(I) isa HasBraiding
        return (eltype(Fsymbol(u, u, u, u, u, u))<:Real) && (eltype(Rsymbol(u, u, u))<:Real)
    else
        return (eltype(Fsymbol(u, u, u, u, u, u))<:Real)
    end
end

"""
    struct SectorSet{I<:Sector, F, S}
        f::F
        set::S
    end

An iterator that applies x->convert(I, f(x)) on the elements of set; if f is not provided it
is just taken as the function identity.
"""
struct SectorSet{I<:Sector, F, S}
    f::F
    set::S
end
SectorSet{I}(::Type{F}, set::S) where {I<:Sector, F, S} = SectorSet{I, Type{F}, S}(F, set)
SectorSet{I}(f::F, set::S) where {I<:Sector, F, S} = SectorSet{I, F, S}(f, set)
SectorSet{I}(set) where {I<:Sector} = SectorSet{I}(identity, set)

Base.IteratorEltype(::Type{<:SectorSet}) = HasEltype()
Base.IteratorSize(::Type{SectorSet{I, F, S}}) where {I<:Sector, F, S} = Base.IteratorSize(S)

Base.eltype(::SectorSet{I}) where {I<:Sector} = I
Base.length(s::SectorSet) = length(s.set)
Base.size(s::SectorSet) = size(s.set)

function Base.iterate(s::SectorSet{I}, args...) where {I<:Sector}
    next = iterate(s.set, args...)
    next === nothing && return nothing
    val, state = next
    return convert(I, s.f(val)), state
end

# possible sectors
include("trivial.jl")
include("groups.jl")
include("irreps.jl") # irreps of symmetry groups, with bosonic braiding
include("fermions.jl") # irreps with defined fermionparity and fermionic braiding
include("anyons.jl") # non-group sectors
include("product.jl") # direct product of different sectors
