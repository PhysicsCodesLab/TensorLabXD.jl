"""
    struct CartesianSpace <: EuclideanSpace{ℝ}

A real Euclidean space `ℝ^d`, which is therefore self-dual. `CartesianSpace` has no
additonal structure and is completely characterised by its dimension `d`. This is the
vector space that is implicitly assumed in most of matrix algebra.
"""
struct CartesianSpace <: EuclideanSpace{ℝ}
    d::Int
end

CartesianSpace(d::Integer = 0; dual = false) = CartesianSpace(Int(d))

function CartesianSpace(dim::Pair; dual = false)
    if dim.first === Trivial()
        return CartesianSpace(dim.second; dual = dual)
    else
        msg = "$(dim) is not a valid dimension for CartesianSpace"
        throw(SectorMismatch(msg))
    end
end

function CartesianSpace(dims::AbstractDict; kwargs...)
    if length(dims) == 0
        return CartesianSpace(0; kwargs...)
    elseif length(dims) == 1
        return CartesianSpace(first(dims); kwargs...)
    else
        msg = "$(dims) is not a valid dimension dictionary for CartesianSpace"
        throw(SectorMismatch(msg))
    end
end

# convenience constructor
"""
    Base.getindex(::RealNumbers) -> CartesianSpace

Make ℝ[] a synonyms for CartesianSpace.
"""
Base.getindex(::RealNumbers) = CartesianSpace

"""
    Base.:^(::RealNumbers, d::Int) -> CartesianSpace

Return a CartesianSpace with dimension `d`.
"""
Base.:^(::RealNumbers, d::Int) = CartesianSpace(d)

# Corresponding methods:
#------------------------
Base.oneunit(::Type{CartesianSpace}) = CartesianSpace(1)
dim(V::CartesianSpace) = V.d
Base.axes(V::CartesianSpace) = OneTo(dim(V)) # Base.OneTo(dim(V))
flip(V::CartesianSpace) = V
⊕(V1::CartesianSpace, V2::CartesianSpace) = CartesianSpace(V1.d+V2.d)
fuse(V1::CartesianSpace, V2::CartesianSpace) = CartesianSpace(V1.d*V2.d)

infimum(V1::CartesianSpace, V2::CartesianSpace) = CartesianSpace(min(V1.d, V2.d))
supremum(V1::CartesianSpace, V2::CartesianSpace) = CartesianSpace(max(V1.d, V2.d))

Base.show(io::IO, V::CartesianSpace) = print(io, "ℝ^$(V.d)")
