# Define a sector for ungraded vector spaces
struct Trivial <: Sector end
Base.iterate(::SectorValues{Trivial}, i = false) = return i ? nothing : (Trivial(), true)
Base.IteratorSize(::Type{SectorValues{Trivial}}) = HasLength()
Base.length(::SectorValues{Trivial}) = 1
Base.getindex(::SectorValues{Trivial}, i::Int) = i == 1 ? Trivial() :
    throw(BoundsError(values(Trivial), i))
findindex(::SectorValues{Trivial}, c::Trivial) = 1
Base.show(io::IO, ::Trivial) = print(io, "Trivial()")


Base.one(::Type{Trivial}) = Trivial()
Base.conj(::Trivial) = Trivial()
Base.isreal(::Type{Trivial}) = true
Base.isless(::Trivial, ::Trivial) = false

FusionStyle(::Type{Trivial}) = UniqueFusion()
⊗(::Trivial, ::Trivial) = (Trivial(),)
Nsymbol(::Trivial, ::Trivial, ::Trivial) = true
Fsymbol(::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial) = 1


BraidingStyle(::Type{Trivial}) = Bosonic()
Rsymbol(::Trivial, ::Trivial, ::Trivial) = 1














