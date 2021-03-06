using Test
using TestExtras
using Random
using TensorLabXD
using Combinatorics
using TensorLabXD: ProductSector, fusiontensor
using TensorContractionsXD
TensorContractionsXD.disable_cache() # avoids memory overflow during CI?
using Base.Iterators: take, product
#using SUNRepresentations: SUNIrrep
#const SU3Irrep = SUNIrrep{3}
import LinearAlgebra

include("newsectors.jl")
using .NewSectors

const TK = TensorLabXD

Random.seed!(1234)

smallset(::Type{I}) where {I<:Sector} = take(values(I), 5)
smallset(::Type{FermionNumber}) = FermionNumber.((0, +1, -1, +2, -2))
function smallset(::Type{ProductSector{Tuple{I1,I2}}}) where {I1,I2}
    iter = product(smallset(I1), smallset(I2))
    s = collect(i ⊠ j for (i,j) in iter if dim(i)*dim(j) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function smallset(::Type{ProductSector{Tuple{I1,I2,I3}}}) where {I1,I2,I3}
    iter = product(smallset(I1), smallset(I2), smallset(I3))
    s = collect(i ⊠ j ⊠ k for (i,j,k) in iter if dim(i)*dim(j)*dim(k) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function randsector(::Type{I}) where {I<:Sector}
    s = collect(smallset(I))
    a = rand(s)
    while a == one(a) # don't use trivial label
        a = rand(s)
    end
    return a
end
function hasfusiontensor(I::Type{<:Sector})
    try
        fusiontensor(one(I), one(I), one(I))
        return true
    catch e
        if e isa MethodError
            return false
        else
            rethrow(e)
        end
    end
end

sectorlist = (Z2Irrep, Z3Irrep, Z4Irrep, U1Irrep, CU1Irrep, SU2Irrep, NewSU2Irrep,
              FibonacciAnyon, IsingAnyon, FermionParity, FermionNumber, FermionSpin,
              FermionParity ⊠ FermionParity, Z3Irrep ⊠ Z4Irrep, FermionNumber ⊠ SU2Irrep,
              FermionSpin ⊠ SU2Irrep, NewSU2Irrep ⊠ NewSU2Irrep, NewSU2Irrep ⊠ SU2Irrep,
              FermionSpin ⊠ NewSU2Irrep, Z2Irrep ⊠ FibonacciAnyon ⊠ FibonacciAnyon)

Ti = time()
include("sectors.jl")
include("fusiontrees.jl")
include("spaces.jl")
include("tensors.jl")
Tf = time()
printstyled("Finished all tests in ",
            string(round((Tf-Ti)/60; sigdigits=3)),
            " minutes."; bold = true, color = Base.info_color())
println()
