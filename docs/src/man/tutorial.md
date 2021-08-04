# [Tutorial](@id s_tutorial)

Before discussing at length all aspects of this package, both its usage and implementation,
we start with a short tutorial to sketch the main capabilities. Thereto, we start by
loading TensorXD.jl

```@repl tutorial
using TensorXD
```

## Cartesian tensors

The most important objects in TensorXD.jl are tensors, which we now create with random
normally distributed entries in the following manner
```@repl tutorial
A = Tensor(randn, ℝ^3 ⊗ ℝ^2 ⊗ ℝ^4)
```
The tensor is created by specifying the vector space associated to each of the tensor
indices, in this case `ℝ^n` (`\bbR+TAB`). The tensor lives in the tensor product of the
index spaces, which can be obtained by typing `⊗` (`\otimes+TAB`) or `*`. The tensor `A`
is printed as an instance of a parametric type `TensorMap`.

Briefly sidetracking into the nature of `ℝ^n`:
```@repl tutorial
V = ℝ^3
typeof(V)
V == CartesianSpace(3)
supertype(CartesianSpace)
supertype(EuclideanSpace)
supertype(InnerProductSpace)
supertype(ElementarySpace)
```
We have seen that `ℝ^n` can also be created using the longer syntax `CartesianSpace(n)`.
It is subtype of `EuclideanSpace{ℝ}`, a space with a standard Euclidean inner product
over the real numbers. Furthermore,
```@repl tutorial
W = ℝ^3 ⊗ ℝ^2 ⊗ ℝ^4
typeof(W)
supertype(ProductSpace)
supertype(CompositeSpace)
```
The tensor product of a number of `CartesianSpace`s is some generic parametric
`ProductSpace` type, specifically `ProductSpace{CartesianSpace,N}` for the tensor product
of `N` instances of `CartesianSpace`.

Tensors behave like vectors (but not `VectorSpace` instance), so we can compute linear
combinations provided they live in the same space.
```@repl tutorial
B = Tensor(randn, ℝ^3 * ℝ^2 * ℝ^4);
C = 0.5*A + 2.5*B
```

Tensors also have inner product and norm, which they inherit from the Euclidean inner
product on the individual `ℝ^n` spaces:
```@repl tutorial
scalarBA = dot(B,A)
scalarAA = dot(A,A)
normA² = norm(A)^2
```

If two tensors live in different spaces, these operations have no meaning and are thus not
allowed
```@repl tutorial
B′ = Tensor(randn, ℝ^4 * ℝ^2 * ℝ^3);
space(B′) == space(A)
C′ = 0.5*A + 2.5*B′
scalarBA′ = dot(B′,A)
```

However, in this particular case, we can reorder the indices of `B′` to match space of `A`,
using the routine `permute` (we deliberately choose not to overload `permutedims` from
Julia Base):
```@repl tutorial
space(permute(B′,(3,2,1))) == space(A)
```

We can contract two tensors using Einstein summation convention, which takes the interface
from [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl). TensorXD.jl
reexports the `@tensor` macro
```@repl tutorial
@tensor D[a,b,c,d] := A[a,b,e]*B[d,c,e]
@tensor d = A[a,b,c]*A[a,b,c]
d ≈ scalarAA ≈ normA²
```
The `:=` is to create a new tensor `D`. The `=` write the contraction result in an existing
tensor `d`, which would yield an error if no tensor `d` exists. If the contraction yields a
scalar, regular assignment with `=` can be used.

We can also factorize a tensor. With a plain Julia `Array`, one would apply `permutedims`
and `reshape` to cast the array into a matrix before applying e.g. the singular value
decomposition. With TensorXD.jl, one just specifies which indices go to the left (rows)
and right (columns)
```@repl tutorial
U, S, Vd = tsvd(A, (1,3), (2,));
@tensor A′[a,b,c] := U[a,c,d]*S[d,e]*Vd[e,b];
A ≈ A′
U
```
The `tsvd` routine returns the decomposition of the linear map as three factors,
`U`, `S` and `Vd`, each of them a `TensorMap`, such that `Vd` is what is commonly
called `V'`.

Notice that `U` is printed as `TensorMap((ℝ^3 ⊗ ℝ^4) ← ProductSpace(ℝ^2))`, which is
a linear map between two `ProductSpace` instances, with
```@repl tutorial
codomain(U)
domain(U)
codomain(A)
domain(A)
```
Hence, a `Tensor` instance such as `A` is just a specific case of `TensorMap` with an empty
domain, i.e. a `ProductSpace{CartesianSpace,0}` instance. For example, we can represent a
vector `v` and matrix `m` as
```@repl tutorial
v = Tensor(randn, ℝ^3)
m1 = TensorMap(randn, ℝ^4, ℝ^3)
m2 = TensorMap(randn, ℝ^4 → ℝ^2) # alternative syntax for TensorMap(randn, ℝ^2, ℝ^4)
w = m1 * v # matrix vector product
m3 = m2 * m1 # matrix matrix product
```
Note that for the construction of `m1`, in accordance with how one specifies the dimensions
of a matrix (e.g. `randn(4,3)`), the first space is the codomain and the second the domain.
This is somewhat opposite to the general notation for a function `f:domain→codomain`, so
that we also support this more mathematical notation, as illustrated in the construction of
`m2`. There is a third syntax which mixes both like `TensorMap(randn, codomain←domain)`.

This 'matrix vector' or 'matrix matrix' product can be computed between any two `TensorMap`
instances for which the domain of the first matches with the codomain of the second, e.g.
```@repl tutorial
v′ = v ⊗ v
m1′ = m1 ⊗ m1
w′ = m1′ * v′
w′ ≈ w ⊗ w
```

Another example involves checking that `U` from the singular value decomposition is a left
isometric tensor
```@repl tutorial
codomain(U)
domain(U)
space(U)
U'*U # should be the identity on the corresponding domain = codomain
U'*U ≈ one(U'*U)
P = U*U' # should be a projector
P*P ≈ P
```
Here, the adjoint of a `TensorMap` results in a new tensor map (actually a simple wrapper
of type `AdjointTensorMap <: AbstractTensorMap`) with domain and codomain interchanged.

Our original tensor `A` living in `ℝ^4 * ℝ^2 * ℝ^3` is isomorphic to a linear map
`ℝ^3 → ℝ^4 * ℝ^2`. This is where the full power of `permute` emerges. It allows to
specify a permutation where some indices go to the codomain, and others go to the domain,
as
```@repl tutorial
A2 = permute(A,(1,2),(3,))
codomain(A2)
domain(A2)
```
In fact, `tsvd(A, (1,3),(2,))` is a shorthand for `tsvd(permute(A,(1,3),(2,)))`, where
`tsvd(A::TensorMap)` will just compute the singular value decomposition according to the
given codomain and domain of `A`.

The `@tensor` macro treats all indices at the same footing and thus does not distinguish
between codomain and domain. The linear numbering is first all indices in the codomain,
followed by all indices in the domain. However, when `@tensor` creates a new tensor
(using `:=`), the default syntax creates a `Tensor`, i.e. with all indices in the codomain.
```@repl tutorial
@tensor A′[a,b,c] := U[a,c,d]*S[d,e]*Vd[e,b];
codomain(A′)
domain(A′)
@tensor A2′[(a,b);(c,)] := U[a,c,d]*S[d,e]*Vd[e,b];
codomain(A2′)
domain(A2′)
@tensor A2′′[a b; c] := U[a,c,d]*S[d,e]*Vd[e,b];
A2 ≈ A2′ == A2′′
```
As illustrated for `A2′` and `A2′′`, additional syntax is available that enables one to
immediately specify the desired codomain and domain indices.

## Complex tensors

To create a complex tensor, we work with complex vector spaces
```@repl tutorial
A = Tensor(randn, ComplexF64, ℂ^3 ⊗ ℂ^2 ⊗ ℂ^4)
```
where `ℂ` is obtained as `\bbC+TAB` and we also have the non-Unicode alternative
`ℂ^n == ComplexSpace(n)`. Most functionality works exactly the same with real tensors
```@repl tutorial
B = Tensor(randn, ℂ^3 * ℂ^2 * ℂ^4);
C = im*A + (2.5-0.8im)*B
scalarBA = dot(B,A)
scalarAA = dot(A,A)
normA² = norm(A)^2
U,S,Vd = tsvd(A,(1,3),(2,));
@tensor A′[a,b,c] := U[a,c,d]*S[d,e]*Vd[e,b];
A′ ≈ A
permute(A,(1,3),(2,)) ≈ U*S*Vd
```

However, trying the following
```@repl tutorial
@tensor D[a,b,c,d] := A[a,b,e]*B[d,c,e]
@tensor d = A[a,b,c]*A[a,b,c]
```
we obtain `SpaceMismatch` errors. The reason for this is that, with `ComplexSpace`, an
index in a space `ℂ^n` can only be contracted with an index in the dual space
`dual(ℂ^n) == (ℂ^n)'`. Because of the complex Euclidean inner product, the dual space is
equivalent to the complex conjugate space, but not the the space itself.
```@repl tutorial
dual(ℂ^3) == conj(ℂ^3) == (ℂ^3)'
(ℂ^3)' == ℂ^3
@tensor d = conj(A[a,b,c])*A[a,b,c]
d ≈ normA²
```
This might seem overly strict or puristic, but we believe that it can help to catch errors,
e.g. unintended contractions. In particular, contracting two indices both living in `ℂ^n`
would represent an operation that is not invariant under arbitrary unitary basis changes.

It also makes clear the isomorphism between linear maps `ℂ^n → ℂ^m` and tensors in
`ℂ^m ⊗ (ℂ^n)'`:
```@repl tutorial
m = TensorMap(randn, ComplexF64, ℂ^3, ℂ^4)
m2 = permute(m, (1,2), ())
codomain(m2)
space(m, 1)
space(m, 2)
```
Hence, spaces become their corresponding dual space if they are 'permuted' from the domain
to the codomain or vice versa. Also, spaces in the domain are reported as their dual when
probing them with `space(A, i)`. Generalizing matrix vector and matrix matrix multiplication
to arbitrary tensor contractions require that the two indices to be contracted have spaces
which are each others dual. Knowing this, all the other functionality of tensors with
`CartesianSpace` indices remains the same for tensors with `ComplexSpace` indices.

## Symmetries
So far, the functionality that we have illustrated seems to be just a convenience (or
inconvenience?) wrapper around dense multidimensional arrays, e.g. Julia's Base `Array`.
More power becomes visible when involving symmetries. With symmetries, we imply that there
is some symmetry action defined on every vector space associated with each of the indices
of a `TensorMap`, and the `TensorMap` is then required to be equivariant, i.e. it acts as
an intertwiner between the tensor product representation on the domain and that on the
codomain. By Schur's lemma, this means that the tensor is block diagonal in some basis
corresponding to the irreducible representations that can be coupled to by combining the
different representations on the different spaces in the domain or codomain. For Abelian
symmetries, this does not require a basis change and it just imposes that the tensor has
some block sparsity. Let's clarify all of this with some examples.

We start with a simple ``ℤ₂`` symmetry:
```@repl tutorial
V1 = ℤ₂Space(0=>3,1=>2)
dim(V1)
V2 = ℤ₂Space(0=>1,1=>1)
dim(V2)
A = Tensor(randn, V1*V1*V2')
convert(Array, A)
```
Here, we create a space 5-dimensional space `V1`, which has a three-dimensional subspace
associated with charge 0 (the trivial irrep of ``ℤ₂``) and a two-dimensional subspace with
charge 1 (the non-trivial irrep). Similar for `V2`, where both subspaces are one-
dimensional. Representing the tensor as a dense `Array`, we see that it is zero in those
regions where the charges don't add to zero (modulo 2). The `Tensor(Map)` type
in TensorXD.jl won't store these zero blocks, and only stores the non-zero information,
which we can recognize in the full `Array` representation.

From there on, the resulting tensors support all of the same operations as the ones we
encountered in the previous examples.
```@repl tutorial
B = Tensor(randn, V1'*V1*V2);
@tensor C[a,b] := A[a,c,d]*B[c,b,d]
U,S,V = tsvd(A,(1,3),(2,));
U'*U # should be the identity on the corresponding domain = codomain
U'*U ≈ one(U'*U)
P = U*U' # should be a projector
P*P ≈ P
```

We also support other abelian symmetries, e.g.
```@repl tutorial
V = U₁Space(0=>2,1=>1,-1=>1)
dim(V)
A = TensorMap(randn, V*V, V)
dim(A)
convert(Array, A)

V = GradedSpace[Irrep[U₁×ℤ₂]]((0,0)=>2,(1,1)=>1,(-1,0)=>1)
dim(V)
A = TensorMap(randn, V*V, V)
dim(A)
convert(Array, A)
```
Here, the `dim` of a `TensorMap` returns the number of linearly independent components,
i.e. the number of non-zero entries in the case of an abelian symmetry. Note that we
can use `×` (obtained as `\times+TAB`) to combine different symmetries. The general space
associated with symmetries is a `GradedSpace`, which is the access point for users to
construct spaces with arbitrary symmetries, and `ℤ₂Space` (or `Z2Space`) and `U₁Space` (or
`U1Space`) are just convenient synonyms, e.g.
```@repl tutorial
GradedSpace[Irrep[U₁]](0=>3,1=>2,-1=>1) == U1Space(-1=>1,1=>2,0=>3)
V = U₁Space(1=>2,0=>3,-1=>1)
for s in sectors(V)
  @show s, dim(V, s)
end
U₁Space(-1=>1,0=>3,1=>2) == GradedSpace(Irrep[U₁](1)=>2, Irrep[U₁](0)=>3, Irrep[U₁](-1)=>1)
supertype(GradedSpace)
```
The `GradedSpace` is not immediately parameterized by some group `G`, but actually by the
set of irreducible representations of `G`, denoted as `Irrep[G]`. Generally, `GradedSpace`
supports a grading that is derived from the fusion ring of a (unitary) pre-fusion category.
The order in which the charges and their corresponding subspace dimensionality are specified
is irrelevant. The `GradedSpace[I]` constructor automatically converts the keys in the list
of `Pair`s it receives to the correct sector type. Alternatively, we can directly create
the sectors of the correct type and use the generic `GradedSpace` constructor. We can probe
the subspace dimension of a certain sector `s` in a space `V` with `dim(V, s)`.

The `GradedSpace` is a subtype of `EuclideanSpace{ℂ}`, i.e., it has the standard Euclidean
inner product and we assume all representations to be unitary.

TensorXD.jl also allows for non-abelian symmetries such as `SU₂`. In this case, the vector
space is characterized via the spin quantum number (i.e. the irrep label of `SU₂`) for each
of its subspaces, and is created using `SU₂Space` (or `SU2Space` or
`GradedSpace[Irrep[SU₂]]`)
```@repl tutorial
V = SU₂Space(0=>2,1/2=>1,1=>1)
dim(V)
V == GradedSpace[Irrep[SU₂]](0=>2, 1=>1, 1//2=>1)
```
`V` has a two-dimensional subspace with spin zero, and two one-dimensional subspaces with
spin 1/2 and spin 1. A subspace with spin `j` has an additional `2j+1` dimensional
degeneracy on which the irreducible representation acts. This brings the total dimension to
`2*1 + 1*2 + 1*3`. Creating a tensor with `SU₂` symmetry yields
```@repl tutorial
A = TensorMap(randn, V*V, V)
dim(A)
convert(Array, A)
norm(A) ≈ norm(convert(Array, A))
```
In this case, the full `Array` representation of the tensor has again many zeros, but it is
less obvious to recognize the dense blocks, as there are additional zeros and the numbers
in the original tensor data do not match with those in the `Array`. The reason is of course
that the original tensor data now needs to be transformed with a construction known as
fusion trees, which are made up out of the Clebsch-Gordan coefficients of the group.
Indeed, note that the non-zero blocks are also no longer labeled by a list of sectors, but
by pair of fusion trees. This will be explained further in the manual. However, the
Clebsch-Gordan coefficients of the group are only needed to actually convert a tensor to an
`Array`. For working with tensors with `SU₂Space` indices, e.g. contracting or factorizing
them, the Clebsch-Gordan coefficients are never needed explicitly. Instead, recoupling
relations are used to symbolically manipulate the basis of fusion trees, and this only
requires what is known as the topological data of the group (or its representation theory).

In fact, this formalism extends beyond the case of group representations on vector spaces,
and can also deal with super vector spaces (to describe fermions) and more general
(unitary) fusion categories. Preliminary support for these generalizations is present in
TensorXD.jl and will be extended in the near future.

All of these concepts will be explained throughout the remainder of this manual, including
several details regarding their implementation. However, to just use tensors and their
manipulations (contractions, factorizations, ...) in higher level algorithms (e.g. tensoer
network algorithms), one does not need to know or understand most of these details, and one
can immediately refer to the general interface of the `TensorMap` type, discussed on the
[last page](@ref s_tensors). Adhering to this interface should yield code and algorithms
that are oblivious to the underlying symmetries and can thus work with arbitrary symmetric
tensors.
