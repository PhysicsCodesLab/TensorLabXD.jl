# [Introduction](@id s_intro)

Before providing a typical "user guide" and discussing the implementation of TensorXD.jl,
let us discuss some of the rationale behind this package.

## [What is a tensor?](@id ss_whatistensor)

At the very start we should ponder about the most suitable and sufficiently general
definition of a tensor. A good starting point is the following:

*   A tensor ``t`` is an element from the tensor product of ``N`` vector spaces
    ``V_1 , V_2, …, V_N``, where ``N`` is referred to as the *rank* or *order* of the
    tensor, i.e.

    ``t ∈ V_1 ⊗ V_2 ⊗ … ⊗ V_N.``

If you think of a tensor as an object with indices, a rank ``N`` tensor has ``N`` indices.  
Each index labels a particular basis in its corresponding vector space. The tensor product
is only defined for vector spaces over the same field of scalars, e.g. ``ℝ^5 ⊗ ℂ^3`` is not
allowed.

Since the tensor product of vector spaces is itself a vector space, a tensor behaves as a
vector, i.e., a tensor can be multiplied by scalars and tensors in the same tensor product
space can be added. When all the vector spaces in the tensor product have an inner product,
the tensor product space also has an inner product.

Aside from interpreting a tensor as a vector, we can also interpret it as a linear map, and
call it as a tensor map:

*   A tensor map ``t`` is a linear map from a source or *domain*
    ``W_1 ⊗ W_2 ⊗ … ⊗ W_{N_2}`` to a target or *codomain* ``V_1 ⊗ V_2 ⊗ … ⊗ V_{N_1}``, i.e.

    ``t:W_1 ⊗ W_2 ⊗ … ⊗ W_{N_2} → V_1 ⊗ V_2 ⊗ … ⊗ V_{N_1}.``

A *tensor* of rank ``N`` is then just a special case of a tensor map with ``N_1 = N`` and
``N_2 = 0``.

A contraction between two tensor maps is just a composition of linear maps (i.e.
matrix multiplication), where the contracted indices correspond to the domain of the first
tensor and the codomain of the second tensor.

We can also decompose tensor maps using linear algebra factorisations (e.g. eigenvalue or
singular value decomposition).

In order to allow for arbitrary tensor contractions or decompositions, we need to be able to
reorganise the positions of vector spaces appear in the domain and the codomain of the
tensor map. This amounts to defining canonical isomorphisms between the different ways
to order and partition the tensor indices. For example, a tensor map ``W → V`` can also be
denoted as a rank 2 tensor in ``V ⊗ W^*``, where ``W^*`` is the dual space of ``W``. This
simple example introduces two new concepts:

1.  Typical vector spaces can appear in the domain and codomain in different related forms.

    In general, every vector space ``V`` has a dual space ``V^*``, a conjugate space
    ``\overline{V}``, and a conjugate dual space ``\overline{V}^*``.

    The spaces ``V^*`` and ``\overline{V}`` correspond respectively to the
    representation spaces of the dual and complex conjugate representation of the general
    linear group ``\mathsf{GL}(V)``.

    For complex vector spaces with an inner product ``\overline{V} ⊗ V → ℂ``, the inner
    product allows to define an isomorphism from the conjugate space to the dual space.

    In spaces with a Euclidean inner product (the setting of quantum mechanics), the
    conjugate and dual space are naturally isomorphic (because the dual and conjugate
    representation of the unitary group are the same).

    In cartesian space ``ℝ^d`` with a Euclidean inner product, these four different spaces
    are all equivalent. The space is completely characterized by its dimension ``d``. This
    is the setting of much of classical mechanics. The tensors in ``ℝ^d`` can equally well
    be represented as multidimensional arrays (i.e. using `AbstractArray{<:Real,N}` in
    Julia) without loss of structure.

2.  In general, the identification between maps ``W → V`` and tensors in
    ``V ⊗ W^*`` is not an equivalence but an isomorphism, which needs to be defined.
    Similarly, there is an isomorphism between between ``V ⊗ W`` and ``W ⊗ V`` that can be
    non-trivial (e.g. in the case of fermions / super vector spaces). The correct formalism
    here is provided by theory of monoidal categories.

This brings us to our final formal definition of a tensor map:

*   A tensor map is a homomorphism between two objects from the category ``\mathbf{Vect}``
    (or some subcategory thereof). In practice, this will be ``\mathbf{FinVect}``, the
    category of finite dimensional vector spaces. More generally, our tensor maps make sense
    for any linear (a.k.a. ``\mathbf{Vect}``-enriched) monoidal category.

## [Symmetries and block sparsity](@id ss_symmetries)

Physical problems often have some symmetries, i.e., the system is invariant under the action
of a group ``\mathsf{G}`` which acts on the vector spaces ``V`` of the system according to a
certain representation. Having quantum mechanics in mind, TensorXD.jl is so far restricted
to unitary representations.

A general representation space ``V`` can be specified as the number of times every
irreducible representation (irrep) ``a`` of ``\mathsf{G}`` appears, i.e.,

``V = \bigoplus_{a} ℂ^{n_a} ⊗ R_a``

with ``R_a`` the space associated with irrep ``a`` of ``\mathsf{G}``, which itself has
dimension ``d_a`` (often called the quantum dimension), and ``n_a`` the number of times
this irrep appears in ``V``. The total dimension of ``V`` is given by ``∑_a n_a d_a``.

If the unitary irrep ``a`` for ``g ∈ \mathsf{G}`` is given by ``u_a(g)``, then there exists
a specific basis for ``V`` such that the group action of ``\mathsf{G}`` on ``V`` is given
by the unitary representation

``u(g) = \bigoplus_{a}  𝟙_{n_a} ⊗ u_a(g)``

with ``𝟙_{n_a}`` the ``n_a × n_a`` identity matrix.

The reason for implementing symmetries is to exploit the computation and memory gains
by restricting to tensor maps that are equivariant under the symmetry. The symmetric tensors
act as intertwiners between the symmetry action on the domain and the codomain. (This is the
key point to understand the concept of symmetric tensor maps!!!)

We could change the basis of the domain (or codomain) from the tensor product of the
uncoupled irreps ``a_1\otimes a_2\otimes ...\otimes a_N``, where ``a_i\in W_i`` (or ``V_i``),
to coupled irreps. The basis change is implemented by a sequence of Clebsch–Gordan
coefficients, also known as a fusion (or splitting) tree.

The symmetric tensors should be **block diagonal** in the basis of coupled irreps because of
Schur's lemma.

We implement the necessary machinery to manipulate these fusion trees for arbitrary groups
``\mathsf{G}``. Generally, this fits with the formalism of monoidal fusion categories,
and only requires the *topological* data of the group, i.e., the fusion rules of the irreps,
their quantum dimensions, and the F-symbol (6j-symbol or more precisely Racah's W-symbol in
the case of ``\mathsf{SU}_2``).
