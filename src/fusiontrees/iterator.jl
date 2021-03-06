"""
	struct FusionTreeIterator{I<:Sector, N}
    	uncoupled::NTuple{N, I}
    	coupled::I
    	isdual::NTuple{N, Bool}
	end

Iterate over all possible splitting trees for fixed uncoupled and coupled sectors.
"""
struct FusionTreeIterator{I<:Sector, N}
    uncoupled::NTuple{N, I}
    coupled::I
    isdual::NTuple{N, Bool}
end

Base.IteratorSize(::FusionTreeIterator) = Base.HasLength()
Base.IteratorEltype(::FusionTreeIterator) = Base.HasEltype()
Base.eltype(T::Type{FusionTreeIterator{I, N}}) where {I<:Sector, N} =
    fusiontreetype(I, N)

Base.length(iter::FusionTreeIterator) = _fusiondim(iter.uncoupled, iter.coupled)
"""
	_fusiondim(u::Tuple{I, I, Vararg{I}}, c::I) where {I<:Sector}

Return the number of fusiontrees with fixed uncoupled objects `u` and coupled object `c`.
"""
_fusiondim(u::Tuple{}, c::I) where {I<:Sector} = Int(one(c) == c)
_fusiondim(u::Tuple{I}, c::I) where {I<:Sector} = Int(u[1] == c)
_fusiondim((a, b)::Tuple{I, I}, c::I) where {I<:Sector} = Int(Nsymbol(a, b, c))
function _fusiondim(u::Tuple{I, I, Vararg{I}}, c::I) where {I<:Sector}
    a = u[1]
    b = u[2]
    d = 0
    for c′ in a ⊗ b
        d += Nsymbol(a, b, c′)*_fusiondim((c′, TupleLabXD.tail2(u)...), c)
    end
    return d
end

# * Iterator methods:
#   Start with special cases:
function Base.iterate(it::FusionTreeIterator{I, 0},
                        state = (it.coupled != one(I))) where {I<:Sector}
    state && return nothing
    T = vertex_labeltype(I)
    tree = FusionTree{I, 0, 0, 0, T}((), one(I), (), (), ())
    return tree, true
end

function Base.iterate(it::FusionTreeIterator{I, 1},
                        state = (it.uncoupled[1] != it.coupled)) where {I<:Sector}
    state && return nothing
    T = vertex_labeltype(I)
    tree = FusionTree{I, 1, 0, 0, T}(it.uncoupled, it.coupled, it.isdual, (), ())
    return tree, true
end

#   General case:
"""
	labelvertices(uncoupled::NTuple{N, I}, coupled::I, lines, vertices) where {I<:Sector, N}

Return the vertices with the label type of its elements defined by the function
`vertex_ind2label`. It is used to construct the instances of `FusionTree`.
"""
labelvertices(uncoupled::NTuple{2, I}, coupled::I,
				lines::Tuple{}, vertices::Tuple{Int}) where {I<:Sector} =
    (vertex_ind2label(vertices[1], uncoupled..., coupled),)

function labelvertices(uncoupled::NTuple{N, I}, coupled::I, lines,
                        vertices) where {I<:Sector, N}
    c = lines[1]
	l = vertex_ind2label(vertices[1], uncoupled[1], uncoupled[2], c)
    resttree = tuple(c, TupleLabXD.tail2(uncoupled)...)
    rest = labelvertices(resttree, coupled, tail(lines), tail(vertices))
    return (l, rest...)
end

@inline function _iterate(uncoupled::NTuple{2, I}, coupled::I, lines = (),
                            vertices = (0,), states = ()) where {I<:Sector}
    a, b = uncoupled
    n = vertices[1] + 1
    n > Nsymbol(a, b, coupled) && return nothing
    return (), (n,), ()
end

function _iterate(uncoupled::NTuple{N, I}, coupled::I) where {N, I<:Sector}
    a, b, = uncoupled
    it = a ⊗ b
    next = iterate(it)
    c, s = next
    resttree = tuple(c, TupleLabXD.tail2(uncoupled)...)
    rest = _iterate(resttree, coupled)
    while rest === nothing
        next = iterate(it, s)
        next === nothing && return nothing
        c, s = next
        resttree = tuple(c, TupleLabXD.tail2(uncoupled)...)
        rest = _iterate(resttree, coupled)
    end
    n = 1
    restlines, restvertices, reststates = rest
    lines = (c, restlines...)
    vertices = (n, restvertices...)
    states = (s, reststates...)
    return lines, vertices, states
end

# In the iterating, for each node, change the vertices label first, until Nsymbol(a,b,c) is
# reached, then change the coupled object of that node.
# The states record the iterator state of c in a ⊗ b, and keep unchanged when changing the
# vertices label.
function _iterate(uncoupled::NTuple{N, I}, coupled::I, lines, vertices,
                            states) where {N, I<:Sector}
    a, b, = uncoupled
    it = a ⊗ b
    c = lines[1]
    n = vertices[1]
    s = states[1]
    restlines = tail(lines)
    restvertices = tail(vertices)
    reststates = tail(states)
    if n < Nsymbol(a, b, c)
        n += 1
        return lines, (n, restvertices...), states
    end
    n = 1
    resttree = tuple(c, TupleLabXD.tail2(uncoupled)...)
    rest = _iterate(resttree, coupled, restlines, restvertices, reststates)
    while rest === nothing
        next = iterate(it, s)
        next === nothing && return nothing
        c, s = next
        resttree = tuple(c, TupleLabXD.tail2(uncoupled)...)
        rest = _iterate(resttree, coupled)
    end
    restlines, restvertices, reststate = rest
    lines = (c, restlines...)
    vertices = (n, restvertices...)
    states = (s, reststate...)
    return lines, vertices, states
end

function Base.iterate(it::FusionTreeIterator{I, N} where {N}) where {I<:Sector}
    next = _iterate(it.uncoupled, it.coupled)
    next === nothing && return nothing
    lines, vertices, states = next
    vertexlabels = labelvertices(it.uncoupled, it.coupled, lines, vertices)
    f = FusionTree(it.uncoupled, it.coupled, it.isdual, lines, vertexlabels)
    return f, (lines, vertices, states)
end

function Base.iterate(it::FusionTreeIterator{I, N} where {N}, state) where {I<:Sector}
    next = _iterate(it.uncoupled, it.coupled, state...)
    next === nothing && return nothing
    lines, vertices, states = next
    vertexlabels = labelvertices(it.uncoupled, it.coupled, lines, vertices)
    f = FusionTree(it.uncoupled, it.coupled, it.isdual, lines, vertexlabels)
    return f, (lines, vertices, states)
end

"""
	fusiontrees(uncoupled::NTuple{N, I}, coupled::I = one(I),
				isdual::NTuple{N, Bool} = ntuple(n->false, Val(N))) where {N, I<:Sector}

Return the iterator `FusionTreeIterator` for fixed uncoupled and coupled sectors.
"""
function fusiontrees(uncoupled::NTuple{N, I}, coupled::I = one(I),
			isdual::NTuple{N, Bool} = ntuple(n->false, Val(N))) where {N, I<:Sector}
    FusionTreeIterator{I, N}(uncoupled, coupled, isdual)
end
