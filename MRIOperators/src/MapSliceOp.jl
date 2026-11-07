export MapSliceOp

function mapSliceForeward(A, x::AbstractVector, size1::Tuple, dim::Int)
  return vec( mapslices(u->A*u, reshape(x, size1), dims=dim) )
end

function mapSliceBackward(A, y::AbstractVector, size2::Tuple, dim::Int)
  return vec( mapslices(v->adjoint(A)*v, reshape(y, size2), dims=dim) )
end

# An empty array on the backend of a transformation, used to derive the operator's storage type.
mapSliceStorageProbe(trafo::AbstractArray) = trafo
mapSliceStorageProbe(trafo) = LinearOperators.storage_type(trafo)(undef, 0)

"""
    MapSliceOp(trafo, dim::Int64, size1::Tuple, size2::Tuple; T=ComplexF64, S)

generates an operator that applies `trafo` to each slice of dimension `dim` of an array

# Arguments
* `trafo`           - transformation to apply
* `size1`           - size of the Array to be transformed
* `size2`           - size of the resulting Array of applying trafo to all slices
* (`T=ComplexF64`)  - type of the transformation
* (`S`)             - storage type of the operator; defaults to vectors of element type `T` on the backend of `trafo`

# Moving to another array backend

    Adapt.adapt(CuArray, op::MapSliceOp)

returns a new `MapSliceOp` with a new parent operator, built from `trafo` adapted to the target and the
same `dim`, `size1`, `size2` and `T`, so it applies the same operation as `op` at the time of the call.
`trafo` can be an array, which is adapted like the inputs of `SensitivityOp`, or an operator with an Adapt
rule such as the operators of LinearOperatorCollection. For an operator without such a rule, e.g. a
`LinearOperator` built from functions, `adapt` throws an `ArgumentError`. Adapt changes the backend, not
the element type: targets that change the element type of `trafo` or of the storage throw an `ArgumentError`.
`copy(op)` returns a new `MapSliceOp` with `copy(trafo)`. `op` itself is not modified.
`SensitivityOp`, `SubspaceOp`, `MapSliceOp` and `ExplicitOp` (also as `adjoint`) are the MRI operators
that support `Adapt.adapt` so far.
"""
struct MapSliceOp{T, O} <: WrappedMRIOperator{T, O}
  op::O
  trafo::Any
  dim::Int
  size1::Tuple
  size2::Tuple
  function MapSliceOp(trafo, dim::Int64, size1::Tuple, size2::Tuple; T=ComplexF64,
                      S = typeof(similar(mapSliceStorageProbe(trafo), T, 0)))
    op = LinearOperator{T}(prod(size2), prod(size1), false, false
              , (res,x) -> ( res .= mapSliceForeward(trafo, x, size1, dim) )
              , nothing
              , (res,y) -> ( res .= mapSliceBackward(trafo, y, size2, dim) )
              , S = S)
    return new{T, typeof(op)}(op, trafo, dim, size1, size2)
  end
end
parent(op::MapSliceOp) = getfield(op, :op)

function checkMapSliceSource(op::MapSliceOp{T}) where T
  S = LinearOperators.storage_type(op)
  eltype(S) == T || throw(ArgumentError("cannot adapt or copy MapSliceOp: its storage type $S does not hold its element type $T"))
  return S
end

function Base.copy(op::MapSliceOp{T}) where T
  S = checkMapSliceSource(op)
  return MapSliceOp(copy(getfield(op, :trafo)), getfield(op, :dim), getfield(op, :size1), getfield(op, :size2); T, S)
end

adaptMapSliceChild(to, trafo::AbstractArray) = adaptRetained(to, trafo, "MapSliceOp transformation")
function adaptMapSliceChild(to, trafo)
  moved = adapt(to, trafo)
  moved === trafo && throw(ArgumentError("cannot adapt MapSliceOp: its transformation of type $(typeof(trafo)) has no Adapt rule"))
  return moved
end

function Adapt.adapt_structure(to, op::MapSliceOp{T}) where T
  checkMapSliceSource(op)
  storage = adaptRetained(to, similar(parent(op).Mv, 0), "MapSliceOp storage")
  eltype(storage) == T ||
    throw(ArgumentError("cannot adapt MapSliceOp: the target changes the element type of its storage from $T to $(eltype(storage))"))
  trafo = getfield(op, :trafo)
  moved = adaptMapSliceChild(to, trafo)
  eltype(moved) == eltype(trafo) ||
    throw(ArgumentError("cannot adapt MapSliceOp: the target changes the element type of its transformation from $(eltype(trafo)) to $(eltype(moved))"))
  return MapSliceOp(moved, getfield(op, :dim), getfield(op, :size1), getfield(op, :size2); T, S = typeof(storage))
end

# The adjoint of an MRI operator is rebuilt from its adapted parent.
Adapt.adapt_structure(to, op::LinearOperators.AdjointLinearOperator{T, <:Union{MapSliceOp, ExplicitOp}}) where T =
  adjoint(Adapt.adapt(to, op.parent))
