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
rule such as the operators of LinearOperatorCollection; `to` is passed to these rules as given. If `trafo` is
or contains an operator without such a rule, e.g. a `LinearOperator` built from functions, `adapt` throws an
`ArgumentError`. Contained operators are the operator children of `ProdOp`, `DiagOp`, `RowBlockOp`, their normal
operators, `MapSliceOp` and adjoints, except below an operator for which `to` has a rule of its own; other operators
are rebuilt by their own rules. If the result still contains an operator of `trafo`, which would share its workspaces
with `op`, `adapt` throws an `ArgumentError`. Array operands may be shared on the same backend; they are only read. Adapt changes the backend, not
the element type: targets that change the element type of `trafo` or of the storage throw an `ArgumentError`.
`copy(op)` returns a new `MapSliceOp` with a copy of `trafo`: an operator with an Adapt rule is rebuilt by
that rule on its backend from copies of its arrays. Operators without an Adapt rule, also inside `trafo`, are
copied with `copy`; if one of them has no `copy` method, `copy(op)` throws an `ArgumentError`. `op` itself is
not modified.
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

# The operators that compute `op`: `op` and, recursively, the operator children of the composites below.
# Array operands are only read and are not included.
function childOperators(op, fields...)
  children = Any[]
  for f in fields
    x = getfield(op, f)
    xs = x isa Union{Tuple, AbstractVector} && !(x isa AbstractArray{<:Number}) ? x : (x,)
    append!(children, Iterators.filter(c -> c isa AbstractLinearOperator, xs))
  end
  return children
end
operatorChildren(op) = ()
operatorChildren(op::ProdOp) = childOperators(op, :A, :B)
operatorChildren(op::Union{DiagOp, RowBlockOp}) = childOperators(op, :ops)
operatorChildren(op::Union{LinearOperatorCollection.DiagNormalOp, LinearOperatorCollection.RowBlockNormalOp}) = childOperators(op, :normalOps)
operatorChildren(op::LinearOperatorCollection.ProdNormalOp) = childOperators(op, :opOuter, :normalOpInner)
operatorChildren(op::LinearOperatorCollection.NormalOpImpl) = childOperators(op, :parent, :weights)
operatorChildren(op::Union{LinearOperators.AdjointLinearOperator, LinearOperators.TransposeLinearOperator,
                           LinearOperators.ConjugateLinearOperator}) = childOperators(op, :parent)
operatorChildren(op::MapSliceOp) = childOperators(op, :trafo)
function operatorTree(op, found = Base.IdSet{Any}())
  op in found && return found
  push!(found, op)
  foreach(child -> operatorTree(child, found), operatorChildren(op))
  return found
end

# Whether `adapt(to, op)` uses a rule rather than Adapt's fallback, which returns `op` unchanged.
isAdaptFallback(f, to, op) = which(f, Tuple{Core.Typeof(to), typeof(op)}) === which(f, Tuple{Any, Any})
hasAdaptRule(to, op) = !(isAdaptFallback(Adapt.adapt_structure, to, op) && isAdaptFallback(Adapt.adapt_storage, to, op))
# Whether `to` has a rule of its own for `op` (not one for any adaptor), which then decides how its children are rebuilt.
ownAdaptRule(to, op) = Base.unwrap_unionall(which(Adapt.adapt_structure, Tuple{Core.Typeof(to), typeof(op)}).sig).parameters[2] !== Any

function checkAdaptRules(to, op)
  hasAdaptRule(to, op) || throw(ArgumentError("cannot adapt MapSliceOp: its transformation is or contains an operator of type " *
                                              "$(typeof(op)) that has no Adapt rule"))
  ownAdaptRule(to, op) || foreach(child -> checkAdaptRules(to, child), operatorChildren(op))
end

# A result that still contains an operator of the source would share its workspaces with it.
function checkRebuilt(moved, trafo, verb)
  source = operatorTree(trafo)
  for op in operatorTree(moved)
    op in source && throw(ArgumentError("cannot $verb MapSliceOp: the new transformation still contains the operator of type " *
                                        "$(typeof(op)) of the source, which would be shared"))
  end
  return moved
end

adaptMapSliceChild(to, trafo::AbstractArray) = adaptRetained(to, trafo, "MapSliceOp transformation")
function adaptMapSliceChild(to, trafo)
  checkAdaptRules(to, trafo)
  return checkRebuilt(adapt(to, trafo), trafo, "adapt")
end

# Adapts to the backend `to` and copies the arrays that are already on it, so Adapt rules rebuild an operator
# with its own arrays and workspaces. Operators reach `adapt_storage` only if they have no Adapt rule, also
# inside a composite such as `ProdOp`; they are copied with their `copy` method.
struct MapSliceCopyAdaptor{A}
  to::A
end
Adapt.adapt_storage(a::MapSliceCopyAdaptor, x::AbstractArray) = (y = adapt(a.to, x); y === x ? copy(x) : y)
Adapt.adapt_storage(::MapSliceCopyAdaptor, op::AbstractLinearOperator) = hasmethod(copy, Tuple{typeof(op)}) ? copy(op) :
  throw(ArgumentError("cannot copy MapSliceOp: its transformation is or contains an operator of type $(typeof(op)) that has no Adapt rule and no `copy` method"))

copyMapSliceChild(trafo::AbstractArray) = copy(trafo)
function copyMapSliceChild(trafo)
  moved = adapt(MapSliceCopyAdaptor(stripParameters(LinearOperators.storage_type(trafo))), trafo)
  moved === trafo && return copy(trafo) # neither an operator nor adaptable
  return checkRebuilt(moved, trafo, "copy")
end

function Base.copy(op::MapSliceOp{T}) where T
  S = checkMapSliceSource(op)
  return MapSliceOp(copyMapSliceChild(getfield(op, :trafo)), getfield(op, :dim), getfield(op, :size1), getfield(op, :size2); T, S)
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
