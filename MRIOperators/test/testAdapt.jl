using MRIOperators.Adapt

# CPU runs move between Array and JLArray; GPU runs (arrayTypes set by test/gpu/*.jl) between Array and the device array.
adaptTargets = arrayTypes == [Array] ? [Array, JLArray] : arrayTypes
adaptDirections = vcat([(Array, to) for to in adaptTargets], [(to, Array) for to in adaptTargets if to !== Array])
adaptRtol(T) = real(T) == Float32 ? 5e-5 : 1e-11

storageType(op) = MRIOperators.LinearOperators.storage_type(op)
testData(T, dims...) = reshape(T[T <: Real ? cos(0.3k + 0.1) : complex(cos(0.3k + 0.1), sin(0.7k) + 0.5) for k in 1:prod(dims)], dims)
onTarget(x, to) = x isa to && x isa DenseArray

# A host matrix that is isbits like StaticArrays' SMatrix: device array adaptors leave such arrays unchanged.
struct IsbitsMatrix{T, L} <: AbstractMatrix{T}
  data::NTuple{L, T}
  rows::Int
end
IsbitsMatrix(a::AbstractMatrix) = IsbitsMatrix(Tuple(a), size(a, 1))
Base.size(m::IsbitsMatrix{T, L}) where {T, L} = (m.rows, L ÷ m.rows)
Base.getindex(m::IsbitsMatrix, i::Int, j::Int) = m.data[(j - 1) * m.rows + i]

# An adaptor that records the arrays it converts and delegates them to Adapt's rules for `to`.
struct DelegatingAdaptor{A}
  to::A
  seen::Vector{Any}
end
DelegatingAdaptor(to) = DelegatingAdaptor(to, Any[])
Adapt.adapt_storage(a::DelegatingAdaptor, x::AbstractArray) = (push!(a.seen, x); adapt(a.to, x))
reached(a::DelegatingAdaptor, x) = any(y -> y === x, a.seen)
# An adaptor with its own rule for WeightingOp, which rejects it.
struct RejectWeighting{A}
  to::A
end
Adapt.adapt_storage(a::RejectWeighting, x::AbstractArray) = adapt(a.to, x)
Adapt.adapt_structure(::RejectWeighting, ::WeightingOp) = throw(ArgumentError("unsupported WeightingOp target"))

# A wrapper whose metadata captures an operator that it never applies.
const LO = MRIOperators.LinearOperators
struct MetadataOp{T, O, M} <: LO.AbstractLinearOperator{T}
  op::O
  metadata::M
end
MetadataOp(op::LO.AbstractLinearOperator{T}, meta) where T = MetadataOp{T, typeof(op), typeof(meta)}(op, meta)
Base.getproperty(x::MetadataOp, s::Symbol) = s in (:op, :metadata) ? getfield(x, s) : getproperty(getfield(x, :op), s)
Base.setproperty!(x::MetadataOp, s::Symbol, v) = setproperty!(getfield(x, :op), s, v)
LO.storage_type(x::MetadataOp) = LO.storage_type(getfield(x, :op))
Adapt.adapt_structure(to, x::MetadataOp) = MetadataOp(adapt(to, getfield(x, :op)), getfield(x, :metadata))

# Forward and adjoint results with three- and five-argument `mul!`, collected on the CPU.
function adaptActions(op, conv, x, y)
  T = eltype(op)
  α, β = T(0.5), T(0.75)
  f3, f5, a3, a5 = conv(zeros(T, size(op, 1))), conv(copy(y)), conv(zeros(T, size(op, 2))), conv(copy(x))
  mul!(f3, op, conv(x)); mul!(f5, op, conv(x), α, β)
  mul!(a3, adjoint(op), conv(y)); mul!(a5, adjoint(op), conv(y), α, β)
  return map(Array, (f3, f5, a3, a5))
end
adaptActions(op, conv) = adaptActions(op, conv, testData(eltype(op), size(op, 2)), testData(eltype(op), size(op, 1)) .+ 1)

function testAdaptActions(moved, to, ref, refConv, rtol)
  for (m, r) in zip(adaptActions(moved, to), adaptActions(ref, refConv))
    @test isapprox(m, r; rtol)
  end
end

function testMovedWrapper(moved, src, to)
  @test typeof(moved).name.wrapper === typeof(src).name.wrapper
  @test moved !== src && parent(moved) !== parent(src)
  @test eltype(moved) == eltype(src) && size(moved) == size(src)
  @test storageType(parent(moved)) <: to
  @test moved.nrow == src.nrow && moved.ncol == src.ncol # property forwarding to the parent
end
function testSensitivityLeaves(op, to)
  @test fieldnames(typeof(op)) == (:op, :sensMaps, :sensMapsC, :numContr)
  @test onTarget(getfield(op, :sensMaps), to)
  @test onTarget(getfield(op, :sensMapsC), to)
end
function testSubspaceLeaf(op, to)
  @test fieldnames(typeof(op)) == (:op, :basis, :shape, :numContr)
  @test onTarget(getfield(op, :basis), to)
end

# Contents and counters of an operator's workspaces (`tmp` for LOC products).
workspaceState(op) = (Array(op.Mv), Array(op.Mtu), op.nprod, op.ntprod, op.nctprod,
                      hasfield(typeof(op), :tmp) ? Array(op.tmp) : nothing)

@testset "Adapt $(nameof(from)) -> $(nameof(to))" for (from, to) in adaptDirections
  @testset "Wrappers $T $shape numContr=$numContr" for T in (ComplexF32, ComplexF64), shape in ((3, 2), (2, 2, 2)), numContr in (1, 3)
    rtol = adaptRtol(T)
    numVox, numChan = prod(shape), 2
    maps4 = testData(T, shape..., ntuple(_ -> 1, 3 - length(shape))..., numChan)
    maps = reshape(maps4, numVox, numChan)
    for src in (SensitivityOp(from(maps), numContr), SensitivityOp(from(maps4), numContr))
      moved = adapt(to, src)
      testMovedWrapper(moved, src, to)
      testSensitivityLeaves(moved, to)
      testAdaptActions(moved, to, src, from, rtol)
    end
    src = SubspaceOp(from(testData(T, numContr, 2)), shape, numContr)
    moved = adapt(to, src)
    testMovedWrapper(moved, src, to)
    testSubspaceLeaf(moved, to)
    @test getfield(moved, :shape) == shape && getfield(moved, :numContr) == numContr
    testAdaptActions(moved, to, src, from, rtol)
  end

  @testset "Nested product and source workspaces $T" for T in (ComplexF32, ComplexF64)
    shape, numContr = (2, 2, 2), 3
    maps, basis = from(testData(T, prod(shape), 2)), from(testData(T, numContr, 2))
    S = SensitivityOp(maps, numContr)
    B = SubspaceOp(basis, shape, numContr)
    W = WeightingOp(from(real(T)[0.5 + k / 10 for k in 1:size(S, 1)]))
    inner = ProdOp(S, B)
    P = ProdOp(W, inner)
    sources = (S, B, W, inner, P)
    # Exercising the sources allocates and writes their workspaces; the results are the references.
    refs = map(op -> adaptActions(op, from), sources)
    holders = (parent(S), parent(B), getfield(W, :op), inner, P)
    before = map(workspaceState, holders)
    inputs = map(Array, (maps, getfield(S, :sensMapsC), basis, W.weights))

    moved = adapt(to, P)
    @test moved.B.A isa SensitivityOp && moved.B.B isa SubspaceOp
    @test onTarget(moved.A.weights, to)
    testSensitivityLeaves(moved.B.A, to)
    testSubspaceLeaf(moved.B.B, to)
    for op in (moved, moved.A, moved.B, parent(moved.B.A), parent(moved.B.B))
      @test storageType(op) <: to
    end
    for (op, ref) in zip((moved.B.A, moved.B.B, moved.A, moved.B, moved), refs)
      for (m, r) in zip(adaptActions(op, to), ref)
        @test isapprox(m, r; rtol = adaptRtol(T))
      end
    end
    @test map(workspaceState, holders) == before
    @test map(Array, (maps, getfield(S, :sensMapsC), basis, W.weights)) == inputs
  end

  @testset "Sensitivity maps changed after construction" begin
    T = ComplexF32
    maps = testData(T, 4, 2)
    x, y = testData(T, 4), testData(T, 8)
    m = from(copy(maps))
    src = SensitivityOp(m, 1)
    m .*= 2 # the forward operation reads the live maps; the adjoint keeps the conjugate taken at construction
    srcFwd, srcAdj = Array(src * from(x)), Array(adjoint(src) * from(y))
    @test srcFwd ≈ 2 .* (SensitivityOp(maps, 1) * x)
    @test srcAdj ≈ adjoint(SensitivityOp(maps, 1)) * y
    @test !isapprox(Array(adjoint(SensitivityOp(m, 1)) * from(y)), srcAdj; rtol = 0.1) # rebuilding from the maps alone differs
    moved = adapt(to, src)
    @test isapprox(Array(moved * to(x)), srcFwd; rtol = adaptRtol(T))
    @test isapprox(Array(adjoint(moved) * to(y)), srcAdj; rtol = adaptRtol(T))
  end
end

@testset "Adapt wrapped inputs Array -> $(nameof(to)) $T" for to in adaptTargets, T in (ComplexF32, ComplexF64)
  rtol = adaptRtol(T)
  maps, big, maps4 = testData(T, 6, 2), testData(T, 8, 2), testData(T, 3, 2, 1, 2)
  for maps in (view(maps, :, :), view(big, 2:7, :), reshape(view(maps4, :, :, :, :), 6, 2), Diagonal(testData(T, 3)),
               IsbitsMatrix(maps))
    src = SensitivityOp(maps, 1)
    moved = adapt(to, src)
    testMovedWrapper(moved, src, to)
    testSensitivityLeaves(moved, to)
    testAdaptActions(moved, to, src, Array, rtol)
  end
  basis, bigBasis = testData(T, 3, 2), testData(T, 5, 2)
  for basis in (view(basis, :, :), view(bigBasis, 2:4, :), Diagonal(testData(T, 3)), IsbitsMatrix(basis))
    src = SubspaceOp(basis, (3, 1), 3)
    moved = adapt(to, src)
    testMovedWrapper(moved, src, to)
    testSubspaceLeaf(moved, to)
    testAdaptActions(moved, to, src, Array, rtol)
  end
  # Structured and isbits inputs inside a product
  src = ProdOp(SensitivityOp(Diagonal(testData(T, 3)), 1), SubspaceOp(IsbitsMatrix(testData(T, 3, 3)), (1, 1), 3))
  moved = adapt(to, src)
  testSensitivityLeaves(moved.A, to)
  testSubspaceLeaf(moved.B, to)
  @test storageType(moved) <: to
  testAdaptActions(moved, to, src, Array, rtol)
end

@testset "Adapt wrapped inputs $(nameof(from)) -> Array" for from in filter(!=(Array), adaptTargets)
  T = ComplexF32
  maps, basis = testData(T, 6, 2), testData(T, 3, 2)
  for (src, ref) in ((SensitivityOp(view(from(maps), :, :), 1), SensitivityOp(maps, 1)),
                     (SensitivityOp(Diagonal(from(testData(T, 3))), 1), SensitivityOp(Matrix(Diagonal(testData(T, 3))), 1)),
                     (SubspaceOp(view(from(basis), :, :), (3, 1), 3), SubspaceOp(basis, (3, 1), 3)))
    moved = adapt(Array, src)
    testMovedWrapper(moved, src, Array)
    src isa SensitivityOp ? testSensitivityLeaves(moved, Array) : testSubspaceLeaf(moved, Array)
    testAdaptActions(moved, Array, ref, Array, adaptRtol(T))
  end
end

if arrayTypes == [Array]
  @testset "Adapt SubspaceOp with explicit storage type $T" for T in (ComplexF32, ComplexF64)
    basis = testData(T, 3, 2)
    src = SubspaceOp(basis, (2, 2), 3, JLArray{T, 1})
    ref = SubspaceOp(basis, (2, 2), 3)
    @test storageType(parent(src)) == JLArray{T, 1}
    testAdaptActions(src, JLArray, ref, Array, adaptRtol(T))
    for to in (Array, JLArray)
      moved = adapt(to, src) # the storage type follows the adapted basis
      testMovedWrapper(moved, src, to)
      testSubspaceLeaf(moved, to)
      testAdaptActions(moved, to, ref, Array, adaptRtol(T))
    end
  end
end

# MapSliceOp and ExplicitOp: actions are compared with dense references on the CPU.
leafClose(a, b) = isapprox(a, b; rtol = 2e-4, atol = 1e-5)
# Forward/adjoint references with three- and five-argument `mul!` in the order of `adaptActions`.
function denseActions(M, x, y)
  T = promote_type(eltype(M), eltype(x))
  α, β = T(0.5), T(0.75)
  return (M * x, α .* (M * x) .+ β .* y, M' * y, α .* (M' * y) .+ β .* x)
end
testLeafActions(op, conv, M, x, y) = foreach(((m, r),) -> @test(leafClose(m, r)), zip(adaptActions(op, conv, x, y), denseActions(M, x, y)))
# Error raised by `f()`, or `nothing`.
caught(f) = try f(); nothing catch e; e end
vectorType(conv, T) = typeof(conv(T[]))

# Dense matrix of applying `A` to the slices along `dim` of arrays of size `size1`.
mapSliceDense(A, dim, size1) = kron(Matrix(I, prod(size1[dim+1:end]), prod(size1[dim+1:end])), Array(A),
                                    Matrix(I, prod(size1[1:dim-1]), prod(size1[1:dim-1])))
mapSliceState(op) = (workspaceState(parent(op)), Array(getfield(op, :trafo)))
function mapSliceFixture(T, Tt, dim, layout)
  A = testData(Tt, 2, 3) ./ 3
  size1 = dim == 1 ? (layout == 2 ? (3, 2) : (3, 2, 2)) : (layout == 2 ? (2, 3) : (2, 3, 2))
  size2 = Base.setindex(size1, 2, dim)
  return A, size1, size2
end

# Adapts and copies a MapSliceOp with operator transformation `child(src)`; `leaves(child)` are its retained arrays.
function testOperatorChild(src, from, to, M, leaves; child = op -> getfield(op, :trafo))
  x, y = testData(eltype(src), size(src, 2)), testData(eltype(src), size(src, 1)) .+ 1
  testLeafActions(src, from, M, x, y)
  inner(c) = c isa MRIOperators.LinearOperators.AdjointLinearOperator ? c.parent : c
  state(op) = (workspaceState(parent(op)), workspaceState(inner(child(op))), map(Array, leaves(child(op))))
  before = state(src)
  moved, copied = adapt(to, src), copy(src)
  for (op, dest) in ((moved, to), (copied, from))
    testMovedWrapper(op, src, dest)
    @test typeof(child(op)).name.wrapper === typeof(child(src)).name.wrapper && child(op) !== child(src)
    @test all(l -> onTarget(l, dest), leaves(child(op))) && storageType(child(op)) <: dest
    testLeafActions(op, dest, M, x, y)
  end
  @test state(src) == before
  @test all(((a, b),) -> a !== b, zip(leaves(child(copied)), leaves(child(src))))
  foreach(l -> l .*= 2, leaves(child(src))) # the copy keeps the arrays it was made from
  testLeafActions(copied, from, M, x, y)
end

@testset "Adapt MapSliceOp $(nameof(from)) -> $(nameof(to))" for (from, to) in adaptDirections
  @testset "$T, trafo $Tt, dim=$dim, $(layout)D" for (T, Tt) in ((Float32, Float32), (ComplexF32, ComplexF32), (ComplexF32, Float32), (ComplexF64, Float32)),
                                                     dim in (1, 2), layout in (2, 3)
    A, size1, size2 = mapSliceFixture(T, Tt, dim, layout)
    src = T == ComplexF64 ? MapSliceOp(from(A), dim, size1, size2) : MapSliceOp(from(A), dim, size1, size2; T)
    @test eltype(src) == T && eltype(storageType(src)) == T && storageType(src) <: from
    M = mapSliceDense(A, dim, size1)
    x, y = testData(T, prod(size1)), testData(T, prod(size2)) .+ 1
    testLeafActions(src, from, M, x, y)
    before = mapSliceState(src)
    for (moved, dest) in ((adapt(to, src), to), (copy(src), from))
      testMovedWrapper(moved, src, dest)
      @test onTarget(getfield(moved, :trafo), dest) && eltype(getfield(moved, :trafo)) == Tt
      @test (getfield(moved, :dim), getfield(moved, :size1), getfield(moved, :size2)) == (dim, size1, size2)
      testLeafActions(moved, dest, M, x, y)
    end
    @test mapSliceState(src) == before
  end

  @testset "Transformation changed before adapting" begin
    T = ComplexF32
    A, size1, size2 = mapSliceFixture(T, T, 2, 3)
    trafo = from(copy(A))
    src = MapSliceOp(trafo, 2, size1, size2; T)
    trafo .*= 2 # the operation reads the live transformation
    M = mapSliceDense(2 .* A, 2, size1)
    x, y = testData(T, prod(size1)), testData(T, prod(size2)) .+ 1
    testLeafActions(src, from, M, x, y)
    moved, copied = adapt(to, src), copy(src)
    before = mapSliceState(src)
    testLeafActions(moved, to, M, x, y)
    testLeafActions(copied, from, M, x, y)
    @test mapSliceState(src) == before
    trafo .*= 3
    testLeafActions(copied, from, M, x, y)
    from === to || testLeafActions(moved, to, M, x, y)
  end

  @testset "Operator transformation" begin
    T = ComplexF32
    child = SamplingOp(T; pattern = [1, 3], shape = (3,), S = vectorType(from, T))
    src = MapSliceOp(child, 2, (2, 3, 2), (2, 2, 2); T)
    M = mapSliceDense(Matrix{T}(I, 3, 3)[[1, 3], :], 2, (2, 3, 2))
    x, y = testData(T, 12), testData(T, 8) .+ 1
    testLeafActions(src, from, M, x, y)
    moved = adapt(to, src)
    testMovedWrapper(moved, src, to)
    @test getfield(moved, :trafo) isa SamplingOp && getfield(moved, :trafo) !== child && storageType(getfield(moved, :trafo)) <: to
    testLeafActions(moved, to, M, x, y)
    # Operator children with an Adapt rule but no `copy` method are rebuilt with their own arrays and workspaces
    maps, w = testData(T, 3, 2), T[0.5, 1.5 - 1im, 2.5im]
    for (child, A, size2, retained) in ((SensitivityOp(from(maps), 1), reduce(vcat, [Diagonal(maps[:, c]) for c in 1:2]), (6, 2), (:sensMaps, :sensMapsC)),
                                        (WeightingOp(from(w), 1), Diagonal(w), (3, 2), (:weights,)))
      testOperatorChild(MapSliceOp(child, 1, (3, 2), size2; T), from, to, mapSliceDense(A, 1, (3, 2)),
                        op -> map(f -> getfield(op, f), retained))
    end
    # Inside a composite child, an operator without an Adapt rule is copied with `copy`, or rejected without one
    f = (res, x) -> (res .= 2 .* x)
    opaque = LinearOperator{T}(3, 3, false, false, f, nothing, f; S = vectorType(from, T))
    weighting = WeightingOp(from(w), 1)
    src = MapSliceOp(ProdOp(weighting, opaque), 1, (3, 2), (3, 2); T)
    M = mapSliceDense(2 .* Diagonal(w), 1, (3, 2))
    x, y = testData(T, 6), testData(T, 6) .+ 1
    state() = (workspaceState(parent(src)), workspaceState(getfield(src, :trafo)), workspaceState(opaque), workspaceState(weighting), Array(getfield(weighting, :weights)))
    before = state()
    copied = copy(src)
    @test getfield(copied, :trafo).A !== weighting && getfield(copied, :trafo).B !== opaque
    testLeafActions(copied, from, M, x, y)
    @test state() == before
    for dest in unique((from, to)) # `adapt` cannot move it, also on the same backend
      err = caught(() -> adapt(dest, src))
      @test err isa ArgumentError && occursin("cannot adapt", err.msg) && occursin("no Adapt rule", err.msg) && occursin("LinearOperator", err.msg)
    end
    # The given adaptor's own rule for a child applies, also inside a composite
    for child in (weighting, ProdOp(WeightingOp(from(T[1, 1im, 2]), 1), weighting))
      err = caught(() -> adapt(RejectWeighting(to), MapSliceOp(child, 1, (3, 2), (3, 2); T)))
      @test err isa ArgumentError && err.msg == "unsupported WeightingOp target"
    end
    @test state() == before
    # Operators that a child holds but does not apply are not inspected
    aux = WeightingOp(from(T[2, 2, 2]), 1)
    meta = let ref = aux; () -> ref; end
    src = MapSliceOp(MetadataOp(weighting, meta), 1, (3, 2), (3, 2); T)
    auxBefore = workspaceState(aux)
    M = mapSliceDense(Diagonal(w), 1, (3, 2))
    for (moved, dest) in ((adapt(to, src), to), (copy(src), from))
      @test getfield(moved, :trafo) isa MetadataOp && getfield(getfield(moved, :trafo), :op) !== weighting
      testLeafActions(moved, dest, M, x, y)
    end
    @test workspaceState(aux) == auxBefore
    src = MapSliceOp(ProdOp(weighting, adjoint(opaque)), 1, (3, 2), (3, 2); T)
    before = state()
    err = caught(() -> copy(src))
    @test err isa ArgumentError && occursin("cannot copy", err.msg) && occursin("AdjointLinearOperator", err.msg)
    @test state() == before
  end

  @testset "Rejected sources and targets" begin
    A, size1, size2 = mapSliceFixture(Float32, Float32, 1, 2)
    for (src, needle) in ((MapSliceOp(LinearOperator(from(A)), 1, size1, size2; T = Float32, S = vectorType(from, Float32)), "no Adapt rule"),
                          (MapSliceOp(from(A), 1, size1, size2; T = ComplexF32, S = vectorType(from, Float32)), "does not hold its element type"))
      before = workspaceState(parent(src))
      err = caught(() -> adapt(to, src))
      @test err isa ArgumentError && occursin(needle, err.msg)
      needle == "no Adapt rule" || (err = caught(() -> copy(src)); @test err isa ArgumentError && occursin(needle, err.msg))
      @test workspaceState(parent(src)) == before
    end
    src = MapSliceOp(from(A), 1, size1, size2; T = Float32)
    before = mapSliceState(src)
    err = caught(() -> adapt(to{Float64}, src))
    @test err isa ArgumentError && occursin("element type", err.msg)
    @test mapSliceState(src) == before
  end
end

# Dense explicit encoding matrix, evaluated in double precision.
function explicitDense(shape, tr, cmap, echoOffset)
  nodes, times = Float64.(kspaceNodes(tr)), Float64.(readoutTimes(tr))
  idx = CartesianIndices(shape)
  return [exp(-2im * pi * sum(nodes[d, k] * (idx[l][d] - shape[d] / 2 - 1) for d in 1:length(shape)) -
              (times[k] - echoOffset) * ComplexF64(cmap[l])) for k in 1:size(nodes, 2), l in 1:prod(shape)]
end
explicitState(op) = (workspaceState(op), map(f -> Array(getfield(op, f)), (:nodes, :times, :disturbanceTerm)))
explicitTrajectory(R, D) = D == 2 ? CartesianTrajectory(R, 3, 3; TE = R(0.001), AQ = R(0.004)) :
                                   CartesianTrajectory3D(R, 3, 3; numSlices = 3, TE = R(0.001), AQ = R(0.004))
explicitMap(T, shape) = reshape(T[complex(20 + 3k, 300 - 10k) for k in 1:prod(shape)], shape)
function testExplicitLeaf(op, to, src)
  @test op isa ExplicitOp && eltype(op) == eltype(src) && size(op) == size(src) && storageType(op) <: to
  @test onTarget(getfield(op, :nodes), to) && eltype(getfield(op, :nodes)) == real(eltype(src))
  @test onTarget(getfield(op, :times), to) && eltype(getfield(op, :times)) == real(eltype(src))
  @test onTarget(getfield(op, :disturbanceTerm), to) && eltype(getfield(op, :disturbanceTerm)) == eltype(src)
  @test getfield(op, :shape) == getfield(src, :shape) && getfield(op, :echoOffset) == getfield(src, :echoOffset)
end

@testset "Adapt ExplicitOp $(nameof(from)) -> $(nameof(to))" for (from, to) in adaptDirections
  @testset "$T $(D)D echoImage=$echo map=$kind" for T in (ComplexF32, ComplexF64), D in (2, 3), echo in (false, true),
                                                   kind in (:nonzero, :empty, :view, :isbits)
    D == 3 && kind == :isbits && continue
    shape = ntuple(_ -> 3, D)
    tr = explicitTrajectory(real(T), D)
    big = explicitMap(T, shape .+ 1)
    cmap = kind == :nonzero ? from(explicitMap(T, shape)) : kind == :empty ? Array{T}(undef, ntuple(_ -> 0, D)) :
           kind == :view ? view(big, ntuple(_ -> 2:4, D)...) : IsbitsMatrix(explicitMap(T, shape))
    src = ExplicitOp(shape, tr, cmap; echoImage = echo, S = vectorType(from, T))
    M = explicitDense(shape, tr, kind == :empty ? zeros(T, shape) : Array(cmap), echo ? echoTime(tr) : 0.0)
    x, y = testData(T, prod(shape)), testData(T, size(M, 1)) .+ 1
    testExplicitLeaf(src, from, src) # retained inputs are plain arrays of the storage backend
    testLeafActions(src, from, M, x, y)
    before = explicitState(src)
    moved, copied = adapt(to, src), copy(src)
    @test moved !== src && copied !== src
    testExplicitLeaf(moved, to, src)
    testExplicitLeaf(copied, from, src)
    for (op, conv) in ((moved, to), (copied, from))
      testLeafActions(op, conv, M, x, y)
    end
    for adj in (adapt(to, adjoint(src)), adjoint(moved))
      @test adj isa MRIOperators.LinearOperators.AdjointLinearOperator && adj.parent isa ExplicitOp && adj.parent !== src
      @test storageType(adj) <: to && adjoint(adj) === adj.parent
      testLeafActions(adj, to, Matrix(M'), y, x)
    end
    @test explicitState(src) == before
  end

  @testset "Correction map changed before adapting" begin
    T, shape = ComplexF32, (3, 3)
    tr = explicitTrajectory(Float32, 2)
    cmap = from(explicitMap(T, shape))
    src = ExplicitOp(shape, tr, cmap; echoImage = true, S = vectorType(from, T))
    cmap .*= 2 # the operation reads the live correction map
    M = explicitDense(shape, tr, 2 .* explicitMap(T, shape), echoTime(tr))
    x, y = testData(T, prod(shape)), testData(T, size(M, 1)) .+ 1
    testLeafActions(src, from, M, x, y)
    moved, copied = adapt(to, src), copy(src)
    before = explicitState(src)
    testLeafActions(moved, to, M, x, y)
    testLeafActions(copied, from, M, x, y)
    @test explicitState(src) == before
    cmap .*= 3
    testLeafActions(copied, from, M, x, y)
    from === to || testLeafActions(moved, to, M, x, y)
  end

  @testset "Rejected targets and storage types" begin
    tr = explicitTrajectory(Float32, 2)
    src = ExplicitOp((3, 3), tr, from(explicitMap(ComplexF32, (3, 3))); S = vectorType(from, ComplexF32))
    before = explicitState(src)
    err = caught(() -> adapt(to{ComplexF64}, src))
    @test err isa ArgumentError && occursin("element types", err.msg)
    @test explicitState(src) == before
    # The storage element type must match the trajectory's precision.
    @test caught(() -> ExplicitOp((3, 3), tr, explicitMap(ComplexF32, (3, 3)); S = vectorType(from, ComplexF64))) isa TypeError
  end

  @testset "Graphs with MapSliceOp, SensitivityOp and adjoint(ExplicitOp)" begin
    T = ComplexF32
    numVox, numChan, K, numContr = 4, 2, 2, 3
    Φ = MapSliceOp(from(testData(T, numContr, K)), 3, (numVox, numChan, K), (numVox, numChan, numContr); T)
    P = ProdOp(Φ, SensitivityOp(from(testData(T, numVox, numChan)), K))
    E = ExplicitOp((3, 3), explicitTrajectory(Float32, 2), from(explicitMap(T, (3, 3))); S = vectorType(from, T))
    G = ProdOp(adjoint(E), WeightingOp(from(Float32[0.5 + k / 10 for k in 1:size(E, 1)])))
    for src in (P, G)
      refs = adaptActions(src, from)
      before = src === P ? (mapSliceState(Φ), workspaceState(P)) : (explicitState(E), workspaceState(G))
      moved = adapt(to, src)
      @test storageType(moved) <: to && storageType(moved.A) <: to && storageType(moved.B) <: to
      if src === P
        @test moved.A isa MapSliceOp && moved.B isa SensitivityOp && onTarget(getfield(moved.A, :trafo), to)
      else
        @test moved.A isa MRIOperators.LinearOperators.AdjointLinearOperator && moved.A.parent isa ExplicitOp && moved.A.parent !== E
        testExplicitLeaf(moved.A.parent, to, E)
      end
      foreach(((m, r),) -> @test(isapprox(m, r; rtol = 2e-4, atol = 1e-5)), zip(adaptActions(moved, to), refs))
      @test (src === P ? (mapSliceState(Φ), workspaceState(P)) : (explicitState(E), workspaceState(G))) == before
    end
  end

  @testset "Adjoint ExplicitOp as MapSliceOp transformation" begin
    T = ComplexF32
    tr, cmap = explicitTrajectory(Float32, 2), explicitMap(T, (3, 3))
    E = ExplicitOp((3, 3), tr, from(cmap); S = vectorType(from, T))
    M = mapSliceDense(explicitDense((3, 3), tr, cmap, getfield(E, :echoOffset))', 1, (size(E, 1), 2))
    testOperatorChild(MapSliceOp(adjoint(E), 1, (size(E, 1), 2), (size(E, 2), 2); T), from, to, M,
                      A -> map(f -> getfield(A.parent, f), (:nodes, :times, :disturbanceTerm)))
  end

  @testset "Custom adaptor" begin
    T = ComplexF32
    A, size1, size2 = mapSliceFixture(T, T, 2, 2)
    maps = from(testData(T, 3, 2))
    sens = SensitivityOp(maps, 1)
    E = ExplicitOp((3, 3), explicitTrajectory(Float32, 2), from(explicitMap(T, (3, 3))); S = vectorType(from, T))
    for (src, retained) in ((MapSliceOp(from(A), 2, size1, size2; T), op -> (getfield(op, :trafo),)),
                            (MapSliceOp(sens, 1, (3, 2), (6, 2); T), op -> (getfield(getfield(op, :trafo), :sensMaps), getfield(getfield(op, :trafo), :sensMapsC))),
                            (adjoint(E), op -> map(f -> getfield(op.parent, f), (:nodes, :times, :disturbanceTerm))))
      adaptor = DelegatingAdaptor(to)
      moved = adapt(adaptor, src)
      for x in retained(src) # each retained input is converted by the given adaptor
        @test reached(adaptor, x)
      end
      @test all(x -> onTarget(x, to), retained(moved)) && storageType(moved) <: to
      foreach(((a, r),) -> @test(isapprox(a, r; rtol = 2e-4, atol = 1e-5)), zip(adaptActions(moved, to), adaptActions(src, from)))
    end
  end
end

@testset "Explicit Fourier encoding factory $(nameof(from)) -> $(nameof(to))" for (from, to) in adaptDirections
  T = ComplexF32
  for D in (2, 3)
    shape = ntuple(_ -> 3, D)
    tr = explicitTrajectory(Float32, D)
    cmap = explicitMap(T, D == 2 ? (shape..., 2) : shape)
    S = vectorType(from, T)
    subsampleIdx = [1, 2, 4, 6, 7]
    for (op, map, rows) in ((fourierEncodingOp(shape, tr, "explicit"; correctionMap = cmap, slice = D == 2 ? 2 : 1, S), D == 2 ? cmap[:, :, 2] : cmap, :),
                            (fourierEncodingOp(shape, tr, "explicit"; S), zeros(T, shape), :),
                            (fourierEncodingOp(shape, tr, "explicit"; subsampleIdx, S), zeros(T, shape), subsampleIdx))
      M = explicitDense(shape, tr, map, echoTime(tr))[rows, :]
      x, y = testData(T, prod(shape)), testData(T, size(M, 1)) .+ 1
      @test storageType(op) <: from
      testLeafActions(op, from, M, x, y)
      moved = adapt(to, op)
      @test storageType(moved) <: to
      testLeafActions(moved, to, M, x, y)
    end
  end
end

if arrayTypes == [Array]
  @testset "Low-rank encoding operator" begin
    shape, numContr, K = (4, 4), 3, 2
    N = prod(shape)
    tr = CartesianTrajectory(Float64, 4, 4)
    idx = [[1, 2, 5, 7, 8, 11, 12, 16], [2, 3, 4, 6, 9, 10, 13, 15], [1, 3, 6, 8, 10, 12, 14, 16]]
    kdata = Array{Matrix{ComplexF64}, 3}(undef, numContr, 1, 1)
    for c in 1:numContr; kdata[c, 1, 1] = zeros(ComplexF64, length(idx[c]), 1); end
    acqData = AcquisitionData(tr, kdata; idx, encodingSize = shape)
    phi = reshape(testData(ComplexF64, numContr, K), numContr, K, 1)
    E = lrEncodingOp(acqData, shape, Dict{Symbol, Any}(:phi => phi); numContr)
    Φ = E.B.A
    @test Φ isa MapSliceOp && getfield(Φ, :dim) == 2 && eltype(Φ) == ComplexF64 && storageType(Φ) == Vector{ComplexF64}
    @test (getfield(Φ, :size1), getfield(Φ, :size2)) == ((N, K, 1), (N, numContr, 1))
    ft = fourierEncodingOp(shape, tr, "fast")
    F = reduce(hcat, [ft * Vector{ComplexF64}(1:N .== j) for j in 1:N])
    x = testData(ComplexF64, N * K)
    Z = reduce(hcat, [F * x[(k - 1) * N + 1:k * N] for k in 1:K]) # images transformed per basis component
    Y = Z * transpose(phi[:, :, 1])                                # contrasts
    ref = reduce(vcat, [Y[idx[c], c] for c in 1:numContr])
    @test size(E) == (length(ref), N * K)
    @test leafClose(E * x, ref)
    composed = SamplingOp(ComplexF64; pattern = hcat(idx...), shape = (N, numContr, 1)) ∘ (MapSliceOp(phi[:, :, 1], 2, (N, K, 1), (N, numContr, 1)) ∘ DiagOp(ft, ft))
    @test leafClose(composed * x, ref)
    @test leafClose(E' * ref, composed' * ref)
  end
end
