export ExplicitOp

mutable struct ExplicitOp{T, vecT <: AbstractVector{T}, F1, F2} <: AbstractMRIOperator{T}
  const nrow :: Int
  const ncol :: Int
  const symmetric :: Bool
  const hermitian :: Bool
  const prod! :: Function
  const tprod! :: F1
  const ctprod! :: F2
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  Mv :: vecT
  Mtu :: vecT
  # Inputs of `prod!` and `ctprod!`, retained to rebuild the operator
  const shape :: Tuple
  const nodes :: AbstractMatrix
  const times :: AbstractVector
  const echoOffset :: Real
  const disturbanceTerm :: AbstractArray
end

LinearOperators.storage_type(::ExplicitOp{T, vecT}) where {T, vecT} = vecT

"""
    ExplicitOp(shape::NTuple{D,Int64}, tr::Trajectory, correctionmap::Array{ComplexF64,D}
            ; echoImage::Bool=false, kargs...) where D

generates a `ExplicitOp` which explicitely evaluates the MRI Fourier signal encoding operator.

# Arguments:
* `shape::NTuple{D,Int64}`             - size of image to encode/reconstruct
* `tr::Trajectory`                     - Trajectory with the kspace nodes to sample
* `correctionmap::Array{ComplexF64,D}` - fieldmap for the correction of off-resonance effects
* `echoImage::Bool=false`              - if true sampling times will only be considered relative to the echo time
                                         this results in complex valued image even for real-valued input.
* `kargs`                              - additional keyword arguments

The adjoint `adjoint(op)` is an `AdjointLinearOperator` with parent `op`.

# Moving to another array backend

    Adapt.adapt(CuArray, op::ExplicitOp)
    Adapt.adapt(CuArray, adjoint(op::ExplicitOp))

returns a new `ExplicitOp` (or the adjoint of one) built from the k-space nodes, readout times and correction
map of `op` adapted to the target, with the same `shape` and echo time offset, so it applies the same operation
as `op` at the time of the call. The trajectory is not used again. Inputs that Adapt does not return as plain
arrays of the target (views, structured or static arrays) are copied into plain arrays on the target. On the
same backend, plain arrays may be shared with `op` and are only read. Adapt changes the backend, not the
element type: targets that change the element type of these inputs or of the storage throw an `ArgumentError`.
`copy(op)` returns a new `ExplicitOp` with copies of these inputs. `op` itself is not modified.
`SensitivityOp`, `SubspaceOp`, `MapSliceOp` and `ExplicitOp` (also as `adjoint`) are the MRI operators
that support `Adapt.adapt` so far.
"""
function ExplicitOp(shape::NTuple{D,Int64}, tr::Trajectory{T}, correctionmap::AbstractArray{Tc,D}
                        ; echoImage::Bool=false, S = Vector{Complex{T}}, kargs...) where {T, Tc <: Union{Complex{T}, T}, D}

  nodes,times = kspaceNodes(tr), readoutTimes(tr)

  if isempty(correctionmap)
    disturbanceTerm = zeros(Complex{T}, shape...)
  else
    disturbanceTerm = correctionmap
  end

  baseArrayType = stripParameters(S)
  nodes = adaptRetained(baseArrayType, nodes, "k-space nodes")
  times = adaptRetained(baseArrayType, times, "readout times")
  disturbanceTerm = adaptRetained(baseArrayType, disturbanceTerm, "correction map")

  # if echo image is desired echo time is needed as an offset
  if echoImage
      echoOffset = echoTime(tr)
  else
      echoOffset = 0.0
  end
  echoOffset = T(echoOffset)

  return ExplicitOp(Complex{T}, S, shape, nodes, times, echoOffset, disturbanceTerm)
end

# Builds the operator from its effective inputs; used by the public constructor, copy and Adapt.
function ExplicitOp(::Type{Tc}, S::Type, shape::Tuple, nodes::AbstractMatrix, times::AbstractVector, echoOffset::Real,
                    disturbanceTerm::AbstractArray) where Tc
  tmp = S(undef, 0)
  return ExplicitOp{Tc, S, Nothing, Function}(size(nodes,2), prod(shape), false, false
            , (res,x)->(produ!(res, x, shape, nodes, times, echoOffset, disturbanceTerm))
            , nothing
            , (res,y)->(ctprodu!(res, y, shape, nodes, times, echoOffset, disturbanceTerm))
            , 0,0,0, tmp, tmp, shape, nodes, times, echoOffset, disturbanceTerm)
end

function checkExplicitSource(op::ExplicitOp{T}) where T
  R = real(T)
  eltype(getfield(op, :nodes)) == R && eltype(getfield(op, :times)) == R && eltype(getfield(op, :disturbanceTerm)) == T ||
    throw(ArgumentError("cannot adapt or copy ExplicitOp: its nodes, times and correction map do not have the element types $R, $R and $T"))
  return op
end

function Base.copy(op::ExplicitOp{T}) where T
  checkExplicitSource(op)
  return ExplicitOp(T, LinearOperators.storage_type(op), getfield(op, :shape), copy(getfield(op, :nodes)),
                    copy(getfield(op, :times)), getfield(op, :echoOffset), copy(getfield(op, :disturbanceTerm)))
end

function Adapt.adapt_structure(to, op::ExplicitOp{T}) where T
  checkExplicitSource(op)
  storage = adaptRetained(to, similar(op.Mv, 0), "ExplicitOp storage")
  nodes = adaptRetained(to, getfield(op, :nodes), "k-space nodes")
  times = adaptRetained(to, getfield(op, :times), "readout times")
  disturbanceTerm = adaptRetained(to, getfield(op, :disturbanceTerm), "correction map")
  R = real(T)
  eltype(storage) == T && eltype(nodes) == R && eltype(times) == R && eltype(disturbanceTerm) == T ||
    throw(ArgumentError("cannot adapt ExplicitOp: the target changes the element types of its storage, nodes, times or correction map"))
  return ExplicitOp(T, typeof(storage), getfield(op, :shape), nodes, times, getfield(op, :echoOffset), disturbanceTerm)
end

function produ!(out::Vector{Tc}, x::Vector{Tc}, shape::NTuple{2,Int64},
                    nodes::Matrix{T}, times::Vector{T}, echoOffset::T,
                    disturbanceTerm::Matrix{Tc}) where {T, Tc <: Union{T, Complex{T}}}

   x= reshape(x,shape)
   fill!(out, zero(Tc))
   for nx=1:shape[1]
       for ny=1:shape[2]
           for k=1:size(nodes,2)
               phi = (nodes[1,k]*(nx-shape[1]/2-1)+
                      nodes[2,k]*(ny-shape[2]/2-1))
               e = exp(-2*1im*pi*phi - (times[k]-echoOffset)*disturbanceTerm[nx,ny])
               out[k] += x[nx,ny] * e
           end
       end
   end
   return out
end

function produ!(out::Vector{Tc}, x::Vector{Tc}, shape::NTuple{3,Int64},
                    nodes::Matrix{T}, times::Vector{T}, echoOffset::T,
                    disturbanceTerm::Array{Tc,3}) where {T, Tc <: Union{T, Complex{T}}}


   x = reshape(x,shape)
   fill!(out, zero(Tc))
   for nx=1:shape[1]
       for ny=1:shape[2]
           for nz=1:shape[3]
               for k=1:size(nodes,2)
                   phi = (nodes[1,k]*(nx-shape[1]/2-1)+nodes[2,k]*(ny-shape[2]/2-1)+nodes[3,k]*(nz-shape[3]/2-1))
                   e = exp(-2*1im*pi*phi - (times[k]-echoOffset)*disturbanceTerm[nx,ny,nz])
                   out[k] += x[nx,ny,nz] * e
               end
           end
       end
   end
   return out
end

function ctprodu!(out::Vector{Tc}, x::Vector{Tc}, shape::NTuple{2,Int64},
                        nodes::Matrix{T}, times::Vector{T}, echoOffset::T,
                        disturbanceTerm::Matrix{Tc}) where {T, Tc <: Union{T, Complex{T}}}


  out = reshape(out, shape)
  fill!(out, zero(Tc))
  for nx=1:shape[1]
      for ny=1:shape[2]
          for k=1:size(nodes,2)
              phi = (nodes[1,k]*(nx-shape[1]/2-1)+nodes[2,k]*(ny-shape[2]/2-1))
              e = exp(-2*1im*pi*phi - (times[k]-echoOffset)*disturbanceTerm[nx,ny])
              out[nx,ny] += x[k] * conj(e)
          end
      end
  end
  return out
end

function ctprodu!(out::Vector{Tc}, x::Vector{Tc}, shape::NTuple{3,Int64},
                        nodes::Matrix{T}, times::Vector{T}, echoOffset::T,
                        disturbanceTerm::Array{Tc,3}) where {T, Tc <: Union{T, Complex{T}}}

  out = reshape(out, shape)
  fill!(out, zero(Tc))
  for nx=1:shape[1]
      for ny=1:shape[2]
          for nz=1:shape[3]
              for k=1:size(nodes,2)
                 phi = (nodes[1,k]*(nx-shape[1]/2-1)+nodes[2,k]*(ny-shape[2]/2-1)+nodes[3,k]*(nz-shape[3]/2-1))
                 e = exp(-2*1im*pi*phi - (times[k]-echoOffset)*disturbanceTerm[nx,ny,nz])
                 out[nx,ny,nz] += x[k] * conj(e)
              end
          end
      end
  end
  return out
end
