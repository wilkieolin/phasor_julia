using Functors
using Flux: glorot_uniform

include("vsa.jl")

struct PhasorDense{M<:AbstractMatrix, B}
    weight::M
    bias::B

    function PhasorDense(W::M, b::B) where {M<:AbstractMatrix, B<:AbstractVector}
      new{M,typeof(b)}(W, b)
    end
end
  
function PhasorDense(W::AbstractMatrix)
    b = ones(ComplexF32, size(W,1))
    return PhasorDense(W, b)
end

function PhasorDense((in, out)::Pair{<:Integer, <:Integer};
                init = glorot_uniform)

    w = convert(Matrix{ComplexF32}, init(out, in))
    PhasorDense(w)
end

@functor PhasorDense

function (a::PhasorDense)(x::AbstractVecOrMat)
    y = bundle_project(x, a.weight', a.bias')
    return y
end

function Base.show(io::IO, l::PhasorDense)
    print(io, "PhasorDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    print(io, ")")
end