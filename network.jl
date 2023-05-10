using Functors
using Flux: glorot_uniform

function atan2_normal(x::AbstractVecOrMat)
    return angle.(x) ./ pi
end

struct PhasorDense{M<:AbstractMatrix, B}
    weight::M
    bias::B

    function PhasorDense(W::M, bias = true) where {M<:AbstractMatrix}
      if bias
        b = ones(ComplexF32, size(W,1))
      else
        b = zeros(ComplexF32, size(W,1))
      end

      new{M,typeof(b)}(W, b)
    end
  end
  
  function PhasorDense((in, out)::Pair{<:Integer, <:Integer};
                 init = glorot_uniform, bias = true)

    w = convert(Matrix{ComplexF32}, init(out, in))
    PhasorDense(w, bias)
  end

  @functor PhasorDense
  
  function (a::PhasorDense)(x::AbstractVecOrMat)

    #convert angles to complex
    k = convert(ComplexF32, pi * (0.0 + 1.0im))
    xz = a.weight * exp.(k .* x') .+ a.bias
    y = atan2_normal(xz) 

    return y
  end
  
  function Base.show(io::IO, l::PhasorDense)
    print(io, "PhasorDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    l.bias == false && print(io, "; bias=false")
    print(io, ")")
  end