
struct PhasorDense{F, M<:AbstractMatrix, B}
    weight::M
    bias::B

    function PhasorDense(W::M, bias = true) where {M<:AbstractMatrix, F}
      if bias
        b = ones(ComplexF32, size(W,1))
      else
        b = zeros(ComplexF32, size(W,1))
      end

      new{F,M}(W, b)
    end
  end
  
  function PhasorDense((in, out)::Pair{<:Integer, <:Integer};
                 init = glorot_uniform, bias = true)

    w = convert(Matrix{ComplexF32}, init(out, in))
    PhasorDense(w, bias)
  end
  
  @functor PhasorDense
  
  function (a::PhasorDense)(x::AbstractVecOrMat)
    _size_check(a, x, 1 => size(a.weight, 2))
    xT = _match_eltype(a, x)  # fixes Float64 input, etc.

    #convert angles to complex
    wz = #PICKUPg
    return σ.(a.weight * xT .+ a.bias)
  end
  
  function (a::PhasorDense)(x::AbstractArray)
    _size_check(a, x, 1 => size(a.weight, 2))
    reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
  end
  
  function Base.show(io::IO, l::PhasorDense)
    print(io, "PhasorDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    l.bias == false && print(io, "; bias=false")
    print(io, ")")
  end