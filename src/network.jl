using Lux: glorot_uniform, truncated_normal
using Random: AbstractRNG

include("vsa.jl")
include("spiking.jl")

struct PhasorDense{M<:AbstractMatrix, B} <: Lux.AbstractExplicitLayer
    shape::Tuple{<:Int, <:Int}
    init_weight::Function
    init_bias::Function

    function PhasorDense(W::M, b::B) where {M<:AbstractMatrix, B<:AbstractVector}
      new{M,typeof(b)}(size(W), () -> copy(W), () -> copy(b))
    end
end
  
function PhasorDense(W::AbstractMatrix)
    b = ones(axes(W,1))
    return PhasorDense(W, b)
end

function PhasorDense((in, out)::Pair{<:Integer, <:Integer};
                init = variance_scaling)

    w = init(out, in)
    PhasorDense(w)
end

function Lux.initialparameters(rng::AbstractRNG, layer::PhasorDense)
    params = (weight = layer.init_weight(), bias = layer.init_bias())
end

function (a::PhasorDense)(x::AbstractVecOrMat, params::NamedTuple)
    y = bundle_project(x, params.weight', params.bias)
    return y
end

function (a::PhasorDense)(x::SpikingCall, params::NamedTuple; return_solution::Bool=false)
    y = bundle_project(x.train, params.weight', params.bias, x.t_span, x.spk_args, return_solution=return_solution)
    return y
end

function (a::PhasorDense)(x::CurrentCall, params::NamedTuple; return_solution::Bool=false)
    y = bundle_project(x.current, params.weight', params.bias, x.t_span, x.spk_args, return_solution=return_solution)
    return y
end

function Base.show(io::IO, l::PhasorDense)
    print(io, "PhasorDense(", l.shape)
    print(io, ")")
end

function quadrature_loss(phases::AbstractMatrix, truth::AbstractMatrix)
    targets = 0.5 .* truth
    sim = similarity(phases, targets)
    return 1.0 .- sim
end

function accuracy(data_loader, model, spk_args::SpikingArgs, t_span::Tuple{<:Real, <:Real})
    acc = []
    n_phases = []
    num = 0

    for (x, y) in data_loader
        train = phase_to_train(x', spk_args, repeats=3)
        call = SpikingCall(train, spk_args, t_span)
        spk_output = model(call)
        ŷ = train_to_phase(spk_output)
        
        append!(acc, sum.(accuracy_quadrature(ŷ, y))') ## Decode the output of the model
        num +=  size(x)[end]
    end

    return acc, num
end

function accuracy_quadrature(phases::AbstractMatrix, truth::AbstractMatrix)
    predictions = getindex.(argmin(abs.(phases .- 0.5), dims=2), 2)
    labels = getindex.(findall(truth), 1)
    return predictions .== labels
end

function accuracy_quadrature(phases::Array{<:Real,3}, truth::AbstractMatrix)
    return [accuracy_quadrature(phases[i,:,:], truth) for i in axes(phases,1)]
end

function variance_scaling(shape::Integer...; mode::String = "fan_in", scale::Real = 1.0)
    fan_in, fan_out = shape
    if mode == "fan_in"
        scale /= max(1.0, fan_in)
    elseif mode == "fan_out"
        scale /= max(1.0, fan_out)
    else
        scale /= max(1.0, (fan_in + fan_out) / 2.0)
    end

    stddev = sqrt(scale) / 0.87962566103423978
    return truncated_normal(shape..., mean = 0.0, std = stddev)
end