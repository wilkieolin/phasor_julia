using Functors
using Flux: glorot_uniform, truncated_normal, nfan

include("vsa.jl")
include("spiking.jl")

struct PhasorDense{M<:AbstractMatrix, B}
    weight::M
    bias::B

    function PhasorDense(W::M, b::B) where {M<:AbstractMatrix, B<:AbstractVector}
      new{M,typeof(b)}(W, b)
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

@functor PhasorDense

function (a::PhasorDense)(x::AbstractVecOrMat)
    y = bundle_project(x, a.weight', a.bias)
    return y
end

function (a::PhasorDense)(x::SpikingCall; return_solution::Bool=false)
    y = bundle_project(x.train, a.weight', a.bias, x.t_span, x.spk_args, return_solution=return_solution)
    return y
end

function (a::PhasorDense)(x::CurrentCall; return_solution::Bool=false)
    y = bundle_project(x.current, a.weight', a.bias, x.t_span, x.spk_args, return_solution=return_solution)
    return y
end

function Base.show(io::IO, l::PhasorDense)
    print(io, "PhasorDense(", size(l.weight, 2), " => ", size(l.weight, 1))
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
    fan_in, fan_out = nfan(shape)
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