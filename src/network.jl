using ComponentArrays, SciMLSensitivity, OrdinaryDiffEq
using Random: AbstractRNG
using Lux: glorot_uniform, truncated_normal

include("vsa.jl")
include("spiking.jl")

LuxParams = Union{NamedTuple, ComponentArray}

###
### Phasor Dense definitions
###

struct PhasorDense{M<:AbstractMatrix, B} <: Lux.AbstractExplicitLayer
    shape::Tuple{<:Int, <:Int}
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function

    function PhasorDense(W::M, b::B) where {M<:AbstractMatrix, B<:AbstractVector}
      new{M,typeof(b)}(size(W), size(W,2), size(W,1), () -> copy(W), () -> copy(b))
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

function (a::PhasorDense)(x::AbstractVecOrMat, params::LuxParams, state::NamedTuple)
    y = bundle_project(x, params.weight, params.bias)
    return y, state
end

function (a::PhasorDense)(x::SpikingCall, params::LuxParams, state::NamedTuple; return_solution::Bool=false)
    y = bundle_project(x.train, params.weight, params.bias, x.t_span, x.spk_args, return_solution=return_solution)
    return y, state
end

function (a::PhasorDense)(x::CurrentCall, params::LuxParams, state::NamedTuple; return_solution::Bool=false)
    y = bundle_project(x.current, params.weight, params.bias, x.t_span, x.spk_args, return_solution=return_solution)
    return y, state
end

function Base.show(io::IO, l::PhasorDense)
    print(io, "PhasorDense(", l.shape)
    print(io, ")")
end

###
### PhasorODE Layer
###

struct PhasorODE{M <: Lux.AbstractExplicitLayer, So, Se, T} <: Lux.AbstractExplicitContainerLayer{(:model,)}
    model::M
    solver::So
    sensealg::Se
    tspan::T
    spk_args::SpikingArgs
    dt::Real
end

#constructor
function PhasorODE(model::Lux.AbstractExplicitLayer; 
    solver=Tsit5(),
    sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
    tspan=(0.0, 30.0),
    spk_args=SpikingArgs(),
    dt=0.1)

    return PhasorODE(model, solver, sensealg, tspan, spk_args, dt)
end

#forward pass
function (n::PhasorODE)(currents, ps, st)
    #define the function which updates neurons' potentials
    function dudt(u, p, t)
        du_real, _ = n.model(currents(t), p, st)
        constant = n.spk_args.leakage + 2*pi*im / n.spk_args.t_period
        du = constant .* u .+ du_real
        return du
    end

    #sample the input to determine size of the state
    i0 = currents(0.0)
    u0 = zeros(ComplexF32, (n.model[end].out_dims, size(i0)[end]))
    prob = ODEProblem(dudt, u0, n.tspan, ps)
    soln = solve(prob, n.solver, 
        adaptive = false, 
        dt = n.dt, 
        saveat = n.tspan[2], 
        sensealg = n.sensealg,
        save_start = false)
    return soln, st
end

"""
Phasor QKV Attention
"""
struct PhasorAttention{M<:AbstractArray, B} <: Lux.AbstractExplicitLayer
    shape::Tuple{<:Int, <:Int}
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end

function attend(q::Array{<:Real, 3}, k::Array{<:Real, 3}, v::Array{<:Real, 3})
    #compute qk scores
    #(c b a) x (c b a) -> (b b a)
    scores = similarity_outer(q, k, 2)
    #do complex-domain matrix multiply of values by scores
    values = angle_to_complex(v)
    #(b b a) * (c b a)
    output = scores * values
end



function (a::PhasorAttention)(query::AbstractArray, keyvalue::AbstractArray)
    q = a.query_network(query)
    k = a.key_network(keyvalue)
    v = a.value_network(keyvalue)

    result = attend(q, k, v)

    output = a.output_network(result)

    return output
end
    

function (a::PhasorDense)(x::SpikingCall, params::LuxParams, state::NamedTuple; return_solution::Bool=false)
    y = bundle_project(x.train, params.weight, params.bias, x.t_span, x.spk_args, return_solution=return_solution)
    return y, state
end

    


"""
Phasor Self-Attention Module
"""
struct PhasorSA{M<:AbstractArray, B} <: Lux.AbstractExplicitLayer
    shape::Tuple{<:Int, <:Int}
    n_heads::Int
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function

    function PhasorDense(W::M, b::B) where {M<:AbstractArray, B<:AbstractVector}
      new{M,typeof(b)}(size(W), size(W,2), size(W,1), () -> copy(W), () -> copy(b))
    end
end

"""
Other utilities
"""


function quadrature_loss(phases::AbstractArray, truth::AbstractArray)
    targets = 0.5 .* truth
    sim = similarity(phases, targets, dim = 1)
    return 1.0 .- sim
end

function similarity_loss(phases::AbstractArray, truth::AbstractArray, dim::Int)
    sim = similarity(phases, truth, dim = dim)
    return 1.0 .- sim
end

function accuracy(data_loader, model, spk_args::SpikingArgs, t_span::Tuple{<:Real, <:Real})
    acc = []
    n_phases = []
    num = 0

    for (x, y) in data_loader
        train = phase_to_train(x, spk_args, repeats=3)
        call = SpikingCall(train, spk_args, t_span)
        spk_output = model(call)
        ŷ = train_to_phase(spk_output)
        
        append!(acc, sum.(accuracy_quadrature(ŷ, y))') ## Decode the output of the model
        num +=  size(x)[end]
    end

    return acc, num
end

function accuracy_quadrature(phases::AbstractMatrix, truth::AbstractMatrix)
    predictions = getindex.(argmin(abs.(phases .- 0.5), dims=1), 1)'
    labels = getindex.(findall(truth), 1)
    return predictions .== labels
end

function accuracy_quadrature(phases::Array{<:Real,3}, truth::AbstractMatrix)
    return [accuracy_quadrature(phases[i,:,:], truth) for i in axes(phases,1)]
end

function variance_scaling(shape::Integer...; mode::String = "fan_in", scale::Real = 1.0)
    fan_in = shape[1]
    fan_out = shape[end]

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