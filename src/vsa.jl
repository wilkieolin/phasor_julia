using Statistics: mean
using DifferentialEquations

include("spiking.jl")

function angle_to_complex(x::AbstractArray)
    k = convert(ComplexF32, pi * (0.0 + 1.0im))
    return exp.(k .* x)
end
function bind(x::AbstractArray; dims)
    bz = sum(x, dims = dims)
    y = remap_phase(bz)
    return y
end

function bind(x::AbstractArray, y::AbstractArray)
    y = remap_phase(x + y)
    return y
end

function bind(x::SpikeTrain, y::SpikeTrain, tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    #set up functions to define the neuron's differential equations
    k = neuron_constant(spk_args)
    #get the number of batches & output neurons
    output_shape = size(x)

    #integrate current through the first compartment that will flow through into the second
    #set up the first compartment
    u0 = zeros(ComplexF32, output_shape)
    dzdt_x(u, p, t) = k .* u + spike_current(x, t, spk_args)
    #solve the ODE for the first compartment over the given time span
    prob_x = ODEProblem(dzdt_x, u0, tspan)
    sol_x = solve(prob_x, Heun(), adaptive=false, dt=spk_args.dt)
    return sol_x
    u_x = Array(sol_x)

    #set up the second compartment
    #current will flow into the second compartment enabled by the second input and flow backards in time
    dzdt_y(u, p, t) = -k .* u + u_x * spike_current(y, t, spk_args)
    #solve the second compartment
    prob_y = ODEProblem(dzdt_y, u0, tspan)
    sol_y = solve(prob_y, Heun(), adaptive=false, dt=spk_args.dt)
    indices, times = find_spikes_rf(sol_y, spk_args)
    #construct the spike train and call for the next layer
    train = SpikeTrain(indices, times, output_shape, x.offset + spk_args.t_period / 4.0)
    next_call = SpikingCall(train, spk_args, tspan)
    return next_call

end

function bundle(x::AbstractMatrix; dims)
    xz = angle_to_complex(x)
    bz = sum(xz, dims = dims)
    y = complex_to_angle(bz)
    return y
end

function bundle_project(x::AbstractMatrix, w::AbstractMatrix, b::AbstractVecOrMat)
    xz = w * angle_to_complex(x) .+ b
    y = complex_to_angle(xz)
    return y
end

#TODO - make dimensions constant with static call
function bundle_project(x::SpikeTrain, w::AbstractMatrix, b::AbstractVecOrMat, tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs; return_solution::Bool=false)
    #set up functions to define the neuron's differential equations
    k = neuron_constant(spk_args)
    #get the number of batches & output neurons
    output_shape = (size(w, 1), x.shape[2])
    u0 = zeros(ComplexF32, output_shape)
    dzdt(u, p, t) = k .* u + w * spike_current(x, t, spk_args) .+ bias_current(b, t, x.offset, spk_args)
    #solve the ODE over the given time span
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, Heun(), adaptive=false, dt=spk_args.dt)
    #option for early exit (mostly for debug)
    if return_solution return sol end
    #convert the full solution (potentials) to spikes
    indices, times = find_spikes_rf(sol, spk_args)
    #construct the spike train and call for the next layer
    train = SpikeTrain(indices, times, output_shape, x.offset + spk_args.t_period / 4.0)
    next_call = SpikingCall(train, spk_args, tspan)
    return next_call
end

function bundle_project(x::LocalCurrent, w::AbstractMatrix, b::AbstractVecOrMat, tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs; return_solution::Bool=false)
    #set up functions to define the neuron's differential equations
    angular_frequency = 2 * pi / spk_args.t_period
    k = (spk_args.leakage + 1im * angular_frequency)
    output_shape = (size(w, 1), x.shape[2])
    u0 = zeros(ComplexF32, output_shape)
    dzdt(u, p, t) = k .* u + w * x.current_fn(t) .+ bias_current(b, t, x.offset, spk_args)
    #solve the ODE over the given time span
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, Heun(), adaptive=false, dt=spk_args.dt)
    #option for early exit (mostly for debug)
    if return_solution return sol end
    #convert the full solution (potentials) to spikes
    indices, times = find_spikes_rf(sol, spk_args)
    #construct the spike train and call for the next layer
    train = SpikeTrain(indices, times, output_shape, x.offset + spk_args.t_period / 4.0)
    next_call = SpikingCall(train, spk_args, tspan)
    return next_call
end

function bind(x::AbstractMatrix; dims)
    bz = sum(x, dims = dims)
    y = remap_phase(bz)
    return y
end

function bind(x::AbstractMatrix, y::AbstractMatrix;)
    y = remap_phase(x .+ y)
    return y
end

function complex_to_angle(x::AbstractVecOrMat)
    return angle.(x) ./ pi
end

function random_symbols(size::Tuple{Vararg{Int}})
    y = 2 .* rand(Float32, size) .- 1.0
    return y
end

function remap_phase(x::AbstractVecOrMat)
    x = x .+ 1
    x = mod.(x, 2.0)
    x = x .- 1
    return x
end

function similarity(x::AbstractArray, y::AbstractArray, dims::Int)
    dx = cos.(pi .* (x .- y))
    s = mean(dx, dims = dims)
    return s
end

function similarity_self(x::AbstractMatrix)
    return similarity_outer(x, x)
end

function similarity_outer(x::AbstractMatrix, y::AbstractMatrix)
    s = [similarity(x[i:i, :], y[j:j,:], 2)[1] for i in 1:size(x)[1], j in 1:size(y)[1]]
    return s
end