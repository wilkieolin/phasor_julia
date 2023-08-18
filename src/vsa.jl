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

function bind(x::SpikeTrain, y::SpikeTrain, tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs; return_solution::Bool = false)
    #set up functions to define the neuron's differential equations
    k = neuron_constant(spk_args)
    k_osc = imag(k)
    k_mem = real(k) + 0.0im
    k_rvs = real(k) + -1.0im * imag(k)

    #get the number of batches & output neurons
    output_shape = x.shape

    #create a reference oscillator to generate complex values for each moment in time
    sol_ref(t) = exp.(1im .* k_osc .* (t .- x.offset))

    #set up the memory compartment
    u0_mem = zeros(ComplexF32, output_shape)
    dzdt_mem(u, p, t) = k_mem .* u .+ sol_ref(t) .* spike_current(x, t, spk_args)
    #solve the memory compartment
    prob_mem = ODEProblem(dzdt_mem, u0_mem, tspan)
    sol_mem = solve(prob_mem, Heun(), adaptive=false, dt=spk_args.dt)

    #set up the countdown compartment
    u0_rvs = zeros(ComplexF32, output_shape)
    dzdt_rvs(u, p, t) = k_rvs .* u .+ sol_mem(t) .* spike_current(y, t, spk_args)
    prob_rvs = ODEProblem(dzdt_rvs, u0_rvs, tspan)
    sol_rvs = solve(prob_rvs,  Heun(), adaptive=false, dt=spk_args.dt)

    if return_solution
        return sol_mem, sol_rvs
    end

    indices, times = find_spikes_rf(sol_rvs, spk_args, reverse=true)
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

function similarity(x::AbstractArray, y::AbstractArray; dim::Int = 1)
    dx = cos.(pi .* (x .- y))
    s = mean(dx, dims = dim)
    return s
end

function similarity_self(x::AbstractMatrix, dims::Int...)
    return similarity_outer(x, x, dims...)
end

function similarity_outer(x::AbstractArray, y::AbstractArray, dims::Int...)
    s = [similarity(xs, ys) for xs in eachslice(x, dims=dims), ys in eachslice(y, dims=dims)]
    return s
end