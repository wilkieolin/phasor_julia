using DifferentialEquations: ODESolution
using Statistics: cor

struct SpikeTrain
    indices::Array{<:Union{Int, CartesianIndex},1}
    times::Array{<:Real,1}
    shape::Tuple
    offset::Real
end

struct LocalCurrent
    current_fn::Function
    shape::Tuple
    offset::Real
end

function Base.show(io::IO, train::SpikeTrain)
    print(io, "Spike Train: ", train.shape, " with ", length(train.times), " spikes.")
end

struct SpikingArgs
    leakage::Real
    t_period::Real
    t_window::Real
    threshold::Real
    dt::Real
end

function Base.show(io::IO, spk_args::SpikingArgs)
    print(io, "Neuron parameters: Period ", spk_args.t_period, " (s)\n")
    print(io, "Current kernel duration: ", spk_args.t_window, " (s)\n")
    print(io, "Threshold: ", spk_args.threshold, " (V)\n")
end

struct SpikingCall
    train::SpikeTrain
    spk_args::SpikingArgs
    t_span::Tuple{<:Real, <:Real}
end

struct CurrentCall
    current::LocalCurrent
    spk_args::SpikingArgs
    t_span::Tuple{<:Real, <:Real}
end

function cor_realvals(x, y)
    is_real = x -> .!isnan.(x)
    x_real = is_real(x)
    y_real = is_real(y)
    reals = x_real .* y_real
    
    cor_val = cor(x[reals], y[reals])
    return cor_val
end

function count_nans(phases::Array{<:Real,3})
    return mapslices(x->sum(isnan.(x)), phases, dims=(2,3)) |> vec
end

function cycle_correlation(static_phases::Matrix{<:Real}, dynamic_phases::Array{<:Real,3})
    n_cycles = axes(dynamic_phases, 1)
    cor_vals = [cor_realvals(static_phases |> vec, dynamic_phases[i,:,:] |> vec) for i in n_cycles]
    return cor_vals
end

function default_spk_args()
    args = SpikingArgs(-0.2,
                    1.0,
                    0.03,
                    0.02,
                    0.01)
    return args
end

function spike_current(train::SpikeTrain, t::Real, spk_args::SpikingArgs)
    #get constants
    t_window = spk_args.t_window + 1e-4
    dt = spk_args.dt

    #determine which synapses will have incoming currents
    #snap spike times to the grid points
    times = train.times .- mod.(train.times, dt)
    active = (times .> (t - t_window)) .* (times .< (t + t_window))
    # relative_time = abs.(times .- t)
    # active = relative_time .<= t_window
    active_inds = train.indices[active]

    #add currents into the active synapses
    current = zeros(Float32, train.shape)
    current[active_inds] .+= (1.0)

    return current
end

function bias_current(bias::AbstractVecOrMat, t::Real, t_offset::Real, spk_args::SpikingArgs)
    #get constants
    t_bias = spk_args.t_period / 2.0
    t_window = spk_args.t_window
    #determine the time within the cycle
    t_relative = mod((t - t_offset), spk_args.t_period)
    #determine if the bias is active
    active = (t_relative > (t_bias - t_window)) && (t_relative < (t_bias + t_window))

    if active
        return bias
    else
        return zero(bias)
    end
end

function find_spikes_rf(sol::ODESolution, spk_args::SpikingArgs)
    t = sol.t
    u = sol.u

    @assert typeof(sol.u) <: Vector{<:Matrix{<:Complex}} "This method is for R&F neurons with complex potential"
    #if potential is from an R&F neuron, it is complex and voltage is the imaginary part
    voltage = imag.(u)
    current = real.(u)

    #convert vector of matrices to 2D matrix
    function rearrange_sol(x)
        #rearrange to (time, batch, neuron)
        x = reshape(reduce(vcat, x), (axes(sol.u[1], 1), axes(sol.t, 1), axes(sol.u[1], 2)))
        x = permutedims(x, (2, 1, 3))
        return x
    end
    
    voltage = rearrange_sol(voltage)
    current = rearrange_sol(current)

    #find the local voltage maxima through the first derivative (current)
    maxima = findall(diff(sign.(current), dims=1) .< 0)
    zero_i = current[maxima]
    peak_voltages = voltage[maxima]
    #return maxima, peak_voltages
    #check voltages at these peaks are above the threshold
    above_threshold = peak_voltages .> spk_args.threshold
    spikes = maxima[above_threshold]

    #retrieve the indices of the spiking neurons
    batch = getindex.(spikes, 2)
    neuron = getindex.(spikes, 3)
    channels = CartesianIndex.(batch, neuron)
    #retrieve the times they spiked at
    times = t[getindex.(spikes, 1)]
    
    return channels, times

end

"""
Convert a static phase to the complex potential of an R&F neuron
"""
function phase_to_potential(phase::Real, t::Array{<:Real}, offset::Real, spk_args::SpikingArgs)
    period = spk_args.t_period
    potential = exp.(1im .* (2*pi.*(t .- offset)/(period) .+ phase))
    return potential
end

"""
Converts a matrix of phases into a spike train via phase encoding

phase_to_train(phases::AbstractMatrix, spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0)
"""
function phase_to_train(phases::AbstractMatrix, spk_args::SpikingArgs; repeats::Int = 1, offset::Real = 0.0)
    t_phase0 = spk_args.t_period / 2.0
    shape = phases |> size
    indices = collect(CartesianIndices(shape)) |> vec
    times = phases .* t_phase0 .+ t_phase0 |> vec

    if repeats > 1
        n_t = times |> length
        offsets = repeat(0:repeats-1, inner=n_t)
        times = repeat(times, repeats) .+ offsets
        indices = repeat(indices, repeats)
    end

    train = SpikeTrain(indices, times, shape, offset)
    return train
end

function time_to_phase(times::AbstractVecOrMat, period::Real, offset::Real)
    times = mod.((times .- offset), period) ./ period
    times = (times .- 0.5) .* 2.0
    return times
end

function train_to_phase(train::SpikeTrain, spk_args::SpikingArgs)
    if length(train.times) == 0
        return missing
    end

    #decode each spike's phase within a cycle
    relative_phase = time_to_phase(train.times, spk_args.t_period, train.offset)
    relative_time = train.times .- train.offset
    #what is the number of cycles in this train?
    n_cycles = maximum(relative_time) ÷ spk_args.t_period + 1
    #what is the cycle in which each spike occurs?
    cycle = floor.(Int, relative_time .÷ spk_args.t_period .+ 1)
    phases = [NaN .* zeros(train.shape...) for i in 1:n_cycles]

    for i in eachindex(relative_phase)
        phases[cycle[i]][train.indices[i]] = relative_phase[i]
    end

    #stack the arrays to cycle, batch, neuron
    phases = mapreduce(x->reshape(x, 1, train.shape...), vcat, phases)
    return phases

end

function train_to_phase(call::SpikingCall)
    return train_to_phase(call.train, call.spk_args)
end

function zero_nans(phases::AbstractArray)
    nans = isnan.(phases)
    phases[nans] .= 0.0
    return phases
end