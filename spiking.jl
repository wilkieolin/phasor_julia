using DifferentialEquations: ODESolution

struct SpikeTrain
    indices::Array{<:Int,1}
    times::Array{<:Real,1}
    shape::Tuple
    offset::Real
end

struct SpikingArgs
    angular_frequency::Real
    leakage::Real
    t_period::Real
    t_window::Real
    threshold::Real
end

struct SpikingCall
    input::SpikeTrain
    spk_args::SpikingArgs
    t_span::Tuple{<:Real, <:Real}
end

function default_spk_args()
    args = SpikingArgs(2.0 * pi,
                    -0.2,
                    1.0,
                    0.03,
                    0.05)
    return args
end

function spike_current(train::SpikeTrain, t::Real, spk_args::SpikingArgs)
    #get constants
    t_window = spk_args.t_window

    #determine which synapses will have incoming currents
    times = train.times
    relative_time = abs.(times .- t)
    active = relative_time .< t_window
    active_inds = train.indices[active]

    #add currents into the active synapses
    current = zeros(ComplexF32, train.shape)
    current[active_inds] .+= (1.0 + 0im)

    return current
end

function bias_current(bias::AbstractVecOrMat, t::Real, t_offset::Real, spk_args::SpikingArgs)
    #get constants
    t_period = spk_args.t_period
    t_window = spk_args.t_window
    full_shape = bias |> size

    #determine the time within the cycle
    t_relative = (t - t_offset) % t_period
    #determine which inputs are ative
    bias_relative = abs.(t_relative .- bias)
    active_inds = bias_relative .< t_window

    #produce the resulting currents
    current = zeros(ComplexF32, full_shape)
    current[active_inds] .= 1.0 + 0.0im

    return current
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
    channels = getindex.(spikes, 2)
    #retrieve the times they spiked at
    times = t[getindex.(spikes, 1)]
    
    return channels, times

end

"""
Converts a matrix of phases into a spike train via phase encoding

phase_to_train(phases::AbstractMatrix, spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0)
"""
function phase_to_train(phases::AbstractMatrix, spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0)
    t_phase0 = spk_args.t_period / 2.0
    shape = phases |> size

    phases = phases |> vec

    indices = collect(1:length(phases))
    times = phases .* t_phase0 .+ t_phase0

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
    times = (times .- offset) .% period
    times = (times .- 0.5) .* 2.0
    return times
end

function train_to_phase(train::SpikeTrain, spk_args::SpikingArgs)
    if length(train.times) == 0
        return missing
    end

    phases_vec = time_to_phase(train.times, spk_args.t_period, train.offset)
    n_cycles = maximum(train.times) ÷ spk_args.t_period + 1
    cycle = floor.(Int, train.times .÷ spk_args.t_period .+ 1)
    phases = [NaN .* zeros(train.shape...) for i in 1:n_cycles]

    for i in eachindex(phases_vec)
        phases[cycle[i]][train.indices[i]] = phases_vec[i]
    end

    #stack the arrays
    phases = mapreduce(x->reshape(x, 1, st.shape...), vcat, phases)
    phases = permutedims(phases, (2, 3, 1))
    return phases

end