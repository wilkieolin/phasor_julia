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

function find_spikes_rf(sol::ODESolution, spk_args::SpikingArgs)
    t = sol.t
    u = sol.u

    @assert typeof(sol.u) <: Vector{<:Matrix{<:Complex}} "This method is for R&F neurons with complex potential"
    #if potential is from an R&F neuron, it is complex and voltage is the imaginary part
    voltage = imag.(u)
    current = real.(u)

    #convert vector of matrices to 2D matrix
    make2d = x -> reduce(vcat, x)
    voltage = make2d(voltage)
    current = make2d(current)

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
    current[active_inds] = 1.0

    return current
end

function find_spikes(sol, offset::Real, spk_args::SpikingArgs)
    #TODO
    return
end