
function resonate_and_fire(x::SpikeTrain, w::AbstractMatrix, b::AbstractVecOrMat, tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    #set up functions to define the neuron's differential equations
    k = (spk_args.leakage + 1im * spk_args.angular_frequency)
    u0 = zeros(ComplexF32, (x.shape[1], axes(w, 2)))
    dzdt(u, p, t) = k .* u + spike_current(x, t, spk_args) * w .+ bias_current(b, t, x.offset, spk_args)'
    #solve the ODE over the given time span
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, Heun(), adaptive=false, dt=0.01)
    return sol

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