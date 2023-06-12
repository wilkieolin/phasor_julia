
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