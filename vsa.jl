using Statistics: mean
using DifferentialEquations

include("spiking.jl")

function angle_to_complex(x::AbstractVecOrMat)
    k = convert(ComplexF32, pi * (0.0 + 1.0im))
    return exp.(k .* x)
end

function bundle(x::AbstractMatrix; dims)
    xz = angle_to_complex(x)
    bz = sum(xz, dims = dims)
    y = complex_to_angle(bz)
    return y
end

function bundle_project(x::AbstractMatrix, w::AbstractMatrix, b::AbstractVecOrMat)
    xz = angle_to_complex(x) * w .+ b'
    y = complex_to_angle(xz)
    return y
end

function bundle_project(x::SpikeTrain, w::AbstractMatrix, b::AbstractVecOrMat, tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    #set up functions to define the neuron's differential equations
    k = (spk_args.leakage + 1im * spk_args.angular_frequency)
    u0 = zeros(ComplexF32, (st.shape[1], axes(w, 2)))
    dzdt(u, p, t) = k .* u + spike_current(x, t, spk_args) * w .+ bias_current(b, t, x.offset, spk_args)'
    #solve the ODE over the given time span
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, Heun(), adaptive=false, dt=0.01)
    #convert the full solution (potentials) to spikes
    train = find_spikes_rf(sol, spk_args)

    return train
end

function bind(x::AbstractMatrix; dims)
    bz = sum(x, dims = dims)
    y = remap_phase(bz)
    return y
end

function bind(x::AbstractMatrix, y::AbstractMatrix; dims)
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

function similarity(x::AbstractMatrix, y::AbstractMatrix)
    dx = cos.(pi .* (x .- y))
    s = mean(dx, dims = ndims(dx))
    return s
end

function similarity(x::AbstractVecOrMat, y::AbstractVecOrMat)
    dx = cos.(pi .* (x .- y))
    s = mean(dx, dims = ndims(dx))
    return s
end

function similarity_self(x::AbstractMatrix)
    return similarity_outer(x, x)
end

function similarity_outer(x::AbstractMatrix, y::AbstractMatrix)
    s = [similarity(x[i:i, :], y[j:j,:])[1] for i in 1:size(x)[1], j in 1:size(y)[1]]
    return s
end