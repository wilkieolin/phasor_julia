using Pkg
Pkg.activate(".")

using DifferentialEquations, PhasorNetworks, Lux, NNlib, Zygote, ComponentArrays, Optimisers, OneHotArrays
using MLUtils: DataLoader
using Random: Xoshiro
using ChainRulesCore: ignore_derivatives
using Statistics: mean

n_epochs = parse(Int, ARGS[1])

include("pixel_data.jl")
data_dir = "pixel_data/"
file_pairs = get_dataset(data_dir)

#load_file(file_pairs[1])

q, ylocal, pt = get_samples(file_pairs[1:2]);
q_test, ylocal_test, pt_test = get_samples(file_pairs[3:3]);

@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 128    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = false   ## use gpu (if cuda available)
end

args = Args(batchsize = 128)

test_loader = DataLoader((q_test, ylocal_test, pt_test), batchsize=args.batchsize)
train_loader = DataLoader((q, ylocal, pt), batchsize=args.batchsize)

function interpolate_2D(t::Real, times::Vector{<:Real}, values::AbstractArray{<:Real,4})
    n_steps, n_y, n_x, n_batch = size(values)
    #extrapolate to zeros
    charge = zeros((n_y, n_x, n_batch),)

    ignore_derivatives() do
        if t > times[1] && t < times[end]
            i_next = findfirst(times .> t)
            i_prev = i_next - 1

            t_next = times[i_next]
            t_prev = times[i_prev]
            proportion = (t - t_prev) / (t_next - t_prev)

            mixture = proportion .* values[i_next,:,:,:] .+ (1 - proportion) .* values[i_prev,:,:,:]
            charge .+= mixture
        end
    end

    return charge
end

using PhasorNetworks: gaussian_kernel

function ylocal_to_current(t::Real, y_local::AbstractArray, spk_args::SpikingArgs; sigma::Real = 9.0, y_range::Real = 32.5)
    output = zero(y_local)

    ignore_derivatives() do
        y_local /= y_range
        phases = (y_local ./ 2.0) .+ 0.5
        times = phases .* spk_args.t_period
        times = mod.(times, spk_args.t_period)

        #add currents into the active synapses
        current_kernel = x -> gaussian_kernel(x, t, spk_args.t_window)
        impulses = current_kernel(times)
        output .+= impulses
    end

    return output
end

function process_inputs(x::AbstractArray, y_local::AbstractArray, spk_args::SpikingArgs)
    v_fn = t -> sum(scale_charge(interpolate_2D(t, x_tms, x)), dims=2)[:,1,:]
    y_fn = t -> ylocal_to_current(t, y_local, spk_args)

    x_fn = t -> cat(v_fn(t), reshape(y_fn(t), (1,:)), dims=1)
    return x_fn
end

sa = SpikingArgs()
rng = Xoshiro(42)

ode_fn = Chain(BatchNorm(n_in),
                x -> tanh.(x),
                Dense(n_in => 128))


ode_model = Chain(PhasorODE(ode_fn, tspan=(0.0, 1.0), dt=0.01),
                x -> complex_to_angle(Array(x)[:,:,end]),
                PhasorDenseF32(128 => 3))

ps, st = Lux.setup(rng, ode_model)
psa = ComponentArray(ps)

function get_truth(pt, threshold::Real = 0.2)
    return 1 .* (pt .> threshold) .+ 2 .* (pt .< -threshold)
end

function momentum_to_label(pt, threshold::Real = 0.2)
    y = onehotbatch(get_truth(pt, threshold), (0, 1, 2))
    return y
end

function loss(x, xl, y, model, ps, st, threshold)
    drive_fn = process_inputs(x, xl, sa)
    y_pred, st = model(drive_fn, ps, st)
    y = momentum_to_label(y, threshold)
    loss = quadrature_loss(y_pred, y) |> mean
    return loss, st
end

function train(model, ps, st, train_loader, threshold::Real = 0.2; kws...)
    args = Args(; kws...) ## Collect options in a struct for convenience

    device = cpu

    @info "Constructing model and starting training"
    ## Construct model
    #model = build_model() |> device

    ## Optimizer
    opt_state = Optimisers.setup(Adam(3e-4), ps)
    losses = []
    i = 0

    ## Training
    for epoch in 1:args.epochs
        println("Epoch ", epoch)
        epoch_losses = []
        for (x, xl, y) in train_loader
            (loss_val, st), gs = withgradient(p -> loss(x, xl, y, model, p, st, threshold), ps)
            append!(epoch_losses, loss_val)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
        end
        append!(losses, mean(epoch_losses))
        jldsave(joinpath("parameters", "epoch_") * string(epoch) * ".jld2"; params=ps, state=st)
    end

    return losses, ps, st
end

@time lhist, pst, stt = train(ode_model, psa, st, train_loader, epochs=n_epochs)