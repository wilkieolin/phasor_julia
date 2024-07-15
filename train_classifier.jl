using Pkg
Pkg.activate(".")

using DifferentialEquations, PhasorNetworks, Lux, NNlib, Zygote, ComponentArrays, Optimisers, OneHotArrays, JLD2
using MLUtils: DataLoader
using Random: Xoshiro
using ChainRulesCore: ignore_derivatives
using Statistics: mean

include("pixel_data.jl")

@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 128    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = false   ## use gpu (if cuda available)
end

#13 input columns, plus y-local are used to define the input data
n_in = 14
#set the oscillator/spiking config
sa = SpikingArgs()
repeats = 20

function get_truth(pt, threshold::Real = 0.2)
    return 1 .* (pt .> threshold) .+ 2 .* (pt .< -threshold)
end

function momentum_to_label(pt, threshold::Real = 0.2)
    y = onehotbatch(get_truth(pt, threshold), (0, 1, 2))
    return y
end

###
### Code for conventional multi-layer perceptron
###
mlp_model = Chain(
                    BatchNorm(n_in),
                    x -> tanh.(x),
                    Dense(n_in => 128, relu),
                    Dense(128 => 3) 
                    )

              

function process_inputs_mlp(x, y_local)
    x = scale_charge(x)
    x = sum(x, dims=(1,3))
    n_px = size(x,2)
    n_batch = size(x, 4)
    x = reshape(x, (n_px, n_batch))
    y_local = reshape(y_local, (1, n_batch))

    x = cat(x, y_local, dims = 1)
    return x
end

logitcrossentropy(y_pred, y) = mean(-1 * sum(y .* logsoftmax(y_pred); dims=1))

function loss_mlp(x, xl, y, model, ps, st, threshold)
    y_pred, st = model(process_inputs_mlp(x, xl), ps, st)
    y = momentum_to_label(y, threshold)
    loss = logitcrossentropy(y_pred, y)
    return loss, st
end

function train_mlp(model, ps, st, train_loader, threshold::Real = 0.2; id::Int, kws...)
    args = Args(; kws...) ## Collect options in a struct for convenience

    device = cpu

    @info "Constructing model and starting training"
    ## Construct model
    #model = build_model() |> device

    ## Optimizer
    opt_state = Optimisers.setup(Adam(3e-4), ps)
    losses = []

    ## Training
    for epoch in 1:args.epochs
        epoch_losses = []
        print("Epoch ", epoch)
        for (x, xl, y) in train_loader
            (loss_val, st), gs = withgradient(p -> loss_mlp(x, xl, y, model, p, st, threshold), ps)
            append!(epoch_losses, loss_val)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
        end
        append!(losses, mean(epoch_losses))
        println(" mean loss ", string(mean(epoch_losses)))
    end

    filename = joinpath("parameters", "mlp_id_") * string(id) * "_epoch_" * string(args.epochs) * ".jld2"
    jldsave(filename; params=ps, state=st)

    jldsave(joinpath("parameters", "mlp_losses_id") * string(id) * ".jld2"; losses = losses)

    return losses, ps, st
end


###
### Code for phasor-based mlp
###
pmlp_model = Chain(
                        BatchNorm(n_in),
                        x -> tanh.(x),
                        PhasorDense(n_in => 128),
                        PhasorDense(128 => 3) 
                    )

pmlp_model_spk = Chain(
                        BatchNorm(n_in),
                        x -> tanh.(x),
                        MakeSpiking(sa, repeats),
                        PhasorDense(n_in => 128),
                        PhasorDense(128 => 3) 
                    )                    

function convert_pmlp_params(pmlp_ps)
    # Add a dummy layer of params for the make_spiking layer
    spk_ps = (layer_1 = pmlp_ps.layer_1, 
            layer_2 = pmlp_ps.layer_2,
            layer_3 = NamedTuple(),
            layer_4 = pmlp_ps.layer_3,
            layer_5 = pmlp_ps.layer_4)
    return spk_ps
end      

function process_inputs_pmlp(x, y_local)
    x = scale_charge(x)
    x = sum(x, dims=(1,3))
    n_batch = size(x, 4)
    n_px = size(x,2)
    x = reshape(x, (n_px, n_batch))
    y_local = reshape(y_local, (1, n_batch))

    x = cat(x, y_local, dims = 1)
    return x
end

function loss_pmlp(x, xl, y, model, ps, st, threshold)
    y_pred, st = model(process_inputs_pmlp(x, xl), ps, st)
    y = momentum_to_label(y, threshold)
    loss = quadrature_loss(y_pred, y) |> mean
    return loss, st
end

function train_pmlp(model, ps, st, train_loader, threshold::Real = 0.2; id::Int, kws...)
    args = Args(; kws...) ## Collect options in a struct for convenience

    device = cpu

    @info "Constructing model and starting training"
    ## Construct model
    #model = build_model() |> device

    ## Optimizer
    opt_state = Optimisers.setup(Adam(3e-4), ps)
    losses = []

    ## Training
    for epoch in 1:args.epochs
        print("Epoch ", epoch)
        epoch_losses = []
        for (x, xl, y) in train_loader
            (loss_val, st), gs = withgradient(p -> loss_pmlp(x, xl, y, model, p, st, threshold), ps)
            append!(epoch_losses, loss_val)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
        end
        append!(losses, mean(epoch_losses))
        println(" mean loss ", string(mean(epoch_losses)))
        filename = joinpath("parameters", "pmlp_id_") * string(id) * "_epoch_" * string(epoch) * ".jld2"
        jldsave(filename; params=ps, state=st)
    end

    jldsave(joinpath("parameters", "pmlp_losses_id") * string(id) * ".jld2"; losses = losses)
    return losses, ps, st
end

###
### Code for analog, continuous-time driven phasor NN
###
ode_fn = Chain(BatchNorm(n_in),
                        x -> tanh.(x),
                        Dense(n_in => 128))


ode_model = Chain(PhasorODE(ode_fn, tspan=(0.0, 1.0), dt=0.01),
                x -> complex_to_angle(Array(x)[:,:,end]),
                PhasorDenseF32(128 => 3))

ode_model_spk = Chain(PhasorODE(ode_fn, tspan=(0.0, 1.0), dt=0.01),
                x -> complex_to_angle(Array(x)[:,:,end]),
                MakeSpiking(sa, repeats),
                PhasorDenseF32(128 => 3))

function convert_ode_params(ode_ps)
    # Add a dummy layer of params for the make_spiking layer
    spk_ps = (layer_1 = ode_ps.layer_1, 
            layer_2 = ode_ps.layer_2,
            layer_3 = NamedTuple(),
            layer_4 = ode_ps.layer_3)
    return spk_ps
end      

function process_inputs_ode(x::AbstractArray, x_tms::AbstractVector, y_local::AbstractArray, spk_args::SpikingArgs)
    v_fn = t -> sum(scale_charge(interpolate_2D(t, x_tms, x)), dims=2)[:,1,:]
    y_fn = t -> ylocal_to_current(t, y_local, spk_args)

    x_fn = t -> cat(v_fn(t), reshape(y_fn(t), (1,:)), dims=1)
    return x_fn
end

function loss_ode(x, x_tms, xl, y, model, ps, st, threshold, spk_args::SpikingArgs=SpikingArgs())
    drive_fn = process_inputs_ode(x, x_tms, xl, spk_args)
    y_pred, st = model(drive_fn, ps, st)
    y = momentum_to_label(y, threshold)
    loss = quadrature_loss(y_pred, y) |> mean
    return loss, st
end

function train_ode(model, ps, st, train_loader, x_tms, threshold::Real = 0.2; id::Int=1, kws...)
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
        print("Epoch ", epoch)
        epoch_losses = []
        for (x, xl, y) in train_loader
            (loss_val, st), gs = withgradient(p -> loss_ode(x, x_tms, xl, y, model, p, st, threshold), ps)
            append!(epoch_losses, loss_val)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
        end
        append!(losses, mean(epoch_losses))
        println(" mean loss ", string(mean(epoch_losses)))
        filename = joinpath("parameters", "ode_id_") * string(id) * "_epoch_" * string(epoch) * ".jld2"
        jldsave(filename; params=ps, state=st)
    end

    jldsave(joinpath("parameters", "ode_losses_id") * string(id) * ".jld2"; losses = losses)

    return losses, ps, st
end