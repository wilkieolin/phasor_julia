using Pkg
Pkg.activate(".")

using DifferentialEquations, PhasorNetworks, Lux, NNlib, Zygote, ComponentArrays, Optimisers, OneHotArrays, JLD2
using MLUtils: DataLoader
using Random: Xoshiro
using ChainRulesCore: ignore_derivatives
using Statistics: mean
using QuadGK: quadgk

include("pixel_data.jl")

@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 128    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = false   ## use gpu (if cuda available)
end

#13 input columns, plus y-local are used to define the input data
n_in = 14

function get_truth(pt, threshold::Real = 0.2)
    return 1 .* (pt .> threshold) .+ 2 .* (pt .< -threshold)
end

function momentum_to_label(pt, threshold::Real = 0.2)
    y = onehotbatch(get_truth(pt, threshold), (0, 1, 2))
    return y
end

function accuracy_phasor(x, y, model_call::Function, threshold::Real)
    y_truth = get_truth(y, threshold)
    y_pred = model_call(x)
    y_labels = predict_quadrature(y_pred) .- 1
    right = sum(y_truth .== y_labels)
    return right
end

function accuracy_phasor_compare(y_pred, y, threshold::Real)
    y_labels = predict_quadrature(y_pred) .- 1
    y_truth = get_truth(y, threshold)
    right = sum(y_truth .== y_labels)
    return right
end

function calc_auroc(yh, pt)
    roc_spk = tpr_fpr(yh, pt)
    roc_fn_spk = linear_interpolation(average_duplicate_knots(roc_spk[2], roc_spk[1])...);
    auc, _ = quadgk(roc_fn_spk, 0.0, 1.0)
    return auc
end

###
### Code for conventional multi-layer perceptron
###
mlp_model = Chain(  x -> process_inputs_mlp(x[1], x[2]),
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
    y_pred, st = model((x, xl), ps, st)
    y = momentum_to_label(y, threshold)
    loss = logitcrossentropy(y_pred, y)
    return loss, st
end

function train_mlp(model, ps, st, train_loader; threshold::Real = 0.2, id::Int, kws...)
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
        append!(losses, epoch_losses)
        println(" mean loss ", string(mean(epoch_losses)))
        
    end

    #filename = joinpath("parameters", "mlp_id_") * string(id) * ".jld2"
    #jldsave(filename; params=ps, state=st)
    #jldsave(joinpath("parameters", "mlp_losses_id") * string(id) * ".jld2"; losses = losses)
    return losses, ps, st
end

function test_mlp_static(ps, st, test_loader)
    println("Testing MLP model (static)...")
    yth = cat([mlp_model((x[1], x[2]), ps, st)[1] for x in test_loader]..., dims=2)
    pt = cat([x[3] for x in test_loader]..., dims=1)
    auroc = calc_auroc(yth, pt)
    println("S" * string(auroc))
    return auroc
end


###
### Code for phasor-based mlp
###

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

pmlp_model = Chain(
                        x -> process_inputs_pmlp(x[1], x[2]),
                        BatchNorm(n_in),
                        x -> tanh.(x),
                        x -> x,
                        PhasorDense(n_in => 128),
                        PhasorDense(128 => 3) 
                    )

pmlp_model_spk(spk_args::SpikingArgs, repeats::Int=repeats) = Chain(
                        x -> process_inputs_pmlp(x[1], x[2]),
                        BatchNorm(n_in),
                        x -> tanh.(x),
                        MakeSpiking(spk_args, repeats),
                        PhasorDense(n_in => 128),
                        PhasorDense(128 => 3) 
                    )                    


function loss_pmlp(x, y, model, ps, st, threshold)
    y_pred, st = model(x, ps, st)
    y = momentum_to_label(y, threshold)
    loss = quadrature_loss(y_pred, y) |> mean
    return loss, st
end

function train_pmlp(model, ps, st, train_loader; threshold::Real = 0.2, id::Int, kws...)
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
            (loss_val, st), gs = withgradient(p -> loss_pmlp((x, xl), y, model, p, st, threshold), ps)
            append!(epoch_losses, loss_val)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
        end
        append!(losses, epoch_losses)
        println(" mean loss ", string(mean(epoch_losses)))
    end

    #filename = joinpath("parameters", "pmlp_id_") * string(id) * ".jld2"
    #jldsave(filename; params=ps, state=st)
    #jldsave(joinpath("parameters", "pmlp_losses_id") * string(id) * ".jld2"; losses = losses)
    return losses, ps, st
end

function test_pmlp(ps, st, test_loader; spk_args::SpikingArgs, repeats::Int=10)
    auroc_static = test_pmlp_static(pmlp_model, ps, st, test_loader)
    auroc_dynamic = maximum(test_pmlp_dynamic(pmlp_model_spk(spk_args, repeats), ps, st, test_loader))
    println("S: " * string(auroc_static) * " D: " * string(auroc_dynamic))
    return auroc_static, auroc_dynamic
end

function test_pmlp_static(model, ps, st, test_loader)
    println("Testing PMLP model (static)...")
    yth = cat([model((x[1], x[2]), ps, st)[1] for x in test_loader]..., dims=2)
    pt = cat([x[3] for x in test_loader]..., dims=1)
    auroc = calc_auroc(yth, pt)
    return auroc
end

function test_pmlp_dynamic(model, ps, st, test_loader)
    println("Testing PMLP model (dynamic)...")
    yspk = [model((x[1], x[2]), ps, st)[1] for x in test_loader]
    yth = cat([train_to_phase(st) for st in yspk]..., dims=3)
    pt = cat([x[3] for x in test_loader]..., dims=1)
    #map the auroc calculation for each cycle of the spiking network
    aurocs = map(x -> calc_auroc(x, pt), eachslice(yth, dims=1))
    return aurocs
end

###
### Code for analog, continuous-time driven phasor NN
###

function process_sample(x; spk_args::SpikingArgs, tspan::Tuple, kwargs...)
    charge, ylocal = x
    x1 = charge_to_current(charge, spk_args=spk_args, tspan=tspan)
    x2 = ylocal_to_current(ylocal, spk_args=spk_args, tspan=tspan)
    xf = cat_currents(x1, x2, dim=1)
    
    return xf
end

resonant_layer(spk_args::SpikingArgs) = Chain(PhasorResonant(n_in, spk_args),)

function data_to_phase(q, yl; spk_args::SpikingArgs, tspan::Tuple, clock_amp::Real = 0.01)
    x = process_sample((q, yl), spk_args=spk_args, tspan=tspan, clock_amp=clock_amp)
    ode_front = resonant_layer(spk_args)

    #parameters are constant so rng doesn't matter
    rng = Xoshiro(42)
    ps_ode, st_ode = Lux.setup(rng, ode_front)
    sol = ode_front(x, ps_ode, st_ode)[1]
    mp = mean_phase(sol, 1, spk_args=spk_args, offset=0.0, threshold=false)
    train = solution_to_train(sol, tspan, spk_args=spk_args, offset=0.0)

    return mp, train
end

function resonate_on_current(q, ylocal, spk_args::SpikingArgs, tspan::Tuple, batchsize::Int, clock_amp::Real = 0.01, parallel::Bool = false, n::Int=-1)
    #reduce the data to the requested amount
    if n > 0
        q = q[:,:,:,1:n]
        ylocal = ylocal[1:n]
    end

    conversion_loader = DataLoader((q, ylocal), partial = false, batchsize=batchsize)
    #pmap if requested
    if parallel
        res = pmap(x -> data_to_phase(x[1], x[2], spk_args=spk_args, tspan=tspan, clock_amp=0.01), conversion_loader)
    else
        res = map(x -> data_to_phase(x[1], x[2], spk_args=spk_args, tspan=tspan, clock_amp=0.01), conversion_loader)
    end
    mean_phase = cat([r[1] for r in res]..., dims=2)
    test_trains = [r[2] for r in res]

    return mean_phase, test_trains
end

ode_model() = Chain(
                PhasorDenseF32(n_in => 128),
                PhasorDenseF32(128 => 3)
                )

function loss_ode(x, y, model, ps, st, threshold)
    y_pred, st = model(x, ps, st)
    y = momentum_to_label(y, threshold)
    loss = quadrature_loss(y_pred, y) |> mean
    return loss, st
end

function train_ode(model, ps, st, train_loader; threshold::Real = 0.2, id::Int=1, verbose::Bool = false, kws...)
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
        for (x, y) in train_loader
            (loss_val, st), gs = withgradient(p -> loss_ode(x, y, model, p, st, threshold), ps)
            append!(epoch_losses, loss_val)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
            if verbose
                println(reduce(*, ("Epoch ", string(epoch), ", loss ", string(loss_val))))
            end
        end
        append!(losses, epoch_losses)
        println(" mean loss ", string(mean(epoch_losses)))
        #filename = joinpath("parameters", "id_") * string(id) * "_epoch_" * string(epoch) * ".jld2"
        #jldsave(filename; params=ps, state=st)
    end

    return losses, ps, st
end

function test_ode(model, ps, st, test_loader, test_trains; spk_args::SpikingArgs, tspan::Tuple)
    auroc_static = test_ode_static(model, ps, st, test_loader)
    auroc_dynamic = maximum(test_ode_dynamic(model, ps, st, test_trains, test_loader, spk_args=spk_args, tspan=tspan))
    println("S: " * string(auroc_static) * " D: " * string(auroc_dynamic))
    return auroc_static, auroc_dynamic
end

function test_ode_static(model, ps, st, test_loader)
    println("Testing ODE model (static)...")
    yth = cat([model(x[1], ps, st)[1] for x in test_loader]..., dims=2)
    pt = cat([x[2] for x in test_loader]..., dims=1)
    auroc = calc_auroc(yth, pt)
    return auroc
end

function test_ode_dynamic(model, ps, st, test_trains, test_loader; spk_args::SpikingArgs, tspan::Tuple)
    println("Testing ODE model (dynamic)...")
    test_calls = [SpikingCall(t, spk_args, tspan) for t in test_trains];
    yspk = [model(c, ps, st)[1] for c in test_calls]
    yth = cat([train_to_phase(st) for st in yspk]..., dims=3)
    pt = cat([x[2] for x in test_loader]..., dims=1)
    #map the auroc calculation for each cycle of the spiking network
    aurocs = map(x -> calc_auroc(x, pt), eachslice(yth, dims=1))
    return aurocs
end