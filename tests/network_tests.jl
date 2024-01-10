using Lux, MLUtils, MLDatasets, OneHotArrays, Plots, Statistics
using Random: Xoshiro
using Base: @kwdef
using Zygote: withgradient
using Optimisers, ComponentArrays

epsilon = 0.10
repeats = 6
epsilon = 0.02
spk_args = default_spk_args()
tspan = (0.0, repeats*1.0)

@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 256    ## batch size
    epochs::Int = 3        ## number of epochs
    use_cuda::Bool = false   ## use gpu (if cuda available)
    rng::Xoshiro = Xoshiro(42) ## global rng
end

function network_tests()
    #load the dataset and a single batch for testing
    args = Args()
    train_loader, test_loader = getdata(args)
    x, y = first(train_loader)

    model, ps, st = build_mlp(args)
    pretrain_chk = correlation_test(model, ps, st, x)
    train_chk, ps_train, st_train = train_test(model, args, ps, st, train_loader, test_loader)
    acc_test = accuracy_test(model, ps_train, st_train, test_loader)
    posttrain_chk = correlation_test(model, ps_train, st_train, x)
    spk_acc_test = spiking_accuracy_test(model, ps_train, st_train, (x, y))





end

"""
Check if a value is within the bounds determined by epsilon
"""
function in_tolerance(x)
    return abs(x) < epsilon ? true : false
end

function getdata(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    ## Load dataset
    xtrain, ytrain = MLDatasets.FashionMNIST(:train)[:]
    xtest, ytest = MLDatasets.FashionMNIST(:test)[:]

    ## Reshape input data to flatten each image into a linear array
    xtrain = MLUtils.flatten(xtrain)
    xtest = MLUtils.flatten(xtest)

    ## One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    ## Create two DataLoader objects (mini-batch iterators)
    train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, rng=args.rng, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_loader, test_loader
end

function build_mlp(args)
    phasor_model = Chain(PhasorDense(784 => 128), PhasorDense(128 => 10))
    ps, st = Lux.setup(args.rng, phasor_model)
    return phasor_model, ps, st
end

function test_correlation(model, ps, st, x)
    #make the regular (static) call
    y, _ = model(x, ps, st)
    #make the spiking (dynamic) call
    x_train = phase_to_train(x, spk_args, repeats = repeats)
    x_call = SpikingCall(x_train, spk_args, tspan)
    y_spk, _ = model(x_call, ps, st)
    yp = train_to_phase(y_spk)
    #measure the correlation between results
    c = cycle_correlation(y, yp)
    return c
end

function correlation_test(model, ps, st, x)
    @info "Running spiking correlation test..."
    c_naive = test_correlation(model, ps, st, x)
    #test the final full cycle of the network - use 70% correlation as the baseline
    corr_chk = c_naive[end-1] > 0.70
    @assert corr_chk "Correlation of network's spiking values out of expected range"
    return corr_chk
end

function train_test(model, args, ps, st, train_loader, test_loader)
    @info "Running training test..."

    # if CUDA.functional() && args.use_cuda
    #     @info "Training on CUDA GPU"
    #     CUDA.allowscalar(false)
    #     device = gpu
    # else
        @info "Training on CPU"
        device = cpu
    # end

    ## Construct model
    # model = model |> device

    ## Optimizer
    opt_state = Optimisers.setup(Adam(args.η), ps)
    losses = []

    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            x, y = device(x), device(y) ## transfer data to device
            loss, gs = withgradient(p -> mean(quadrature_loss(model(x, p, st)[1], y)), ps) ## compute gradient of the loss
            append!(losses, loss)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
        end

        ## Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, model, ps, st)
        test_loss, test_acc = loss_and_accuracy(test_loader, model, ps, st)
    end

    #check the final loss against the usual ending value
    loss_error = 0.03 - losses[end] 
    loss_check = in_tolerance(loss_error)
    @assert loss_check "Final network loss not within expected bounds"

    return loss_check, ps, st
end

function accuracy_test(model, ps, st, test_loader)
    _, accuracy = loss_and_accuracy(test_loader, model, ps, st)
    #usually reaches ~83% after 6 epochs
    acc_error = 0.827 - accuracy
    acc_check = in_tolerance(acc_error)
    return acc_check
end

function spiking_accuracy_test(model, ps, st, test_batch)
    @info "Running spiking accuracy test..."
    acc = spiking_accuracy(test_batch, model, ps, st, spk_args, tspan, repeats)
    print(acc)
    #make sure accuracy is above the baseline (~80%)
    acc_check = acc[end-1] > 0.80
    @assert acc_check "Accuracy of spiking result too low"
    return acc_check
end

 