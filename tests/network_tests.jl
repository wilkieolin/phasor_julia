using Pkg
Pkg.activate(".")

using Lux, MLUtils, MLDatasets, OneHotArrays, Plots, Statistics
using Random: Xoshiro
using Base: @kwdef

include("module.jl")
using .PhasorNetworks

epsilon = 0.10

function network_tests()
    @kwdef mutable struct Args
        η::Float64 = 3e-4       ## learning rate
        batchsize::Int = 256    ## batch size
        epochs::Int = 10        ## number of epochs
        use_cuda::Bool = false   ## use gpu (if cuda available)
        rng::Xoshiro = Xoshiro(42) ## global rng
    end

    #load the dataset and a single batch for testing
    args = Args()
    train_loader, test_loader = getdata(args)
    x, y = first(train_loader)

    model, ps, st = build_mlp(args)




end

function check_bounds(x)
    return abs(x) < epsilon ? true : false
end

function getdata(args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    @info "Getting and transforming data"

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

