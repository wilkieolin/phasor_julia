using Interpolations: linear_interpolation
using LinearAlgebra: diag

include("network.jl")

function quadrature_loss(phases::AbstractArray, truth::AbstractArray)
    targets = 0.5 .* truth
    sim = similarity(phases, targets, dim = 1)
    return 1.0 .- sim
end

function similarity_loss(phases::AbstractArray, truth::AbstractArray, dim::Int)
    sim = similarity(phases, truth, dim = dim)
    return 1.0 .- sim
end

function loss_and_accuracy(data_loader, model, ps, st; 
                            mode="static", 
                            spk_args::SpikingArgs = default_spk_args(), 
                            t_span::Tuple{<:Real, <:Real} = (0.0, 6.0))

    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        if mode != "static"
            x = phase_to_train(x, spk_args, repeats=repeats)
            x = SpikingCall(x, spk_args, t_span)
        end
        ŷ, _ = model(x, ps, st)
        
        ls += sum(quadrature_loss(ŷ, y))
        acc += sum(accuracy_quadrature(ŷ, y)) ## Decode the output of the model
        num +=  size(y)[2]
    end
    return ls/num, acc/num
end

function spiking_accuracy(data_loader, model, ps, st, spk_args::SpikingArgs, t_span::Tuple{<:Real, <:Real}; repeats::Int = 3)
    acc = []
    n_phases = []
    num = 0

    for (x, y) in data_loader
        train = phase_to_train(x, spk_args, repeats=repeats)
        call = SpikingCall(train, spk_args, t_span)
        spk_output, _ = model(call, ps, st)
        ŷ = train_to_phase(spk_output)
        
        append!(acc, sum.(accuracy_quadrature(ŷ, y))) ## Decode the output of the model
        num +=  size(x)[end]
    end

    return acc, num
end

function accuracy_quadrature(phases::AbstractMatrix, truth::AbstractMatrix)
    predictions = getindex.(argmin(abs.(phases .- 0.5), dims=1), 1)'
    labels = getindex.(findall(truth), 1)
    return predictions .== labels
end

function accuracy_quadrature(phases::Array{<:Real,3}, truth::AbstractMatrix)
    return [accuracy_quadrature(phases[i,:,:], truth) for i in axes(phases,1)]
end

function confusion_matrix(sim, truth, threshold::Real)
    truth = hcat(truth .== 1, truth .== 0)
    prediction = hcat(sim .> threshold, sim .<= threshold)

    confusion = truth' * prediction
    return confusion
end

function OvR_matrices(predictions, labels, threshold::Real)
    #get the confusion matrix for each class verus the rest
    mats = diag([confusion_matrix(ys, ts, threshold) for ys in eachslice(predictions, dims=1), ts in eachslice(labels, dims=1)])
    return mats
end

function tpr_fpr(prediction, labels, points::Int = 201, epsilon::Real = 0.01)
    test_points = range(start = 0.0, stop = -20.0, length = points)
    test_points = vcat(exp.(test_points), 0.0, reverse(-1 .* exp.(test_points)))

    fn = x -> sum(OvR_matrices(prediction, labels, x))
    confusion = cat(fn.(test_points)..., dims=3)

    classifications = dropdims(sum(confusion, dims=2), dims=2)
    cond_true = classifications[1,:]
    cond_false = classifications[2,:]

    #return cond_true, cond_false

    true_positives = confusion[1,1,:]
    false_positives = confusion[2,1,:]

    #return true_positives, false_positives

    tpr = true_positives ./ cond_true
    fpr = false_positives ./ cond_false

    return tpr, fpr
end
    
function interpolate_roc(roc)
    tpr, fpr = roc
    interp = linear_interpolation(fpr, tpr)
    return interp
end