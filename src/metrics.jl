using Interpolations: linear_interpolation
using LinearAlgebra: diag

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