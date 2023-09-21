using Interpolations: linear_interpolation

function confusion_matrix(prediction, truth, threshold::Real)
    #how close are the predictions to positive?
    sim = similarity(prediction, zero(prediction) .+ 0.5, dim=2)
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

function tpr_fpr(prediction, truth, threshold::Real = 0.2, points::Int = 201, epsilon::Real = 0.01)
    labels = momentum_to_label(truth, threshold)
    test_points = range(start = 1.0, stop = -1.0, length = points)

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
    tpr = cat(0.0, tpr, 1.0, dims=1)
    fpr = cat(0.0, fpr, 1.0, dims=1)
    interp = linear_interpolation(fpr, tpr)
    return interp
end