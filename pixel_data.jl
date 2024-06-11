using CSV, DataFrames, Interpolations
using Parquet2: Dataset
using Interpolations: gradient

const X_PIXELS = 21
const Y_PIXELS = 13
const T_STEPS = 20

"""
Read a directory with smart pixel data and return the paired label-data files.
"""
function get_dataset(directory::String)
    label_file = r"labels_d(\d+).parquet"
    data_file = r"recon3D_d(\d+).parquet"

    data_files = readdir(joinpath(directory, "recon3D"))
    label_files = readdir(joinpath(directory, "labels"))

    #find the label and data files by name
    reject_empty = y -> filter(x -> x != nothing, y)
    labels = reject_empty(match.(label_file, label_files))
    data = reject_empty(match.(data_file, data_files))

    #convert the string matches to ints
    get_ids = x -> parse.(Int, getindex.(x, 1))
    label_ids = get_ids(labels)
    data_ids = get_ids(data)

    #match the label and data ids
    matches = findall(label_ids .== data_ids')
    pairs = [(labels[idx[1]].match, data[idx[2]].match) for idx in matches]
    add_dir = x -> (joinpath(directory, "labels", x[1]), joinpath(directory, "recon3D", x[2]))
    pairs = add_dir.(pairs)
    
    return pairs
end

"""
Using a pair of files returned from get_dataset, load the momentum and deposited charge.
"""
function load_file(file_pair::Tuple{<:String, <:String})
    #load the pixel charge data and particle truth
    label = DataFrame(Dataset(file_pair[1]))
    #load all charge data into a matrix and remove the row id
    data = 1.0 .* Matrix(DataFrame(Dataset(file_pair[2])))[:,1:end-1]
    #get the momentum
    momentum = 1.0 .* Vector(label[!, "pt"])
    y_local = 1.0 .* Vector(label[!, "y-local"])
    n_ex = size(data, 1)
    #reshape the 2D matrix into a 4D tensor - (n, x, y, t)
    charge = reshape(data, n_ex, X_PIXELS, Y_PIXELS, T_STEPS)
    #reverese the order to meet JuliaML's convention
    charge = permutedims(charge, (4,3,2,1))

    return charge, y_local, momentum
end

function get_samples(data_dir, files)
    all_files = get_dataset(data_dir)
    data = [load_file(all_files[f]) for f in files]

    #read a set of charge over time samples
    charge = cat([d[1] for d in data]..., dims=4)
    ylocal = cat([d[2] for d in data]..., dims=1)
    momentum = cat([d[3] for d in data]..., dims=1)

    return charge, ylocal, momentum
end

"""
Using a single example of charge in the detector over time, return a quadratic B-spline
interpolation with zeros outside the data range.
"""
function interpolate_charge(charge::AbstractArray{<:Real})
    #interpolate the charge with a quadratic spline
    interpolation = interpolate(charge, BSpline(Quadratic(Reflect(OnCell()))))
    #return 0s outside of the data range
    extrapolation = extrapolate(interpolation, 0.0)
    return extrapolation
end

"""
Using an interpolated or fit version of the charge data, take the gradient to estimate
the current deposited with respect to time.
"""
function interpolate_current(fit::Interpolations.FilledExtrapolation{<:Any, 3}, t::Real)
    current = [gradient(fit, t, y, x)[1] for x in 1:X_PIXELS, y in 1:Y_PIXELS]
    return current
end

function interpolate_current(fit::Interpolations.FilledExtrapolation{<:Any, 4}, t::Real)
    n_batch = size(fit, 4)
    current = [gradient(fit, t, y, x, b)[1] for y in 1:Y_PIXELS, x in 1:X_PIXELS, b in 1:n_batch]
    return current
end

function log_scale_current(i::AbstractArray)
    lo = -11500
    hi = 11500
    #clamp to within (0.001, 0.999 quantiles)
    i = clamp.(i, lo, hi)
    #log-scale & include sign
    i = sign.(i) .* log1p.(abs.(i))
    return i
end

function scale_current(i::AbstractArray)
    lo = -11500
    hi = 11500
    #clamp to within (0.001, 0.999 quantiles)
    i = clamp.(i, lo, hi)
    #log-scale & include sign
    i = i ./ (hi)
    return i
end

function accuracy(x, xl, y, model, ps, st, threshold::Real)
    y_truth = get_truth(y, threshold)
    y_pred, _ = model(process_inputs(x, xl), ps, st)
    y_labels = onecold(y_pred, (0, 1, 2))
    right = sum(y_truth .== y_labels)
    return right
end

function confusion_matrix(prediction, truth, threshold::Real)
    truth = hcat(truth .== 1, truth .== 0)
    prediction = hcat(prediction .> threshold, prediction .<= threshold)

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
    test_points = range(start = 5.0, stop = -5.0, length = points)

    
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
