using CSV, DataFrames
using Interpolations
using Parquet2: Dataset
import Interpolations.gradient as interp_gradient
using PhasorNetworks: gaussian_kernel
using LinearAlgebra: diag

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


function get_samples(files)
    data = [load_file(f) for f in files]

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
    current = [interp_gradient(fit, t, y, x)[1] for x in 1:X_PIXELS, y in 1:Y_PIXELS]
    return current
end

function interpolate_current(fit::Interpolations.FilledExtrapolation{<:Any, 4}, t::Real)
    n_batch = size(fit, 4)
    current = [interp_gradient(fit, t, y, x, b)[1] for y in 1:Y_PIXELS, x in 1:X_PIXELS, b in 1:n_batch]
    return current
end

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
        elseif t >= times[end]
            charge .+= values[end,:,:,:]
        end
    end

    return charge
end


function interpolate_2D_derivative(t::Real, times::Vector{<:Real}, values::AbstractArray{<:Real,4})
    n_steps, n_y, n_x, n_batch = size(values)
    #extrapolate to zeros
    current = zeros((n_y, n_x, n_batch),)

    ignore_derivatives() do
        if t > times[1] && t < times[end]
            i_next = findfirst(times .> t)
            i_prev = i_next - 1

            t_next = times[i_next]
            t_prev = times[i_prev]
            dt = t_next - t_prev
            d = (values[i_next,:,:,:] .- values[i_prev,:,:,:]) ./ dt
            current .+= d
        end
    end

    return current
end

function charge_to_current(values::AbstractArray; spk_args::SpikingArgs, tspan::Tuple, clock_amp::Real = 0.01)
    x_tms = range(start=0.0, stop=1.0, length=size(values, 1)) |> collect
    
    function current_fn(t)
        #scale the charge for each pixel using dataset stats (Y X B)
        q = scale_charge(interpolate_2D_derivative(t, x_tms, values))
        #take the mean charge accumulated over each row (X)
        q = mean(q, dims=2)[:,1,:]
    end

    clock_call = phase_to_current(zeros(size(values,2)), spk_args=spk_args, offset=0.0, tspan=tspan, repeat=false)
    clock_fn = clock_call.current.current_fn
    clocked_fn = t -> current_fn(t) .+ clock_amp .* clock_fn(t)

    current = LocalCurrent(clocked_fn, (size(values,2), size(values,4)), 0.0)
    call = CurrentCall(current, spk_args, tspan)

    return call
end

function ylocal_to_current(y_local::AbstractArray; spk_args::SpikingArgs, tspan::Tuple, sigma::Real = 9.0, y_range::Real = 32.5)
    y_local /= y_range
    phases = (y_local ./ 2.0) .+ 0.5
    phases = reshape(phases, (1, :))
    current_fn = phase_to_current(phases, spk_args = spk_args, tspan = tspan, offset = 0.0, repeat=false)

    return current_fn
end

function cat_currents(x::CurrentCall, y::CurrentCall; dim::Int)
    @assert x.spk_args == y.spk_args "Spiking args must match"
    new_tspan = (min(x.t_span[1], y.t_span[1]), max(x.t_span[2], y.t_span[2]))
    x_i = x.current
    y_i = y.current
    @assert x_i.offset == y_i.offset "Current offsets must match to concatenate"

    new_shape = [i == dim ? x_i.shape[i] + y_i.shape[i] : x_i.shape[i] for i in 1:length(x_i.shape)] |> Tuple
    new_current = t -> cat(x_i.current_fn(t), y_i.current_fn(t), dims=dim)
    new_offset = x_i.offset

    current = LocalCurrent(new_current, new_shape, new_offset)
    call = CurrentCall(current, x.spk_args, new_tspan)
    return call
end

function process_sample(x; spk_args::SpikingArgs, tspan::Tuple, clock_amp::Real = 0.01, kwargs...)
    charge, ylocal = x
    x1 = charge_to_current(charge, spk_args=spk_args, tspan=tspan, clock_amp=clock_amp)
    x2 = ylocal_to_current(ylocal, spk_args=spk_args, tspan=tspan)
    xf = cat_currents(x1, x2, dim=1)
    
    return xf
end

function data_to_phase(q, yl; spk_args::SpikingArgs, tspan::Tuple, clock_amp::Real = 0.01)
    x = process_sample((q, yl), spk_args=spk_args, tspan=tspan, clock_amp=clock_amp)
    ode_front = Chain(PhasorResonant(n_in, spk_args),)
    rng = Xoshiro(42)
    ps_ode, st_ode = Lux.setup(rng, ode_front)
    sol = ode_front(x, ps_ode, st_ode)[1]
    mp = mean_phase(sol, 1, spk_args=spk_args, offset=0.0, threshold=false)
    train = solution_to_train(sol, tspan, spk_args=spk_args, offset=0.0)
    return mp, train
end

function scale_charge(i::AbstractArray)
    #lo = -690.0
    hi = 15000.0
    #scale with 99.99% at 6 sigma
    i = 6.0 .* i ./ hi
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

function find_duplicates(x::AbstractVector)
    u = sort!(unique(x))
    matches = [x .== uval for uval in u]
    return u, matches
end

function average_duplicate_knots(x, y)
    x_unique, x_duplicates = find_duplicates(x)
    y_mean = map(x -> sum(x .* y)/sum(x), x_duplicates)
    y_mean[end] = 1.0
    if x_unique[end] != 1.0
        append!(x_unique, 1.0)
        append!(y_mean, 1.0)
    end
    return x_unique, y_mean
end  

function interpolate_roc(roc)
    tpr, fpr = roc
    tpr = cat(0.0, tpr, 1.0, dims=1)
    fpr = cat(0.0, fpr, 1.0, dims=1)
    interp = linear_interpolation(fpr, tpr)
    return interp
end
