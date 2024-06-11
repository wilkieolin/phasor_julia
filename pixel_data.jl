using CSV, DataFrames, Interpolations
using Parquet2: Dataset
using Interpolations: gradient

const X_PIXELS = 21
const Y_PIXELS = 13
const T_STEPS = 20

"""
Read a directory with smart pixel data and return the paired label-data files.
"""
function get_dataset(directory::String, type::String)
    if type == "csv"
        label_file = r"labels_d(\d+).csv"
        data_file = r"recon8t_d(\d+).csv"
    elseif type == "parquet2D"
        label_file = r"labels_d(\d+).parquet"
        data_file = r"recon2D_d(\d+).parquet"
    elseif type == "parquet3D"
        label_file = r"labels_d(\d+).parquet"
        data_file = r"recon3D_d(\d+).parquet"
    else
        print("Unrecognized type")
        return
    end

    files = readdir(directory)

    #find the label and data files by name
    reject_empty = y -> filter(x -> x != nothing, y)
    labels = reject_empty(match.(label_file, files))
    data = reject_empty(match.(data_file, files))

    #convert the string matches to ints
    get_ids = x -> parse.(Int, getindex.(x, 1))
    label_ids = get_ids(labels)
    data_ids = get_ids(data)

    #match the label and data ids
    matches = findall(label_ids .== data_ids')
    pairs = [(labels[idx[1]].match, data[idx[2]].match) for idx in matches]
    
    return pairs
    
end

"""
Using a pair of files returned from get_dataset, load the momentum and deposited charge.
"""
function load_file(directory::String, file_pair::Tuple{<:SubString, <:SubString}, type::String="3D")
    #load the pixel charge data and particle truth
    label = DataFrame(Dataset(directory * "/" * file_pair[1]))
    #load all charge data into a matrix and remove the row id
    data = 1.0 .* Matrix(DataFrame(Dataset(directory * "/" * file_pair[2])))[:,1:end-1]
    #get the momentum
    momentum = 1.0 .* Vector(label[!, "pt"])
    y_local = 1.0 .* Vector(label[!, "y-local"])
    n_ex = size(data, 1)
    if type == "3D"
        #reshape the 2D matrix into a 4D tensor - (n, x, y, t)
        charge = reshape(data, n_ex, X_PIXELS, Y_PIXELS, T_STEPS)
        #reverese the order to meet JuliaML's convention
        charge = permutedims(charge, (4,3,2,1))
    else
        #reshape the 2D matrix into a 3D tensor - (n, x, y)
        charge = reshape(data, n_ex, X_PIXELS, Y_PIXELS)
        #reverese the order to meet JuliaML's convention
        charge = permutedims(charge, (3,2,1))
    end

    return charge, y_local, momentum
end

function load_file_csv(directory::String, file_pair::Tuple{<:SubString, <:SubString})
    #load the pixel charge data and particle truth
    label = CSV.read(directory * "/" * file_pair[1], DataFrame)
    data = CSV.read(directory * "/" * file_pair[2], DataFrame) |> Matrix
    #get the momentum
    momentum = label[!, "pt"]
    #reshape the 2D matrix into a 4D tensor - (n, x, y, t)
    n_ex = size(data, 1)
    charge = reshape(data, n_ex, X_PIXELS, Y_PIXELS, T_STEPS)
    #reverese the order to meet JuliaML's convention
    charge = permutedims(charge, (4,3,2,1))

    return charge, momentum
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
