struct SpikeTrain
    times::Array{<:Real,1}
    indices::Array{<:Int,1}
    shape::Tuple
    offset::Real
end

function spike_current(train::SpikeTrain, t::Real, t_window::Real = 0.03)
    current = zeros(ComplexF32, train.shape)

    times = train.times
    relative_time = abs.(times .- t)
    active = relative_time .< t_window
    active_inds = train.indices[active]
    current[active_inds] .+= (1.0 + 0im)
    return current
end