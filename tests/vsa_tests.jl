using Statistics: mean
using LinearAlgebra: diag

function vsa_tests()
    #test functions
    t1 = test_orthogonal()
    t2 = test_outer()
end

"""
Basic test that random VSA symbols are orthogonal
"""
function test_orthogonal()
    x = random_symbols((100, 1024))
    y = random_symbols((100, 1024))

    s = similarity(x, y)
    pass = mean(s) < 0.1 ? true : false
    @assert pass "Orthogonality test failed"
    return pass
end

"""
Test the outer similarity function with normal and spiking arguments
"""
function test_outer()
    n_x = 101
    n_y = 101
    n_vsa = 1
    repeats = 6
    epsilon = 0.02
    spk_args = default_spk_args()
    tspan = (0.0, repeats*1.0)

    function check_phase(matrix)
        in_phase = diag(matrix)
        anti_phase = diag(matrix, convert(Int, round(n_x / 2)))

        v1 = reduce(*, map(x -> x > 1.0 - epsilon, in_phase))
        v2 = reduce(*, map(x -> x < -1.0 + epsilon, anti_phase))
        return v1, v2
    end

    #check the non-spiking implementation
    phase_x = reshape(range(-1.0, 1.0, n_x), (1, n_x, n_vsa)) |> collect
    phase_y = reshape(range(-1.0, 1.0, n_y), (1, n_y, n_vsa)) |> collect
    sims = similarity_outer(phase_x, phase_y, dims= 2, reduce_dim=1)[1,1,:,:]
    v1, v2 = check_phase(sims)
    @assert v1 "In-phase values producing incorrect similarity"
    @assert v2 "Anti-phase values producing incorrect similarity"

    #check the spiking implementation
    st_x = phase_to_train(phase_x, spk_args, repeats = repeats)
    st_y = phase_to_train(phase_y, spk_args, repeats = repeats)
    sims_2 = stack(similarity_outer(st_x, st_y, dims=2, reduce_dim=3, tspan=tspan));
    #check at the last time step
    sims_spk = sims_2[1,1,end,:,:]
    v1s, v2s = check_phase(sims_spk)
    @assert v1s "In-phase spiking values producing incorrect similarity"
    @assert v2s "Anti-phase spiking values producing incorrect similarity"

    #check the cross-implementation error
    avg_error = mean(sims .- sims_spk)
    error_check = avg_error < epsilon
    @assert error_check "Poor match between similarity implementations"

    return v1, v2, v1s, v2s, error_check
end