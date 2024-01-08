using Statistics: mean
using LinearAlgebra: diag
using .PhasorNetworks: bind

#global args for all tests
n_x = 101
n_y = 101
n_vsa = 1
repeats = 6
epsilon = 0.02
spk_args = default_spk_args()
tspan = (0.0, repeats*1.0)
tbase = collect(tspan[1]:spk_args.dt:tspan[2])

"""
Run all the basic VSA tests and check they pass
"""
function vsa_tests()
    #test functions
    tests = [test_orthogonal(),
            test_outer(),
            test_binding(),
            test_bundling(),
            ]

    all_pass = reduce(*, tests)
    if all_pass
        println("All VSA tests passed.")
    else
        println("VSA test failed.")
    end

    return all_pass
end

"""
Check if a value is within the bounds determined by epsilon
"""
function in_tolerance(x)
    return abs(x) < epsilon ? true : false
end

"""
Produce all possible pairs of angles
"""
function make_angle_pairs()
    phases = collect([[x, y] for x in range(-1.0, 1.0, n_x), y in range(-1.0, 1.0, n_y)]) |> stack
    phases = reshape(phases, (1,2,:))
    return phases
end

"""
Remove NaN values
"""
function remove_nan(x)
    y = filter(x -> !isnan(x), x)
    return y
end

"""
Produce the error of angular values from zero
"""
function sin_error(x)
    return sin.(pi .* x) |> mean
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

    pass = reduce(*, [v1, v2, v1s, v2s, error_check])
    return pass
end


"""
Test the binding/unbinding functions in both versions & cross-check errors
"""
function test_binding()
    phases = make_angle_pairs()
    #check binding and unbinding functions
    b = bind(phases, dims=2)
    ub = unbind(phases[1:1,1:1,:], phases[1:1,2:2,:])

    #check binding via oscillators
    st_x = phase_to_train(phases[1:1,1:1,:], spk_args, repeats = repeats)
    st_y = phase_to_train(phases[1:1,2:2,:], spk_args, repeats = repeats)
    uout = bind(st_x, st_y, tspan=tspan, return_solution=true);
    decoded = potential_to_phase(uout, tbase, dim=4, spk_args=spk_args);
    u_err = mean(decoded[1,:,:,:] .- b[1,:,:], dims=(1,2))[end]
    u_check = in_tolerance(u_err)
    @assert u_check "Incorrect potential resulting from binding"

    #check with spiking outputs
    b2 = bind(st_x, st_y, tspan=tspan, return_solution=false)
    b2d = train_to_phase(b2, spk_args)
    enc_error = remove_nan(vec(b2d[5,:,:,:]) .- vec(b)) |> mean
    enc_check = in_tolerance(enc_error)
    @assert enc_check "Incorrect spiking outputs from binding"

    #check unbinding operation
    ubout = unbind(st_x, st_y, tspan=tspan, return_solution=true)
    decoded = potential_to_phase(ubout, tbase, dim=4, spk_args=spk_args)
    err = mean(decoded[1,:,:,:] .- ub[1,:,:], dims=(1,2))[end]
    unbind_chk = in_tolerance(err)
    @assert unbind_chk "Unbinding resulted in incorrect potential"

    #check unbinding with spiking outputs
    ub2 = unbind(st_x, st_y, tspan=tspan, return_solution=false)
    ub2d = train_to_phase(ub2, spk_args)
    ub_enc_error = remove_nan(vec(ub2d[5,:,:,:]) .- vec(ub)) |> mean
    ub_enc_check = in_tolerance(ub_enc_error)
    @assert ub_enc_check "Unbinding resulted in incorrect spiking outputs"

    pass = reduce(*, [u_check, enc_check, unbind_chk, ub_enc_check])
    return pass
end

function test_bundling()
    phases = make_angle_pairs()
    #check bundling function
    b = bundle(phases, dims=2);

    st = phase_to_train(phases, spk_args, repeats=6)
    #check potential encodings
    b2_sol = bundle(st, dims=2, spk_args=spk_args, tspan=tspan, return_solution=true)
    b2_phase = potential_to_phase(b2_sol, tbase, dim=4, spk_args=spk_args, offset=0.0)
    b2_phase_error = vec(b2_phase[1,1,:,end]) .- vec(b) |> mean
    b2_phase_check = in_tolerance(b2_phase_error)
    @assert b2_phase_check "Bundling phase error out of tolerance"

    #check spiking encoding
    b2 = bundle(st, dims=2, spk_args=spk_args, tspan=tspan, return_solution=false)
    decoded = train_to_phase(b2, spk_args)
    b2_error = remove_nan(vec(decoded[end-1,:,:,:]) .- vec(b)) |> sin_error
    b2_spike_check = in_tolerance(b2_error)
    @assert b2_spike_check "Bundling spiking error out of tolerance"

    pass = b2_phase_check * b2_spike_check
    return pass
    
end