using Distributed, Pkg
Pkg.activate(".")

n_samples = parse(Int, ARGS[1])
seeds = collect(43:43 + n_samples - 1)
type_chk = ARGS[2]
length(ARGS) > 2 ? n_procs = parse(Int, ARGS[3]) : n_procs = n_samples
@assert type_chk in ("mlp", "pmlp", "ode") "Unrecognized network type requested"
type = type_chk

addprocs(n_procs)

@everywhere include("classifier.jl")

args = Args(batchsize = 128)
@everywhere n_epochs = 10
@everywhere ode_spk_args = SpikingArgs(leakage=-0.2, solver=Tsit5())
@everywhere ode_tspan = (0.0, 25.0)

@everywhere normal_spk_args = SpikingArgs()
@everywhere normal_tspan = (0.0, 10.0)

# """
# Check if the data necessary for ODE training and testing is present, 
# generate if not (make sure workers don't end up racing on this)
# """
@everywhere function check_ode_data(ode_tspan::Tuple, args::Args)
    storage_dir = "data"
    test_file = joinpath(storage_dir, "test_phase.jld2")
    train_file = joinpath(storage_dir, "train_phase.jld2")
    data_dir = "pixel_data/"
    file_pairs = get_dataset(data_dir)
    clock_amp = 0.01
    batchsize = args.batchsize
    ode_spk_args = SpikingArgs(leakage=-0.2, solver=Tsit5())

    if !isfile(test_file)
        @info "Generating test ODE data..."
        q_test, ylocal_test, pt_test = get_samples(file_pairs[3:3]);
        test_mp, test_train = resonate_on_current(q_test, ylocal_test, ode_spk_args, ode_tspan, batchsize, clock_amp)
        #cut down pt to what matches the test train
        pt_test = pt_test[1:size(test_mp,2)]
        save_object(test_file, Dict("phase" => test_mp, "momentum" => pt_test, "spikes" => test_train))
    end
    
    if !isfile(train_file)
        @info "Generating training ODE data..."
        q, ylocal, pt = get_samples(file_pairs[1:2]);
        x_mp, _ = resonate_on_current(q, ylocal, ode_spk_args, ode_tspan, batchsize, clock_amp)
        #cut down momentum to what matches x_mp
        pt = pt[1:size(x_mp,2)]
        save_object(train_file, Dict("phase" => x_mp, "momentum" => pt))
    end

    return test_file, train_file
end

@everywhere function exec_training(type::String, seed::Int, args::Args)
    data_dir = "pixel_data/"
    file_pairs = get_dataset(data_dir)

    n_px = 13
    n_in = n_px + 1
    rng = Xoshiro(seed)
    global ode_tspan

    if type == "ode"
        tspan = ode_tspan

        #load data from the stored phasor representation
        test_file, train_file = check_ode_data(tspan, args)
        #training data
        train_data = load_object(train_file)
        x_mp = train_data["phase"]
        pt = train_data["momentum"]
        train_loader = DataLoader((x_mp, pt), partial=false, batchsize=args.batchsize)

        #testing data
        test_data = load_object(test_file)
        test_mp = test_data["phase"]
        test_pt = test_data["momentum"]
        train_test = test_data["spikes"]
        test_loader = DataLoader((test_mp, test_pt), partial = false, batchsize=args.batchsize)
        println(test_loader)

        #models defined in classifier.jl
        model = ode_model()
        ps, st = Lux.setup(rng, model)
        psa = ComponentArray(ps)
        lhist, pst, stt = train_ode(model, psa, st, train_loader, id=seed, epochs=n_epochs)
        result = Dict("loss" => lhist, "params" => pst, "state" => stt)

        #test the model
        auroc_static, auroc_dynamic = test_ode(model, pst, stt, test_loader, train_test, spk_args=spk_args, tspan=tspan)
        result["auroc static"] = auroc_static
        result["auroc dynamic"] = auroc_dynamic 

        #save the results
        filename = reduce(*, [joinpath("parameters", "final_id_"), string(seed), ".jld2"])
        save_object(filename, result)
    else
        #set parameters
        spk_args = SpikingArgs(leakage = -0.10, threshold=1e-5)
        repeats = 10

        #load data from the direct input files
        q, ylocal, pt = get_samples(file_pairs[1:2]);
        q_test, ylocal_test, pt_test = get_samples(file_pairs[3:3]);
        train_loader = DataLoader((q, ylocal, pt), partial=false, batchsize=args.batchsize)

        if type == "pmlp"
            ps, st = Lux.setup(rng, pmlp_model)
            model = pmlp_model
            lhist, pst, stt = train_pmlp(pmlp_model, ps, st, train_loader, id=seed, epochs=n_epochs)
        else # type is "mlp"
            ps, st = Lux.setup(rng, mlp_model)
            model = mlp_model
            lhist, pst, stt = train_mlp(mlp_model, ps, st, train_loader, id=seed, epochs=n_epochs) 
        end
    end

    return model, pst, stt
end

check_ode_data(ode_tspan, args)
exec = pmap(x -> exec_training(type, x, args), seeds)

for i in workers()
	rmprocs(i)
end
