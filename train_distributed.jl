using Distributed
n_procs = parse(Int, ARGS[1])
type_chk = ARGS[2]
n_samples = 6
@assert type_chk in ("mlp", "pmlp", "ode") "Unrecognized network type requested"

seeds = collect(43:43+n_samples)
addprocs(n_procs)

@everywhere include("train_classifier.jl")
@everywhere n_epochs = 100
@everywhere type = $type_chk

@everywhere function exec_training(seed::Int)
    global type
    
    data_dir = "pixel_data/"
    file_pairs = get_dataset(data_dir)

    q, ylocal, pt = get_samples(file_pairs[1:2]);
    q_test, ylocal_test, pt_test = get_samples(file_pairs[3:3]);

    args = Args(batchsize = 128)

    test_loader = DataLoader((q_test, ylocal_test, pt_test), batchsize=args.batchsize)
    train_loader = DataLoader((q, ylocal, pt), batchsize=args.batchsize)

    x, xl, y = first(train_loader)
    x_tms = range(start=0.0, stop=1.0, length=size(x, 1)) |> collect
    n_px = size(x, 2) 
    n_in = n_px + 1
    sa = SpikingArgs()
    rng = Xoshiro(seed)

    if type == "ode"
        #models defined in train_classifier.jl
        ps, st = Lux.setup(rng, ode_model)
        psa = ComponentArray(ps)
        @time lhist, pst, stt = train_ode(ode_model, psa, st, train_loader, x_tms, id=seed, epochs=n_epochs)

    elseif type == "pmlp"
        ps, st = Lux.setup(rng, pmlp_model)
        @time lhist, pst, stt = train_pmlp(pmlp_model, ps, st, train_loader, id=seed, epochs=n_epochs)

    elseif type == "mlp"
        ps, st = Lux.setup(rng, mlp_model)
        @time lhist, pst, stt = train_mlp(mlp_model, ps, st, train_loader, id=seed, epochs=n_epochs)        
    end

    return 1
end

exec = pmap(exec_training, seeds)

for i in workers()
	rmprocs(i)
end
