using Distributed
n_samples = parse(Int, ARGS[1])
type_chk = ARGS[2]
@assert type_chk in ("mlp", "pmlp", "ode") "Unrecognized network type requested"

seeds = collect(43:43 + n_samples - 1)
addprocs(n_samples)

@everywhere include("train_classifier.jl")
@everywhere n_epochs = 10
@everywhere type = $type_chk

@everywhere function exec_training(seed::Int)
    global type
    
    data_dir = "pixel_data/"
    file_pairs = get_dataset(data_dir)

    q, ylocal, pt = get_samples(file_pairs[1:2]);
    q_test, ylocal_test, pt_test = get_samples(file_pairs[3:3]);

    args = Args(batchsize = 128)
    spk_args = SpikingArgs(leakage = -0.10)
    repeats = 10

    test_loader = DataLoader((q_test, ylocal_test, pt_test), partial=false, batchsize=args.batchsize)
    train_loader = DataLoader((q, ylocal, pt), partial=false, batchsize=args.batchsize)

    x, xl, y = first(train_loader)
    n_px = size(x, 2) 
    n_in = n_px + 1
    rng = Xoshiro(seed)

    if type == "ode"
        #models defined in train_classifier.jl
        model = ode_model(spk_args)
        ps, st = Lux.setup(rng, model)
        psa = ComponentArray(ps)
        lhist, pst, stt = train_ode(model, psa, st, train_loader, repeats = 10, id=seed, epochs=n_epochs, spk_args = spk_args)

    elseif type == "pmlp"
        ps, st = Lux.setup(rng, pmlp_model)
        lhist, pst, stt = train_pmlp(pmlp_model, ps, st, train_loader, id=seed, epochs=n_epochs)

    elseif type == "mlp"
        ps, st = Lux.setup(rng, mlp_model)
        lhist, pst, stt = train_mlp(mlp_model, ps, st, train_loader, id=seed, epochs=n_epochs)        
    end

    return 1
end

exec = pmap(exec_training, seeds)

for i in workers()
	rmprocs(i)
end
