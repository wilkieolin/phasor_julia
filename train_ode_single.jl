include("train_ode_classifier.jl")

seed = parse(Int, ARGS[1])
n_epochs = parse(Int, ARGS[2])

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

ode_fn = Chain(BatchNorm(n_in),
                x -> tanh.(x),
                Dense(n_in => 128))


ode_model = Chain(PhasorODE(ode_fn, tspan=(0.0, 1.0), dt=0.01),
                x -> complex_to_angle(Array(x)[:,:,end]),
                PhasorDenseF32(128 => 3))

ps, st = Lux.setup(rng, ode_model)
psa = ComponentArray(ps)

@time lhist, pst, stt = train(ode_model, psa, st, train_loader, x_tms, epochs=n_epochs)