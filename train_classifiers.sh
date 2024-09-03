N_SAMPLES=24
N_PROCS=8
julia train_distributed.jl $N_SAMPLES mlp $N_PROCS
julia train_distributed.jl $N_SAMPLES pmlp $N_PROCS
julia train_distributed.jl $N_SAMPLES ode $N_PROCS