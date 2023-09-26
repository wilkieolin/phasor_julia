using Pkg
Pkg.activate(".")

using Distributed, ClusterManagers, JLD2, Dates
using Base: @kwdef
using Random: Xoshiro, AbstractRNG

n_procs = parse(Int, ARGS[1])

# Arguments to the Slurm srun(1) command can be given as keyword
# arguments to addprocs.  The argument name and value is translated to
# a srun(1) command line argument as follows:
# 1) If the length of the argument is 1 => "-arg value",
#    e.g. t="0:1:0" => "-t 0:1:0"
# 2) If the length of the argument is > 1 => "--arg=value"
#    e.g. time="0:1:0" => "--time=0:1:0"
# 3) If the value is the empty string, it becomes a flag value,
#    e.g. exclusive="" => "--exclusive"
# 4) If the argument contains "_", they are replaced with "-",
#    e.g. mem_per_cpu=100 => "--mem-per-cpu=100"
#addprocs(SlurmManager(2), partition="bdwall", t="00:5:00")
addprocs(n_procs)
@everywhere include("graph_functions.jl")

@everywhere @kwdef struct Args
	nodes::Int = 20
	p_edge::Real = 0.1
	d_vsa::Int = 1024
	rng::AbstractRNG
end

key = Xoshiro(42)
n_trials = 6
n_nodes = [25]
p_edge = collect(0.1:0.1:0.9)
d_vsa = [512]

all_args = stack([[Args(nodes = n, p_edge = p, d_vsa = d, rng = Xoshiro(rand(key, UInt32))) for i in 1:n_trials] for n in n_nodes, p in p_edge, d in d_vsa]) |> vec

@everywhere function save(args, result)
    name = "data/result_" * string(now()) * ".jld2"
    dict = ["arguments" => args, "results" => result]
    save_object(name, dict)
end

@everywhere function call_test(args::Args)
    aurocs = test_methods(args.nodes, args.p_edge, args.d_vsa, args.rng)
	save(args, aurocs)
    return aurocs
end

result = pmap(call_test, all_args)
print(result)

save(all_args, result)

# The Slurm resource allocation is released when all the workers have
# exited
for i in workers()
	rmprocs(i)
end