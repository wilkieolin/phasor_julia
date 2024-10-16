using Pkg
Pkg.activate(".")

using DifferentialEquations: Tsit5, Heun
using Distributed, ClusterManagers
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
@everywhere using Dates: now
@everywhere using JLD2: save_object

@everywhere @kwdef struct Args
	nodes::Int = 20
	p_edge::Real = 0.1
	d_vsa::Int = 1024
	rng::AbstractRNG
	spk_args::SpikingArgs
end

key = Xoshiro(52)
n_trials = 12
n_nodes = [25]
p_edge = collect(0.1:0.1:0.9)
d_vsa = [1024]
spk_args = SpikingArgs(solver = Tsit5(),
						solver_args = Dict(:adaptive => false, :dt => 0.01),)

all_args = stack([[Args(nodes = n, p_edge = p, d_vsa = d, rng = Xoshiro(rand(key, UInt32)), spk_args = spk_args) for i in 1:n_trials] for n in n_nodes, p in p_edge, d in d_vsa]) |> vec

@everywhere function save(args, result, final::Bool=false)
	if !final
    	name = "data/graph/result_" * string(now()) * ".jld2"
	else
		name = "data/graph/final_" * string(now()) * ".jld2"
	end
    dict = ["arguments" => args, "results" => result]
    save_object(name, dict)
end

@everywhere function call_test(args::Args)
    aurocs = test_methods(args.nodes, args.p_edge, args.d_vsa, args.rng, args.spk_args)
	save(args, aurocs)
    return aurocs
end

result = pmap(call_test, all_args)
print(result)

save(all_args, result, true)

# The Slurm resource allocation is released when all the workers have
# exited
for i in workers()
	rmprocs(i)
end
