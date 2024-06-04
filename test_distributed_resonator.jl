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
@everywhere include("resonator.jl")
@everywhere using Dates: now
@everywhere using JLD2: save_object

@everywhere @kwdef struct Args
	n_cb::Int = 20
	d_vsa::Int = 1024
	n_iters::Int = 20
	repeats::Int = 20
	rng::AbstractRNG
	spk_args::SpikingArgs
end

key = Xoshiro(42)
n_trials = 100
spk_args = SpikingArgs(solver = Tsit5(),
						solver_args = Dict(:adaptive => false, :dt => 0.01),)

all_args = [Args(rng = Xoshiro(rand(key, UInt32)), spk_args = spk_args) for _ in 1:n_trials]

@everywhere function save(args, result, final::Bool=false)
	if !final
    	name = "data/resonator/result_" * string(now()) * ".jld2"
	else
		name = "data/resonator/final_" * string(now()) * ".jld2"
	end
    dict = ["arguments" => args, "results" => result]
    save_object(name, dict)
end

@everywhere function call_test(args::Args)
    acc, trends = factor3_test_spiking(args.rng, args.n_cb, args.n_iters, args.spk_args, args.repeats)
	result = Dict("accuracy" => acc, "trends" => trends)
	save(args, result)
    return aurocs
end

all_results = pmap(call_test, all_args)

save(all_args, all_results, true)

# The Slurm resource allocation is released when all the workers have
# exited
for i in workers()
	rmprocs(i)
end
