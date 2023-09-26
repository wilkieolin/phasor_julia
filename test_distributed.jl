using Pkg
Pkg.activate(".")

using Distributed, ClusterManagers
using Base: @kwdef

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
addprocs(2)
@everywhere include("module.jl")

@kwdef struct Args
	nodes::Int = 20
	p_edge::Real = 0.1
	d_vsa::Int = 1024
end

n_trials = 6

n_nodes = [20]
p_edge = [0.1]
d_vsa = [1024]

all_args = [repeat(Args(nodes = n, p_edge = p, d_vsa = d), n_trials) for n in n_nodes, p in p_edge, d in d_vsa]

for i in workers()
	host, pid = fetch(@spawnat i (gethostname(), getpid()))
	push!(hosts, host)
	push!(pids, pid)
end

print(hosts)
print(pids)

# The Slurm resource allocation is released when all the workers have
# exited
for i in workers()
	rmprocs(i)
end