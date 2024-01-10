using Pkg
Pkg.activate(".")

include("module.jl")
using .PhasorNetworks

println("Skipping VSA tests")
#include("tests/vsa_tests.jl")
#vsa_tests()

include("tests/network_tests.jl")
network_tests()