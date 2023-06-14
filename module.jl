module PhasorNetworks

using Flux, Functors, DifferentialEquations, Statistics

export 
include("src/vsa.jl")
include("src/spiking.jl")
include("src/network.jl")

end;