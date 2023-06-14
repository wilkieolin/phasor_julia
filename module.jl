module PhasorNetworks

using Flux, Functors, DifferentialEquations, Statistics

export 
#types
PhasorDense, 
SpikeTrain, 
SpikingArgs, 
SpikingCall, 
CurrentCall,

#spiking
default_spk_args,
phase_to_train,
train_to_phase,
zero_nans,
cycle_correlation,
cor_realvals,

#vsa
bundle,
bundle_project,
bind,
angle_to_complex,
complex_to_angle,
random_symbols,
similarity,
similarity_self,
similarity_outer

include("src/vsa.jl")
include("src/spiking.jl")
include("src/network.jl")

end;