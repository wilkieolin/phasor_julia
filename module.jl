module PhasorNetworks

using Lux, DifferentialEquations, Statistics

export 
#types
PhasorDense, 
PhasorODE,
SpikeTrain, 
LocalCurrent,
SpikingArgs, 
SpikingCall, 
CurrentCall,

#spiking
default_spk_args,
phase_to_train,
phase_to_potential,
potential_to_phase,
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
similarity_outer,
similarity_loss

include("src/vsa.jl")
include("src/spiking.jl")
include("src/network.jl")

end;