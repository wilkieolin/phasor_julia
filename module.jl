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
unbind,
angle_to_complex,
chance_level,
complex_to_angle,
random_symbols,
similarity,
similarity_self,
similarity_outer,
similarity_loss,

#metrics
confusion_matrix,
OvR_matrices,
tpr_fpr,
interpolate_roc

include("src/network.jl")
include("src/metrics.jl")

end;