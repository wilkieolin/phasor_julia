using Pkg
Pkg.activate(".")

using DifferentialEquations, PhasorNetworks, Lux, NNlib, Zygote, ComponentArrays, Optimisers
using MLUtils: DataLoader
using Random: Xoshiro
using ChainRulesCore: ignore_derivatives

include("pixel_data.jl")
data_dir = "pixel_data/"
file_pairs = get_dataset(data_dir)