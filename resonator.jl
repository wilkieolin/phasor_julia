using Random: Xoshiro, AbstractRNG
using PhasorNetworks

function select_composition(rng::AbstractRNG, shape::Array)
    indices = [rand(rng, 1:n) for n in shape]
    return indices
end

function generate_composition(rng::AbstractRNG, codebooks::AbstractArray...)
    ns = [size(cb,1) for cb in codebooks]
    indices = select_composition(rng, ns)
    symbols = [codebooks[i][indices[i],:] for i in 1:length(codebooks)]
    factors = stack(symbols, dims=1)
    symbol = v_bind(factors, dims=1)
    return indices, factors, symbol
end


function generate_composition(rng::AbstractRNG, spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}, codebooks::SpikeTrain...)
    ns = [size(cb)[1] for cb in codebooks]
    indices = select_composition(rng, ns)
    factors = [codebooks[i][indices[i],:] for i in 1:length(codebooks)]
    symbol = reduce((a, b) -> v_bind(a, b, tspan=tspan, spk_args=spk_args), factors)

    return indices, factors, symbol
end

function generate_trains(codebook::AbstractArray, spk_args::SpikingArgs, repeats::Int)
    trains = [phase_to_train(codebook[i,:], spk_args=spk_args, repeats=repeats) for i in 1:size(codebook,1)]
    return trains
end


function initialize_guesses(codebooks::AbstractArray...)
    function inner(codebook::AbstractArray)
        return v_bundle(codebook, dims=1)
    end

    guesses = collect(map(inner, codebooks))
    return guesses
end

function initialize_guesses(spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}, codebooks::SpikeTrain...)
    function inner(codebook::SpikeTrain)
        return v_bundle(codebook, dims=1, tspan=tspan, spk_args=spk_args)
    end

    guesses = collect(map(inner, codebooks))
    return guesses
end

function normalize(x::AbstractArray)
    m = maximum(x)
    if abs(m) > 0.0
        return x ./ m
    else
        return m
    end
end

function refine(composite::AbstractArray, factor_codebook::AbstractArray, external::AbstractMatrix)
    #bind the symbols for external factors
    external = v_bind(external, dims=1)

    #unbind external factors from the composite symbol
    factor = v_unbind(composite, external)

    #calculate the similarity to the codebook
    s = similarity_outer(factor, factor_codebook, dims=1)
    s = abs.(dropdims(s, dims=1))
    w = normalize(s)
    new_guess = v_bundle_project(factor_codebook, w, zeros((size(s,1), size(factor_codebook,2))))
    return new_guess
end 

function refine(composite::SpikeTrain, factor_codebook::SpikeTrain, external::Array{<:SpikeTrain}, spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real})
    #bind the symbols for external factors
    bindfn = (x, y) -> v_bind(x, y, spk_args=spk_args, tspan=tspan)
    external = reduce(bindfn, external)
    #return external

    #unbind external factors from the composite symbol
    factor = v_unbind(composite, external, spk_args=spk_args, tspan=tspan)

    #calculate the similarity to the codebook
    s = similarity_outer(factor_codebook, factor, dims=1, reduce_dim=2, spk_args=spk_args, tspan=tspan)
    w = reshape(abs.([x[end] for x in vec(s)]), (1, :))
    w = normalize(w)
    new_guess = v_bundle_project(factor_codebook, w, zeros((size(w,1), size(factor_codebook)[2])), spk_args=spk_args, tspan=tspan)
    return new_guess
end

function resonate(composite::AbstractArray, iterations::Int, codebooks::AbstractArray...)
    n_factors = length(codebooks)
    i_factors = collect(1:n_factors)
    #create the initial guesses for the symbol components
    components = initialize_guesses(codebooks...)
    guesses = [cat(components..., dims=1),]

    #refine the guess on one factor
    function refine_inner(ind::Int, components)
        #what is the factor we are refining
        factor = components[ind]
        #what are the other factors
        external_i = setdiff(i_factors, ind)
        externals = getindex(components, external_i, :)
        #refine the factor
        refined_factor = refine(composite, codebooks[ind], externals)
        return refined_factor
    end

    for iter in 1:iterations
        new_guesses = [refine_inner(i, guesses[iter]) for i in 1:n_factors]
        new_guesses = cat(new_guesses..., dims=1)
        push!(guesses, new_guesses)
    end

    return guesses
end

function resonate(composite::SpikeTrain, spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}, iterations::Int, codebooks::SpikeTrain...)
    n_factors = length(codebooks)
    i_factors = collect(1:n_factors)
    #create the initial guesses for the symbol components
    components = initialize_guesses(spk_args, tspan, codebooks...)
    guesses = [components, ]

    #refine the guess on one factor
    function refine_inner(ind::Int, components)
        #what is the factor we are refining
        factor = components[ind]
        #what are the other factors
        external_i = setdiff(i_factors, ind)
        externals = components[external_i]
        #refine the factor
        refined_factor = refine(composite, codebooks[ind], externals, spk_args, tspan)
        return refined_factor
    end

    for iter in 1:iterations
        new_guesses = [refine_inner(i, guesses[iter]) for i in 1:n_factors]
        push!(guesses, new_guesses)
    end

    return guesses
end

function extract_trends(factor, sims)
    n = size(sims,1)
    others = setdiff(1:n, factor)

    correct = sims[factor,:]
    incorrect = sims[others,:]

    return correct, incorrect
end

function extract_all_trends(factors, sims...)
    results = [extract_trends(factors[i], sims[i]) for i in 1:length(sims)]
    correct = stack([r[1] for r in results], dims=1)
    incorrect = cat([r[2] for r in results]..., dims=1)
    return correct, incorrect
end

function check(factors, sims...)
    function inner(factor, sim)
        return factor == argmax(sim[:,end]) ? true : false
    end

    correct = [inner(factors[i], sims[i]) for i in 1:length(sims)]
    return correct
end

function factor3_test(rng::AbstractRNG, n_cb::Int, n_vsa::Int, n_iters::Int)
    #generate the codebooks and composition given the rng
    X_cb = random_symbols((n_cb, n_vsa), rng)
    Y_cb = random_symbols((n_cb, n_vsa), rng)
    Z_cb = random_symbols((n_cb, n_vsa), rng)
    fac_i, fac, sym = generate_composition(rng, X_cb, Y_cb, Z_cb)
    
    #initialize the guesses
    x_cb, y_cb, z_cb = initialize_guesses(X_cb, Y_cb, Z_cb)
    #resonate the factors
    g = resonate(sym, n_iters, X_cb, Y_cb, Z_cb)

    #measure the result's similarity to the original symbols
    xmapfn = x -> vec(similarity_outer(x[1:1,:], X_cb, dims=1))
    ymapfn = x -> vec(similarity_outer(x[2:2,:], Y_cb, dims=1))
    zmapfn = x -> vec(similarity_outer(x[3:3,:], Z_cb, dims=1))
    
    xsims = cat(collect(map(xmapfn, g))..., dims=2)
    ysims = cat(collect(map(ymapfn, g))..., dims=2)
    zsims = cat(collect(map(zmapfn, g))..., dims=2)

    #check the correctness of the resonated factors
    acc = check(fac_i, xsims, ysims, zsims)
    trends = extract_all_trends(fac_i, xsims, ysims, zsims)

    return acc, trends
        
end

function factor3_test_spiking(rng::AbstractRNG, n_cb::Int, n_vsa::Int, n_iters::Int, spk_args::SpikingArgs, repeats::Int)
    #set the simulation timespan
    tspan = (0.0, spk_args.t_period * repeats)
    #generate the codebooks and composition given the rng & convert to spikes to drive oscillators
    p2t = x -> phase_to_train(x, spk_args=spk_args, repeats = repeats)
    X_cb = random_symbols((n_cb, n_vsa), rng) |> p2t
    Y_cb = random_symbols((n_cb, n_vsa), rng) |> p2t
    Z_cb = random_symbols((n_cb, n_vsa), rng) |> p2t
    
    fac_i, fac, sym = generate_composition(rng, spk_args, tspan, X_cb, Y_cb, Z_cb)
    
    #initialize the guesses
    x_cb, y_cb, z_cb = initialize_guesses(spk_args, tspan, X_cb, Y_cb, Z_cb)
    #resonate the factors
    g = resonate(sym, spk_args, tspan, n_iters, X_cb, Y_cb, Z_cb)

    function final_similarity(train::SpikeTrain, codebook::SpikeTrain, spk_args::SpikingArgs)
        sim = similarity_outer(train, codebook, dims=1, reduce_dim=2, spk_args=spk_args, tspan=tspan)
        sim_final = [s[end] for s in sim]
        return sim_final
    end

    #measure the result's similarity to the original symbols
    xmapfn = x -> final_similarity(x[1], X_cb, spk_args)
    ymapfn = x -> final_similarity(x[2], Y_cb, spk_args)
    zmapfn = x -> final_similarity(x[3], Z_cb, spk_args)
    
    xsims = cat(collect(map(xmapfn, g))..., dims=1)'
    ysims = cat(collect(map(ymapfn, g))..., dims=1)'
    zsims = cat(collect(map(zmapfn, g))..., dims=1)'

    #check the correctness of the resonated factors
    acc = check(fac_i, xsims, ysims, zsims)
    trends = extract_all_trends(fac_i, xsims, ysims, zsims)

    return acc, trends
    
end