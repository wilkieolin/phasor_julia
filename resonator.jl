function generate_composition(rng::AbstractRNG, codebooks...)
    ns = [size(cb,1) for cb in codebooks]
    indices = [rand(rng, 1:n) for n in ns]
    symbols = [codebooks[i][indices[i],:] for i in 1:length(codebooks)]
    factors = stack(symbols, dims=1)
    symbol = v_bind(factors, dims=1)
    return factors, symbol
end

function initialize_guesses(codebooks::AbstractArray...)
    function inner(codebook::AbstractArray)
        return v_bundle(codebook, dims=1)
    end

    guesses = collect(map(inner, codebooks))
    return guesses
end

function refine(composite::AbstractArray, factor_codebook::AbstractArray, external::AbstractMatrix)
    #bind the symbols for external factors
    external = v_bind(external, dims=1)

    #unbind external factors from the composite symbol
    factor = v_unbind(composite, external)

    #calculate the similarity to the codebook
    s = similarity_outer(factor, factor_codebook, dims=1)
    s = dropdims(s, dims=1)
    new_guess = v_bundle_project(factor_codebook, s, zeros((size(s,1), size(X_cb,2))))
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

    guesses = 
    return guesses
end