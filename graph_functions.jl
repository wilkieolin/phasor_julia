using Pkg
Pkg.activate(".")

using PhasorNetworks, QuadGK
using LinearAlgebra: triu, diagm, diag
using Statistics: std, median
using Random: Xoshiro, AbstractRNG

function auroc(graph::AbstractMatrix, reconstruction::AbstractMatrix)
    tpr, fpr = tpr_fpr(vec(reconstruction), vec(graph))
    roc_fn = interpolate_roc((tpr, fpr))
    auc, err = quadgk(x -> roc_fn(x), 0.0, 1.0)
    return auc, err
end

function define_node_symbols(graph::AbstractMatrix, nd::Int, rng::AbstractRNG)
    @assert size(graph,1) == size(graph,2) "Takes an adjacency matrix as the input"
    n = size(graph,1)

    #create vectors representing the nodes
    node_values = random_symbols((n, nd), rng)
    return node_values
end

function generate_er_graph(n::Int, p::Real, rng::AbstractRNG, self_loops::Bool = false)
    adj = rand(rng, Float64, (n, n)) .< p
    #make undirected
    adj = triu(adj)
    adj =  (adj .+ adj') .> 0

    if !self_loops
        #remove self-loops
        for i in 1:n
            adj[i,i] = 0
        end
    end

    return adj
end

function graph_to_vector(graph::AbstractMatrix, node_values::AbstractMatrix)
    @assert size(graph,1) == size(graph,2) "Takes an adjacency matrix as the input"
    n = size(graph,1)
    nd = size(node_values, 2)

    #get cartesian coordinates representing each edge
    edges = findall(graph)
    n_edges = length(edges)
    edge_values = zeros(Float64, n_edges, nd)

    #iterate through the edges
    for (i, edge) in enumerate(edges)
        tx = edge[1]
        rx = edge[2]

        tx_symbol = node_values[tx,:]
        rx_symbol = node_values[rx,:]
        #create a representation for that edge by binding its incident nodes
        edge_symbol = v_bind(tx_symbol, rx_symbol)
        edge_values[i,:] = edge_symbol
    end

    #combine the edges in the graph to the single embedding via bundling
    graph_embedding = v_bundle(edge_values, dims=1)
    return edge_values, graph_embedding
end

function graph_to_vector(graph::AbstractMatrix, node_values::AbstractMatrix, spk_args::SpikingArgs; repeats::Int=15)
    @assert size(graph,1) == size(graph,2) "Takes an adjacency matrix as the input"
    n = size(graph,1)
    nd = size(node_values, 2)
    
    #slice each node symbol into a spike train
    train_values = [phase_to_train(reshape(node, (1,:)), spk_args=spk_args, repeats=repeats) for node in eachslice(node_values, dims=1)]
    tspan = (0.0, repeats * 1.0)
    
    #get cartesian coordinates representing each edge
    edges = findall(graph)
    n_edges = length(edges)
    edge_values = []

    #iterate through the edges
    function edge_to_train(edge)
        tx = edge[1]
        rx = edge[2]

        tx_symbol = train_values[tx]
        rx_symbol = train_values[rx]
        #create a representation for that edge by binding its incident nodes
        edge_symbol = v_bind(tx_symbol, rx_symbol, spk_args=spk_args, tspan=tspan)
        return edge_symbol
    end

    edge_values = map(edge_to_train, edges)
    #combine the edges in the graph to the single embedding via bundling
    combined = vcat_trains(edge_values)
    graph_embedding = v_bundle(combined, dims=1, spk_args=spk_args, tspan=tspan)
    return train_values, edge_values, graph_embedding, tspan
end

function query_edges(graph::AbstractMatrix, nodes::AbstractMatrix)
    n = size(nodes, 1)
    nd = size(graph, 2)

    adj_rec = zeros(Float64, n, n)
    for (i,node) in enumerate(eachslice(nodes, dims=1))
        #add a dimension for consistency
        node = reshape(node, (1, :))
        query = v_unbind(graph, node)
        s = similarity_outer(query, nodes, dims=1) |> vec
        adj_rec[i,:] = s
    end

    return adj_rec
end

function query_edges(graph::SpikeTrain, nodes::Vector{<:SpikeTrain}, spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real})
    all_nodes = vcat_trains(nodes)
    
    function query_edge(node)
        query = v_unbind(graph, node, tspan=tspan, spk_args=spk_args)
        s = similarity_outer(query, all_nodes, dims=1, reduce_dim=2, spk_args=spk_args, tspan=tspan)
        return s
    end
    
    similarity = map(query_edge, nodes)
    adj_rec = stack(map(x -> last.(x), similarity))[1,:,:]

    return adj_rec
end

function test_methods(n::Int, p::Real, d_vsa::Int, rng::AbstractRNG, sa::SpikingArgs, repeats::Int = 15)
    graph = generate_er_graph(n, p, rng)
    node_symbols = define_node_symbols(graph, d_vsa, rng)

    #test with the floating-point method
    _, graph_static = graph_to_vector(graph, node_symbols)
    recon_static = query_edges(graph_static, node_symbols)
    auroc_static = auroc(graph, recon_static)

    #test with the oscillator-based method
    nodes_dynamic, _, graph_dynamic, tspan = graph_to_vector(graph, node_symbols, sa, repeats = repeats)
    recon_dynamic = query_edges(graph_dynamic, nodes_dynamic, sa, tspan)
    auroc_dynamic = auroc(graph, recon_dynamic)

    #return graph, recon_static, recon_dynamic
    return auroc_static, auroc_dynamic
end