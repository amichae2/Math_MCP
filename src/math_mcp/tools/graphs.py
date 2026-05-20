"""Graph theory tools using NetworkX."""

from typing import Any

import networkx as nx
import numpy as np

from ..utils.errors import tool_error_handler


def _make_graph(graph_type: str) -> nx.Graph | nx.DiGraph:
    normalized = graph_type.lower()
    if normalized in {"undirected", "graph"}:
        return nx.Graph()
    if normalized in {"directed", "digraph"}:
        return nx.DiGraph()
    raise ValueError("graph_type must be 'undirected' or 'directed'")


def _normalize_edge(edge: list[Any] | tuple[Any, ...]) -> tuple[Any, Any]:
    if len(edge) != 2:
        raise ValueError("each edge must contain exactly two endpoints")
    return edge[0], edge[1]


def _build_graph(
    edges: list[list[Any]],
    *,
    directed: bool = False,
    weights: list[float] | None = None,
    edge_weights_attr: str = "weight",
) -> nx.Graph | nx.DiGraph:
    graph = nx.DiGraph() if directed else nx.Graph()
    normalized_edges = [_normalize_edge(edge) for edge in edges]
    if weights is not None and len(weights) != len(normalized_edges):
        raise ValueError("weights must match the number of edges")
    for index, (u_node, v_node) in enumerate(normalized_edges):
        attributes = {}
        if weights is not None:
            attributes[edge_weights_attr] = float(weights[index])
        graph.add_edge(u_node, v_node, **attributes)
    return graph


def _sample_edges(graph: nx.Graph | nx.DiGraph, limit: int = 10) -> list[list[Any]]:
    return [[u_node, v_node] for u_node, v_node in list(graph.edges())[:limit]]


@tool_error_handler("graph_create")
async def graph_create(
    specification: str,
    graph_type: str = "undirected",
    n: int | None = None,
    edges: list[list[Any]] | None = None,
    adjacency: list[list[float]] | None = None,
    edge_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Create a graph from a named specification, edge list, or adjacency matrix."""
    graph = _make_graph(graph_type)
    spec = specification.lower()

    if spec == "edge_list":
        if edges is None:
            raise ValueError("edges are required for specification='edge_list'")
        weighted_graph = _build_graph(edges, directed=graph.is_directed(), weights=edge_weights)
        graph = weighted_graph
    elif spec == "adjacency":
        if adjacency is None:
            raise ValueError("adjacency is required for specification='adjacency'")
        adjacency_matrix = np.asarray(adjacency, dtype=float)
        if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        graph = nx.from_numpy_array(adjacency_matrix, create_using=graph)
    elif spec == "path":
        if n is None or n < 1:
            raise ValueError("n must be a positive integer for a path graph")
        graph = nx.path_graph(n, create_using=graph)
    elif spec == "cycle":
        if n is None or n < 3:
            raise ValueError("n must be at least 3 for a cycle graph")
        graph = nx.cycle_graph(n, create_using=graph)
    elif spec == "complete":
        if n is None or n < 1:
            raise ValueError("n must be a positive integer for a complete graph")
        graph = nx.complete_graph(n, create_using=graph)
    elif spec == "star":
        if n is None or n < 2:
            raise ValueError("n must be at least 2 for a star graph")
        graph = nx.star_graph(n - 1)
        if graph_type.lower() in {"directed", "digraph"}:
            graph = nx.DiGraph(graph)
    elif spec == "empty":
        if n is None or n < 0:
            raise ValueError("n must be non-negative for an empty graph")
        graph.add_nodes_from(range(n))
    else:
        raise ValueError("specification must be one of: edge_list, adjacency, path, cycle, complete, star, empty")

    return {
        "result": f"Created {graph_type} graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges",
        "graph_type": graph_type.lower(),
        "nodes": list(graph.nodes())[:20],
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "edges_sample": _sample_edges(graph),
        "latex": None,
    }


@tool_error_handler("shortest_path")
async def shortest_path(
    edges: list[list[Any]],
    source: int | str,
    target: int | str | None = None,
    weights: list[float] | None = None,
    algorithm: str = "dijkstra",
    edge_weights_attr: str = "weight",
) -> dict[str, Any]:
    """Compute shortest paths on an edge-list graph."""
    algo = algorithm.lower()
    if algo not in {"dijkstra", "bellman-ford", "unweighted"}:
        raise ValueError("algorithm must be one of: dijkstra, bellman-ford, unweighted")
    graph = _build_graph(edges, directed=False, weights=weights, edge_weights_attr=edge_weights_attr)
    if source not in graph:
        raise ValueError("source node is not present in the graph")
    if target is not None and target not in graph:
        raise ValueError("target node is not present in the graph")

    try:
        if target is None:
            if algo == "dijkstra":
                lengths, paths = nx.single_source_dijkstra(graph, source, weight=edge_weights_attr)
            elif algo == "bellman-ford":
                lengths, paths = nx.single_source_bellman_ford(graph, source, weight=edge_weights_attr)
            else:
                lengths = nx.single_source_shortest_path_length(graph, source)
                paths = nx.single_source_shortest_path(graph, source)
            return {
                "result": f"Shortest paths from {source} to {len(lengths) - 1} reachable nodes",
                "source": source,
                "distances": {str(node): float(distance) if isinstance(distance, (int, float)) else distance for node, distance in lengths.items()},
                "paths": {str(node): path for node, path in list(paths.items())[:10]},
                "latex": None,
            }

        if algo == "dijkstra":
            distance = nx.dijkstra_path_length(graph, source, target, weight=edge_weights_attr)
            path = nx.dijkstra_path(graph, source, target, weight=edge_weights_attr)
        elif algo == "bellman-ford":
            distance = nx.bellman_ford_path_length(graph, source, target, weight=edge_weights_attr)
            path = nx.bellman_ford_path(graph, source, target, weight=edge_weights_attr)
        else:
            distance = nx.shortest_path_length(graph, source, target)
            path = nx.shortest_path(graph, source, target)
    except nx.NetworkXNoPath as error:
        raise ValueError("no path exists between the requested source and target") from error
    except nx.NetworkXUnbounded as error:
        raise ValueError("negative cycle detected; shortest path is undefined") from error

    return {
        "result": f"Shortest path from {source} to {target} has length {distance}",
        "source": source,
        "target": target,
        "distance": float(distance),
        "path": path,
        "algorithm": algo,
        "latex": None,
    }


@tool_error_handler("spanning_tree")
async def spanning_tree(
    edges: list[list[Any]],
    weights: list[float] | None = None,
    maximize: bool = False,
    edge_weights_attr: str = "weight",
) -> dict[str, Any]:
    """Compute a minimum or maximum spanning tree."""
    graph = _build_graph(edges, directed=False, weights=weights, edge_weights_attr=edge_weights_attr)
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one node")
    if not nx.is_connected(graph):
        raise ValueError("graph must be connected to compute a spanning tree")

    if maximize:
        tree = nx.maximum_spanning_tree(graph, weight=edge_weights_attr)
        mode = "maximum"
    else:
        tree = nx.minimum_spanning_tree(graph, weight=edge_weights_attr)
        mode = "minimum"
    total_weight = sum(data.get(edge_weights_attr, 1.0) for _, _, data in tree.edges(data=True))
    return {
        "result": f"Computed {mode} spanning tree with total weight {total_weight}",
        "edges": [[u_node, v_node, float(data.get(edge_weights_attr, 1.0))] for u_node, v_node, data in tree.edges(data=True)],
        "total_weight": float(total_weight),
        "num_edges": int(tree.number_of_edges()),
        "latex": None,
    }


@tool_error_handler("graph_metrics")
async def graph_metrics(
    edges: list[list[Any]],
    weights: list[float] | None = None,
    metrics: list[str] | None = None,
    edge_weights_attr: str = "weight",
) -> dict[str, Any]:
    """Compute common graph-theoretic summary metrics."""
    graph = _build_graph(edges, directed=False, weights=weights, edge_weights_attr=edge_weights_attr)
    requested = {metric.lower() for metric in (metrics or ["density", "degree_centrality", "clustering", "connected_components", "diameter"])}
    result: dict[str, Any] = {
        "result": f"Computed {len(requested)} graph metrics",
        "latex": None,
    }

    if "density" in requested:
        result["density"] = float(nx.density(graph))
    if "degree_centrality" in requested:
        result["degree_centrality"] = {str(node): float(value) for node, value in nx.degree_centrality(graph).items()}
    if "clustering" in requested:
        result["clustering"] = {str(node): float(value) for node, value in nx.clustering(graph, weight=edge_weights_attr).items()}
    if "connected_components" in requested:
        result["connected_components"] = [sorted(component, key=str) for component in nx.connected_components(graph)]
    if "diameter" in requested:
        if nx.is_connected(graph):
            result["diameter"] = int(nx.diameter(graph))
        else:
            result["diameter"] = None
            result["diameter_warning"] = "graph is disconnected; diameter is undefined"
    if "betweenness_centrality" in requested:
        result["betweenness_centrality"] = {
            str(node): float(value)
            for node, value in nx.betweenness_centrality(graph, weight=edge_weights_attr).items()
        }
    return result


@tool_error_handler("max_flow")
async def max_flow(
    edges: list[list[Any]],
    source: int | str,
    sink: int | str,
    capacities: list[float],
) -> dict[str, Any]:
    """Compute maximum flow in a capacitated directed graph."""
    if source == sink:
        raise ValueError("source and sink must be different")
    graph = _build_graph(edges, directed=True, weights=capacities, edge_weights_attr="capacity")
    if source not in graph:
        raise ValueError("source node is not present in the graph")
    if sink not in graph:
        raise ValueError("sink node is not present in the graph")
    flow_value, flow_dict = nx.maximum_flow(graph, source, sink, capacity="capacity")
    cut_value, partition = nx.minimum_cut(graph, source, sink, capacity="capacity")
    return {
        "result": f"Maximum flow from {source} to {sink} is {flow_value}",
        "max_flow": float(flow_value),
        "flow_dict": {str(node): {str(target): float(value) for target, value in targets.items()} for node, targets in flow_dict.items()},
        "min_cut_value": float(cut_value),
        "min_cut_partition": [sorted(list(part), key=str) for part in partition],
        "latex": None,
    }


def register(server: Any) -> None:
    """Register all graph tools."""
    server.tool("graph_create")(graph_create)
    server.tool("shortest_path")(shortest_path)
    server.tool("spanning_tree")(spanning_tree)
    server.tool("graph_metrics")(graph_metrics)
    server.tool("max_flow")(max_flow)