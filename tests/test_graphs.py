"""Tests for graph theory tools."""

import pytest

from math_mcp.tools.graphs import (
    graph_create,
    graph_metrics,
    max_flow,
    shortest_path,
    spanning_tree,
)


# --- graph_create ---


@pytest.mark.asyncio
async def test_graph_create_path() -> None:
    result = await graph_create(specification="path", n=5, graph_type="undirected")
    assert result["num_nodes"] == 5
    assert result["num_edges"] == 4


@pytest.mark.asyncio
async def test_graph_create_complete() -> None:
    result = await graph_create(specification="complete", n=4, graph_type="undirected")
    assert result["num_nodes"] == 4
    assert result["num_edges"] == 6


@pytest.mark.asyncio
async def test_graph_create_edge_list() -> None:
    result = await graph_create(
        specification="edge_list",
        edges=[[0, 1], [1, 2], [2, 0]],
        graph_type="undirected",
    )
    assert result["num_nodes"] == 3
    assert result["num_edges"] == 3


# --- shortest_path ---


@pytest.mark.asyncio
async def test_shortest_path_simple() -> None:
    result = await shortest_path(
        edges=[[0, 1], [1, 2], [2, 3]],
        source=0,
        target=3,
        algorithm="unweighted",
    )
    assert result["distance"] == 3.0
    assert result["path"] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_shortest_path_weighted() -> None:
    result = await shortest_path(
        edges=[[0, 1], [1, 2], [0, 2]],
        source=0,
        target=2,
        weights=[10.0, 2.0, 1.0],
    )
    assert result["distance"] == 1.0
    assert result["path"] == [0, 2]


# --- spanning_tree ---


@pytest.mark.asyncio
async def test_spanning_tree_minimum() -> None:
    edges = [[0, 1], [1, 2], [2, 0]]
    weights = [1.0, 2.0, 3.0]
    result = await spanning_tree(edges=edges, weights=weights)
    assert result["num_edges"] == 2


# --- graph_metrics ---


@pytest.mark.asyncio
async def test_graph_metrics_path() -> None:
    result = await graph_metrics(edges=[[0, 1], [1, 2], [2, 3]])
    assert result["diameter"] == 3
    assert len(result["connected_components"]) == 1


# --- max_flow ---


@pytest.mark.asyncio
async def test_max_flow_simple() -> None:
    result = await max_flow(
        edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
        source=0,
        sink=3,
        capacities=[10.0, 5.0, 4.0, 8.0, 6.0],
    )
    assert result["max_flow"] == 14.0
    assert result["min_cut_value"] == 14.0
