import pytest

from math_mcp.tools.graphs import (
    graph_create,
    graph_metrics,
    max_flow,
    shortest_path,
    spanning_tree,
)


@pytest.mark.asyncio
async def test_graph_create_path_graph_returns_counts() -> None:
    result = await graph_create("path", n=4)

    assert result["num_nodes"] == 4
    assert result["num_edges"] == 3


@pytest.mark.asyncio
async def test_shortest_path_returns_weighted_path() -> None:
    result = await shortest_path(
        [["A", "B"], ["B", "C"], ["A", "C"]],
        source="A",
        target="C",
        weights=[1.0, 1.0, 3.0],
    )

    assert result["distance"] == pytest.approx(2.0)
    assert result["path"] == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_spanning_tree_returns_minimum_weight_tree() -> None:
    result = await spanning_tree(
        [[0, 1], [1, 2], [0, 2]],
        weights=[1.0, 2.0, 10.0],
    )

    assert result["num_edges"] == 2
    assert result["total_weight"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_graph_metrics_returns_density_and_components() -> None:
    result = await graph_metrics([[0, 1], [1, 2], [3, 4]], metrics=["density", "connected_components"])

    assert result["density"] == pytest.approx(0.3)
    assert len(result["connected_components"]) == 2


@pytest.mark.asyncio
async def test_max_flow_returns_expected_value() -> None:
    result = await max_flow(
        [["s", "a"], ["s", "b"], ["a", "t"], ["b", "t"], ["a", "b"]],
        source="s",
        sink="t",
        capacities=[3.0, 2.0, 2.0, 3.0, 1.0],
    )

    assert result["max_flow"] == pytest.approx(5.0)