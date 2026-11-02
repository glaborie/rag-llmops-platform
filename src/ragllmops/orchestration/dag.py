"""Declarative DAG definition.

An asset is a unit of work with a name, dependencies, and a function. The
engine decides the order, the cache key and the retry policy; the asset
itself only describes *what* it does.

This is deliberately small. The same concepts (idempotent assets, partitioned
runs, retries, lineage, sensors) are what Dagster, Prefect and Airflow give
you; the point of writing them out is that you can see exactly what your
pipeline guarantees.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AssetContext:
    """What an asset function receives.

    ``results`` holds the outputs of the assets that ran before it, keyed by
    asset name.
    """

    platform: Any
    results: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def result(self, name: str, default: Any = None) -> Any:
        """Fetch an upstream result."""
        return self.results.get(name, default)


@dataclass(frozen=True, slots=True)
class Asset:
    """A single node in the pipeline."""

    name: str
    fn: Callable[[AssetContext], Any]
    deps: tuple[str, ...] = ()
    schedule: str | None = None
    retries: int = 0
    retry_delay_seconds: float = 1.0
    cache: bool = True
    tags: tuple[str, ...] = ()
    description: str = ""

    def code_fingerprint(self) -> str:
        """Hash of the asset function's source code."""
        try:
            source = inspect.getsource(self.fn)
        except (OSError, TypeError):  # pragma: no cover - builtins
            source = repr(self.fn)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


class DAG:
    """A directed acyclic graph of assets."""

    def __init__(self, name: str, assets: list[Asset]) -> None:
        self.name = name
        self.assets = {asset.name: asset for asset in assets}
        self._validate()

    def _validate(self) -> None:
        """Fail fast on unknown dependencies and cycles."""
        for asset in self.assets.values():
            for dep in asset.deps:
                if dep not in self.assets:
                    raise ValueError(f"asset {asset.name!r} depends on unknown asset {dep!r}")

        self.topological_order()  # raises on a cycle

    def topological_order(self) -> list[Asset]:
        """Assets ordered so that dependencies come first."""
        visited: dict[str, int] = {}
        order: list[Asset] = []

        def visit(name: str, stack: tuple[str, ...]) -> None:
            state = visited.get(name, 0)
            if state == 1:
                cycle = " -> ".join([*stack, name])
                raise ValueError(f"cycle detected in DAG {self.name!r}: {cycle}")
            if state == 2:
                return
            visited[name] = 1
            for dep in self.assets[name].deps:
                visit(dep, (*stack, name))
            visited[name] = 2
            order.append(self.assets[name])

        for asset_name in self.assets:
            visit(asset_name, ())
        return order

    def get(self, name: str) -> Asset:
        return self.assets[name]

    def upstream_of(self, name: str) -> set[str]:
        """All transitive dependencies of an asset."""
        seen: set[str] = set()
        stack = list(self.assets[name].deps)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.assets[current].deps)
        return seen

    def upstream_of_all(self, names: set[str]) -> set[str]:
        """All transitive dependencies of a set of assets."""
        seen: set[str] = set()
        for name in names:
            seen |= self.upstream_of(name)
        return seen

    def downstream_of(self, name: str) -> set[str]:
        """All assets that (transitively) depend on an asset."""
        seen: set[str] = set()
        for asset in self.assets.values():
            if name in self.upstream_of(asset.name):
                seen.add(asset.name)
        return seen

    def describe(self) -> list[dict[str, Any]]:
        """A serialisable description of the graph."""
        return [
            {
                "name": asset.name,
                "deps": list(asset.deps),
                "schedule": asset.schedule,
                "retries": asset.retries,
                "cache": asset.cache,
                "tags": list(asset.tags),
                "description": asset.description,
            }
            for asset in self.topological_order()
        ]
