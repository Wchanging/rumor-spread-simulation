from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from data.loaders import RawComment, RawPost


@dataclass
class Network:
    adjacency: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, node_id: str) -> None:
        self.adjacency.setdefault(node_id, set())

    def add_edge(self, src: str, dst: str) -> None:
        self.add_node(src)
        self.add_node(dst)
        if src != dst:
            self.adjacency[src].add(dst)

    def neighbors(self, node_id: str) -> list[str]:
        return list(self.adjacency.get(node_id, set()))

    def nodes(self) -> list[str]:
        return list(self.adjacency.keys())

    def edge_count(self) -> int:
        return sum(len(targets) for targets in self.adjacency.values())

    def to_dict(self) -> dict[str, list[str]]:
        return {node: sorted(list(targets)) for node, targets in self.adjacency.items()}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "Network":
        network = cls()
        for src, targets in data.items():
            network.add_node(src)
            for dst in targets:
                network.add_edge(src, dst)
        return network


class NetworkBuilder:
    def build_from_user_follow(self, follow_map: dict[str, Iterable[str]]) -> Network:
        network = Network()
        for user_id, followees in follow_map.items():
            network.add_node(user_id)
            for target in followees:
                network.add_edge(user_id, target)
        return network

    def build_from_interaction(self, posts: list[RawPost], comments: list[RawComment]) -> Network:
        network = Network()
        post_authors: dict[str, str] = {}

        for post in posts:
            network.add_node(post.user_id)
            post_authors[post.post_id] = post.user_id

        for comment in comments:
            network.add_node(comment.user_id)
            post_author = post_authors.get(comment.post_id)
            if post_author:
                network.add_edge(comment.user_id, post_author)
                network.add_edge(post_author, comment.user_id)

        return network

    def build_synthetic(
        self,
        network_type: str,
        n: int,
        params: dict | None = None,
        seed: int | None = None,
    ) -> Network:
        params = params or {}
        if seed is not None:
            random.seed(seed)

        normalized_type = str(network_type or "").strip().lower()
        alias_map = {
            "erdos_renyi": "random",
            "er": "random",
        }
        # get normalized type from alias map, default to itself if not found
        normalized_type = alias_map.get(normalized_type, normalized_type)

        if normalized_type == "small_world":
            return self._build_small_world(
                n=n,
                k=int(params.get("k", 4)),
                p=float(params.get("p", 0.1)),
            )
        if normalized_type == "scale_free":
            return self._build_scale_free(
                n=n,
                m=int(params.get("m", 2)),
            )
        if normalized_type == "random":
            return self._build_random(
                n=n,
                p=float(params.get("p", 0.01)),
            )
        raise ValueError(
            f"Unsupported network_type: {network_type}. "
            "Supported: small_world, scale_free, random (aliases: erdos_renyi, er)"
        )

    def _build_small_world(self, n: int, k: int, p: float) -> Network:
        network = Network()
        k = max(2, k)
        if k % 2 == 1:
            k += 1

        nodes = [f"u_{idx}" for idx in range(n)]
        for node in nodes:
            network.add_node(node)

        half_k = k // 2
        for i, src in enumerate(nodes):
            for offset in range(1, half_k + 1):
                dst = nodes[(i + offset) % n]
                network.add_edge(src, dst)
                network.add_edge(dst, src)

        for src in nodes:
            neighbors = list(network.adjacency[src])
            for dst in neighbors:
                if random.random() < p:
                    network.adjacency[src].discard(dst)
                    candidates = [node for node in nodes if node != src and node not in network.adjacency[src]]
                    if candidates:
                        new_dst = random.choice(candidates)
                        network.add_edge(src, new_dst)
        return network

    def _build_scale_free(self, n: int, m: int) -> Network:
        m = max(1, m)
        network = Network()
        nodes = [f"u_{idx}" for idx in range(n)]
        if n == 0:
            return network

        initial = min(max(2, m + 1), n)
        for i in range(initial):
            src = nodes[i]
            network.add_node(src)
            for j in range(initial):
                if i != j:
                    network.add_edge(src, nodes[j])

        for idx in range(initial, n):
            new_node = nodes[idx]
            network.add_node(new_node)
            degree_map = {node: len(network.adjacency[node]) for node in network.nodes()}
            total_degree = sum(degree_map.values())

            targets: set[str] = set()
            while len(targets) < min(m, len(degree_map)):
                if total_degree <= 0:
                    candidates = [node for node in network.nodes() if node != new_node]
                    if not candidates:
                        break
                    targets.add(random.choice(candidates))
                    continue

                threshold = random.uniform(0, total_degree)
                cumulative = 0.0
                for node, degree in degree_map.items():
                    if node == new_node:
                        continue
                    cumulative += degree
                    if cumulative >= threshold:
                        targets.add(node)
                        break

            for target in targets:
                network.add_edge(new_node, target)
                network.add_edge(target, new_node)

        return network

    def _build_random(self, n: int, p: float) -> Network:
        network = Network()
        nodes = [f"u_{idx}" for idx in range(n)]
        for node in nodes:
            network.add_node(node)

        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                if random.random() < p:
                    network.add_edge(src, dst)

        return network


def indegree_map(network: Network) -> dict[str, int]:
    indegree = defaultdict(int)
    for node in network.nodes():
        indegree[node] += 0
    for _, targets in network.adjacency.items():
        for target in targets:
            indegree[target] += 1
    return dict(indegree)


def relabel_network_nodes(network: Network, new_node_ids: list[str]) -> Network:
    old_nodes = network.nodes()
    if len(old_nodes) != len(new_node_ids):
        raise ValueError(
            f"relabel_network_nodes requires node lists of equal length; current old={len(old_nodes)}, new={len(new_node_ids)}"
        )

    mapping = {old: new for old, new in zip(old_nodes, new_node_ids)}
    relabeled = Network()
    for old_src, old_targets in network.adjacency.items():
        src = mapping[old_src]
        relabeled.add_node(src)
        for old_dst in old_targets:
            relabeled.add_edge(src, mapping[old_dst])
    return relabeled


def save_network(network: Network, file_path: str | Path, metadata: dict | None = None) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "adjacency": network.to_dict(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_network(file_path: str | Path) -> Network:
    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    adjacency = payload.get("adjacency", payload)
    if not isinstance(adjacency, dict):
        raise ValueError("Invalid network file format: adjacency field not found")
    return Network.from_dict(adjacency)
