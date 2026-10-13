"""Atribui o tempo do perfil por camada (trtexec --exportProfile) aos blocos do modelo.

Cada camada da engine traz em "Metadata" (inspector do TensorRT 10) os nós ONNX que foram
fundidos nela. Cada nó ONNX é classificado pelo escopo Keras do nome
(stem/b#/last_conv/gap_time -> backbone, ord_ -> orders, sc_ -> scalars, fusion,
kan_/mlp_ -> head). Nós sem escopo (Transpose/LayerNorm gerados pelo tf2onnx) herdam o bloco
do vizinho escopado mais próximo no grafo (primeiro a jusante, depois a montante).
Camadas fundidas que cruzam blocos dividem o tempo na proporção dos nós de cada bloco;
camadas de reformatação (troca de formato/precisão, sem nó ONNX) vão para "reformat".

O profiler sincroniza a cada camada e infla o total; por isso o resultado é usado como
fração do tempo, reescalada para o tempo de GPU medido sem profiler (trtexec GPU Compute).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

import onnx

BLOCKS = ("backbone", "orders", "scalars", "fusion", "head", "reformat", "other")
META_RE = re.compile(r"\[ONNX Layer: ([^\]]+)\]")


def scope_block(name: str):
    parts = name.split("/")
    if len(parts) < 2:
        return None
    s = parts[1]
    if re.match(r"(stem|b\d+_|last_conv|gap_time)", s):
        return "backbone"
    if s.startswith("ord_"):
        return "orders"
    if s.startswith("sc_"):
        return "scalars"
    if s.startswith("fusion"):
        return "fusion"
    if s.startswith(("kan_", "mlp_")):
        return "head"
    return None


def onnx_node_blocks(onnx_path: Path) -> dict:
    g = onnx.load(str(onnx_path)).graph
    producer = {o: n.name for n in g.node for o in n.output}
    consumers = defaultdict(list)
    for n in g.node:
        for i in n.input:
            consumers[i].append(n.name)
    nodes = {n.name: n for n in g.node}
    block = {n.name: scope_block(n.name) for n in g.node}

    def bfs(start, downstream: bool):
        seen, q = {start}, deque([start])
        while q:
            cur = q.popleft()
            n = nodes[cur]
            nxt = ([c for o in n.output for c in consumers[o]] if downstream
                   else [producer[i] for i in n.input if i in producer])
            for x in nxt:
                if x in seen:
                    continue
                if block.get(x):
                    return block[x]
                seen.add(x)
                q.append(x)
        return None

    resolved = dict(block)
    for name, b in block.items():
        if b is None:
            resolved[name] = bfs(name, True) or bfs(name, False) or "other"
    return resolved


def layer_fractions(layer: dict, node_block: dict) -> dict:
    nodes = META_RE.findall(layer.get("Metadata") or "")
    if nodes:
        c = Counter(node_block.get(n, "other") for n in nodes)
        tot = sum(c.values())
        return {k: v / tot for k, v in c.items()}
    name = layer.get("Name", "")
    if "Reformat" in name or "CopyNode" in name or name.startswith("__myl_Mov"):
        return {"reformat": 1.0}
    for tok in re.findall(r"[A-Za-z]+__\d+", name):      # ex.: reshape_before_Gemm__7816
        if tok in node_block:
            return {node_block[tok]: 1.0}
    return {"other": 1.0}


def breakdown(profile_json: Path, layers_json: Path, node_block: dict) -> dict:
    """Devolve {bloco: fração do tempo profilado} e o total profilado (ms por inferência)."""
    prof = [r for r in json.loads(Path(profile_json).read_text()) if "name" in r]
    info = json.loads(Path(layers_json).read_text())
    by_name = {l["Name"]: l for l in info["Layers"] if isinstance(l, dict)}
    acc = defaultdict(float)
    unmatched = 0.0
    for r in prof:
        layer = by_name.get(r["name"])
        if layer is None:
            fr = {"other": 1.0}
            unmatched += r["averageMs"]
        else:
            fr = layer_fractions(layer, node_block)
        for b, f in fr.items():
            acc[b] += f * r["averageMs"]
    total = sum(acc.values())
    return {"fractions": {b: acc.get(b, 0.0) / total for b in BLOCKS}, "profiled_ms": total,
            "unmatched_ms": unmatched}
