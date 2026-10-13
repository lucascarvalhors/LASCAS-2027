#!/usr/bin/env python3
"""Diagnóstico das engines TensorRT do mlp_pure_tribranch_18cls (saídas FP32/FP16 erradas na b1).

Na campanha c1 (rep 1) as engines fp32, fp16 e fp16auto deste modelo concordaram com o ONNX
Runtime em só ~7% das previsões, enquanto a int8auto_cA (toda em FP32, mas com kernels Cask em vez
do Myelin) reproduziu o ORT até 4e-6. Este script compara cada engine com o ORT CPU (mesmo ONNX,
sem otimizações de grafo) nas 900 amostras e testa:
  1. a engine b1: determinismo, dependência da entrada não usada `time`, ordem das amostras,
     contexto novo por amostra;
  2. trtexec com as entradas da amostra 0 (harness independente do nosso runner);
  3. rebuilds com: nível de otimização 0–5, tensores intermediários marcados como saída (o que
     impede a fusão do Myelin naquele ponto), ONNX sem a entrada `time`, flag INT8 sem forçar camadas.
Rodar SÓ com a GPU livre (fora das campanhas). Saída: ~/LASCAS-2027/diag/mlp_pure_tribranch_18cls/
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import cudart  # noqa: E402
from build_engines import force_fp16_float_layers, make_calibrator  # noqa: E402

M = "mlp_pure_tribranch_18cls"
OUT = C.BASE / "diag" / M
P = "tri_none_mlp/"
INTER = {  # tensores intermediários (nome no ONNX) -> rótulo
    P + "ord_gap/Mean:0": "gap",
    P + "ord_fc/Relu:0": "ord_fc",
    P + "sc_fc2/Relu:0": "sc_fc2",
    P + "fusion/concat:0": "concat",
    P + "mlp_h0/Relu:0": "h0",
    P + "mlp_h1/Relu:0": "h1",
    P + "mlp_h2/Relu:0": "h2",
}
OUTPUT = "mlp_logits"


class MultiRunner:
    """Como runners.TrtRunner, mas aceita várias saídas (tensores intermediários marcados)."""

    def __init__(self, plan: bytes):
        self.rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.rt.deserialize_cuda_engine(plan)
        self.ctx = self.engine.create_execution_context()
        self.stream = cudart.stream_create()
        self.ev0, self.ev1 = cudart.event_create(), cudart.event_create()
        self.buf, self.inputs, self.outputs = {}, [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.ctx.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host, hptr = cudart.pinned_empty(shape, dtype)
            dev = cudart.malloc(host.nbytes)
            self.ctx.set_tensor_address(name, dev)
            self.buf[name] = (host, dev, hptr)
            is_in = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_in else self.outputs).append(name)

    def infer(self, feed: dict):
        s = self.stream
        for n in self.inputs:
            host, dev, _ = self.buf[n]
            np.copyto(host, feed[n])
            cudart.memcpy_htod_async(dev, host, s)
        cudart.event_record(self.ev0, s)
        if not self.ctx.execute_async_v3(s):
            raise RuntimeError("execute_async_v3 falhou")
        cudart.event_record(self.ev1, s)
        for n in self.outputs:
            host, dev, _ = self.buf[n]
            cudart.memcpy_dtoh_async(host, dev, s)
        cudart.stream_sync(s)
        return {n: self.buf[n][0].reshape(-1).copy() for n in self.outputs}, cudart.event_elapsed_ms(self.ev0, self.ev1)

    def run_all(self, feeds):
        outs, gpu = {n: [] for n in self.outputs}, []
        for f in feeds:
            o, g = self.infer(f)
            for n in self.outputs:
                outs[n].append(o[n])
            gpu.append(g)
        return {n: np.stack(v) for n, v in outs.items()}, float(np.median(gpu[min(50, len(gpu) // 2):]))

    def close(self):
        for _, dev, hptr in self.buf.values():
            cudart.free(dev)
            cudart.free_host(hptr)


def ort_reference(model: onnx.ModelProto, feeds) -> dict:
    """ORT CPU sem otimizações de grafo, com os intermediários de INTER como saídas extras."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    for name in INTER:
        if name not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    in_names = [i.name for i in sess.get_inputs()]
    res = {n: [] for n in names}
    for f in feeds:
        for n, v in zip(names, sess.run(None, {k: f[k] for k in in_names})):
            res[n].append(np.asarray(v, dtype=np.float32).reshape(-1))
    return {n: np.stack(v) for n, v in res.items()}


def build(onnx_bytes: bytes, prec: str = "fp32", opt_level: int | None = None, mark=(), folds=None):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(onnx_bytes):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    for i in range(net.num_inputs):
        t = net.get_input(i)
        t.shape = (1,) + tuple(t.shape)[1:]
    if mark:
        tensors = {}
        for i in range(net.num_layers):
            layer = net.get_layer(i)
            for j in range(layer.num_outputs):
                tensors[layer.get_output(j).name] = layer.get_output(j)
        for name in mark:
            net.mark_output(tensors[name])
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.clear_flag(trt.BuilderFlag.TF32)
    if opt_level is not None:
        cfg.builder_optimization_level = opt_level
    calib = None
    if prec in ("fp16", "fp16auto"):
        cfg.set_flag(trt.BuilderFlag.FP16)
        if prec == "fp16":
            cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            force_fp16_float_layers(net)
    elif prec == "int8auto":
        xs, _, _, _ = C.load_eval_set(M)
        calib = make_calibrator("entropy", {k: np.ascontiguousarray(v[folds["A"]]) for k, v in xs.items()},
                                OUT / "tmp_int8auto.calib")
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = calib
    t0 = time.monotonic()
    ser = builder.build_serialized_network(net, cfg)
    build_s = time.monotonic() - t0
    if calib is not None:
        calib.release()
    if ser is None:
        raise RuntimeError("build falhou")
    return bytes(ser), build_s


def engine_info(plan: bytes) -> dict:
    eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = [l for l in info.get("Layers", []) if isinstance(l, dict)]
    return {"n_layers": len(layers), "layer_types": dict(Counter(l.get("LayerType") for l in layers)),
            "kgen": [l["Name"].split("_myl")[0] for l in layers if l.get("LayerType") == "kgen"]}


def evaluate(label: str, plan: bytes, feeds, ref: dict, y, tol: float) -> dict:
    r = MultiRunner(plan)
    outs, gpu_ms = r.run_all(feeds)
    r.close()
    lg, rl = outs[OUTPUT], ref[OUTPUT]
    d = np.abs(lg.astype(np.float64) - rl)
    res = {"label": label, "max_abs": float(d.max()), "argmax_agree": float(np.mean(lg.argmax(1) == rl.argmax(1))),
           "accuracy": float(np.mean(lg.argmax(1) == y)), "gpu_ms_median": round(gpu_ms, 4), **engine_info(plan)}
    inter = {lab: float(np.abs(outs[n].astype(np.float64) - ref[n]).max()) for n, lab in INTER.items() if n in outs}
    if inter:
        res["inter_max_abs"] = inter
    res["ok"] = bool(res["max_abs"] < tol and res["argmax_agree"] >= 0.99)
    fused = [k for k in res["kgen"] if "RelResMulSum" in k]
    print(f"  {'OK ' if res['ok'] else 'ERR'} {label:38s} |Δ|máx={res['max_abs']:9.2e} argmax={res['argmax_agree']:.4f} "
          f"acc={res['accuracy']:.4f} gpu={gpu_ms:.4f} ms kgen={len(res['kgen'])} fusões_duplas={fused}"
          + (f" inter={ {k: f'{v:.1e}' for k, v in inter.items()} }" if inter else ""), flush=True)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plans").mkdir(exist_ok=True)
    xs, y, ref_keras, _ = C.load_eval_set(M)
    N = len(y)
    feeds = [{n: xs[n][i:i + 1] for n in C.INPUT_NAMES} for i in range(N)]
    _, y0, _, _ = C.load_eval_set(next(iter(C.MODELS)))
    fa, fb = C.make_folds(y0)
    folds = {"A": fa, "B": fb}
    model = onnx.load(str(C.EXPORT / M / "model.onnx"))
    report = {"trt": trt.__version__, "ort": ort.__version__, "versions": C.versions(), "env": C.snapshot(),
              "nvpmodel": C.nvpmodel_mode(), "clocks": C.clocks_locked(), "tests": {}, "builds": []}

    print("referência ORT CPU (sem otimizações de grafo)...", flush=True)
    ref = ort_reference(model, feeds)
    k = np.abs(ref[OUTPUT].astype(np.float64) - ref_keras).max()
    print(f"  ORT vs Keras: |Δ|máx = {k:.2e}, acc = {np.mean(ref[OUTPUT].argmax(1) == y):.4f}", flush=True)
    report["ort_vs_keras_max_abs"] = float(k)

    print("\n1) engines da b1", flush=True)
    b1 = C.ENGINES / "b1" / M
    plans = {v: (b1 / f"{v}.plan").read_bytes() for v in ("fp32", "fp16", "fp16auto", "int8auto_cA")}
    for v, plan in plans.items():
        report["builds"].append(evaluate(f"b1/{v}", plan, feeds, ref, y, 1e-3 if v != "fp16" else 0.2))

    r = MultiRunner(plans["fp32"])
    base = r.run_all(feeds)[0][OUTPUT]
    again = r.run_all(feeds)[0][OUTPUT]
    rng = np.random.default_rng(0)
    t_zero = r.run_all([{**f, "time": np.zeros_like(f["time"])} for f in feeds])[0][OUTPUT]
    t_rand = r.run_all([{**f, "time": rng.normal(0, 100, f["time"].shape).astype(np.float32)} for f in feeds])[0][OUTPUT]
    rev = r.run_all(feeds[::-1])[0][OUTPUT][::-1]
    r.close()
    fresh = []
    for i in range(20):
        rr = MultiRunner(plans["fp32"])
        fresh.append(rr.infer(feeds[i])[0][OUTPUT])
        rr.close()
    fresh = np.stack(fresh)
    tests = {"repetição bit a bit idêntica": bool(np.array_equal(base, again)),
             "|Δ| com time=0": float(np.abs(base - t_zero).max()),
             "|Δ| com time aleatório": float(np.abs(base - t_rand).max()),
             "|Δ| ordem inversa": float(np.abs(base - rev).max()),
             "|Δ| contexto novo por amostra (20)": float(np.abs(base[:20] - fresh).max())}
    for kk, vv in tests.items():
        print(f"  b1/fp32 — {kk}: {vv}", flush=True)
    report["tests"]["b1_fp32"] = tests

    print("\n2) trtexec com as entradas da amostra 0", flush=True)
    for n in C.INPUT_NAMES:
        feeds[0][n].astype(np.float32).tofile(OUT / f"s0_{n}.bin")
    for v in ("fp32", "int8auto_cA"):
        oj = OUT / f"trtexec_out_s0_{v}.json"
        cmd = [C.TRTEXEC, f"--loadEngine={b1 / f'{v}.plan'}",
               "--loadInputs=" + ",".join(f"{n}:{OUT / f's0_{n}.bin'}" for n in C.INPUT_NAMES),
               f"--exportOutput={oj}", "--iterations=1", "--warmUp=0", "--duration=0", "--avgRuns=1"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        try:
            vals = np.array([o for o in json.loads(oj.read_text()) if o["name"] == OUTPUT][0]["values"], dtype=np.float64)
            dt = float(np.abs(vals - ref[OUTPUT][0]).max())
            dr = float(np.abs(vals - (base[0] if v == "fp32" else vals)).max())
            print(f"  trtexec {v}: |Δ vs ORT| = {dt:.2e}; |Δ vs nosso runner| = {dr:.2e} (rc={p.returncode})", flush=True)
            report["tests"][f"trtexec_s0_{v}"] = {"vs_ort": dt, "vs_runner": dr, "rc": p.returncode}
        except Exception as e:  # noqa: BLE001
            print(f"  trtexec {v}: falhou ({e}); rc={p.returncode}\n{p.stdout[-1500:]}{p.stderr[-800:]}", flush=True)

    print("\n3) rebuilds (sem timing cache)", flush=True)
    raw = (C.EXPORT / M / "model.onnx").read_bytes()
    nt = onnx.ModelProto()
    nt.CopyFrom(model)
    keep = [i for i in nt.graph.input if i.name != "time"]
    del nt.graph.input[:]
    nt.graph.input.extend(keep)
    notime = nt.SerializeToString()
    h1 = P + "mlp_h1/Relu:0"
    exps = [("fp32", {}), *[(f"fp32 opt{k}", {"opt_level": k}) for k in (0, 1, 2, 4, 5)],
            ("fp32 sem entrada time", {"onnx": notime}),
            ("fp32 marca todos intermediários", {"mark": list(INTER)}),
            *[(f"fp32 marca {lab}", {"mark": [n]}) for n, lab in INTER.items()],
            ("fp32 flag INT8 (=int8auto)", {"prec": "int8auto"}),
            ("fp16auto", {"prec": "fp16auto"}), ("fp16auto marca h1", {"prec": "fp16auto", "mark": [h1]}),
            ("fp16", {"prec": "fp16"}), ("fp16 marca h1", {"prec": "fp16", "mark": [h1]}),
            ("fp16 opt0", {"prec": "fp16", "opt_level": 0})]
    for label, kw in exps:
        try:
            plan, bs = build(kw.pop("onnx", raw), folds=folds, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"  FALHOU build {label}: {e}", flush=True)
            report["builds"].append({"label": label, "error": str(e)})
            continue
        (OUT / "plans" / (label.replace(" ", "_") + ".plan")).write_bytes(plan)
        tol = 0.2 if label.startswith("fp16") and "auto" not in label else 1e-3
        res = evaluate(label, plan, feeds, ref, y, tol)
        res["build_s"] = round(bs, 2)
        report["builds"].append(res)

    (OUT / "diag_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\nrelatório: {OUT / 'diag_report.json'}", flush=True)


if __name__ == "__main__":
    main()
