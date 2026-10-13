#!/usr/bin/env python3
"""Constrói as engines TensorRT (batch = 1, formas estáticas) dos 6 modelos.

Precisões (TF32 desligado em todas, para que "FP32" seja FP32 de verdade):
  fp32         FP32
  fp16         FP16 forçado: toda camada de ponto flutuante com restrição FP16
               (PREFER_PRECISION_CONSTRAINTS; o TensorRT só cai para FP32 onde não há kernel)
  fp16auto     só a flag FP16 (como trtexec --fp16): o TensorRT escolhe por camada, pelo tempo
  int8_cA      INT8 PTQ forçado: Conv/MatMul obrigatoriamente em INT8 (escalas do
               IInt8EntropyCalibrator2, fold A), demais camadas em FP32
  int8_cB      idem, calibrado no fold B
  int8auto_cA  só a flag INT8 (como trtexec --int8), calibrado no fold A: o TensorRT decide.
  int8mm_cA/B  como int8_cA/B, mas com IInt8MinMaxCalibrator (ablação do método de calibração)
Em batch 1, os modos "auto" escolheram FP32 em todas as camadas do KAN full (FP16/INT8 não
compensam para camadas tão pequenas) — por isso as variantes forçadas medem a sensibilidade.
Folds A/B: divisão estratificada das 900 amostras (25 por classe em cada fold,
seed fixa). A accuracy INT8 "fora da amostra" usa a engine cA no fold B e a cB no fold A,
de modo que nenhuma amostra é avaliada por uma engine calibrada com ela.

Saída: engines/<build-id>/<modelo>/<precisão>.plan, .layers.json (precisão por camada),
.build.json (tempo, tamanho, avisos) e int8*.calib (cache de calibração).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

import common as C
import cudart


class CaptureLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)
        self.msgs = []

    def log(self, severity, msg):
        if severity <= trt.ILogger.Severity.WARNING:
            self.msgs.append(f"{severity.name}: {msg}")
            print(f"    [TRT {severity.name}] {msg}", flush=True)


def make_calibrator(algo: str, xs: dict, cache_path: Path):
    """Calibrador que alimenta as amostras do fold, uma por vez (batch 1).
    algo: "entropy" (IInt8EntropyCalibrator2, KL) ou "minmax" (IInt8MinMaxCalibrator)."""
    base = trt.IInt8MinMaxCalibrator if algo == "minmax" else trt.IInt8EntropyCalibrator2

    class NpyCalibrator(base):
        def __init__(self):
            base.__init__(self)
            self.algo = algo
            self.xs = xs
            self.n = len(next(iter(xs.values())))
            self.i = 0
            self.dev = {name: cudart.malloc(arr[0:1].nbytes) for name, arr in xs.items()}

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= self.n:
                return None
            ptrs = []
            for name in names:
                cudart.memcpy_htod(self.dev[name], self.xs[name][self.i:self.i + 1])
                ptrs.append(int(self.dev[name]))
            self.i += 1
            return ptrs

        def read_calibration_cache(self):
            return None  # sempre calibra do zero

        def write_calibration_cache(self, cache):
            cache_path.write_bytes(bytes(cache))

        def release(self):
            for p in self.dev.values():
                cudart.free(p)

    return NpyCalibrator()


def layer_dtype(fmt: str) -> str:
    f = fmt.lower()
    for key, name in (("int8", "int8"), ("fp16", "fp16"), ("half", "fp16"), ("fp32", "fp32"),
                      ("float", "fp32"), ("int32", "int32"), ("int64", "int64"), ("bool", "bool")):
        if key in f:
            return name
    return "other"


COMPUTE_TYPES = ("Convolution", "Gemm", "MatrixMultiply", "FullyConnected")

# Contorno de um defeito do TensorRT 10.3 (Myelin) no mlp_pure: em FP32/FP16 o Myelin funde as
# camadas ocultas h1 e h2 num único kernel que produz logits errados (concordância de argmax com o
# ONNX Runtime de ~7%, em todos os níveis de otimização; ver diag_mlp_pure.py). Marcar a ativação h1
# como saída auxiliar impede essa fusão e o resultado volta a bater com o ORT (|Δ| ≤ 6e-6). O runner
# não copia saídas auxiliares de volta. Nas variantes INT8 o Myelin não é usado, então nada muda.
AUX_OUTPUTS = {"mlp_pure_tribranch_18cls": ("tri_none_mlp/mlp_h1/Relu:0",)}


def mark_aux_outputs(network, names) -> list[str]:
    tensors = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            tensors[layer.get_output(j).name] = layer.get_output(j)
    for name in names:
        network.mark_output(tensors[name])
    return list(names)


def summarize_layers(info: dict) -> dict:
    """Conta camadas por dtype de saída e quantas camadas de cálculo (conv/gemm) recebem INT8/FP16."""
    counts, compute = {}, {"n": 0, "int8_in": 0, "fp16_in": 0}
    for layer in info.get("Layers", []):
        if not isinstance(layer, dict):
            continue
        outs = layer.get("Outputs") or []
        ins = layer.get("Inputs") or []
        dt = layer_dtype(outs[0].get("Format/Datatype", "")) if outs else "none"
        counts[dt] = counts.get(dt, 0) + 1
        if any(k in str(layer.get("LayerType")) for k in COMPUTE_TYPES):
            in_dts = {layer_dtype(i.get("Format/Datatype", "")) for i in ins}
            compute["n"] += 1
            compute["int8_in"] += "int8" in in_dts
            compute["fp16_in"] += "fp16" in in_dts
    return {"n_layers": sum(counts.values()), "out_dtype_counts": counts, "compute_layers": compute}


def force_int8_compute(network) -> int:
    """Marca Conv/MatMul para executar em INT8 (com PREFER_PRECISION_CONSTRAINTS)."""
    n = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in (trt.LayerType.CONVOLUTION, trt.LayerType.MATRIX_MULTIPLY):
            layer.precision = trt.int8
            n += 1
    return n


def force_fp16_float_layers(network) -> int:
    """Marca para FP16 toda camada cujas entradas e saídas são float32 (exceto constantes).
    Comparações, casts e cálculos de forma (bool/int) ficam de fora."""
    n = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type == trt.LayerType.CONSTANT or layer.num_outputs == 0:
            continue
        ins = [layer.get_input(j) for j in range(layer.num_inputs)]
        if all(layer.get_output(j).dtype == trt.float32 for j in range(layer.num_outputs)) and \
                all(t is None or t.dtype == trt.float32 for t in ins):
            layer.precision = trt.float16
            n += 1
    return n


def build_one(model: str, prec: str, out_dir: Path, folds, workspace_mb: int, timing_cache: Path | None) -> dict:
    onnx_path = C.EXPORT / model / "model.onnx"
    logger = CaptureLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"falha no parser ONNX: {errs}")
    for i in range(network.num_inputs):          # batch dinâmico -> 1 (engine estática)
        t = network.get_input(i)
        t.shape = (1,) + tuple(t.shape)[1:]
    aux = mark_aux_outputs(network, AUX_OUTPUTS.get(model, ())) if not prec.startswith("int8") else []

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    if timing_cache is not None:
        cache = config.create_timing_cache(timing_cache.read_bytes() if timing_cache.exists() else b"")
        config.set_timing_cache(cache, ignore_mismatch=False)
    calib, n_forced = None, 0
    if prec.startswith("fp16"):
        config.set_flag(trt.BuilderFlag.FP16)
        if prec == "fp16":
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            n_forced = force_fp16_float_layers(network)
    elif prec.startswith("int8"):
        fold = folds["A"] if prec.endswith("A") else folds["B"]
        xs, _, _, _ = C.load_eval_set(model)
        calib = make_calibrator("minmax" if prec.startswith("int8mm") else "entropy",
                                {k: np.ascontiguousarray(v[fold]) for k, v in xs.items()},
                                out_dir / f"{prec}.calib")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
        if not prec.startswith("int8auto"):
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            n_forced = force_int8_compute(network)

    t0 = time.monotonic()
    ser = builder.build_serialized_network(network, config)
    build_s = time.monotonic() - t0
    if calib is not None:
        calib.release()
    if ser is None:
        raise RuntimeError("build_serialized_network devolveu None")
    if timing_cache is not None:
        timing_cache.write_bytes(bytes(config.get_timing_cache().serialize()))
    plan = bytes(ser)
    (out_dir / f"{prec}.plan").write_bytes(plan)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    summary = summarize_layers(info)
    (out_dir / f"{prec}.layers.json").write_text(json.dumps(info, indent=1))
    io = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        io.append({"name": name, "mode": engine.get_tensor_mode(name).name,
                   "shape": list(engine.get_tensor_shape(name)),
                   "dtype": engine.get_tensor_dtype(name).name})
    dev_mem = getattr(engine, "device_memory_size_v2", None) or engine.device_memory_size
    rec = {"model": model, "precision": prec, "trt": trt.__version__, "build_s": round(build_s, 2),
           "engine_bytes": len(plan), "device_memory_bytes": int(dev_mem), "io": io,
           "network_inputs": [network.get_input(i).name for i in range(network.num_inputs)],
           "flags": {"tf32": False, "fp16": prec.startswith("fp16"), "int8": prec.startswith("int8"),
                     "forced_layers": n_forced, "workspace_mb": workspace_mb,
                     "timing_cache": timing_cache is not None, "aux_outputs": aux},
           "calib_fold": prec[-1] if prec.startswith("int8") else None,
           "calib_algo": calib.algo if calib is not None else None,
           "n_layers": summary["n_layers"], "out_dtype_counts": summary["out_dtype_counts"],
           "compute_layers": summary["compute_layers"],
           "warnings": [m for m in logger.msgs if "Missing scale and zero-point" not in m],
           "n_missing_scale_warnings": sum("Missing scale and zero-point" in m for m in logger.msgs)}
    (out_dir / f"{prec}.build.json").write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-id", required=True)
    ap.add_argument("--models", nargs="*", default=list(C.MODELS))
    ap.add_argument("--precisions", nargs="*", default=list(C.TRT_VARIANTS))
    ap.add_argument("--workspace-mb", type=int, default=1024)
    ap.add_argument("--no-timing-cache", action="store_true",
                    help="build independente (sem reaproveitar tempos de tática entre precisões)")
    ap.add_argument("--skip-existing", action="store_true", help="pula engines que já têm .plan e .build.json")
    a = ap.parse_args()

    root = C.ENGINES / a.build_id
    root.mkdir(parents=True, exist_ok=True)
    _, y, _, _ = C.load_eval_set(next(iter(C.MODELS)))
    fa, fb = C.make_folds(y)
    folds = {"A": fa, "B": fb}
    (root / "folds.json").write_text(json.dumps({"seed": C.FOLD_SEED, "A": fa.tolist(), "B": fb.tolist()}))
    print(f"folds: A={len(fa)} B={len(fb)} (seed {C.FOLD_SEED})", flush=True)

    results = []
    for model in a.models:
        d = root / model
        d.mkdir(exist_ok=True)
        for prec in a.precisions:
            if a.skip_existing and (d / f"{prec}.plan").exists() and (d / f"{prec}.build.json").exists():
                rec = json.loads((d / f"{prec}.build.json").read_text())
                print(f"[{time.strftime('%H:%M:%S')}] {model} / {prec}: já existe, pulando", flush=True)
                results.append({k: rec.get(k) for k in ("model", "precision", "build_s", "engine_bytes",
                                                        "device_memory_bytes", "out_dtype_counts", "compute_layers")})
                continue
            print(f"[{time.strftime('%H:%M:%S')}] build {model} / {prec}", flush=True)
            try:
                tc = None if a.no_timing_cache else d / "timing.cache"
                rec = build_one(model, prec, d, folds, a.workspace_mb, tc)
                print(f"    ok: {rec['build_s']} s, {rec['engine_bytes'] / 1e6:.2f} MB, "
                      f"camadas por dtype={rec['out_dtype_counts']}, conv/gemm={rec['compute_layers']}", flush=True)
                results.append({k: rec[k] for k in ("model", "precision", "build_s", "engine_bytes",
                                                    "device_memory_bytes", "out_dtype_counts", "compute_layers")})
            except Exception as e:  # noqa: BLE001
                print(f"    FALHOU: {e}", flush=True)
                results.append({"model": model, "precision": prec, "error": str(e)})
    (root / "build_summary.json").write_text(json.dumps(
        {"trt": trt.__version__, "versions": C.versions(), "snapshot": C.snapshot(), "results": results}, indent=1))
    n_fail = sum("error" in r for r in results)
    print(f"FIM: {len(results) - n_fail} engines ok, {n_fail} falhas", flush=True)


if __name__ == "__main__":
    main()
