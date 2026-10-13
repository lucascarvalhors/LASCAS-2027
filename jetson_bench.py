#!/usr/bin/env python3
"""
Benchmark na Jetson Nano — latência (mean, P50, P95, P99) e energia por inferência.

Roda NA JETSON, sem depender do resto do pacote (só numpy + o runtime escolhido).
Copie esta pasta junto com os artefatos exportados:

    scp -r kanbench_results/export/mafaulda__cnn_kan__raw_fft__seed0 jetson:~/bench/
    scp kanbench/bench/jetson_bench.py jetson:~/bench/

Na Jetson:
    sudo nvpmodel -m 0 && sudo jetson_clocks     # fixa clocks: sem isso a
                                                 # variância de latência estoura
    python3 jetson_bench.py --engine onnxruntime --model model.onnx \
        --calib-dir . --runs 500 --out jetson_bench.csv

Backends suportados:
    trt           TensorRT via engine .plan (caminho principal do artigo)
    onnxruntime   ONNX Runtime (CUDAExecutionProvider ou CPU)
    tflite        tflite_runtime

Energia: lê os rails do INA3221 em /sys/bus/i2c/drivers/ina3221x/*/iio:device0/
em uma thread a ~100 Hz e integra a potência durante a janela de medição.
Em placas sem INA3221 acessível, cai para o parsing do `tegrastats`.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import threading
import time

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Energia
# ──────────────────────────────────────────────────────────────────────────
INA_GLOBS = [
    "/sys/bus/i2c/drivers/ina3221x/*/iio:device0/in_power*_input",  # Nano/TX2
    "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/power*_input",     # Xavier/Orin
]


class PowerSampler(threading.Thread):
    """Amostra potência total (mW) e integra para energia (mJ)."""

    def __init__(self, period_s: float = 0.01):
        super().__init__(daemon=True)
        self.period = period_s
        self.samples = []
        self._stop = threading.Event()
        self.rails = []
        for g in INA_GLOBS:
            self.rails = sorted(glob.glob(g))
            if self.rails:
                break
        self.available = bool(self.rails)

    def _read_mw(self) -> float:
        tot = 0.0
        for r in self.rails:
            try:
                with open(r) as fh:
                    v = float(fh.read().strip())
                tot += v / 1000.0 if v > 1e5 else v   # µW vs mW conforme o driver
            except OSError:
                pass
        return tot

    def run(self):
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self._read_mw()))
            time.sleep(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)

    def energy_mj(self) -> float:
        """Integral trapezoidal de P dt."""
        if len(self.samples) < 2:
            return float("nan")
        t = np.array([s[0] for s in self.samples])
        p = np.array([s[1] for s in self.samples])       # mW
        return float(np.trapz(p, t))                     # mW·s = mJ

    def mean_power_mw(self) -> float:
        if not self.samples:
            return float("nan")
        return float(np.mean([s[1] for s in self.samples]))


def tegrastats_power(duration_s: float = 3.0) -> float:
    """Fallback: potência média (mW) extraída do tegrastats."""
    try:
        p = subprocess.Popen(["tegrastats", "--interval", "100"],
                             stdout=subprocess.PIPE, text=True)
    except FileNotFoundError:
        return float("nan")
    vals, t0 = [], time.time()
    try:
        for line in p.stdout:
            for tok in line.split():
                if "POM_5V_IN" in tok or "VDD_IN" in tok:
                    try:
                        vals.append(float(tok.split("=")[1].split("/")[0]
                                          .replace("mW", "")))
                    except (IndexError, ValueError):
                        pass
            if time.time() - t0 > duration_s:
                break
    finally:
        p.terminate()
    return float(np.mean(vals)) if vals else float("nan")


# ──────────────────────────────────────────────────────────────────────────
# Runners
# ──────────────────────────────────────────────────────────────────────────
class TRTRunner:
    def __init__(self, plan_path: str):
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(plan_path, "rb") as fh, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(fh.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.host, self.dev = [], {}, {}
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.ctx.get_binding_shape(i))
            shape = tuple(1 if s < 0 else s for s in shape)
            size = int(np.prod(shape))
            h = cuda.pagelocked_empty(size, np.float32)
            d = cuda.mem_alloc(h.nbytes)
            self.host[name], self.dev[name] = h, d
            self.bindings.append(int(d))
            if self.engine.binding_is_input(i):
                self.ctx.set_binding_shape(i, shape)

    def __call__(self, inputs):
        names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)
                 if self.engine.binding_is_input(i)]
        for name, x in zip(names, inputs):
            np.copyto(self.host[name], np.ravel(x).astype(np.float32))
            self.cuda.memcpy_htod_async(self.dev[name], self.host[name], self.stream)
        self.ctx.execute_async_v2(self.bindings, self.stream.handle)
        self.stream.synchronize()


class ORTRunner:
    def __init__(self, onnx_path: str, providers=None):
        import onnxruntime as ort
        providers = providers or [p for p in
                                  ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                                   "CPUExecutionProvider"]
                                  if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.names = [i.name for i in self.sess.get_inputs()]
        print(f"  providers: {self.sess.get_providers()}")

    def __call__(self, inputs):
        feed = {n: np.asarray(x, np.float32) for n, x in zip(self.names, inputs)}
        return self.sess.run(None, feed)


class TFLiteRunner:
    def __init__(self, path: str, threads: int = 4):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
        self.it = Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()
        self.out = self.it.get_output_details()

    def __call__(self, inputs):
        for d, x in zip(self.inp, inputs):
            self.it.set_tensor(d["index"], np.asarray(x, d["dtype"]))
        self.it.invoke()
        return [self.it.get_tensor(o["index"]) for o in self.out]


# ──────────────────────────────────────────────────────────────────────────
def load_inputs(calib_dir: str, batch: int = 1):
    files = sorted(glob.glob(os.path.join(calib_dir, "calib_input*.npy")))
    if not files:
        raise SystemExit(f"nenhum calib_input*.npy em {calib_dir}")
    arrs = [np.load(f) for f in files]
    return [a[:batch] for a in arrs], arrs


def benchmark(runner, sample, runs: int, warmup: int, measure_energy: bool):
    for _ in range(warmup):
        runner(sample)

    sampler = PowerSampler() if measure_energy else None
    if sampler and sampler.available:
        sampler.start()
        time.sleep(0.2)
    idle_mw = sampler.mean_power_mw() if (sampler and sampler.available) else float("nan")

    lat = []
    t_start = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        runner(sample)
        lat.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t_start

    energy_mj = power_mw = float("nan")
    if sampler and sampler.available:
        sampler.stop()
        energy_mj = sampler.energy_mj()
        power_mw = sampler.mean_power_mw()
    elif measure_energy:
        power_mw = tegrastats_power(2.0)
        energy_mj = power_mw * wall

    lat = np.array(lat)
    return {
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_std": float(lat.std()),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_p99": float(np.percentile(lat, 99)),
        "throughput_ips": float(runs / wall),
        "mean_power_mw": power_mw,
        "idle_power_mw": idle_mw,
        "energy_mj_total": energy_mj,
        "energy_mj_per_inference": energy_mj / runs if energy_mj == energy_mj else float("nan"),
        "n_runs": runs,
    }


def check_accuracy(runner, arrs, labels_path: str, batch: int = 32) -> float:
    """Confere se a quantização não derrubou o F1 — o número que vai no artigo."""
    if not os.path.exists(labels_path):
        return float("nan")
    from collections import Counter  # noqa: F401
    y = np.load(labels_path)
    preds = []
    n = len(arrs[0])
    for i in range(0, n, batch):
        out = runner([a[i:i + batch] for a in arrs])
        if out is None:
            return float("nan")
        preds.append(np.argmax(np.asarray(out[0]), axis=-1))
    preds = np.concatenate(preds)[:len(y)]
    return float((preds == y[:len(preds)]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="onnxruntime",
                    choices=["trt", "onnxruntime", "tflite"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-dir", default=".")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--precision", default="fp16", help="rótulo para o CSV")
    ap.add_argument("--out", default="jetson_bench.csv")
    ap.add_argument("--tag", default="", help="model/dataset, ex: cnn_kan/mafaulda")
    a = ap.parse_args()

    sample, arrs = load_inputs(a.calib_dir, a.batch)
    runner = {"trt": TRTRunner, "onnxruntime": ORTRunner,
              "tflite": TFLiteRunner}[a.engine](a.model)

    meta = {}
    meta_path = os.path.join(a.calib_dir, "artifacts.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))

    print(f"\nbenchmark: {a.engine} | {os.path.basename(a.model)} | batch={a.batch}")
    res = benchmark(runner, sample, a.runs, a.warmup, not a.no_energy)
    acc = check_accuracy(runner, arrs, os.path.join(a.calib_dir, "calib_labels.npy"))

    model_name, dataset = (a.tag.split("/") + ["", ""])[:2] if a.tag else \
        (meta.get("model", ""), meta.get("dataset", ""))

    row = {"run_id": meta.get("run_id", os.path.basename(a.calib_dir)),
           "model": model_name, "dataset": dataset, "rep": meta.get("rep", ""),
           "engine": a.engine, "precision": a.precision, "batch": a.batch,
           "accuracy_on_calib": acc,
           "model_size_mb": os.path.getsize(a.model) / 1e6, **res}

    print(json.dumps(row, indent=2))
    write_header = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"\nanexado em {a.out}")


if __name__ == "__main__":
    main()
