#!/usr/bin/env python3
"""Mede UMA configuração (modelo × variante) num processo novo.

Fases, nesta ordem:
  0. carrega os dados e inicia o amostrador de potência/memória (processo separado)
  1. linha de base de memória (1,5 s), antes de importar o runtime
  2. carrega runtime + engine/sessão (tempo de carga)
  3. passe de accuracy: as 900 amostras, batch 1 -> logits.npy (inclui a 1ª inferência, "fria")
  4. aquecimento (--warmup-s)
  5. latência: até --lat-iters inferências síncronas (limite --lat-max-s): e2e no host e GPU
  6. ocioso (--idle-s): engine carregada, sem inferência -> potência de repouso
  7. ativo (--active-s): inferência contínua -> potência média e nº de inferências -> energia
O GC do Python fica desligado nas fases 3-7.

Energia por inferência (mJ): total = P_ativo / vazão; dinâmica = (P_ativo - P_ocioso) / vazão,
para o rail VDD_IN (módulo inteiro) e VDD_CPU_GPU_CV. O primeiro --trim-s de cada fase é descartado.
Memória: pegada = MemAvailable(linha de base) - min(MemAvailable) durante o processo
(memória unificada: inclui contexto CUDA, engine e buffers).
"""
from __future__ import annotations

import argparse
import gc
import json
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def meminfo_kb(key: str) -> int:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    raise KeyError(key)


def status_kb(key: str) -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    raise KeyError(key)


def power_window(P: np.ndarray, t0: float, t1: float, trim: float) -> dict | None:
    m = (P[:, 0] >= t0 + trim) & (P[:, 0] <= t1)
    if m.sum() < 3:
        return None
    s = P[m]
    w_in = s[:, 1] * s[:, 2] / 1e6       # mV * mA = µW -> W
    w_cg = s[:, 3] * s[:, 4] / 1e6
    w_soc = s[:, 5] * s[:, 6] / 1e6
    return {"n_samples": int(m.sum()), "duration_s": float(t1 - t0 - trim),
            "vdd_in_w": float(w_in.mean()), "vdd_in_w_std": float(w_in.std()),
            "cpu_gpu_cv_w": float(w_cg.mean()), "soc_w": float(w_soc.mean()),
            "gpu_load_pct": float(s[:, 8].mean() / 10.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--variant", required=True, choices=list(C.ALL_VARIANTS))
    ap.add_argument("--engines", required=True, help="diretório do build (engines/<build-id>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--lat-iters", type=int, default=2000)
    ap.add_argument("--lat-max-s", type=float, default=40.0)
    ap.add_argument("--idle-s", type=float, default=10.0)
    ap.add_argument("--active-s", type=float, default=20.0)
    ap.add_argument("--sample-hz", type=float, default=50.0)
    ap.add_argument("--trim-s", type=float, default=1.0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    backbone, head = C.MODELS[a.model]
    is_trt = a.variant in C.TRT_VARIANTS
    marks = {}

    def mark(k):
        marks[k] = time.monotonic()

    mark("start")
    snap_start = C.snapshot()
    xs, y, ref, meta = C.load_eval_set(a.model)
    N = len(y)
    feeds = [{n: xs[n][i:i + 1] for n in C.INPUT_NAMES} for i in range(N)]

    sampler = subprocess.Popen(
        [sys.executable, str(HERE / "power_sampler.py"), "--out", str(out / "power.csv"), "--hz", str(a.sample_hz)],
        stdout=subprocess.DEVNULL, stderr=open(out / "sampler.err", "w"))
    try:
        time.sleep(1.5)
        mark("baseline")
        mem_base = meminfo_kb("MemAvailable")

        mark("load_start")
        if is_trt:
            from runners import TrtRunner
            runner = TrtRunner(Path(a.engines) / a.model / f"{a.variant}.plan")
            import tensorrt
            runtime_version = f"tensorrt {tensorrt.__version__}"
        else:
            from runners import OrtRunner
            runner = OrtRunner(C.EXPORT / a.model / "model.onnx", a.variant.split("_", 1)[1])
            import onnxruntime
            runtime_version = f"onnxruntime {onnxruntime.__version__}"
        mark("load_end")
        mem_loaded = meminfo_kb("MemAvailable")

        gc.disable()
        # 3. accuracy (as 900 amostras, na ordem)
        logits = np.empty((N, C.N_CLASSES), np.float32)
        lat_acc = np.empty(N)
        mark("acc_start")
        for i in range(N):
            t0 = time.perf_counter_ns()
            o, _ = runner.infer(feeds[i])
            t1 = time.perf_counter_ns()
            logits[i] = o[0]
            lat_acc[i] = (t1 - t0) / 1e3
        mark("acc_end")

        # 4. aquecimento
        k = 0
        tend = time.monotonic() + a.warmup_s
        while time.monotonic() < tend:
            runner.infer(feeds[k % N])
            k += 1
        n_warm = k

        # 5. latência
        e2e = np.empty(a.lat_iters)
        gpu = np.empty(a.lat_iters)
        n = 0
        mark("lat_start")
        tcap = marks["lat_start"] + a.lat_max_s
        while n < a.lat_iters:
            f = feeds[n % N]
            t0 = time.perf_counter_ns()
            _, g = runner.infer(f)
            t1 = time.perf_counter_ns()
            e2e[n] = (t1 - t0) / 1e3
            gpu[n] = g * 1e3
            n += 1
            if time.monotonic() > tcap:
                break
        mark("lat_end")

        # 6. ocioso
        mark("idle_start")
        time.sleep(a.idle_s)
        mark("idle_end")

        # 7. ativo
        n_act = 0
        mark("act_start")
        tend = marks["act_start"] + a.active_s
        while True:
            runner.infer(feeds[n_act % N])
            n_act += 1
            if time.monotonic() >= tend:
                break
        mark("act_end")
        gc.enable()
        vmhwm, vmrss = status_kb("VmHWM"), status_kb("VmRSS")
        time.sleep(0.3)
    finally:
        sampler.send_signal(signal.SIGTERM)
        try:
            sampler.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sampler.kill()
    snap_end = C.snapshot()

    P = np.loadtxt(out / "power.csv", delimiter=",", skiprows=1, ndmin=2)
    idle = power_window(P, marks["idle_start"], marks["idle_end"], a.trim_s)
    act = power_window(P, marks["act_start"], marks["act_end"], a.trim_s)
    act_dur = marks["act_end"] - marks["act_start"]
    thr = n_act / act_dur
    energy = {}
    for rail in ("vdd_in", "cpu_gpu_cv"):
        energy[f"{rail}_total_mj"] = act[f"{rail}_w"] / thr * 1e3
        energy[f"{rail}_dynamic_mj"] = (act[f"{rail}_w"] - idle[f"{rail}_w"]) / thr * 1e3
    after = P[:, 0] >= marks["baseline"]
    mem_min = float(P[after, 7].min())

    pred = logits.argmax(1)
    summary = {
        "model": a.model, "backbone": backbone, "head": head, "variant": a.variant,
        "backend": runner.backend, "runtime": runtime_version,
        "providers": getattr(runner, "providers", None),
        "input_names_fed": runner.input_names,
        "aux_outputs": getattr(runner, "aux_outputs", []),
        "n_eval": int(N),
        "accuracy": float(np.mean(pred == y)), "macro_f1": C.macro_f1(y, pred),
        "vs_keras": C.logit_diff(logits, ref),
        "latency_e2e_us": C.dist_stats(e2e[:n]),
        "latency_gpu_us": C.dist_stats(gpu[:n]) if is_trt else None,
        "latency_first_us": float(lat_acc[0]),
        "latency_accpass_us": C.dist_stats(lat_acc[1:]),
        "load_s": marks["load_end"] - marks["load_start"],
        "n_warmup": n_warm,
        "power_idle": idle, "power_active": act,
        "active_s": act_dur, "n_active": n_act, "throughput_ips": thr,
        "energy": energy,
        "memory": {"avail_baseline_kb": mem_base, "avail_loaded_kb": mem_loaded, "avail_min_kb": mem_min,
                   "footprint_peak_mb": (mem_base - mem_min) / 1024.0,
                   "footprint_loaded_mb": (mem_base - mem_loaded) / 1024.0,
                   "vmhwm_mb": vmhwm / 1024.0, "vmrss_mb": vmrss / 1024.0},
        "engine_mb": runner.engine_bytes / 1e6 if runner.engine_bytes else None,
        "engine_device_mem_mb": runner.device_memory_bytes / 1e6 if runner.device_memory_bytes else None,
        "onnx_mb": (C.EXPORT / a.model / "model.onnx").stat().st_size / 1e6,
        "snapshot_start": snap_start, "snapshot_end": snap_end,
        "marks_s": {k: round(v - marks["start"], 4) for k, v in marks.items()},
        "protocol": vars(a),
    }
    np.save(out / "logits.npy", logits)
    np.save(out / "lat_e2e_us.npy", e2e[:n].astype(np.float32))
    if is_trt:
        np.save(out / "lat_gpu_us.npy", gpu[:n].astype(np.float32))
    np.save(out / "lat_accpass_us.npy", lat_acc.astype(np.float32))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    L = summary["latency_e2e_us"]
    print(f"{a.model} {a.variant}: acc={summary['accuracy']:.4f} F1={summary['macro_f1']:.4f} "
          f"lat_e2e mean={L['mean']:.1f}us p99={L['p99']:.1f}us | {thr:.0f} inf/s | "
          f"P_idle={idle['vdd_in_w']:.2f}W P_act={act['vdd_in_w']:.2f}W "
          f"E={energy['vdd_in_total_mj']:.3f}mJ (din {energy['vdd_in_dynamic_mj']:.3f}) | "
          f"mem={summary['memory']['footprint_peak_mb']:.0f}MB", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
