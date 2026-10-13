"""Definições comuns do benchmark LASCAS-2027 (Artigo C) na Jetson Orin Nano Super."""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("LASCAS_BASE", str(Path.home() / "LASCAS-2027")))
EXPORT = BASE / "export"
ENGINES = BASE / "engines"
RESULTS = BASE / "results"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

# modelo -> (backbone, head)
MODELS = {
    "tribranch18__mlp": ("full", "MLP"),
    "tribranch18__kan": ("full", "KAN"),
    "mobilenetv3_light_mlp_tribranch_18cls": ("light", "MLP"),
    "mobilenetv3_light_kan_tribranch_18cls": ("light", "KAN"),
    "mlp_pure_tribranch_18cls": ("none", "MLP"),
    "kan_pure_tribranch_18cls": ("none", "KAN"),
}
# calib_input0/1/2.npy, na ordem de artifacts.json["input_order"]
INPUT_NAMES = ("time", "orders", "scalars")
N_CLASSES = 18

# Variantes TensorRT (ver build_engines.py): fp16 = FP16 forçado; int8_cA/int8_cB = INT8 forçado
# em Conv/MatMul, calibrado no fold A/B (validação cruzada da calibração); *auto = só a flag,
# o TensorRT decide por camada (em batch 1 ele fica em FP32).
TRT_VARIANTS = ("fp32", "fp16", "fp16auto", "int8_cA", "int8_cB", "int8auto_cA", "int8mm_cA", "int8mm_cB")
ORT_VARIANTS = ("ort_cpu", "ort_cuda")
ALL_VARIANTS = TRT_VARIANTS + ORT_VARIANTS
FOLD_SEED = 2027

INA_NAME = "ina3221"
GPU_DIR = Path("/sys/devices/platform/bus@0/17000000.gpu")
GPU_FREQ = GPU_DIR / "devfreq/17000000.gpu/cur_freq"
GPU_MAXFREQ = GPU_DIR / "devfreq/17000000.gpu/max_freq"
GPU_LOAD = GPU_DIR / "load"


# ── dados ──────────────────────────────────────────────────────────────────
def load_eval_set(model: str):
    """Entradas (dict nome->array float32 contíguo), rótulos, ref_logits do Keras e artifacts.json."""
    d = EXPORT / model
    xs = {name: np.ascontiguousarray(np.load(d / f"calib_input{i}.npy"), dtype=np.float32)
          for i, name in enumerate(INPUT_NAMES)}
    y = np.load(d / "calib_labels.npy")
    ref = np.load(d / "ref_logits.npy").astype(np.float32)
    meta = json.loads((d / "artifacts.json").read_text())
    assert meta["input_order"] == list(INPUT_NAMES), meta["input_order"]
    return xs, y, ref, meta


def make_folds(y: np.ndarray, seed: int = FOLD_SEED):
    """Divide as amostras em dois folds estratificados (metade de cada classe em cada fold)."""
    rng = np.random.default_rng(seed)
    fold_a, fold_b = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        h = len(idx) // 2
        fold_a += idx[:h].tolist()
        fold_b += idx[h:].tolist()
    return np.sort(np.array(fold_a)), np.sort(np.array(fold_b))


# ── métricas ───────────────────────────────────────────────────────────────
def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    """Macro-F1 sobre as classes presentes em y ∪ p (= sklearn average='macro')."""
    scores = []
    for c in np.union1d(y, p):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def logit_diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "argmax_agree": float(np.mean(a.argmax(1) == b.argmax(1))),
            "n_argmax_diff": int(np.sum(a.argmax(1) != b.argmax(1)))}


def dist_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).any():
        return {"n": int(x.size)}
    q = np.percentile(x, [50, 90, 95, 99])
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "min": float(x.min()), "p50": float(q[0]), "p90": float(q[1]), "p95": float(q[2]),
            "p99": float(q[3]), "max": float(x.max())}


# ── sistema ────────────────────────────────────────────────────────────────
def read_text(path, default=None):
    # Alguns sensores (cv*-thermal) respondem EAGAIN; o read devolve None -> TypeError.
    try:
        return Path(path).read_text().strip()
    except (OSError, TypeError):
        return default


def ina3221_dir() -> Path:
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        if read_text(h / "name") == INA_NAME:
            return h
    raise FileNotFoundError("INA3221 (hwmon) não encontrado")


def thermal() -> dict:
    out = {}
    for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        t = read_text(z / "temp")
        if t is not None and t.lstrip("-").isdigit():
            out[read_text(z / "type")] = int(t) / 1000.0
    return out


def snapshot() -> dict:
    """Clocks, temperaturas e memória no instante (sem sudo)."""
    cpus = {}
    for c in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        f = read_text(c / "cpufreq/scaling_cur_freq")
        if f:
            cpus[c.name] = int(f) / 1000.0  # MHz
    fan = None
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        if read_text(h / "name") == "pwmfan":
            fan = read_text(h / "pwm1")
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines()[:5]:
        k, v = line.split(":")
        mem[k] = int(v.split()[0])
    return {"gpu_freq_mhz": int(read_text(GPU_FREQ, "0")) / 1e6,
            "gpu_max_mhz": int(read_text(GPU_MAXFREQ, "0")) / 1e6,
            "cpu_freq_mhz": cpus, "temp_c": thermal(), "fan_pwm": fan, "meminfo_kb": mem}


def nvpmodel_mode() -> str:
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10).stdout
        return out.splitlines()[0].replace("NV Power Mode:", "").strip()
    except Exception as e:  # noqa: BLE001
        return f"? ({e})"


def versions() -> dict:
    v = {"python": platform.python_version(), "numpy": np.__version__,
         "kernel": platform.release(), "l4t": read_text("/etc/nv_tegra_release", "?").splitlines()[0],
         "board": read_text("/proc/device-tree/model", "?").replace("\x00", "")}
    for mod in ("tensorrt", "onnxruntime"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            v[mod] = f"? ({e})"
    return v


def clocks_locked() -> tuple[bool, str]:
    """jetson_clocks aplicado? (GPU e todos os CPUs no máximo permitido pelo modo nvpmodel atual)."""
    s = snapshot()
    gpu_ok = abs(s["gpu_freq_mhz"] - s["gpu_max_mhz"]) < 1
    cpu_max = max(int(read_text(c / "cpufreq/scaling_max_freq", "0"))
                  for c in Path("/sys/devices/system/cpu").glob("cpu[0-9]*")) / 1000.0
    cpu_ok = all(abs(f - cpu_max) < 1 for f in s["cpu_freq_mhz"].values())
    msg = f"GPU {s['gpu_freq_mhz']:.0f}/{s['gpu_max_mhz']:.0f} MHz, CPU {sorted(set(s['cpu_freq_mhz'].values()))} (máx {cpu_max:.0f})"
    return gpu_ok and cpu_ok, msg
