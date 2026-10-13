#!/usr/bin/env python3
"""Campanha de medição: R repetições × (modelos × variantes), em ordem aleatória por repetição.

Cada configuração roda num processo novo (run_config.py). Para engines TensorRT roda também
o trtexec sobre a mesma engine, como validação cruzada da latência (tempo de GPU e
H2D+GPU+D2H medidos pelo próprio TensorRT, em C++).
Retomável: configurações que já têm summary.json são puladas.

Uso típico:
  python3 run_campaign.py --campaign-id c1 --engines b1 --reps 5
  python3 run_campaign.py --campaign-id piloto --engines b1 --reps 1 --pilot --models tribranch18__kan
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

FULL = {"warmup_s": 3.0, "lat_iters": 2000, "lat_max_s": 40.0, "idle_s": 10.0, "active_s": 20.0,
        "trtexec_iters": 2000, "trtexec_warmup_ms": 2000, "profile_iters": 300}
PILOT = {"warmup_s": 1.0, "lat_iters": 300, "lat_max_s": 8.0, "idle_s": 3.0, "active_s": 5.0,
         "trtexec_iters": 300, "trtexec_warmup_ms": 500, "profile_iters": 50}
TELESCOPE = ("telescope-console", "telescope-studio", "telescope-panel", "telescope-mdns", "telescope-indi")


def log(root: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(root / "campaign.log", "a") as fh:
        fh.write(line + "\n")


def env_status() -> dict:
    locked, clk = C.clocks_locked()
    active = {}
    for u in TELESCOPE:
        r = subprocess.run(["systemctl", "is-active", u], capture_output=True, text=True)
        active[u] = r.stdout.strip()
    with open("/proc/loadavg") as fh:
        load1 = float(fh.read().split()[0])
    return {"clocks_locked": locked, "clocks": clk, "nvpmodel": C.nvpmodel_mode(),
            "telescope_services": active, "loadavg_1min": load1}


def parse_trtexec(text: str) -> dict:
    res = {}
    m = re.search(r"\] Throughput: ([\d.eE+-]+) qps", text)
    if m:
        res["throughput_qps"] = float(m.group(1))
    for key, name in (("latency", "Latency"), ("gpu_compute", "GPU Compute Time"),
                      ("h2d", "H2D Latency"), ("d2h", "D2H Latency"), ("enqueue", "Enqueue Time")):
        m = re.search(rf"\] {re.escape(name)}: (.*)", text)
        if m:
            pairs = re.findall(r"([\w]+(?:\(\d+(?:\.\d+)?%\))?) = ([\d.eE+-]+) ms", m.group(1))
            res[key] = {k: float(v) for k, v in pairs}
    return res


def run_trtexec(plan: Path, out: Path, iters: int, warmup_ms: int) -> dict:
    cmd = [C.TRTEXEC, f"--loadEngine={plan}", f"--iterations={iters}", f"--warmUp={warmup_ms}",
           "--duration=0", "--avgRuns=1", "--percentile=50,90,95,99",
           f"--exportTimes={out / 'trtexec_times.json'}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    (out / "trtexec.log").write_text(r.stdout + r.stderr)
    res = parse_trtexec(r.stdout)
    res["returncode"] = r.returncode
    res["cmd"] = " ".join(cmd)
    (out / "trtexec.json").write_text(json.dumps(res, indent=1))
    return res


def run_trtexec_profile(plan: Path, out: Path, iters: int) -> int:
    """Perfil por camada (trtexec --exportProfile). O profiler sincroniza a cada camada, então os
    tempos absolutos ficam inflados; serve para a fração de tempo por bloco (backbone, cabeça KAN...)."""
    cmd = [C.TRTEXEC, f"--loadEngine={plan}", f"--iterations={iters}", "--warmUp=500", "--duration=0",
           "--avgRuns=1", "--dumpProfile", f"--exportProfile={out / 'trt_profile.json'}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    (out / "trt_profile.log").write_text(r.stdout[-20000:] + r.stderr[-5000:])
    return r.returncode


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out


def compile_csv(root: Path) -> int:
    rows = []
    for s in sorted(root.glob("rep*/*/summary.json")):
        d = json.loads(s.read_text())
        for k in ("snapshot_start", "snapshot_end", "protocol", "marks_s"):
            d.pop(k, None)
        row = {"rep": s.parent.parent.name, **flatten(d)}
        t = s.parent / "trtexec.json"
        if t.exists():
            row.update(flatten(json.loads(t.read_text()), "trtexec."))
            row.pop("trtexec.cmd", None)
        rows.append(row)
    if not rows:
        return 0
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(root / "summary_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--engines", required=True, help="build id (engines/<id>)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--models", nargs="*", default=list(C.MODELS))
    ap.add_argument("--variants", nargs="*", default=list(C.ALL_VARIANTS))
    ap.add_argument("--pilot", action="store_true", help="protocolo curto, só para validar o pipeline")
    ap.add_argument("--cooldown-s", type=float, default=3.0)
    ap.add_argument("--skip-trtexec", action="store_true")
    ap.add_argument("--no-profile", action="store_true", help="não roda o perfil por camada na rep 1")
    ap.add_argument("--force", action="store_true", help="roda mesmo sem clocks travados")
    ap.add_argument("--power-mode", default="MAXN_SUPER", help="nome esperado do modo nvpmodel (ex.: 15W)")
    a = ap.parse_args()

    proto = PILOT if a.pilot else FULL
    root = C.RESULTS / a.campaign_id
    root.mkdir(parents=True, exist_ok=True)
    engines = C.ENGINES / a.engines
    env = env_status()
    log(root, f"início: {a.campaign_id} | engines={engines} | reps={a.reps} | protocolo={proto} | env={env}")
    if not a.force:
        problems = []
        if not env["clocks_locked"]:
            problems.append(f"clocks não travados ({env['clocks']})")
        if env["nvpmodel"] != a.power_mode:
            problems.append(f"nvpmodel={env['nvpmodel']} (esperado {a.power_mode})")
        if any(s == "active" for s in env["telescope_services"].values()):
            problems.append(f"serviços do telescópio ativos: {env['telescope_services']}")
        if problems:
            log(root, "ABORTADO: " + "; ".join(problems))
            sys.exit(2)

    configs = [(m, v) for m in a.models for v in a.variants]
    manifest = {"campaign_id": a.campaign_id, "engines": str(engines), "args": vars(a), "protocol": proto,
                "versions": C.versions(), "env_start": env, "snapshot_start": C.snapshot(),
                "configs": configs, "orders": {}, "runs": []}
    t_campaign = time.monotonic()
    durations = []
    total = a.reps * len(configs)
    done = 0
    for rep in range(1, a.reps + 1):
        rng = np.random.default_rng(a.seed * 1000 + rep)
        order = [configs[i] for i in rng.permutation(len(configs))]
        manifest["orders"][rep] = order
        for m, v in order:
            done += 1
            d = root / f"rep{rep:02d}" / f"{m}__{v}"
            if (d / "summary.json").exists():
                log(root, f"[{done}/{total}] rep{rep} {m} {v}: já existe, pulando")
                continue
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            cmd = [sys.executable, "-u", str(HERE / "run_config.py"), "--model", m, "--variant", v,
                   "--engines", str(engines), "--out", str(d),
                   "--warmup-s", str(proto["warmup_s"]), "--lat-iters", str(proto["lat_iters"]),
                   "--lat-max-s", str(proto["lat_max_s"]), "--idle-s", str(proto["idle_s"]),
                   "--active-s", str(proto["active_s"])]
            with open(d / "run.log", "w") as fh:
                try:
                    rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
                except subprocess.TimeoutExpired:
                    rc = -9
            tx = None
            if v in C.TRT_VARIANTS and not a.skip_trtexec and rc == 0:
                try:
                    tx = run_trtexec(engines / m / f"{v}.plan", d, proto["trtexec_iters"], proto["trtexec_warmup_ms"])
                    if rep == 1 and not a.no_profile:
                        run_trtexec_profile(engines / m / f"{v}.plan", d, proto["profile_iters"])
                except Exception as e:  # noqa: BLE001
                    tx = {"error": str(e)}
            dt = time.monotonic() - t0
            durations.append(dt)
            eta_min = np.mean(durations) * (total - done) / 60
            last = (d / "run.log").read_text().strip().splitlines()[-1:] or [""]
            gpu_tx = (tx or {}).get("gpu_compute", {}).get("mean")
            log(root, f"[{done}/{total}] rep{rep} {m} {v}: rc={rc} {dt:.0f}s ETA {eta_min:.0f} min | {last[0][:220]}"
                      + (f" | trtexec GPU mean={gpu_tx:.4f} ms" if gpu_tx else ""))
            manifest["runs"].append({"rep": rep, "model": m, "variant": v, "rc": rc, "seconds": round(dt, 1),
                                     "loadavg_after": open("/proc/loadavg").read().split()[0]})
            time.sleep(a.cooldown_s)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    manifest["env_end"] = env_status()
    manifest["snapshot_end"] = C.snapshot()
    manifest["total_min"] = round((time.monotonic() - t_campaign) / 60, 1)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    n = compile_csv(root)
    n_fail = sum(r["rc"] != 0 for r in manifest["runs"])
    log(root, f"FIM: {n} linhas em summary_runs.csv, {n_fail} falhas, {manifest['total_min']} min")


if __name__ == "__main__":
    main()
