#!/usr/bin/env python3
"""Tabela MAXN_SUPER × 15 W com as mesmas engines (b1): latência, potência e energia por inferência.

Uso:
  python power_modes.py --maxn ../analysis_out/c1 --w15 ../analysis_out/c2_15w --out ../figures/final
Saída: tab_power_modes.tex (FP32, a configuração mais rápida) e power_modes.csv (todas as variantes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import MODELS, fmt  # noqa: E402

COLS = ["e2e_mean_ms", "e2e_p99_ms", "tx_gpu_mean_ms", "p_idle_w", "p_act_w", "e_tot_mj", "e_dyn_mj",
        "throughput_ips", "mem_peak_mb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxn", required=True, help="saída do analyze.py da campanha em MAXN_SUPER")
    ap.add_argument("--w15", required=True, help="saída do analyze.py da campanha em 15 W")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pm, pw = pd.read_csv(Path(a.maxn) / "perf.csv"), pd.read_csv(Path(a.w15) / "perf.csv")
    cols = [c for c in COLS if c in pm and c in pw]
    m = pm[["model", "variant"] + cols].merge(pw[["model", "variant"] + cols], on=["model", "variant"],
                                               suffixes=("_maxn", "_15w"))
    for c in cols:
        m[f"{c}_ratio"] = m[f"{c}_15w"] / m[f"{c}_maxn"]
    m.to_csv(out / "power_modes.csv", index=False)

    rows = []
    for model, b, h in MODELS:
        r = m[(m.model == model) & (m.variant == "fp32")]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(f"{b} & {h} & {fmt(r.e2e_mean_ms_maxn, 3)} & {fmt(r.e2e_mean_ms_15w, 3)} & "
                    f"{fmt(r.p_act_w_maxn)} & {fmt(r.p_act_w_15w)} & {fmt(r.e_tot_mj_maxn)} & {fmt(r.e_tot_mj_15w)} \\\\")
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Power mode: MAXN SUPER vs.\ 15\,W (same TensorRT FP32 engines, batch 1). "
           r"Latency: mean end-to-end; $P$: mean VDD\_IN power while inferring; $E$: VDD\_IN energy per inference.}",
           r"\label{tab:powermodes}", r"\setlength{\tabcolsep}{3pt}", r"\small",
           r"\begin{tabular}{llcccccc}", r"\toprule",
           r" & & \multicolumn{2}{c}{Latency (ms)} & \multicolumn{2}{c}{$P$ (W)} & \multicolumn{2}{c}{$E$ (mJ)} \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
           r"Backbone & Head & MAXN & 15\,W & MAXN & 15\,W & MAXN & 15\,W \\", r"\midrule", *rows,
           r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_power_modes.tex").write_text("\n".join(tex), encoding="utf-8")

    sys.stdout.reconfigure(encoding="utf-8")
    trt = m[~m.variant.str.startswith("ort")]
    print(f"{len(m)} pares (modelo, variante) em comum")
    for c in ("e2e_mean_ms", "tx_gpu_mean_ms", "p_act_w", "e_tot_mj", "e_dyn_mj"):
        if f"{c}_ratio" in m:
            s = trt[f"{c}_ratio"]
            print(f"  TensorRT {c}: 15 W / MAXN = {s.median():.3f} (mín {s.min():.3f}, máx {s.max():.3f})")
    for c in ("e2e_mean_ms", "e_tot_mj"):
        s = m[m.variant.str.startswith("ort")][f"{c}_ratio"]
        if len(s):
            print(f"  ONNX Runtime {c}: 15 W / MAXN = {s.median():.3f} (mín {s.min():.3f}, máx {s.max():.3f})")
    print("tabela:", out / "tab_power_modes.tex")


if __name__ == "__main__":
    main()
