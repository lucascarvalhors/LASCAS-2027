#!/usr/bin/env python3
"""Compara duas campanhas já analisadas (analyze.py) — reprodutibilidade e efeito de condição.

Exemplos:
  c1 × c3: mesmas variantes com engines reconstruídas do zero (b1 × b2) -> efeito do build
  c1 × c2: mesmas engines em MAXN_SUPER × 15 W -> efeito do modo de energia (saídas devem ser idênticas)

Uso:
  python compare_campaigns.py --a ../analysis_out/c1 --b ../analysis_out/c3 --ra ../results/c1 --rb ../results/c3 \
                              --label-a "b1 (c1)" --label-b "b2 (c3)" --out ../analysis_out/cmp_c1_c3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["model", "variant"]
PERF_COLS = [("e2e_mean_ms", "latência e2e média"), ("e2e_p99_ms", "latência e2e P99"),
             ("tx_gpu_mean_ms", "GPU (trtexec)"), ("throughput_ips", "vazão"), ("p_act_w", "potência ativa"),
             ("e_tot_mj", "energia total/inf."), ("e_dyn_mj", "energia dinâmica/inf."), ("mem_peak_mb", "pegada de memória")]


def logits(run_root: Path, model: str, variant: str):
    p = run_root / "rep01" / f"{model}__{variant}" / "logits.npy"
    return np.load(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ra", required=True)
    ap.add_argument("--rb", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pa, pb = pd.read_csv(Path(a.a) / "perf.csv"), pd.read_csv(Path(a.b) / "perf.csv")
    m = pa.merge(pb, on=KEYS, suffixes=("_a", "_b"))
    rows = []
    for _, r in m.iterrows():
        row = {"model": r.model, "variant": r.variant}
        for c, _ in PERF_COLS:
            if f"{c}_a" in r and f"{c}_b" in r and pd.notna(r[f"{c}_a"]) and pd.notna(r[f"{c}_b"]):
                row[f"{c}_a"], row[f"{c}_b"] = r[f"{c}_a"], r[f"{c}_b"]
                row[f"{c}_delta_pct"] = 100 * (r[f"{c}_b"] / r[f"{c}_a"] - 1)
        la, lb = logits(Path(a.ra), r.model, r.variant), logits(Path(a.rb), r.model, r.variant)
        if la is not None and lb is not None:
            row["logit_max_abs"] = float(np.abs(la - lb).max())
            row["argmax_diff_pct"] = 100 * float(np.mean(la.argmax(1) != lb.argmax(1)))
        rows.append(row)
    cmp = pd.DataFrame(rows)
    cmp.to_csv(out / "comparison.csv", index=False)

    aa, ab = pd.read_csv(Path(a.a) / "accuracy.csv"), pd.read_csv(Path(a.b) / "accuracy.csv")
    acc = aa.merge(ab, on=KEYS, suffixes=("_a", "_b"))[KEYS + ["macro_f1_a", "macro_f1_b", "d_f1_pp_a", "d_f1_pp_b"]]
    acc["macro_f1_delta_pp"] = 100 * (acc.macro_f1_b - acc.macro_f1_a)
    acc.to_csv(out / "accuracy_comparison.csv", index=False)

    L = [f"# Comparação: {a.label_a} × {a.label_b}\n",
         f"{len(cmp)} pares (modelo, variante) em comum.\n", "## Desempenho (Δ% = B/A − 1)\n",
         "| métrica | mediana Δ% | mín Δ% | máx Δ% |", "|---|---|---|---|"]
    for c, name in PERF_COLS:
        col = f"{c}_delta_pct"
        if col in cmp:
            v = cmp[col].dropna()
            if len(v):
                L.append(f"| {name} | {v.median():+.2f} | {v.min():+.2f} | {v.max():+.2f} |")
    L += ["\n## Saídas (rep 1)\n"]
    if "logit_max_abs" in cmp:
        same = cmp[cmp.logit_max_abs == 0]
        L.append(f"- logits idênticos em {len(same)}/{cmp.logit_max_abs.notna().sum()} pares")
        diff = cmp[cmp.logit_max_abs > 0].sort_values("logit_max_abs", ascending=False)
        for _, r in diff.head(12).iterrows():
            L.append(f"- {r.model} / {r.variant}: |Δlogit| máx {r.logit_max_abs:.2e}, previsões diferentes {r.argmax_diff_pct:.2f}%")
    L += ["\n## Macro-F1\n", f"- |ΔF1| máximo entre campanhas: {acc.macro_f1_delta_pp.abs().max():.2f} pp "
          f"(mediana {acc.macro_f1_delta_pp.abs().median():.2f} pp)"]
    big = acc[acc.macro_f1_delta_pp.abs() > 0.5]
    for _, r in big.iterrows():
        L.append(f"- {r.model} / {r.variant}: {100 * r.macro_f1_a:.2f} → {100 * r.macro_f1_b:.2f}")
    (out / "comparison.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
