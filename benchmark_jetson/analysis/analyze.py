#!/usr/bin/env python3
"""Análise e validação de uma campanha do benchmark (roda no PC, sobre os resultados copiados da Jetson).

Uso:
  python analyze.py --campaign ../results/c1 --engines ../engines/b1 --export ../.. --out ../analysis_out/c1

Gera em --out:
  validation.md     checagens (PASS/WARN/FAIL) — determinismo, TRT vs ONNX/Keras, trtexec vs eventos CUDA,
                    estabilidade entre repetições, clocks/temperatura, cobertura FP16/INT8 das engines
  accuracy.csv      accuracy/macro-F1 por modelo×variante e sensibilidade vs FP32 (ΔF1, flips, IC bootstrap, McNemar)
  perf.csv          latência/energia/memória agregadas por modelo×variante (média ± dp entre repetições)
  runs.csv          uma linha por (rep, modelo, variante)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "tribranch18__mlp": ("full", "MLP"),
    "tribranch18__kan": ("full", "KAN"),
    "mobilenetv3_light_mlp_tribranch_18cls": ("light", "MLP"),
    "mobilenetv3_light_kan_tribranch_18cls": ("light", "KAN"),
    "mlp_pure_tribranch_18cls": ("none", "MLP"),
    "kan_pure_tribranch_18cls": ("none", "KAN"),
}
TRT = ("fp32", "fp16", "fp16auto", "int8_cA", "int8_cB", "int8auto_cA", "int8mm_cA", "int8mm_cB")
ORT = ("ort_cpu", "ort_cuda")


# ── métricas ───────────────────────────────────────────────────────────────
def macro_f1(y, p):
    s = []
    for c in np.union1d(y, p):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        s.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(s))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def kl(p_logits, q_logits):
    p, q = softmax(p_logits.astype(np.float64)), softmax(q_logits.astype(np.float64))
    return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=1)))


def mcnemar_p(y, p_a, p_b):
    """McNemar exato (binomial) entre dois classificadores pareados."""
    a_ok, b_ok = p_a == y, p_b == y
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * cdf), n01, n10


def bootstrap_delta_f1(y, p_ref, p_x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        d[b] = macro_f1(y[i], p_x[i]) - macro_f1(y[i], p_ref[i])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ── carga ──────────────────────────────────────────────────────────────────
def load_runs(camp: Path) -> pd.DataFrame:
    rows = []
    for s in sorted(camp.glob("rep*/*/summary.json")):
        d = json.loads(s.read_text())
        r = {"rep": int(s.parent.parent.name[3:]), "dir": str(s.parent), "model": d["model"],
             "backbone": d["backbone"], "head": d["head"], "variant": d["variant"],
             "accuracy": d["accuracy"], "macro_f1": d["macro_f1"],
             "vs_keras_max": d["vs_keras"]["max_abs"], "vs_keras_agree": d["vs_keras"]["argmax_agree"],
             "load_s": d["load_s"], "first_us": d["latency_first_us"],
             "throughput": d["throughput_ips"], "n_active": d["n_active"],
             "p_idle_w": d["power_idle"]["vdd_in_w"], "p_act_w": d["power_active"]["vdd_in_w"],
             "p_idle_cg_w": d["power_idle"]["cpu_gpu_cv_w"], "p_act_cg_w": d["power_active"]["cpu_gpu_cv_w"],
             "gpu_load_act": d["power_active"]["gpu_load_pct"],
             "e_tot_mj": d["energy"]["vdd_in_total_mj"], "e_dyn_mj": d["energy"]["vdd_in_dynamic_mj"],
             "e_cg_tot_mj": d["energy"]["cpu_gpu_cv_total_mj"], "e_cg_dyn_mj": d["energy"]["cpu_gpu_cv_dynamic_mj"],
             "mem_peak_mb": d["memory"]["footprint_peak_mb"], "mem_loaded_mb": d["memory"]["footprint_loaded_mb"],
             "vmhwm_mb": d["memory"]["vmhwm_mb"], "engine_mb": d["engine_mb"],
             "engine_dev_mb": d["engine_device_mem_mb"], "onnx_mb": d["onnx_mb"],
             "gpu_mhz_start": d["snapshot_start"]["gpu_freq_mhz"], "gpu_mhz_end": d["snapshot_end"]["gpu_freq_mhz"],
             "gpu_max_mhz": d["snapshot_start"]["gpu_max_mhz"],
             "cpu_mhz_min": min(d["snapshot_end"]["cpu_freq_mhz"].values()),
             "tj_start": d["snapshot_start"]["temp_c"].get("tj-thermal"),
             "tj_end": d["snapshot_end"]["temp_c"].get("tj-thermal"),
             "providers": ",".join(d["providers"]) if d.get("providers") else ""}
        for k, v in d["latency_e2e_us"].items():
            r[f"e2e_{k}"] = v
        for k, v in (d.get("latency_gpu_us") or {}).items():
            r[f"gpu_{k}"] = v
        t = s.parent / "trtexec.json"
        if t.exists():
            tx = json.loads(t.read_text())
            r["tx_rc"] = tx.get("returncode")
            r["tx_thr"] = tx.get("throughput_qps")
            for key in ("latency", "gpu_compute"):
                for k, v in (tx.get(key) or {}).items():
                    r[f"tx_{key}_{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def load_array(run_dir: str, name: str):
    p = Path(run_dir) / name
    return np.load(p) if p.exists() else None


# ── validação ──────────────────────────────────────────────────────────────
class Report:
    def __init__(self):
        self.lines, self.n = [], {"PASS": 0, "WARN": 0, "FAIL": 0}

    def add(self, status, msg):
        self.n[status] += 1
        self.lines.append(f"- **{status}** — {msg}")

    def section(self, title):
        self.lines.append(f"\n## {title}\n")

    def text(self, t):
        self.lines.append(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--engines", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    camp, eng, exp, out = Path(a.campaign), Path(a.engines), Path(a.export), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = Report()

    runs = load_runs(camp)
    runs.to_csv(out / "runs.csv", index=False)
    manifest = json.loads((camp / "manifest.json").read_text()) if (camp / "manifest.json").exists() else {}
    reps = [int(r) for r in sorted(runs["rep"].unique())]
    models = [m for m in MODELS if m in set(runs["model"])]
    variants = [v for v in TRT + ORT if v in set(runs["variant"])]
    folds = json.loads((eng / "folds.json").read_text())
    fold_a, fold_b = np.array(folds["A"]), np.array(folds["B"])

    R.text(f"# Validação — campanha `{camp.name}` (engines `{eng.name}`)\n")
    R.text(f"Repetições: {reps} · modelos: {len(models)} · variantes: {variants} · execuções: {len(runs)}")
    if manifest:
        R.text(f"\nProtocolo: `{manifest.get('protocol')}`  \nVersões: `{manifest.get('versions')}`  \n"
               f"Ambiente no início: `{manifest.get('env_start')}`")

    # 1. completude
    R.section("1. Completude")
    exp_n = len(reps) * len(models) * len(variants)
    fails = [r for r in manifest.get("runs", []) if r.get("rc") != 0]
    R.add("PASS" if len(runs) == exp_n and not fails else "FAIL",
          f"{len(runs)}/{exp_n} execuções com summary.json; {len(fails)} com rc≠0 {fails[:5]}")

    # 2. clocks e temperatura
    R.section("2. Clocks, temperatura e ambiente")
    gpu_ok = (np.abs(runs["gpu_mhz_start"] - runs["gpu_max_mhz"]) < 1) & (np.abs(runs["gpu_mhz_end"] - runs["gpu_max_mhz"]) < 1)
    R.add("PASS" if gpu_ok.all() else "FAIL",
          f"GPU no clock máximo no início e no fim de {gpu_ok.sum()}/{len(runs)} execuções "
          f"({runs['gpu_max_mhz'].iloc[0]:.0f} MHz)")
    cpu_vals = sorted(set(runs["cpu_mhz_min"].round(1)))
    R.add("PASS" if len(cpu_vals) == 1 else "WARN", f"menor clock de CPU observado ao fim de cada execução: {cpu_vals} MHz")
    tmax = runs[["tj_start", "tj_end"]].max().max()
    R.add("PASS" if tmax < 80 else "WARN", f"temperatura de junção (tj): {runs['tj_start'].min():.1f}–{tmax:.1f} °C "
                                          f"(throttling da Orin começa perto de 99 °C)")

    # 3. determinismo das saídas entre repetições
    R.section("3. Determinismo (logits idênticos entre repetições)")
    logits = {}
    for m in models:
        for v in variants:
            sub = runs[(runs.model == m) & (runs.variant == v)].sort_values("rep")
            arrs = [load_array(d, "logits.npy") for d in sub["dir"]]
            arrs = [x for x in arrs if x is not None]
            if not arrs:
                continue
            logits[(m, v)] = arrs[0]
            dmax = max(float(np.abs(x - arrs[0]).max()) for x in arrs)
            status = "PASS" if dmax == 0 else ("WARN" if dmax < 1e-4 else "FAIL")
            if status != "PASS":
                R.add(status, f"{m} / {v}: |Δlogit| máx entre repetições = {dmax:.2e}")
    R.add("PASS", f"demais combinações: logits bit a bit idênticos em todas as {len(reps)} repetições")

    # 4. correção numérica: TRT FP32 vs ONNX Runtime (mesmo grafo) e vs Keras
    R.section("4. Correção numérica do deploy")
    acc_rows = []
    for m in models:
        y = np.load(exp / m / "calib_labels.npy")
        ref = np.load(exp / m / "ref_logits.npy").astype(np.float32)
        fp32 = logits.get((m, "fp32"))
        orc = logits.get((m, "ort_cpu"))
        if fp32 is not None and orc is not None:
            d = float(np.abs(fp32 - orc).max())
            agree = float(np.mean(fp32.argmax(1) == orc.argmax(1)))
            R.add("PASS" if d < 1e-3 and agree == 1 else "FAIL",
                  f"{m}: TRT FP32 vs ONNX Runtime CPU (mesmo ONNX): |Δlogit| máx = {d:.2e}, argmax = {agree:.4f}")
        if fp32 is not None:
            d = float(np.abs(fp32 - ref).max())
            agree = float(np.mean(fp32.argmax(1) == ref.argmax(1)))
            R.add("PASS" if agree == 1 and d < 5e-2 else "FAIL",
                  f"{m}: TRT FP32 vs Keras (ref_logits): |Δlogit| máx = {d:.2e}, argmax = {agree:.4f}")

        # accuracy e sensibilidade
        base_pred = fp32.argmax(1) if fp32 is not None else None
        entries = {}
        for v in variants:
            if (m, v) in logits:
                entries[v] = logits[(m, v)]
        for pre in ("int8", "int8mm"):                         # INT8 fora da amostra (validação cruzada)
            if f"{pre}_cA" in entries and f"{pre}_cB" in entries:
                oof = entries[f"{pre}_cA"].copy()
                oof[fold_a] = entries[f"{pre}_cB"][fold_a]     # fold A avaliado pela engine calibrada no B
                entries[pre] = oof
        for v, lg in entries.items():
            p = lg.argmax(1)
            row = {"model": m, "backbone": MODELS[m][0], "head": MODELS[m][1], "variant": v,
                   "accuracy": float(np.mean(p == y)), "macro_f1": macro_f1(y, p),
                   "vs_keras_max": float(np.abs(lg - ref).max())}
            if base_pred is not None:
                row.update({"d_acc_pp": 100 * (row["accuracy"] - float(np.mean(base_pred == y))),
                            "d_f1_pp": 100 * (row["macro_f1"] - macro_f1(y, base_pred)),
                            "flip_pct": 100 * float(np.mean(p != base_pred)),
                            "logit_max_abs_vs_fp32": float(np.abs(lg - fp32).max()),
                            "logit_mean_abs_vs_fp32": float(np.abs(lg - fp32).mean()),
                            "kl_vs_fp32": kl(fp32, lg)})
                if v not in ("fp32",):
                    lo, hi = bootstrap_delta_f1(y, base_pred, p, a.n_boot)
                    pval, n01, n10 = mcnemar_p(y, base_pred, p)
                    row.update({"d_f1_ci_lo_pp": 100 * lo, "d_f1_ci_hi_pp": 100 * hi,
                                "mcnemar_p": pval, "n_fp32_only_correct": n01, "n_x_only_correct": n10})
            if v.endswith(("_cA", "_cB")):               # engines de um fold: accuracy só fora da amostra
                held = fold_b if v.endswith("A") else fold_a
                row["accuracy_heldout"] = float(np.mean(p[held] == y[held]))
                row["macro_f1_heldout"] = macro_f1(y[held], p[held])
                if base_pred is not None:
                    row["fp32_macro_f1_same_half"] = macro_f1(y[held], base_pred[held])
            acc_rows.append(row)
    acc = pd.DataFrame(acc_rows)
    acc.to_csv(out / "accuracy.csv", index=False)

    # 5. cobertura de precisão nas engines (do build.json)
    R.section("5. Precisão efetiva das engines (inspector do TensorRT)")
    cov_rows = []
    for m in models:
        for v in TRT:
            bj = eng / m / f"{v}.build.json"
            if not bj.exists():
                continue
            b = json.loads(bj.read_text())
            c = b.get("compute_layers", {})
            cnt = b.get("out_dtype_counts", {})
            tot = sum(cnt.values()) or 1
            cov_rows.append({"model": m, "variant": v, "n_layers": b.get("n_layers"),
                             "pct_out_fp16": 100 * cnt.get("fp16", 0) / tot, "pct_out_int8": 100 * cnt.get("int8", 0) / tot,
                             "compute_n": c.get("n"), "compute_int8_in": c.get("int8_in"), "compute_fp16_in": c.get("fp16_in"),
                             "forced_layers": b.get("flags", {}).get("forced_layers"),
                             "aux_outputs": ",".join(b.get("flags", {}).get("aux_outputs") or []),
                             "engine_mb": b["engine_bytes"] / 1e6, "build_s": b["build_s"]})
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(out / "engine_precision.csv", index=False)
    if len(cov):
        for _, r in cov.iterrows():
            v = r["variant"]
            if v == "fp16" and r["pct_out_fp16"] < 50:
                R.add("WARN", f"{r['model']} fp16 (forçado): só {r['pct_out_fp16']:.0f}% das camadas com saída FP16")
            if v.startswith("int8_c") and (r["compute_int8_in"] or 0) == 0:
                R.add("FAIL", f"{r['model']} {v} (forçado): nenhuma camada conv/gemm com entrada INT8")
            if v.endswith("auto") or "auto" in v:
                R.text(f"- {r['model']} {v}: {r['pct_out_fp16']:.0f}% saídas FP16, {r['pct_out_int8']:.0f}% INT8 "
                       f"(modo livre do TensorRT)")
            if r["aux_outputs"]:
                R.text(f"- {r['model']} {v}: saída auxiliar `{r['aux_outputs']}` (contorno do defeito do Myelin, "
                       f"ver METODOLOGIA.md)")
        R.add("PASS", "tabela completa em engine_precision.csv")

    # 6. latência: eventos CUDA vs trtexec; estabilidade entre repetições
    R.section("6. Latência — validação cruzada e estabilidade")
    perf_rows = []
    for (m, v), g in runs.groupby(["model", "variant"]):
        lat = [load_array(d, "lat_e2e_us.npy") for d in g["dir"]]
        lat = np.concatenate([x for x in lat if x is not None])
        gl = [load_array(d, "lat_gpu_us.npy") for d in g["dir"]]
        gl = [x for x in gl if x is not None]
        gl = np.concatenate(gl) if gl else None
        row = {"model": m, "backbone": MODELS[m][0], "head": MODELS[m][1], "variant": v, "n_reps": len(g),
               "e2e_mean_ms": g["e2e_mean"].mean() / 1e3, "e2e_mean_sd_ms": g["e2e_mean"].std(ddof=1) / 1e3 if len(g) > 1 else 0.0,
               "e2e_p50_ms": np.percentile(lat, 50) / 1e3, "e2e_p95_ms": np.percentile(lat, 95) / 1e3,
               "e2e_p99_ms": np.percentile(lat, 99) / 1e3, "e2e_n": len(lat),
               "throughput_ips": g["throughput"].mean(), "fps": 1e3 / (g["e2e_mean"].mean() / 1e3),
               "p_idle_w": g["p_idle_w"].mean(), "p_act_w": g["p_act_w"].mean(),
               "p_act_cg_w": g["p_act_cg_w"].mean(),
               "e_tot_mj": g["e_tot_mj"].mean(), "e_tot_sd_mj": g["e_tot_mj"].std(ddof=1) if len(g) > 1 else 0.0,
               "e_dyn_mj": g["e_dyn_mj"].mean(), "e_dyn_sd_mj": g["e_dyn_mj"].std(ddof=1) if len(g) > 1 else 0.0,
               "e_cg_dyn_mj": g["e_cg_dyn_mj"].mean(),
               "mem_peak_mb": g["mem_peak_mb"].mean(), "mem_peak_sd_mb": g["mem_peak_mb"].std(ddof=1) if len(g) > 1 else 0.0,
               "vmhwm_mb": g["vmhwm_mb"].mean(), "engine_mb": g["engine_mb"].mean(), "engine_dev_mb": g["engine_dev_mb"].mean(),
               "onnx_mb": g["onnx_mb"].mean(), "load_s": g["load_s"].mean(), "first_ms": g["first_us"].mean() / 1e3,
               "gpu_load_act": g["gpu_load_act"].mean()}
        row["fps_per_w"] = row["throughput_ips"] / row["p_act_w"]
        if gl is not None:
            row.update({"gpu_mean_ms": g["gpu_mean"].mean() / 1e3, "gpu_p99_ms": np.percentile(gl, 99) / 1e3})
        if "tx_gpu_compute_mean" in g:
            row.update({"tx_lat_mean_ms": g["tx_latency_mean"].mean(), "tx_lat_p99_ms": g.get("tx_latency_percentile(99%)", pd.Series(dtype=float)).mean(),
                        "tx_gpu_mean_ms": g["tx_gpu_compute_mean"].mean(), "tx_thr": g["tx_thr"].mean()})
        perf_rows.append(row)
        cv = row["e2e_mean_sd_ms"] / row["e2e_mean_ms"] if row["e2e_mean_ms"] else 0
        if len(g) > 1 and cv > 0.05:
            R.add("WARN", f"{m} / {v}: latência média varia {100 * cv:.1f}% entre repetições")
        if gl is not None and "tx_gpu_mean_ms" in row and not math.isnan(row["tx_gpu_mean_ms"]):
            ratio = row["gpu_mean_ms"] / row["tx_gpu_mean_ms"]
            if not 0.8 <= ratio <= 1.25:
                R.add("WARN", f"{m} / {v}: GPU (eventos CUDA) {row['gpu_mean_ms']:.3f} ms vs trtexec {row['tx_gpu_mean_ms']:.3f} ms (razão {ratio:.2f})")
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(out / "perf.csv", index=False)
    if "tx_gpu_mean_ms" in perf:
        t = perf.dropna(subset=["tx_gpu_mean_ms", "gpu_mean_ms"])
        ratio = t["gpu_mean_ms"] / t["tx_gpu_mean_ms"]
        R.add("PASS" if ratio.between(0.8, 1.25).all() else "WARN",
              f"tempo de GPU por eventos CUDA / trtexec GPU Compute: razão {ratio.min():.2f}–{ratio.max():.2f} "
              f"(mediana {ratio.median():.2f}) em {len(t)} engines")
    cvs = (perf["e2e_mean_sd_ms"] / perf["e2e_mean_ms"]).replace([np.inf], np.nan)
    R.add("PASS" if cvs.max() <= 0.05 else "WARN",
          f"variação da latência média entre repetições: CV mediano {100 * cvs.median():.2f}%, máximo {100 * cvs.max():.2f}%")

    # 7. energia e potência
    R.section("7. Potência e energia")
    R.add("PASS" if (runs["p_act_w"] > runs["p_idle_w"]).all() else "WARN",
          f"P_ativo > P_ocioso em {(runs['p_act_w'] > runs['p_idle_w']).sum()}/{len(runs)} execuções; "
          f"P_ocioso (VDD_IN) {runs['p_idle_w'].min():.2f}–{runs['p_idle_w'].max():.2f} W")
    ecv = (perf["e_tot_sd_mj"] / perf["e_tot_mj"]).max()
    R.add("PASS" if ecv < 0.05 else "WARN", f"energia total por inferência: CV máximo entre repetições {100 * ecv:.1f}%")

    R.text(f"\n---\n**Resumo:** {R.n['PASS']} PASS · {R.n['WARN']} WARN · {R.n['FAIL']} FAIL")
    (out / "validation.md").write_text("\n".join(R.lines), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(R.lines))


if __name__ == "__main__":
    main()
