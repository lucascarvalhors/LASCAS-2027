#!/usr/bin/env python3
"""Figuras (PDF vetorial + PNG) e tabelas LaTeX para o artigo, a partir da saída do analyze.py.

Uso:
  python figures.py --analysis ../analysis_out/c1 --campaign ../results/c1 --engines ../engines/b1 \
                    --export ../.. --out ../figures/c1

Cores (paleta categórica checada para daltonismo): KAN = azul, MLP = laranja, reformatação = verde-água;
blocos de contexto em cinzas neutros. Precisão = forma do marcador.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import profile_breakdown as PB  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
HEAD_C = {"KAN": "#2a78d6", "MLP": "#eb6834"}
REFORMAT_C = "#1baf7a"
BLOCK_C = {"backbone": AXIS, "orders": MUTED, "rest": GRID}
COL1, COL2 = 3.5, 7.16  # larguras IEEE (pol.)

MODELS = [  # (modelo, backbone, cabeça) — ordem das linhas
    ("tribranch18__kan", "Full", "KAN"), ("tribranch18__mlp", "Full", "MLP"),
    ("mobilenetv3_light_kan_tribranch_18cls", "Light", "KAN"), ("mobilenetv3_light_mlp_tribranch_18cls", "Light", "MLP"),
    ("kan_pure_tribranch_18cls", "None", "KAN"), ("mlp_pure_tribranch_18cls", "None", "MLP"),
]
BACKBONE_LABEL = {"Full": "Full backbone", "Light": "Light backbone", "None": "No backbone"}
PREC = {  # rótulo, marcador, variantes de desempenho (média), variante de accuracy
    "FP32": ("FP32", "o", ["fp32"], "fp32"),
    "FP16": ("FP16", "s", ["fp16"], "fp16"),
    "INT8": ("INT8 (entropy)", "^", ["int8_cA", "int8_cB"], "int8"),
    "INT8mm": ("INT8 (MinMax)", "D", ["int8mm_cA", "int8mm_cB"], "int8mm"),
}


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def row_axis(ax, labels, groups):
    """Linhas = modelos (de cima para baixo), com separadores finos entre grupos de backbone."""
    n = len(labels)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    for i in range(1, n):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color=GRID, lw=0.6, zorder=0)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def log_ticks(axis):
    axis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    axis.set_minor_locator(mpl.ticker.NullLocator())


DODGE = {"FP32": -0.2, "FP16": 0.0, "INT8": 0.2, "INT8mm": 0.2}


def perf_value(perf, model, variants, col):
    v = perf[(perf.model == model) & (perf.variant.isin(variants))][col]
    return float(v.mean()) if len(v) else np.nan


# ── Fig. 1: sensibilidade de accuracy à precisão ───────────────────────────
def fig_quant_sensitivity(acc, out):
    panels = [p for p in ("FP16", "INT8", "INT8mm") if PREC[p][3] in set(acc.variant)]
    fig, axes = plt.subplots(1, len(panels), figsize=(COL2, 1.75), sharey=True)
    axes = np.atleast_1d(axes)
    labels = [f"{b} · {h}" for _, b, h in MODELS]
    groups = [b for _, b, _ in MODELS]

    def xrange(variants):  # mesma escala nos painéis INT8 (entropy × MinMax são comparados lado a lado)
        s = acc[acc.variant.isin(variants)]
        lo, hi = min(s["d_f1_ci_lo_pp"].min(), -0.5), max(s["d_f1_ci_hi_pp"].max(), 0.5)
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad
    int8_lim = xrange([PREC[p][3] for p in panels if p.startswith("INT8")])
    for ax, p in zip(axes, panels):
        name, marker, _, accv = PREC[p]
        sub = acc[acc.variant == accv].set_index("model")
        row_axis(ax, labels, groups)
        ax.axvline(0, color=INK2, lw=0.7, zorder=1)
        for i, (m, b, h) in enumerate(MODELS):
            if m not in sub.index:
                continue
            r = sub.loc[m]
            ax.plot([r.d_f1_ci_lo_pp, r.d_f1_ci_hi_pp], [i, i], color=HEAD_C[h], lw=1.0, solid_capstyle="round", zorder=2)
            ax.plot(r.d_f1_pp, i, marker=marker, ms=4.5, color=HEAD_C[h], mec="white", mew=0.8, zorder=3)
        ax.set_title(name + (" PTQ" if p.startswith("INT8") else ""), loc="left", color=INK, pad=3)
        ax.set_xlabel("ΔMacro-F1 vs FP32 (pp)")
        ax.set_xlim(*(int8_lim if p.startswith("INT8") else xrange([accv])))
    handles = [Patch(color=HEAD_C[h], label=f"{h} head") for h in ("KAN", "MLP")]
    handles.append(Line2D([], [], color=INK2, lw=1.0, marker="o", ms=3.5, mec="white", label="estimate, 95% CI"))
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.1), handlelength=1.4,
               columnspacing=1.2)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig1_quant_sensitivity")


# ── Fig. 2: latência e energia por precisão ────────────────────────────────
def fig_latency_energy(perf, out):
    precs = [p for p in ("FP32", "FP16", "INT8") if set(PREC[p][2]) & set(perf.variant)]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 1.75), sharey=True)
    labels = [f"{b} · {h}" for _, b, h in MODELS]
    groups = [b for _, b, _ in MODELS]
    for ax, (col, xlabel) in zip(axes, (("e2e_mean_ms", "Latency per inference, mean (ms, log)"),
                                        ("e_tot_mj", "Energy per inference, VDD_IN (mJ, log)"))):
        row_axis(ax, labels, groups)
        ax.set_xscale("log")
        for i, (m, b, h) in enumerate(MODELS):
            vals = [perf_value(perf, m, PREC[p][2], col) for p in precs]
            if np.all(np.isnan(vals)):
                continue
            ys = [i + DODGE[p] for p in precs]
            ok = [k for k, v in enumerate(vals) if not np.isnan(v)]
            for k in ok:
                ax.plot(vals[k], ys[k], marker=PREC[precs[k]][1], ms=4.2, color=HEAD_C[h], mec="white", mew=0.7, zorder=3)
        ax.set_xlabel(xlabel)
        log_ticks(ax.xaxis)
    handles = [Line2D([], [], color=INK2, marker=PREC[p][1], ms=4.5, lw=0, label=PREC[p][0]) for p in precs]
    handles += [Patch(color=HEAD_C[h], label=f"{h} head") for h in ("KAN", "MLP")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.55, 1.08),
               handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig2_latency_energy")


# ── Fig. 3: tempo de GPU por bloco (perfil por camada) ─────────────────────
def profile_table(campaign: Path, engines: Path, export: Path) -> pd.DataFrame:
    rows = []
    for m, b, h in MODELS:
        onnx_path = export / m / "model.onnx"
        if not onnx_path.exists():
            continue
        nb = PB.onnx_node_blocks(onnx_path)
        for p in ("FP32", "FP16", "INT8"):
            v = PREC[p][2][0]
            d = campaign / "rep01" / f"{m}__{v}"
            if not (d / "trt_profile.json").exists():
                continue
            r = PB.breakdown(d / "trt_profile.json", engines / m / f"{v}.layers.json", nb)
            gpu = json.loads((d / "trtexec.json").read_text())["gpu_compute"]["mean"]
            fr = r["fractions"]
            rows.append({"model": m, "backbone": b, "head": h, "precision": p, "gpu_ms": gpu,
                         "profiled_ms": r["profiled_ms"],
                         **{f"ms_{k}": gpu * fr[k] for k in PB.BLOCKS}, **{f"pct_{k}": 100 * fr[k] for k in PB.BLOCKS}})
    return pd.DataFrame(rows)


def fig_profile(prof: pd.DataFrame, out):
    if prof.empty:
        return
    backbones = [b for b in ("Full", "Light", "None") if b in set(prof.backbone)]
    fig, axes = plt.subplots(len(backbones), 1, figsize=(COL1, 0.95 * len(backbones) + 0.55))
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, backbones):
        sub = prof[prof["backbone"] == b]
        rows = [(h, p) for h in ("KAN", "MLP") for p in ("FP32", "FP16", "INT8")
                if len(sub[(sub["head"] == h) & (sub["precision"] == p)])]
        for i, (h, p) in enumerate(rows):
            r = sub[(sub["head"] == h) & (sub["precision"] == p)].iloc[0]
            segs = [("backbone", r.ms_backbone, BLOCK_C["backbone"]), ("orders", r.ms_orders, BLOCK_C["orders"]),
                    ("rest", r.ms_scalars + r.ms_fusion + r.ms_other, BLOCK_C["rest"]),
                    ("head", r.ms_head, HEAD_C[h]), ("reformat", r.ms_reformat, REFORMAT_C)]
            left = 0.0
            for _, w, c in segs:
                if w <= 0:
                    continue
                ax.barh(i, w, left=left, height=0.62, color=c, edgecolor="white", linewidth=0.6, zorder=2)
                left += w
            ax.text(left, i, f" {r.gpu_ms:.2f}", va="center", ha="left", fontsize=6.5, color=INK2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{h} {p}" for h, p in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        for i in range(1, len(rows)):
            if rows[i][0] != rows[i - 1][0]:
                ax.axhline(i - 0.5, color=GRID, lw=0.6)
        ax.set_title(BACKBONE_LABEL[b], loc="left", pad=2)
        xmax = (sub.gpu_ms.max()) * 1.18
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("GPU time per inference (ms), split by block")
    handles = [Patch(color=BLOCK_C["backbone"], label="Time backbone"), Patch(color=BLOCK_C["orders"], label="Orders branch"),
               Patch(color=BLOCK_C["rest"], label="Scalars + fusion"), Patch(color=HEAD_C["KAN"], label="KAN head"),
               Patch(color=HEAD_C["MLP"], label="MLP head"), Patch(color=REFORMAT_C, label="Reformat (precision/layout)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), handlelength=1.0,
               columnspacing=0.9, fontsize=6.5)
    fig.tight_layout(h_pad=0.6)
    save(fig, out, "fig3_profile_breakdown")


# ── Fig. 4: accuracy × latência ────────────────────────────────────────────
def fig_tradeoff(acc, perf, out):
    precs = [p for p in ("FP32", "FP16", "INT8") if PREC[p][3] in set(acc.variant)]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    ax.set_xscale("log")
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    fp32_pts = {}  # backbone -> [(x, y) do FP32 de cada cabeça], para um rótulo por backbone
    for m, b, h in MODELS:
        xs, ys = [], []
        for p in precs:
            x = perf_value(perf, m, PREC[p][2], "e2e_mean_ms")
            a = acc[(acc.model == m) & (acc.variant == PREC[p][3])]
            if np.isnan(x) or a.empty:
                continue
            xs.append(x)
            ys.append(100 * float(a.macro_f1.iloc[0]))
            ax.plot(x, ys[-1], marker=PREC[p][1], ms=5, color=HEAD_C[h], mec="white", mew=0.8, zorder=3, lw=0)
        if xs:
            ax.plot(xs, ys, color=HEAD_C[h], lw=0.8, alpha=0.5, zorder=2)
            fp32_pts.setdefault(b, []).append((xs[0], ys[0]))
    for b, pts in fp32_pts.items():  # rótulo à esquerda do FP32 mais rápido do grupo (mesmos nomes das Figs. 1–2)
        x = min(p[0] for p in pts)
        y = float(np.mean([p[1] for p in pts]))
        ax.annotate(b, (x, y), textcoords="offset points", xytext=(-6, 0), ha="right", va="center",
                    fontsize=6.5, color=INK2)
    if fp32_pts:
        xmin = min(p[0] for pts in fp32_pts.values() for p in pts)
        ax.set_xlim(left=xmin / 10 ** 0.13)  # espaço para o rótulo do grupo mais à esquerda
    ax.set_xlabel("Latency per inference, mean (ms, log)")
    ax.set_ylabel("Macro-F1 (%)")
    log_ticks(ax.xaxis)
    handles = [Line2D([], [], color=INK2, marker=PREC[p][1], ms=5, lw=0, label=PREC[p][0]) for p in precs]
    handles += [Patch(color=HEAD_C[h], label=f"{h} head") for h in ("KAN", "MLP")]
    ax.legend(handles=handles, loc="upper left", ncol=1, handlelength=1.0, fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "fig4_accuracy_latency")


# ── Tabelas LaTeX ──────────────────────────────────────────────────────────
def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def master_csv(acc, perf, out):
    """Tabela completa (todas as variantes) para material suplementar."""
    p = perf.copy()
    a = acc.copy()
    for pre in ("int8", "int8mm"):               # desempenho do INT8 combinado = média das engines cA/cB
        for m in p.model.unique():
            sub = p[(p.model == m) & (p.variant.isin([f"{pre}_cA", f"{pre}_cB"]))]
            if len(sub):
                row = sub.mean(numeric_only=True).to_dict()
                row.update({"model": m, "backbone": sub.backbone.iloc[0], "head": sub["head"].iloc[0], "variant": pre})
                p = pd.concat([p, pd.DataFrame([row])], ignore_index=True)
    merged = a.merge(p, on=["model", "variant"], how="outer", suffixes=("", "_perf"))
    merged.to_csv(out / "master_table.csv", index=False)


def table_main(acc, perf, out):
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Accuracy and deployment cost on the Jetson Orin Nano Super (TensorRT 10.3, batch 1). "
             r"$\Delta$F1: change vs.\ TensorRT FP32 with 95\% bootstrap CI; flips: predictions that change vs.\ FP32. "
             r"INT8: entropy PTQ on Conv/MatMul, cross-calibrated (no sample is evaluated by an engine calibrated on it). "
             r"Engine: H2D+GPU+D2H latency measured by \texttt{trtexec}; E2E: synchronous Python loop "
             r"(mean $\pm$ s.d.\ across runs, and P99). Energy: whole-module VDD\_IN per inference.}",
             r"\label{tab:main}", r"\setlength{\tabcolsep}{3.5pt}", r"\small",
             r"\begin{tabular}{lllccccccc}", r"\toprule",
             r"Backbone & Head & Prec. & Macro-F1 (\%) & $\Delta$F1 (pp) [95\% CI] & Flips (\%) & "
             r"Engine (ms) & E2E (ms) & E2E P99 & Energy (mJ) \\", r"\midrule"]
    first = True
    for m, b, h in MODELS:
        added = False
        for p in ("FP32", "FP16", "INT8"):
            accv, perfv = PREC[p][3], PREC[p][2]
            a = acc[(acc.model == m) & (acc.variant == accv)]
            pf = perf[(perf.model == m) & (perf.variant.isin(perfv))]
            if a.empty or pf.empty:
                continue
            if not added and not first:
                lines.append(r"\addlinespace[2pt]")
            added, first = True, False
            a = a.iloc[0]
            d = "--" if p == "FP32" else f"{a.d_f1_pp:+.2f} [{a.d_f1_ci_lo_pp:+.2f}, {a.d_f1_ci_hi_pp:+.2f}]"
            flips = "--" if p == "FP32" else f"{a.flip_pct:.1f}"
            tx = pf.tx_lat_mean_ms.mean() if "tx_lat_mean_ms" in pf else np.nan
            lines.append(f"{b if p == 'FP32' else ''} & {h if p == 'FP32' else ''} & {PREC[p][0].split(' ')[0]} & "
                         f"{100 * a.macro_f1:.2f} & {d} & {flips} & {fmt(tx, 3)} & "
                         f"{pf.e2e_mean_ms.mean():.3f} $\\pm$ {pf.e2e_mean_sd_ms.mean():.3f} & {pf.e2e_p99_ms.mean():.3f} & "
                         f"{pf.e_tot_mj.mean():.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "tab_main.tex").write_text("\n".join(lines), encoding="utf-8")


def table_calibration(acc, out):
    if "int8mm" not in set(acc.variant):
        return
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{INT8 PTQ calibration ablation (cross-calibrated, Conv/MatMul forced to INT8). "
             r"$\Delta$F1 vs.\ TensorRT FP32 (pp) and prediction flips (\%).}",
             r"\label{tab:calib}", r"\setlength{\tabcolsep}{4pt}", r"\small",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Entropy (KL)} & \multicolumn{2}{c}{MinMax} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Backbone & Head & $\Delta$F1 & Flips & $\Delta$F1 & Flips \\", r"\midrule"]
    for m, b, h in MODELS:
        e = acc[(acc.model == m) & (acc.variant == "int8")]
        mm = acc[(acc.model == m) & (acc.variant == "int8mm")]
        if e.empty or mm.empty:
            continue
        e, mm = e.iloc[0], mm.iloc[0]
        lines.append(f"{b} & {h} & {e.d_f1_pp:+.2f} & {e.flip_pct:.1f} & {mm.d_f1_pp:+.2f} & {mm.flip_pct:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_calibration.tex").write_text("\n".join(lines), encoding="utf-8")


def table_runtimes(perf, out):
    cols = [("ort_cpu", "ORT CPU"), ("ort_cuda", "ORT CUDA"), ("fp32", "TRT FP32")]
    if not {"ort_cpu", "ort_cuda"} <= set(perf.variant):
        return
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Runtime baselines (FP32, batch 1): mean end-to-end latency (ms) and VDD\_IN energy per inference (mJ).}",
             r"\label{tab:runtimes}", r"\setlength{\tabcolsep}{3.5pt}", r"\small",
             r"\begin{tabular}{ll" + "cc" * len(cols) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for _, n in cols) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(cols))),
             "Backbone & Head & " + " & ".join(["ms & mJ"] * len(cols)) + r" \\", r"\midrule"]
    for m, b, h in MODELS:
        vals = []
        for v, _ in cols:
            pf = perf[(perf.model == m) & (perf.variant == v)]
            vals += [fmt(pf.e2e_mean_ms.mean(), 3), fmt(pf.e_tot_mj.mean(), 2)] if len(pf) else ["--", "--"]
        lines.append(f"{b} & {h} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tab_runtimes.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--engines", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    acc = pd.read_csv(Path(a.analysis) / "accuracy.csv")
    perf = pd.read_csv(Path(a.analysis) / "perf.csv")
    fig_quant_sensitivity(acc, out)
    fig_latency_energy(perf, out)
    prof = profile_table(Path(a.campaign), Path(a.engines), Path(a.export))
    # rótulo por extenso: "None" vira NaN ao ler o CSV com pandas
    prof.assign(backbone=prof["backbone"].map(BACKBONE_LABEL)).to_csv(out / "profile_breakdown.csv", index=False)
    fig_profile(prof, out)
    fig_tradeoff(acc, perf, out)
    table_main(acc, perf, out)
    master_csv(acc, perf, out)
    table_calibration(acc, out)
    table_runtimes(perf, out)
    print("figuras e tabelas em", out)


if __name__ == "__main__":
    main()
