# Comparação: b1 (c1) × b2 sem timing cache (c3)

24 pares (modelo, variante) em comum.

## Desempenho (Δ% = B/A − 1)

| métrica | mediana Δ% | mín Δ% | máx Δ% |
|---|---|---|---|
| latência e2e média | -0.33 | -2.55 | +3.25 |
| latência e2e P99 | -0.77 | -13.66 | +32.84 |
| GPU (trtexec) | +0.05 | -3.37 | +5.56 |
| vazão | +0.28 | -2.76 | +2.20 |
| potência ativa | +0.04 | -0.94 | +2.64 |
| energia total/inf. | -0.38 | -2.13 | +2.60 |
| energia dinâmica/inf. | +0.60 | -4.01 | +7.43 |
| pegada de memória | +2.32 | -6.77 | +26.24 |

## Saídas (rep 1)

- logits idênticos em 5/24 pares
- kan_pure_tribranch_18cls / int8_cB: |Δlogit| máx 5.46e+00, previsões diferentes 4.67%
- mobilenetv3_light_kan_tribranch_18cls / int8_cB: |Δlogit| máx 3.13e+00, previsões diferentes 1.11%
- mlp_pure_tribranch_18cls / int8_cB: |Δlogit| máx 3.00e+00, previsões diferentes 2.56%
- kan_pure_tribranch_18cls / int8_cA: |Δlogit| máx 1.33e+00, previsões diferentes 1.33%
- tribranch18__mlp / int8_cB: |Δlogit| máx 1.10e+00, previsões diferentes 0.67%
- mobilenetv3_light_mlp_tribranch_18cls / int8_cB: |Δlogit| máx 1.09e+00, previsões diferentes 0.78%
- mlp_pure_tribranch_18cls / int8_cA: |Δlogit| máx 9.50e-01, previsões diferentes 0.78%
- mobilenetv3_light_mlp_tribranch_18cls / int8_cA: |Δlogit| máx 7.34e-01, previsões diferentes 0.44%
- tribranch18__mlp / int8_cA: |Δlogit| máx 6.39e-01, previsões diferentes 0.33%
- tribranch18__kan / int8_cB: |Δlogit| máx 6.25e-01, previsões diferentes 0.22%
- mobilenetv3_light_kan_tribranch_18cls / int8_cA: |Δlogit| máx 5.63e-01, previsões diferentes 0.78%
- tribranch18__mlp / fp16: |Δlogit| máx 6.25e-02, previsões diferentes 0.00%

## Macro-F1

- |ΔF1| máximo entre campanhas: 0.91 pp (mediana 0.11 pp)
- tribranch18__mlp / int8_cB: 93.08 → 92.36
- mobilenetv3_light_kan_tribranch_18cls / int8: 86.59 → 87.18
- kan_pure_tribranch_18cls / int8_cB: 73.12 → 72.21
- kan_pure_tribranch_18cls / int8: 81.38 → 80.73