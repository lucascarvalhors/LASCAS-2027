# Validação — campanha `c1` (engines `b1`)

Repetições: [1, 2, 3] · modelos: 6 · variantes: ['fp32', 'fp16', 'fp16auto', 'int8_cA', 'int8_cB', 'int8auto_cA', 'int8mm_cA', 'int8mm_cB', 'ort_cpu', 'ort_cuda'] · execuções: 180

Protocolo: `{'warmup_s': 3.0, 'lat_iters': 2000, 'lat_max_s': 40.0, 'idle_s': 10.0, 'active_s': 20.0, 'trtexec_iters': 2000, 'trtexec_warmup_ms': 2000, 'profile_iters': 300}`  
Versões: `{'python': '3.10.12', 'numpy': '2.2.6', 'kernel': '5.15.185-tegra', 'l4t': '# R36 (release), REVISION: 5.0, GCID: 43688277, BOARD: generic, EABI: aarch64, DATE: Fri Jan 16 03:50:45 UTC 2026', 'board': 'NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super', 'tensorrt': '10.3.0', 'onnxruntime': '1.24.0'}`  
Ambiente no início: `{'clocks_locked': True, 'clocks': 'GPU 1020/1020 MHz, CPU [1728.0] (máx 1728)', 'nvpmodel': 'MAXN_SUPER', 'telescope_services': {'telescope-console': 'inactive', 'telescope-studio': 'inactive', 'telescope-panel': 'inactive', 'telescope-mdns': 'inactive', 'telescope-indi': 'inactive'}, 'loadavg_1min': 1.0}`

## 1. Completude

- **PASS** — 180/180 execuções com summary.json; 0 com rc≠0 []

## 2. Clocks, temperatura e ambiente

- **PASS** — GPU no clock máximo no início e no fim de 180/180 execuções (1020 MHz)
- **PASS** — menor clock de CPU observado ao fim de cada execução: [1728.0] MHz
- **PASS** — temperatura de junção (tj): 36.4–40.4 °C (throttling da Orin começa perto de 99 °C)

## 3. Determinismo (logits idênticos entre repetições)

- **PASS** — demais combinações: logits bit a bit idênticos em todas as 3 repetições

## 4. Correção numérica do deploy

- **PASS** — tribranch18__mlp: TRT FP32 vs ONNX Runtime CPU (mesmo ONNX): |Δlogit| máx = 1.40e-04, argmax = 1.0000
- **PASS** — tribranch18__mlp: TRT FP32 vs Keras (ref_logits): |Δlogit| máx = 1.50e-02, argmax = 1.0000
- **PASS** — tribranch18__kan: TRT FP32 vs ONNX Runtime CPU (mesmo ONNX): |Δlogit| máx = 2.13e-04, argmax = 1.0000
- **PASS** — tribranch18__kan: TRT FP32 vs Keras (ref_logits): |Δlogit| máx = 1.32e-02, argmax = 1.0000
- **PASS** — mobilenetv3_light_mlp_tribranch_18cls: TRT FP32 vs ONNX Runtime CPU (mesmo ONNX): |Δlogit| máx = 8.57e-05, argmax = 1.0000
- **PASS** — mobilenetv3_light_mlp_tribranch_18cls: TRT FP32 vs Keras (ref_logits): |Δlogit| máx = 1.20e-02, argmax = 1.0000
- **PASS** — mobilenetv3_light_kan_tribranch_18cls: TRT FP32 vs ONNX Runtime CPU (mesmo ONNX): |Δlogit| máx = 1.67e-04, argmax = 1.0000
- **PASS** — mobilenetv3_light_kan_tribranch_18cls: TRT FP32 vs Keras (ref_logits): |Δlogit| máx = 1.31e-02, argmax = 1.0000
- **PASS** — mlp_pure_tribranch_18cls: TRT FP32 vs ONNX Runtime CPU (mesmo ONNX): |Δlogit| máx = 5.72e-06, argmax = 1.0000
- **PASS** — mlp_pure_tribranch_18cls: TRT FP32 vs Keras (ref_logits): |Δlogit| máx = 7.39e-06, argmax = 1.0000
- **PASS** — kan_pure_tribranch_18cls: TRT FP32 vs ONNX Runtime CPU (mesmo ONNX): |Δlogit| máx = 8.11e-06, argmax = 1.0000
- **PASS** — kan_pure_tribranch_18cls: TRT FP32 vs Keras (ref_logits): |Δlogit| máx = 7.15e-06, argmax = 1.0000

## 5. Precisão efetiva das engines (inspector do TensorRT)

- tribranch18__mlp fp16auto: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- tribranch18__mlp int8auto_cA: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- tribranch18__kan fp16auto: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- tribranch18__kan int8auto_cA: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- mobilenetv3_light_mlp_tribranch_18cls fp16auto: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- mobilenetv3_light_mlp_tribranch_18cls int8auto_cA: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- mobilenetv3_light_kan_tribranch_18cls fp16auto: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- mobilenetv3_light_kan_tribranch_18cls int8auto_cA: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- mlp_pure_tribranch_18cls fp32: saída auxiliar `tri_none_mlp/mlp_h1/Relu:0` (contorno do defeito do Myelin, ver METODOLOGIA.md)
- mlp_pure_tribranch_18cls fp16: saída auxiliar `tri_none_mlp/mlp_h1/Relu:0` (contorno do defeito do Myelin, ver METODOLOGIA.md)
- mlp_pure_tribranch_18cls fp16auto: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- mlp_pure_tribranch_18cls fp16auto: saída auxiliar `tri_none_mlp/mlp_h1/Relu:0` (contorno do defeito do Myelin, ver METODOLOGIA.md)
- mlp_pure_tribranch_18cls int8auto_cA: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- kan_pure_tribranch_18cls fp16auto: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- kan_pure_tribranch_18cls int8auto_cA: 0% saídas FP16, 0% INT8 (modo livre do TensorRT)
- **PASS** — tabela completa em engine_precision.csv

## 6. Latência — validação cruzada e estabilidade

- **PASS** — tempo de GPU por eventos CUDA / trtexec GPU Compute: razão 1.03–1.24 (mediana 1.05) em 48 engines
- **PASS** — variação da latência média entre repetições: CV mediano 0.75%, máximo 2.82%

## 7. Potência e energia

- **PASS** — P_ativo > P_ocioso em 180/180 execuções; P_ocioso (VDD_IN) 6.43–6.56 W
- **PASS** — energia total por inferência: CV máximo entre repetições 4.6%

---
**Resumo:** 22 PASS · 0 WARN · 0 FAIL