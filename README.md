# Benchmark dos Modelos ONNX na Jetson Nano — Artigo C (Quantization and Deployment Sensitivity)

Este guia explica como rodar os 6 modelos exportados (`.onnx`) na Jetson Nano e coletar **todos os dados exigidos pela seção 4.5 do Artigo C**: accuracy pós-deploy, latência, energia, memória e tamanho — usando os dados de calibração já extraídos do MAFAULDA (sem precisar de sensor conectado).

> ⚠️ **Ponto metodológico do Artigo C (seção 4.5):** a accuracy tem que ser medida **na própria Jetson**, não só reaproveitada do servidor — a implementação de operadores pode diferir sutilmente, especialmente para o head KAN (splines). Por isso o passo 6 abaixo não é opcional.

## 1. O que você vai receber

Uma pasta `export/` com 6 subpastas, uma por configuração (a matriz do Artigo B, seção 3.4):

```
export/
├── tribranch18__mlp/                          # full backbone + MLP
├── tribranch18__kan/                          # full backbone + KAN
├── mobilenetv3_light_mlp_tribranch_18cls/     # light backbone + MLP
├── mobilenetv3_light_kan_tribranch_18cls/     # light backbone + KAN
├── kan_pure_tribranch_18cls/                  # sem backbone + KAN (controle)
└── mlp_pure_tribranch_18cls/                  # sem backbone + MLP (controle)
```

Cada subpasta contém:

| Arquivo | Descrição |
|---|---|
| `model.onnx` | O modelo treinado (seed 0), exportado para ONNX, precisão **FP32** |
| `calib_input0.npy` | 200 janelas do sinal temporal — ramo A (entrada 1) |
| `calib_input1.npy` | As mesmas 200 janelas, espectro de ordem — ramo B (entrada 2) |
| `calib_input2.npy` | As mesmas 200 janelas, escalares físicos — ramo C (entrada 3) |
| `calib_labels.npy` | Rótulo verdadeiro de cada uma das 200 amostras (conjunto de teste) |
| `artifacts.json` | Metadados do export (formas de entrada, classes, etc.) |

> Esses `.npy` são as **mesmas 200 amostras de teste** usadas para calcular o macro-F1 reportado no Artigo B. Reproduzi-las na Jetson é o procedimento padrão de benchmark de edge-AI: mede-se a mesma coisa que se mediria com sensor ao vivo, com a vantagem de ser reprodutível por qualquer pessoa.

## 2. Pré-requisitos na Jetson Nano

```bash
# JetPack já deve estar instalado (SDK Manager da NVIDIA)
python3 -m pip install --upgrade pip
python3 -m pip install onnxruntime numpy scikit-learn
```

Confirme a versão do Python e do onnxruntime:
```bash
python3 --version
python3 -c "import onnxruntime; print(onnxruntime.__version__)"
```

## 3. Fixar o power mode ANTES de qualquer medição

O Artigo C (checklist 4.7) exige que **o power mode da Jetson esteja fixo e documentado em todas as medições** — sem isso, latência e energia não são comparáveis entre modelos.

```bash
sudo nvpmodel -m 0        # modo de máxima performance (MAXN)
sudo jetson_clocks        # trava os clocks no máximo, sem variação por thermal throttling
```

Anote no relatório final: `nvpmodel -m 0` + `jetson_clocks`, e a versão do JetPack/L4T (`cat /etc/nv_tegra_release`).

## 4. Transferir os arquivos para a Jetson

Do computador onde está a pasta `export/`, envie tudo de uma vez:

```bash
scp -r export/ usuario@IP_DA_JETSON:/home/usuario/lascas_onnx/
scp jetson_bench.py usuario@IP_DA_JETSON:/home/usuario/lascas_onnx/
```

Troque `usuario` e `IP_DA_JETSON` pelas credenciais de acesso da placa.

## 5. Rodar o benchmark de latência (para cada um dos 6 modelos)

```bash
cd /home/usuario/lascas_onnx/export/tribranch18__mlp
python3 ../../jetson_bench.py \
  --engine onnxruntime \
  --model model.onnx \
  --calib-dir . \
  --runs 500 \
  --precision fp32 \
  --tag tribranch18__mlp
```

Repita trocando a pasta e o `--tag` para os outros 5:
- `tribranch18__kan`
- `mobilenetv3_light_mlp_tribranch_18cls`
- `mobilenetv3_light_kan_tribranch_18cls`
- `kan_pure_tribranch_18cls`
- `mlp_pure_tribranch_18cls`

> **Nota:** se `--tag` não for reconhecido pela versão do script, rode sem ele: `python3 jetson_bench.py --engine onnxruntime --model model.onnx --calib-dir . --runs 500 --precision fp32`. Confira os argumentos exatos em `main()` antes de rodar em lote.

500 execuções por modelo dão amostra suficiente para reportar **latência mean, P95 e P99** (métrica exigida na tabela do Artigo C, seção 4.5) — confirme que o `jetson_bench.py` já calcula os percentis; se só imprimir a média, calcule P95/P99 a partir do CSV bruto de latências por execução (ver passo 8).

## 6. Medir energia por inferência com `tegrastats`

O Artigo C pede **energia por inferência em mJ** (seção 4.5), obtida via `tegrastats` × latência. Rode o `tegrastats` em paralelo ao benchmark, para cada modelo:

```bash
# Terminal 1 — inicia o log de energia (amostra a cada 100 ms)
tegrastats --interval 100 --logfile tegrastats_tribranch18__mlp.log &
TEGRA_PID=$!

# Terminal 2 (ou logo em seguida) — roda o benchmark normalmente
cd /home/usuario/lascas_onnx/export/tribranch18__mlp
python3 ../../jetson_bench.py --engine onnxruntime --model model.onnx \
  --calib-dir . --runs 500 --precision fp32 --tag tribranch18__mlp

# Para o log de energia
kill $TEGRA_PID
```

Do `tegrastats_<tag>.log`, extraia o campo `VDD_IN` (potência total de entrada, em mW) durante a janela de execução do benchmark:

```bash
grep -oP 'VDD_IN \K[0-9]+' tegrastats_tribranch18__mlp.log | \
  awk '{sum+=$1; n++} END {print "Potência média (mW):", sum/n}'
```

Energia por inferência (mJ) = potência média (mW) × latência média (s) por inferência:
```
energia_mJ = potencia_media_mW * (latencia_media_ms / 1000)
```

Repita para os 6 modelos, mantendo um `tegrastats_<tag>.log` por modelo.

## 7. Memória de pico (`tegrastats`)

Do mesmo log, extraia o campo `RAM` (memória usada/total, em MB) e pegue o valor máximo durante a execução:

```bash
grep -oP 'RAM \K[0-9]+(?=/)' tegrastats_tribranch18__mlp.log | sort -n | tail -1
```

Esse é o **peak unified memory (MB)** pedido na tabela do Artigo C.

## 8. Accuracy pós-deploy NA JETSON (não reaproveitar do servidor)

Rode a inferência completa sobre as 200 amostras de calibração e compare com `calib_labels.npy`, dentro de cada pasta de modelo:

```python
# accuracy_jetson.py — rodar dentro de cada pasta de modelo na Jetson
import onnxruntime as ort
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

sess = ort.InferenceSession("model.onnx")
input_names = [i.name for i in sess.get_inputs()]

x0 = np.load("calib_input0.npy").astype("float32")
x1 = np.load("calib_input1.npy").astype("float32")
x2 = np.load("calib_input2.npy").astype("float32")
y  = np.load("calib_labels.npy")

feed = {input_names[0]: x0, input_names[1]: x1, input_names[2]: x2}
logits = sess.run(None, feed)[0]
pred = logits.argmax(axis=1)

print("macro-F1 (Jetson):", f1_score(y, pred, average="macro", zero_division=0))
print("accuracy (Jetson):", accuracy_score(y, pred))
```

```bash
python3 accuracy_jetson.py
```

Rode para os 6 modelos e compare com o macro-F1 já reportado no Artigo B — qualquer divergência relevante deve ser documentada (checklist do Artigo C: *"Operadores KAN (splines) verificados em INT8 — fallback documentado"*, e mais amplamente qualquer diferença de operador FP32 servidor vs Jetson).

## 9. Tamanho do modelo

```bash
ls -lh export/*/model.onnx
```
Ou por modelo individual:
```bash
ls -lh model.onnx
```

## 10. Consolidar a tabela mestre do Artigo C (Fig. 1)

Para cada um dos 6 modelos, reúna em uma linha:

| Modelo | macro-F1 (Jetson) | Accuracy (Jetson) | Latência mean (ms) | Latência P95 (ms) | Latência P99 (ms) | Energia/inferência (mJ) | Peak RAM (MB) | Tamanho (MB) | FPS | FPS/W |
|---|---|---|---|---|---|---|---|---|---|---|
| tribranch18__mlp | | | | | | | | | | |
| tribranch18__kan | | | | | | | | | | |
| mobilenetv3_light_mlp_tribranch_18cls | | | | | | | | | | |
| mobilenetv3_light_kan_tribranch_18cls | | | | | | | | | | |
| kan_pure_tribranch_18cls | | | | | | | | | | |
| mlp_pure_tribranch_18cls | | | | | | | | | | |

Fórmulas derivadas:
- **FPS** = 1000 / latência_mean_ms
- **FPS/W** = FPS / (potência_média_mW / 1000)

Essa tabela corresponde à **Fig. 1 do Artigo C** (seção 4.6) e alimenta diretamente:
- **Fig. 2** — accuracy drop MLP vs KAN (comparar pares nas mesmas linhas de backbone)
- **Fig. 3** — scatter accuracy × latência, tamanho do ponto = memória
- **Fig. 4** — ganho de eficiência entre precisões (quando FP16/INT8 forem adicionados)
- **Fig. 5** — trade-off backbone full vs light (FPS suficiente para justificar a perda de F1?)

> **Nota sobre precisão:** os 6 `model.onnx` atuais são **FP32** (baseline). Rodar FP16/INT8 (linhas adicionais da matriz da seção 4.4 do Artigo C) exige o caminho TensorRT com os 3 perfis de shape explícitos — isso é um passo separado, não coberto pelo ONNX Runtime direto usado aqui.

## Checklist de execução na Jetson (alinhado à seção 4.7 do Artigo C)

- [ ] Power mode fixo (`nvpmodel -m 0` + `jetson_clocks`) documentado
- [ ] Transferir `export/` (6 modelos) + `jetson_bench.py` para a Jetson
- [ ] Rodar latência (`--runs 500`) para os 6 modelos → mean/P95/P99
- [ ] Rodar `tegrastats` em paralelo → energia/inferência (mJ) e peak RAM (MB)
- [ ] Rodar `accuracy_jetson.py` para os 6 modelos → macro-F1/accuracy medidos na própria Jetson
- [ ] Conferir tamanho de cada `model.onnx` (`ls -lh`)
- [ ] Calcular FPS e FPS/W derivados
- [ ] Consolidar tudo na tabela mestre (Fig. 1 do Artigo C)
- [ ] Documentar qualquer divergência de accuracy Jetson vs servidor (especialmente nos modelos com head KAN)
