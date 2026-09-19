# Benchmark dos Modelos ONNX na Jetson Nano — LASCAS 2027 (Experimento B)

Este guia explica como rodar os modelos exportados (`.onnx`) na Jetson Nano para medir **latência e consumo de energia de inferência**, usando os dados de calibração já extraídos do MAFAULDA (sem precisar de sensor conectado).

## 1. O que você vai receber

Uma pasta `export/` com 5 subpastas, uma por modelo:

```
export/
├── kan_pure_tribranch_18cls/
├── mobilenetv3_light_kan_tribranch_18cls/
├── mobilenetv3_light_mlp_tribranch_18cls/
├── tribranch18__kan/
└── tribranch18__mlp/
```

Cada subpasta contém:

| Arquivo | Descrição |
|---|---|
| `model.onnx` | O modelo treinado, exportado para ONNX |
| `calib_input0.npy` | 200 janelas do sinal temporal (entrada 1 do modelo) |
| `calib_input1.npy` | As mesmas 200 janelas, como espectro de ordem (entrada 2) |
| `calib_input2.npy` | As mesmas 200 janelas, como escalares físicos (entrada 3) |
| `calib_labels.npy` | Rótulo verdadeiro de cada uma das 200 amostras |
| `artifacts.json` | Metadados do export (formas de entrada, classes, etc.) |

> Esses `.npy` são amostras reais do conjunto de teste do MAFAULDA — não é dado sintético. O benchmark de latência/energia depende só do par modelo + hardware, então reproduzir essas 200 amostras já gravadas é equivalente a rodar com o sensor ao vivo.

## 2. Pré-requisitos na Jetson Nano

```bash
# JetPack já deve estar instalado (SDK Manager da NVIDIA)
python3 -m pip install --upgrade pip
python3 -m pip install onnxruntime numpy
```

Se for medir energia, você também vai precisar do script/monitor de energia que já usam no projeto (ex.: leitura via `tegrastats` ou um INA219 na alimentação da placa — ajustar conforme o setup do laboratório).

Confirme a versão do Python e do onnxruntime:
```bash
python3 --version
python3 -c "import onnxruntime; print(onnxruntime.__version__)"
```

## 3. Transferir os arquivos para a Jetson

Do computador onde está a pasta `export/`, envie tudo para a Jetson via `scp`:

```bash
scp -r export/ usuario@IP_DA_JETSON:/home/usuario/lascas_onnx/
```

Troque `usuario` e `IP_DA_JETSON` pelas credenciais de acesso da placa.

## 4. Clonar/copiar o script de benchmark

Se `jetson_bench.py` já está no repositório do projeto:

```bash
git clone <URL_DO_REPOSITORIO>
cd <pasta_do_repositorio>
```

Ou copie o arquivo `jetson_bench.py` manualmente para a Jetson, na mesma pasta onde estará o modelo.

## 5. Rodar o benchmark

Para cada uma das 5 pastas de modelo, entre nela e rode:

```bash
cd /home/usuario/lascas_onnx/export/kan_pure_tribranch_18cls
python3 /caminho/para/jetson_bench.py \
  --engine onnxruntime \
  --model model.onnx \
  --calib-dir . \
  --runs 500 \
  --precision fp32 \
  --tag kan_pure_tribranch_18cls
```

Repita trocando o `--calib-dir`/pasta e o `--tag` para os outros 4 modelos:
- `mobilenetv3_light_kan_tribranch_18cls`
- `mobilenetv3_light_mlp_tribranch_18cls`
- `tribranch18__kan`
- `tribranch18__mlp`

> **Nota:** se o `jetson_bench.py` reclamar de argumento desconhecido (`--tag` não existe na versão do colega), rode sem ele:
> ```bash
> python3 jetson_bench.py --engine onnxruntime --model model.onnx --calib-dir . --runs 500 --precision fp32
> ```
> Confirme os nomes exatos dos argumentos olhando a função `main()` do script antes de rodar em lote.

## 6. Onde ficam os resultados

O script deve gravar um CSV (ou similar) com latência por inferência e, se configurado, consumo de energia. Confirmar no próprio `jetson_bench.py` o caminho de saída — geralmente algo como `results/jetson_bench_<tag>.csv` ou impresso no terminal ao final da execução.

Recomenda-se rodar cada modelo 500 vezes (`--runs 500`) para ter uma amostra estatística estável de latência (média, desvio padrão, p95).

## 7. Consolidar os 5 resultados

Depois de rodar os 5 modelos, junte os CSVs (ou saídas) em uma única tabela comparando:
- Latência média por modelo
- Energia média por inferência (se medida)
- Tamanho do modelo (`model.onnx`)
- F1-score / acurácia (já calculados na etapa de treino, para cruzar com os dados de eficiência)

Essa tabela é o que entra na seção de resultados (3.4) do artigo.

## Checklist rápido

- [ ] Transferir `export/` inteira para a Jetson
- [ ] Instalar `onnxruntime` na Jetson
- [ ] Confirmar argumentos do `jetson_bench.py`
- [ ] Rodar os 5 modelos com `--runs 500`
- [ ] Coletar os CSVs de latência/energia
- [ ] Consolidar em uma tabela única para o artigo
