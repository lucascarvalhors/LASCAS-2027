# Benchmark dos 6 modelos ONNX na Jetson Orin Nano Super — metodologia

Esta pasta reúne o código, os resultados brutos e as figuras do benchmark dos modelos exportados
pelo Lucas (pastas de modelo na raiz deste repositório). O texto descreve exatamente o que é medido e como, para servir
de base à seção de métodos do artigo.

## 1. Plataforma

| Item | Valor |
|---|---|
| Placa | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit **Super** (8 GB, Ampere, 1024 CUDA cores) |
| Software | L4T R36.5 (JetPack 6), Ubuntu 22.04, kernel 5.15.185-tegra |
| Runtime | TensorRT 10.3.0.30, CUDA 12.6, cuDNN 9.3, onnxruntime-gpu 1.24.0, Python 3.10.12, NumPy 2.2.6 |
| Modo de energia | `nvpmodel` **MAXN_SUPER** (modo 2) + `jetson_clocks`: CPU 6×1728 MHz, GPU 1020 MHz, EMC 3199 MHz, ventoinha no máximo |
| Ambiente | serviços do projeto do telescópio parados durante as medições; tela de login (gdm) ativa |

**Atenção:** nesta placa o `nvpmodel -m 0` do README é o modo **15 W**, não o MAXN. O rascunho do
LASCAS (`LASCAS_2027_Inf_UFRGS.pdf`) foi medido no Mode 0. A campanha principal (`c1`) usa MAXN_SUPER,
e a campanha `c2_15w` repete tudo em 15 W, com as mesmas engines b1.

**O que muda no modo 15 W:**
- **CPU e GPU.** CPU 6×1497,6 MHz e GPU 612 MHz, em vez de 1728 e 1020 MHz.
- **EMC.** O perfil 15 W do `/etc/nvpmodel.conf` limita o EMC a 2133 MHz. Nesta placa (L4T R36.5),
  porém, o nó de limite `/sys/kernel/nvpmodel_clk_cap/emc` continuou em 3199 MHz depois do
  `nvpmodel -m 0`, e o `jetson_clocks` travou o EMC em 3199 MHz.
- **Consequência.** A memória ficou no mesmo clock do MAXN_SUPER, então a comparação c1 × c2_15w
  isola o efeito dos clocks de CPU e GPU.
- **Não verificado.** Não se sabe se um boot direto no modo 0 aplicaria o limite de 2133 MHz. Testar
  isso exige reiniciar a placa.

## 2. Modelos e dados

- **Modelos.** 6 modelos ONNX (tf2onnx 1.17, opset 13), 18 classes do MAFAULDA, 3 entradas:
  `time` [N,2048,6], `orders` [N,6,21] e `scalars` [N,36]. São 3 backbones (full, light, nenhum),
  cada um com cabeça MLP ou KAN.
- **Conjunto de avaliação.** 900 janelas estratificadas, 50 por classe, do split de teste do Artigo B
  (seed 1024). O arquivo `ref_logits.npy` traz os logits do Keras (seed 0) nessas mesmas janelas.
- **Conferência.** Os arquivos na Jetson foram conferidos por MD5 contra o repositório (46/46 idênticos).

## 3. Engines TensorRT (`bench/build_engines.py`)

- **Build.** API Python do TensorRT 10.3, formas de entrada estáticas com batch = 1, workspace de 1 GiB
  e nível de otimização padrão (3).
- **TF32.** Desligado em todas as precisões, para que o FP32 seja FP32 de verdade. O padrão do
  TensorRT/`trtexec` liga o TF32.
- **Inspeção.** As engines são geradas com `ProfilingVerbosity.DETAILED`. O inspector registra a
  precisão efetiva e os nós ONNX de cada camada (`*.layers.json`).
- **Build b1.** Usa um timing cache por modelo, compartilhado entre as precisões.
- **Build b2 (revalidação).** Feito do zero, sem cache, de modo que a escolha de táticas é independente.

| Variante | Definição |
|---|---|
| `fp32` | FP32 |
| `fp16` | FP16 **forçado**: restrição FP16 em toda camada de ponto flutuante (`PREFER_PRECISION_CONSTRAINTS`) |
| `fp16auto` | Só a flag FP16 (equivale a `trtexec --fp16`). O TensorRT escolhe por camada |
| `int8` (`int8_cA`/`int8_cB`) | INT8 PTQ **forçado** em Conv/MatMul, com escalas do `IInt8EntropyCalibrator2` (KL). As demais camadas ficam em FP32 |
| `int8mm` (`int8mm_cA`/`int8mm_cB`) | Igual ao anterior, com `IInt8MinMaxCalibrator` (ablação do método de calibração) |
| `int8auto_cA` | Só a flag INT8 (equivale a `trtexec --int8`), com calibração por entropia no fold A |

**Calibração cruzada.** As 900 amostras são divididas em dois folds estratificados (25 por classe,
seed 2027). A engine calibrada no fold A avalia o fold B, e vice-versa. Assim a accuracy INT8 cobre as
900 amostras sem que nenhuma seja avaliada por uma engine calibrada com ela.

**Defeito do TensorRT 10.3 no `mlp_pure` (e contorno).**
- **O problema.** No `mlp_pure_tribranch_18cls`, as engines FP32, FP16 e FP16auto da b1 deram logits
  errados: |Δ| máx. de 16,1 e concordância de argmax de 7,2% contra o ONNX Runtime e o Keras. A
  accuracy caiu de 86,6% para 6,7%.
- **Isolamento** (`bench/diag_mlp_pure.py`, relatório em `diag/mlp_pure_tribranch_18cls/`):
  - O erro é determinístico.
  - Não depende da entrada não usada `time`, da ordem das amostras, do contexto de execução nem do
    timing cache.
  - Aparece em todos os níveis de otimização do builder (0–5).
  - O `trtexec` reproduz a mesma saída errada, então não é efeito do nosso runner.
- **Causa.** É o Myelin: ele funde as duas últimas camadas ocultas (h1 e h2, 317×317) num único
  kernel gerado. Só esse modelo tem duas camadas ocultas seguidas na cabeça, e os outros 5 modelos
  não são afetados.
- **Contorno.** Marcar a ativação h1 como saída auxiliar impede essa fusão. O resultado volta a bater
  com o ORT: |Δ| ≤ 6e-6 em FP32 e 0,019 em FP16, com 100% de argmax. Marcar qualquer tensor anterior a
  h1 não resolve.
- **Onde se aplica.** O contorno só vale para as variantes não-INT8 desse modelo (`AUX_OUTPUTS` em
  `build_engines.py`). Com a flag INT8 o TensorRT não usa o Myelin, e as engines INT8 já davam o
  resultado certo.
- **Runner.** Saídas auxiliares recebem buffer na GPU, mas não são copiadas para o host.
- **Custo.** Cerca de +4 a +8 µs de GPU, em ~0,1 ms.
- **Campanha c1.** As 9 execuções afetadas (3 variantes × 3 repetições) foram refeitas com as engines
  corrigidas, logo depois do fim da c1. Os resultados originais ficaram em `results/c1/_myelin_bug/`.
- **No artigo.** O episódio vale como achado: sem a checagem contra o ONNX Runtime, o modelo apareceria
  como "rápido e inútil" em FP32.

**Por que forçar a precisão.** Nos modos livres (`*auto`), o autotuner do TensorRT escolheu FP32 em
**todas** as camadas, para KAN e para MLP. Em batch 1, os kernels FP16 e INT8 e as conversões de formato
que eles exigem não compensam em camadas tão pequenas. Sem forçar, as linhas "FP16" e "INT8" mediriam
FP32.

**Por que FP16/INT8 ficam mais lentos que FP32** (contagem de camadas do inspector, engines b1):

| Engine | Kernels Myelin fundidos (`kgen`) | Camadas | Reformatações |
|---|---|---|---|
| FP32 (full/light) | 37–38 | ~268 | 0 |
| FP16 forçado | 37–38 | 314–324 | 30–58 |
| `int8auto` | 0–5 | 340–367 | ≤ 4 |
| INT8 forçado | 0–5 | 418–449 | 65–70 |

- **FP16 forçado.** O grafo continua fundido, mas ganha de 30 a 58 reformatações. Fica 1,09–1,34×
  mais lento que FP32 (latência e2e).
- **Flag INT8.** Só ligar a flag INT8 (calibração implícita) já tira o grafo do Myelin, mesmo que
  todas as camadas fiquem em FP32. Resultado: a `int8auto` sai 1,1–2,2× mais lenta que o FP32 com
  saídas iguais (ΔF1 = 0).
- **INT8 forçado.** Soma ~65 reformatações a isso e fica 1,5–2,8× mais lento.
- **Conclusão.** Em batch 1 nesta placa, o FP32 com Myelin é a configuração TensorRT mais rápida
  para os 6 modelos.
- **Exceção em energia.** Só nos modelos full o FP16 forçado gasta 1–4% menos energia por inferência
  que o FP32. A potência menor compensa a latência maior.

## 4. Protocolo de medição (`bench/run_config.py`, um processo novo por configuração)

1. **Linha de base de memória (1,5 s).** O amostrador de potência já está rodando, antes de importar o
   runtime.
2. **Carga** da engine ou da sessão ONNX Runtime, com o tempo de carga registrado.
3. **Accuracy.** As 900 amostras passam uma a uma (batch 1), e os logits são salvos.
4. **Aquecimento** de 3 s.
5. **Latência.** São 2000 inferências síncronas, com limite de 40 s:
   - **e2e**: `perf_counter` em volta de cópia H2D → `execute_async_v3` → cópia D2H → sincronização,
     com buffers pinned;
   - **GPU**: eventos CUDA em volta do `execute_async_v3`.
6. **Ocioso** por 10 s, com a engine carregada e sem inferência.
7. **Ativo** por 20 s, com inferência contínua.

**Potência.** O INA3221 do módulo é lido a 50 Hz por um processo separado preso ao CPU 0. O sensor
atualiza a cada ~20 ms. Os rails são VDD_IN (módulo inteiro), VDD_CPU_GPU_CV e VDD_SOC. O 1º segundo
de cada fase é descartado.

**Energia por inferência.**
- Total = P_ativo / vazão.
- Dinâmica = (P_ativo − P_ocioso) / vazão.

**Memória.**
- Pegada = MemAvailable(linha de base) − mín(MemAvailable) durante o processo. A memória é unificada,
  então a pegada inclui o contexto CUDA, os kernels carregados, a engine e os buffers.
- Também são registrados o VmHWM do processo, o tamanho da engine e a memória de ativação da engine.

**Validação cruzada da latência com o `trtexec`** (C++, mesma engine):
- `--iterations=2000 --warmUp=2000 --avgRuns=1 --percentile=50,90,95,99`, com `--exportTimes`.
- Na repetição 1, também `--dumpProfile --exportProfile` (300 iterações).
- O perfil por camada é atribuído aos blocos (backbone temporal, ramo de ordens, ramo escalar, fusão,
  cabeça, reformatação) pelo metadata do inspector (`analysis/profile_breakdown.py`). Em seguida é
  reescalado para o tempo de GPU sem profiler.

**Baselines ONNX Runtime.** CPU EP com o número padrão de threads e CUDA EP, ambos com
`ORT_ENABLE_ALL`.

**Campanha** (`bench/run_campaign.py`).
- R repetições, cada uma em ordem aleatória (seed).
- Pausa de 3 s entre configurações.
- Aborta se os clocks não estiverem travados, se o modo não for o esperado (`--power-mode`) ou se os serviços do telescópio
  estiverem ativos.

## 5. Estatística e validação (`analysis/analyze.py`)

**Métricas de accuracy.**
- Accuracy e macro-F1, equivalentes ao `sklearn`.
- ΔF1 contra o TensorRT FP32, com IC 95% por bootstrap pareado (2000 reamostragens).
- McNemar exato.
- *Flips*: % de previsões que mudam em relação ao FP32.
- Erro de logit (máximo e médio) e divergência KL entre os softmax.

**Métricas de latência.** Média ± desvio-padrão entre repetições (das médias de cada repetição), e
P50/P95/P99 agrupados.

**Checagens automáticas** (`validation.md`):
1. Completude.
2. Clocks no máximo e temperatura.
3. Logits idênticos entre repetições.
4. TRT FP32 contra ONNX Runtime CPU (mesmo grafo) e contra o Keras.
5. Precisão efetiva das camadas.
6. Tempo de GPU (eventos CUDA) contra o `trtexec`.
7. Coeficiente de variação entre repetições.
8. Sanidade de potência e energia.

### Reprodutibilidade do build (c1 × c3)

A campanha c3 repete fp32, fp16, int8_cA e int8_cB (2 repetições) com engines **b2**, reconstruídas
do zero e sem timing cache. A comparação está em `analysis_out/cmp_c1_c3/`.

**Desempenho, b2 em relação a b1:**

| Métrica | Mediana | Faixa |
|---|---|---|
| Latência e2e média | −0,3% | −2,6% a +3,3% |
| GPU (`trtexec`) | +0,05% | −3,4% a +5,6% |
| Energia por inferência | −0,4% | −2,1% a +2,6% |

O P99 varia bem mais (−14% a +33%), porque a cauda depende das rajadas de interferência descritas em 6.

**Saídas:**
- **FP32.** Logits bit a bit idênticos em 4 dos 6 modelos e |Δ| ≤ 7e-6 nos outros 2.
- **FP16.** |Δ| ≤ 0,06, com no máximo 0,11% de previsões diferentes.
- **INT8.** As escalas de calibração são idênticas bit a bit (12/12 caches), e todas as conv/gemm
  forçadas estão em INT8 nos dois builds. Mesmo assim, o autotuner escolhe kernels diferentes e muda
  a precisão das camadas não forçadas: a fração de camadas com saída INT8 muda até 8 pp entre builds
  (p.ex. 35% → 43% no mlp_pure int8_cB).
  Resultado: até 4,7% de previsões diferentes (kan_pure int8_cB) e Macro-F1 com |Δ| ≤ 0,91 pp
  (mediana 0,11 pp).
- **Leitura.** As diferenças de INT8 relatadas no artigo, de vários pp, ficam bem acima dessa
  variação de build.

**Validação.** A c3 passou em todas as checagens: 16 PASS, 0 WARN, 0 FAIL.

### Modo de energia (c1 × c2_15w)

A c2_15w usa as mesmas engines b1, 3 repetições e as 60 configurações. A comparação está em
`analysis_out/cmp_c1_c2_15w/`, e a tabela em `figures/final/tab_power_modes.tex`.

**Saídas.** Os logits são idênticos bit a bit em 60/60 pares, e o ΔF1 é 0. O modo de energia muda o
custo, não o resultado.

**Efeito no TensorRT FP32** (15 W em relação a MAXN_SUPER):
- **Latência e2e:** +14% a +44%.
- **Potência ativa (VDD_IN):** −12% a −22%.
- **Energia total por inferência:** +0,3% a +15%. O ocioso do módulo cai de 6,5 W para 5,9 W, mas
  essa parcela se acumula por mais tempo a cada inferência. Em batch 1, o MAXN_SUPER ("race to idle")
  gasta menos energia por inferência.

**Efeito no ONNX Runtime CPU/CUDA:** latência +6% a +22% e energia −6% a +4%.

**Validação.** A c2_15w teve 21 PASS, 2 WARN e 0 FAIL. Os dois WARN são o mesmo caso: no kan_pure
FP16, o tempo de GPU por eventos CUDA foi 1,27× o do `trtexec`. Num modelo tão pequeno, com a CPU a
1,5 GHz, o laço Python lança os kernels mais devagar do que a GPU os executa, e os eventos medem
também essa espera. O `trtexec` mede só a execução.

## 6. Limitações

- **Um modelo treinado por configuração** (seed 0). Diferenças de sensibilidade entre cabeças KAN e
  MLP podem variar com a seed.
- **Conjunto de avaliação reduzido.** São 900 janelas, um subconjunto estratificado do teste.
- **Calibração INT8 com dados do teste.** As amostras de calibração vêm do mesmo split de teste,
  embora com validação cruzada. O ideal é calibrar com dados de treino.
- **Overhead do Python na e2e.** A latência e2e inclui ~0,1 ms do laço em Python. A latência da
  engine (`trtexec`) não inclui.
- **Escopo do VDD_IN.** Mede o módulo inteiro, não só a GPU. Por isso a energia é reportada como
  total e dinâmica.
- **Controles `pure`.** Não usam a entrada `time`, e o par tem paridade de parâmetros desigual
  (3,6×), como documentado no README.
- **Processo de fundo.** Uma sessão SSH do MobaXterm (desde 09/10) rodou um laço de monitoramento
  durante todas as medições: a cada 1 s, `head`/`tail` em `/proc`, `df` e `who`. A carga é pequena e
  constante, igual em todas as configurações e campanhas. Por isso foi mantida, e não removida no meio
  da campanha.
- **INT8 por quantização implícita.** O INT8 usa a API de calibradores do TensorRT, que está
  depreciada no TRT 10 e é o caminho do "PTQ padrão" do rascunho. A quantização explícita, com nós
  Q/DQ no ONNX (p.ex. TensorRT Model Optimizer), passa pelo Myelin e pode ter outro custo. Os
  resultados de INT8 valem para o caminho implícito.
- **Cauda da latência.** 26 das 180 execuções da c1 tiveram rajadas curtas de inferências ~40% mais
  lentas: 15 a 60 iterações seguidas, quase todas no ONNX Runtime CPU (13). Isso é compatível com
  atividade esporádica do sistema, como logins SSH de monitoramento e o laço do MobaXterm. As médias
  quase não mudam (CV entre repetições ≤ 2,8%), mas o P99 agrupado de algumas configurações sobe, como
  o Light·KAN FP32 (1,47 ms com média de 1,10 ms). O `trtexec` não tem essas rajadas.
- **Reexecução do `mlp_pure`.** As 9 execuções refeitas (ver seção 3) rodaram em sequência, não
  intercaladas com as demais configurações da c1.

## 7. Reprodução

**Layout na Jetson.**
- `~/LASCAS-2027/export/` recebe as 6 pastas de modelo da raiz deste repositório: `model.onnx`,
  `calib_*.npy`, `ref_logits.npy` e `artifacts.json`, baixados com `git lfs pull`.
- `~/LASCAS-2027/bench/` recebe a pasta `bench/`.
- Para outra base, defina `LASCAS_BASE`.
- **Dependências.** O código usa TensorRT e numpy do JetPack 6, mais `onnxruntime-gpu` (baselines
  ORT) e `onnx` (só em `diag_mlp_pure.py`), instalados à parte. Não precisa de pycuda: a CUDA é
  acessada por `ctypes`, via `bench/cudart.py`.

**Dados neste repositório.**
- `results/<campanha>/repNN/<modelo>__<variante>/` guarda, por execução:
  - `summary.json`;
  - logits e latências (`.npy`, via Git LFS);
  - a série de potência (`power.csv`);
  - os arquivos do `trtexec`.
- `engines/` guarda só os metadados das engines (`*.layers.json`, `*.build.json` e os caches de
  calibração). As `.plan` dependem da placa e da versão do TensorRT e devem ser reconstruídas.

```bash
# na Jetson (~/LASCAS-2027/bench)
sudo jetson_clocks --store ~/LASCAS-2027/_jetson_clocks_antes.conf
sudo nvpmodel -m 2 && sudo jetson_clocks && sudo jetson_clocks --fan
python3 build_engines.py --build-id b1                                    # 6 modelos × 8 variantes
python3 run_campaign.py --campaign-id c1 --engines b1 --reps 3            # MAXN_SUPER
python3 build_engines.py --build-id b2 --no-timing-cache --precisions fp32 fp16 int8_cA int8_cB
python3 run_campaign.py --campaign-id c3 --engines b2 --reps 2 --variants fp32 fp16 int8_cA int8_cB
sudo nvpmodel -m 0 && sudo jetson_clocks && sudo jetson_clocks --fan      # 15 W
python3 run_campaign.py --campaign-id c2_15w --engines b1 --reps 3 --power-mode 15W
python3 diag_mlp_pure.py                                                  # diagnóstico do defeito do Myelin
sudo nvpmodel -m 2 && sudo jetson_clocks --restore ~/LASCAS-2027/_jetson_clocks_antes.conf

# no PC (benchmark_jetson/analysis)
python analyze.py --campaign ../results/c1 --engines ../engines/b1 --export ../.. --out ../analysis_out/c1
python analyze.py --campaign ../results/c3 --engines ../engines/b2 --export ../.. --out ../analysis_out/c3
python analyze.py --campaign ../results/c2_15w --engines ../engines/b1 --export ../.. --out ../analysis_out/c2_15w
python compare_campaigns.py --a ../analysis_out/c1 --b ../analysis_out/c3 --ra ../results/c1 --rb ../results/c3 --out ../analysis_out/cmp_c1_c3
python compare_campaigns.py --a ../analysis_out/c1 --b ../analysis_out/c2_15w --ra ../results/c1 --rb ../results/c2_15w --out ../analysis_out/cmp_c1_c2_15w
python figures.py --analysis ../analysis_out/c1 --campaign ../results/c1 --engines ../engines/b1 --export ../.. --out ../figures/final
python power_modes.py --maxn ../analysis_out/c1 --w15 ../analysis_out/c2_15w --out ../figures/final
```

**Estado da Jetson ao final.** MAXN_SUPER, clocks restaurados do arquivo salvo (governor `schedutil`,
GPU 306–1020 MHz, EMC sem override) e ventoinha em `nvfancontrol`. Os 4 serviços do telescópio foram
religados. As engines (`.plan`) e os resultados brutos continuam em `~/LASCAS-2027/` na placa.
