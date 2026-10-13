#!/bin/bash
# Revalidação: engines reconstruídas do zero (build b2, SEM timing cache -> escolha de táticas
# independente da b1) e nova campanha c3 (2 repetições) com as variantes principais.
# Uso: nohup bash pipeline_validacao.sh <pid_para_esperar_ou_0> > log 2>&1 &
set -u
PID_WAIT=${1:-0}
cd "$(dirname "$0")"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
MODELS="tribranch18__kan kan_pure_tribranch_18cls tribranch18__mlp mlp_pure_tribranch_18cls mobilenetv3_light_kan_tribranch_18cls mobilenetv3_light_mlp_tribranch_18cls"
VARIANTS="fp32 fp16 int8_cA int8_cB"

if [ "$PID_WAIT" != "0" ]; then
  echo "[$(ts)] esperando o processo $PID_WAIT terminar"
  while kill -0 "$PID_WAIT" 2>/dev/null; do sleep 30; done
fi
echo "[$(ts)] build b2 (sem timing cache)"
python3 -u build_engines.py --build-id b2 --no-timing-cache --skip-existing --models $MODELS --precisions $VARIANTS 2>&1 \
  | grep -vE "TRT WARNING|Deprecat|int8_calibrator|device_discovery"
missing=0
for m in $MODELS; do for v in $VARIANTS; do
  [ -f "$HOME/LASCAS-2027/engines/b2/$m/$v.plan" ] || { echo "FALTA: $m/$v"; missing=$((missing+1)); }
done; done
if [ "$missing" -gt 0 ]; then echo "[$(ts)] $missing engines faltando; campanha não iniciada"; exit 1; fi
echo "[$(ts)] iniciando campanha c3 (engines b2, 2 reps)"
python3 -u run_campaign.py --campaign-id c3 --engines b2 --reps 2 --variants $VARIANTS 2>&1 | grep -v device_discovery
echo "[$(ts)] pipeline de validação concluído"
