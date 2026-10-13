#!/bin/bash
# Encadeia: espera um build em andamento -> builds que faltam -> confere engines -> campanha.
# Uso: nohup bash pipeline.sh <pid_build_ou_0> <build_id> <campaign_id> <reps> > log 2>&1 &
set -u
PID_WAIT=${1:-0}; BUILD=${2:-b1}; CAMP=${3:-c1}; REPS=${4:-3}
cd "$(dirname "$0")"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
MODELS="tribranch18__kan kan_pure_tribranch_18cls tribranch18__mlp mlp_pure_tribranch_18cls mobilenetv3_light_kan_tribranch_18cls mobilenetv3_light_mlp_tribranch_18cls"
VARIANTS="fp32 fp16 fp16auto int8_cA int8_cB int8auto_cA int8mm_cA int8mm_cB"

if [ "$PID_WAIT" != "0" ]; then
  echo "[$(ts)] esperando o processo $PID_WAIT terminar"
  while kill -0 "$PID_WAIT" 2>/dev/null; do sleep 30; done
fi
echo "[$(ts)] builds que faltam ($BUILD)"
python3 -u build_engines.py --build-id "$BUILD" --skip-existing --models $MODELS --precisions $VARIANTS 2>&1 \
  | grep -vE "TRT WARNING|Deprecat|int8_calibrator|device_discovery"
missing=0
for m in $MODELS; do for v in $VARIANTS; do
  [ -f "$HOME/LASCAS-2027/engines/$BUILD/$m/$v.plan" ] || { echo "FALTA: $m/$v"; missing=$((missing+1)); }
done; done
if [ "$missing" -gt 0 ]; then echo "[$(ts)] $missing engines faltando; campanha não iniciada"; exit 1; fi
echo "[$(ts)] todas as engines ok; iniciando campanha $CAMP ($REPS reps)"
python3 -u run_campaign.py --campaign-id "$CAMP" --engines "$BUILD" --reps "$REPS" 2>&1 | grep -v device_discovery
echo "[$(ts)] pipeline concluído"
