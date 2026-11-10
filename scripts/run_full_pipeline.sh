set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

INPUT="data/full/releases/blackjack-v1-100k-sharded-v1"
MERGED="data/merged"
CONFIG="configs/train_full.yaml"
RUN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)   INPUT="$2";   shift 2 ;;
    --merged)  MERGED="$2";  shift 2 ;;
    --config)  CONFIG="$2";  shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  mkdir -p runs
  if [[ -L runs/full-latest ]]; then
    RUN_DIR="runs/$(readlink runs/full-latest)"
  else
    RUN_STAMP="full-$(date +%Y%m%d-%H%M%S)"
    RUN_DIR="runs/$RUN_STAMP"
    ln -sfn "$RUN_STAMP" runs/full-latest
  fi
fi
NAME="$(basename "$RUN_DIR")"
mkdir -p "$RUN_DIR"

log() { echo "[pipeline] $*"; }

log "run dir: $RUN_DIR"
log "input:   $INPUT"
log "merged:  $MERGED"
log "config:  $CONFIG"

if [[ -f "$MERGED/tokens.npy" && -f "$MERGED/merged_manifest.json" ]]; then
  log "stage 1 merge: SKIP (found $MERGED/merged_manifest.json)"
else
  log "stage 1 merge: RUN"
  "$PYTHON" -m blackjackgpt.data.merge_shards --input "$INPUT" --output "$MERGED"
fi

if [[ -f "$MERGED/signal_report.json" ]]; then
  log "stage 2 signal: SKIP (found $MERGED/signal_report.json)"
else
  log "stage 2 signal: RUN"
  "$PYTHON" -m blackjackgpt.data.validate_signal --dataset-dir "$MERGED"
fi
cp -f "$MERGED/signal_report.json" "$RUN_DIR/signal_report.json"
cp -f "$MERGED/merged_manifest.json" "$RUN_DIR/merged_manifest.json"

if [[ -f "$RUN_DIR/ckpt_last.pt" ]]; then
  log "stage 3 train: SKIP (found $RUN_DIR/ckpt_last.pt)"
else
  log "stage 3 train: RUN"
  "$PYTHON" -m blackjackgpt.train --config "$CONFIG" --name "$NAME"
fi

if [[ -f "$RUN_DIR/behavior_report.json" ]]; then
  log "stage 4 eval: SKIP (found $RUN_DIR/behavior_report.json)"
else
  log "stage 4 eval: RUN"
  "$PYTHON" -m blackjackgpt.eval.behavior --config "$CONFIG" --run-dir "$RUN_DIR"
fi

if [[ -f "$RUN_DIR/probe/probe_report.json" ]]; then
  log "stage 5 probe: SKIP (found $RUN_DIR/probe/probe_report.json)"
else
  log "stage 5 probe: RUN"
  "$PYTHON" -m blackjackgpt.probe --config "$CONFIG" --run-dir "$RUN_DIR"
fi

log "DONE. The answer lives in:"
log "  $RUN_DIR/signal_report.json      (pre-training data signal)"
log "  $RUN_DIR/metrics.csv             (loss curves + bet unigram baseline)"
log "  $RUN_DIR/behavior_report.json    (legality + bet calibration)"
log "  $RUN_DIR/probe/probe_report.json (count decodability, card + bet)"
log "  $RUN_DIR/probe/probe_r2.png      (R^2 vs layer figure)"
