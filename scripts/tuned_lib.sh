# Shared by scripts/g0_tuned_run.sh and scripts/eraser_tuned_run.sh (sourced, not run).
# Linear baselines re-selected by best SCORPION val top-1 (findings.md, finding 18):
# fit (sweep on the H0-mini GAP cache, test read once) -> PLISM (plism_apply on the stored
# raw features, no backbone pass) and HEST (full, --posthoc) in parallel on two GPUs.
#
# Environment (defaults = lxbelshark):
#   GPU0, GPU1        CUDA indices (0, 1): PLISM on GPU0, HEST on GPU1
#   WORKERS           dataloader workers (8; 0 inside claude-box)
#   HEST_BENCH_DATA   HEST bench_data (/mnt/nas7/data/Personal/Valentin/HEST/eval/bench_data)
#   STEPS             "all" (default) or a list of 0 preflight, 1 fit, 2 PLISM + HEST,
#                     3 PLISM only, 4 HEST only; later steps read the manifest
#   DRY=1             print the commands instead of running them
#   CACHE             SCORPION feature cache to fit on (default: H0-mini GAP)
#   PLISM_RAW         raw PLISM run whose stored features plism_apply maps (same encoder
#                     and pooling as CACHE; default: H0-mini GAP)
#   MANIFEST          manifest of run dirs (default runs/tuned_baselines_manifest.env)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
HEST_BENCH_DATA="${HEST_BENCH_DATA:-/mnt/nas7/data/Personal/Valentin/HEST/eval/bench_data}"
STEPS="${STEPS:-all}"
DRY="${DRY:-0}"
CACHE="${CACHE:-runs/bench/2026-10-13_15-44-35_lxbelshark_d6f7_scorpion_softshrink_h0mini_gap}"
PLISM_RAW="${PLISM_RAW:-runs/bench/2026-10-13_10-02-08_lxbelshark_053f_h0_mini_gap_ref}"
MANIFEST="${MANIFEST:-runs/tuned_baselines_manifest.env}"
# One log dir per invocation (pid: parallel invocations started in the same second).
LOGS="runs/tuned_baselines_logs/$(date +%Y-%m-%d_%H-%M-%S)_$(basename "$0" .sh)_$$"
mkdir -p runs/bench "$LOGS"
touch "$MANIFEST"
STAMP="$LOGS/.start"  # outputs must be newer than this invocation
touch "$STAMP"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
run() {  # <log file> <command...>
    local out=$1
    shift
    if [ "$DRY" = 1 ]; then echo "+ $* > $out"; return; fi
    "$@" >"$out" 2>&1
}
record() {  # KEY value; later lines win
    if [ "$DRY" = 1 ]; then log "(dry) $1 = $2"; return; fi
    [ -n "$2" ] && [ -e "$2" ] || { log "no output for $1 ('$2')"; exit 1; }
    echo "$1=$2" >>"$MANIFEST"
    log "$1 = $2"
}
get() {  # value of a manifest key, or fail
    local v
    v=$(grep "^$1=" "$MANIFEST" | tail -1 | cut -d= -f2- || true)
    [ -n "$v" ] || { log "missing $1 in $MANIFEST (run its step first)"; exit 1; }
    echo "$v"
}
newest() {  # <tag> [marker]: newest dir of a tag created by this invocation; with a marker
    # (metrics = PLISM, results = HEST), only dirs containing it: PLISM and HEST of one tag run
    # in parallel and each writes its own dir
    local d
    for d in $(ls -dt runs/bench/*_"$1" 2>/dev/null || true); do
        [ -z "${2:-}" ] || [ -e "$d/$2" ] || continue
        [ "$d" -nt "$STAMP" ] && echo "$d"
        return
    done
}
json_get() {  # <file> <python expression on `s`> (selected.json)
    uv run python -c "import json; s = json.load(open('$1')); print($2)"
}

preflight() {
    log "preflight"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    [ -z "$(git status --porcelain)" ] || log "WARNING: SIPE has uncommitted changes (recorded as dirty)"
    [ -f "$CACHE/features_val.npz" ] || { log "no feature cache $CACHE"; exit 1; }
    [ -f "$PLISM_RAW/provenance.json" ] || { log "no raw PLISM run $PLISM_RAW"; exit 1; }
    [ -d "$HEST_BENCH_DATA" ] || { log "no HEST_BENCH_DATA $HEST_BENCH_DATA"; exit 1; }
    b=$(git -C ../HEST rev-parse --abbrev-ref HEAD)
    [ "$b" = sipe ] || { log "../HEST is on branch $b, not sipe"; exit 1; }
    uv sync -q
    bench/run python -c "import plismbench, hest" || { log "bench env broken"; exit 1; }
}

plism() {  # <gpu> <map file> <tag>
    log "PLISM $3 on GPU $1 (~35 min)"
    run "$LOGS/plism_$3.log" env CUDA_VISIBLE_DEVICES="$1" bench/run python -m \
        sipe.bench.plism_apply --features-from "$PLISM_RAW" --posthoc "$2" --tag "$3"
    record "PLISM_$3" "$(newest "$3" metrics)"
}

hest() {  # <gpu> <map file> <tag>
    log "HEST $3 on GPU $1"
    run "$LOGS/hest_$3.log" env CUDA_VISIBLE_DEVICES="$1" bench/run python -m \
        sipe.bench.hest --posthoc "$2" --tag "$3" --datasets '*' \
        --bench-data-root "$HEST_BENCH_DATA" \
        --num-workers "$( [ "$WORKERS" -eq 0 ] && echo 0 || echo 1 )"
    record "HEST_$3" "$(newest "$3" results)"
}

bench_both() {  # <map file> <tag>: PLISM on GPU0 and HEST on GPU1 in parallel
    local p0 p1 status=0
    [ "$DRY" = 1 ] || [ -f "$1" ] || { log "missing map $1"; exit 1; }
    plism "$GPU0" "$1" "$2" & p0=$!
    hest "$GPU1" "$1" "$2" & p1=$!
    wait "$p0" || status=1
    wait "$p1" || status=1
    [ "$status" = 0 ] || { log "a benchmark failed (logs in $LOGS)"; exit 1; }
}

main_steps() {  # dispatch STEPS to step0..step4 of the calling script
    local s
    if [ "$STEPS" = all ]; then STEPS="0 1 2"; fi
    for s in $STEPS; do "step$s"; done
    log "done; manifest $MANIFEST, logs $LOGS"
}
