#!/usr/bin/env bash
# Other backbones for the ISBI paper: UNI2-h (default) or CONCH v1.5 (ENCODER=conch-v1.5)
# through the same pipeline as H0-mini: SCORPION cache -> G0 (closed form) + identity
# (raw) maps -> CGM V1-S fit (configs/consensus_gen_uni2h.yaml with the encoder and
# pooling overridden on the command line; arm and hyperparameters as for H0-mini,
# nothing re-selected) -> cgm_select (SCORPION val/test) -> PLISM -> HEST for raw / G0 /
# V1-S.
#
# Runs on lxbelshark or lxbelwahoo (needs: uv; ../HEST and ../plism-benchmark on branch
# `sipe` next to this repo; the NAS; an HF token with access to the gated weights:
# MahmoodLab/UNI2-h, MahmoodLab/conchv1_5).
#
#   scripts/uni2h_run.sh                  # every step, two GPUs (see "all" below)
#   STEPS="5 6" scripts/uni2h_run.sh      # some steps; earlier outputs from the manifest
#   STEPS=sync scripts/uni2h_run.sh       # only print the rsync command
#
# Steps: 0 preflight, 1 cache, 2 G0 + identity map, 3 V1-S fit, 4 cgm_select,
# 5 PLISM (3 runs), 6 HEST (3 runs). "all" = 0, then {1, 2} on GPU0 while 3 runs on
# GPU1, then 4, then PLISM and HEST split over both GPUs. Every run dir goes into
# runs/<bk>_manifest.env (KEY=path, re-read by later steps); the end prints the rsync
# command that copies them to the other machine.
#
# Environment (defaults = lxbelshark):
#   GPU0, GPU1        CUDA indices (0, 1)
#   WORKERS           dataloader workers (8; 0 inside claude-box)
#   SCORPION_ROOT     SCORPION tiles (data/processed/SCORPION_tiles_224px_0p5mpp)
#   PLISM_DIR         PLISM .tif.h5 dir (/mnt/nas7/data/Personal/Valentin/plism)
#   HEST_BENCH_DATA   HEST bench_data (/mnt/nas7/data/Personal/Valentin/HEST/eval/bench_data)
#   SYNC_TO           rsync target printed at the end (lxbelshark:workspaces/SIPE/)
#   ENCODER           uni2-h (default; tags uni2h_*) or conch-v1.5 (tags conch15_*)
#   POOLING           the encoder's own embedding by default (uni2-h: cls; conch-v1.5:
#                     head, its attentional pooler), or cls / gap (mean of the patch
#                     tokens, as H0-mini). The default pooling writes runs/<bk>_manifest.env,
#                     another one runs/<bk>_<pooling>_manifest.env with tags
#                     <bk>_<pooling>_*; its step 1 reuses the CACHE of the default
#                     manifest (the cache holds every pooling)
#   SMOKE=1           test the script: 1 slide per split, 2 fit steps, PLISM on the
#                     slides of PLISM_DIR (a dir with 3 slides covering scanner, stain
#                     and joint pairs, e.g. GIVH_AT2, GIVH_GT450, GIV_AT2) with 460
#                     tiles, HEST on SKCM only; tags prefixed smoke_ (excluded from the
#                     doc tables), own manifest
set -euo pipefail
cd "$(dirname "$0")/.."

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
SCORPION_ROOT="${SCORPION_ROOT:-data/processed/SCORPION_tiles_224px_0p5mpp}"
PLISM_DIR="${PLISM_DIR:-/mnt/nas7/data/Personal/Valentin/plism}"
HEST_BENCH_DATA="${HEST_BENCH_DATA:-/mnt/nas7/data/Personal/Valentin/HEST/eval/bench_data}"
SYNC_TO="${SYNC_TO:-lxbelshark:workspaces/SIPE/}"
STEPS="${STEPS:-all}"
SMOKE="${SMOKE:-0}"
ENCODER="${ENCODER:-uni2-h}"
case "$ENCODER" in  # short tag, default pooling, cache poolings, expected (dim, prefix)
    uni2-h) BK=uni2h; DEFAULT_POOLING=cls; CACHE_POOLINGS="cls gap"; EXPECT="1536, 9" ;;
    conch-v1.5) BK=conch15; DEFAULT_POOLING=head; CACHE_POOLINGS="cls gap head"
        EXPECT="1024, 1" ;;
    *) echo "ENCODER must be uni2-h or conch-v1.5" >&2; exit 1 ;;
esac
POOLING="${POOLING:-$DEFAULT_POOLING}"
case " $CACHE_POOLINGS " in *" $POOLING "*) ;;
    *) echo "POOLING must be one of: $CACHE_POOLINGS" >&2; exit 1 ;; esac
P=""  # tag prefix
SFX=""  # tag suffix of the select run and the fit tag (default pooling: none)
[ "$POOLING" = "$DEFAULT_POOLING" ] || SFX="_$POOLING"
MANIFEST_DEFAULT=runs/${BK}_manifest.env
LOGS=runs/${BK}_logs
if [ "$SMOKE" = 1 ]; then
    P=smoke_
    MANIFEST_DEFAULT=runs/${BK}_smoke_manifest.env
    LOGS=runs/${BK}_smoke_logs
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
MANIFEST=$MANIFEST_DEFAULT
[ -z "$SFX" ] || MANIFEST="${MANIFEST_DEFAULT%_manifest.env}${SFX}_manifest.env"
LOGS="$LOGS/$(date +%Y-%m-%d_%H-%M-%S)"  # one log dir per invocation
mkdir -p runs/bench "$LOGS"
touch "$MANIFEST"
STAMP="$LOGS/.start"  # outputs must be newer than this invocation
touch "$STAMP"

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
record() {  # KEY value; later lines win
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
kill_tree() {  # <pid>: SIGTERM the process and all its descendants (uv -> python -> workers)
    local c
    for c in $(pgrep -P "$1" 2>/dev/null || true); do kill_tree "$c"; done
    kill -TERM "$1" 2>/dev/null || true
}
newest() {  # newest dir of a tag touched since this invocation started (one instance at a time)
    local d
    d=$(ls -dt runs/bench/*_"$1" 2>/dev/null | head -1 || true)
    [ -n "$d" ] && [ "$d" -nt "$STAMP" ] && echo "$d" || true
}

step0() {  # preflight: fail early, before hours of GPU time
    log "preflight"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    [ -z "$(git status --porcelain)" ] || log "WARNING: SIPE has uncommitted changes (recorded as dirty)"
    [ -d "$SCORPION_ROOT" ] || { log "no SCORPION_ROOT $SCORPION_ROOT"; exit 1; }
    n=$(find "$PLISM_DIR" -maxdepth 1 -name '*.tif.h5' 2>/dev/null | wc -l || true)
    [ "$n" -eq 91 ] || [ "$SMOKE" = 1 ] || {
        log "PLISM_DIR $PLISM_DIR has $n slides, not 91"; exit 1; }
    [ -d "$HEST_BENCH_DATA" ] || { log "no HEST_BENCH_DATA $HEST_BENCH_DATA"; exit 1; }
    for repo in ../HEST ../plism-benchmark; do
        b=$(git -C "$repo" rev-parse --abbrev-ref HEAD)
        [ "$b" = sipe ] || { log "$repo is on branch $b, not sipe"; exit 1; }
    done
    uv sync -q
    bench/run python -c "import plismbench, hest" || { log "bench env broken"; exit 1; }
    # Downloads the gated weights once (fails without access) and checks them.
    uv run python -c "
from sipe.model.encoders import build_encoder
m = build_encoder('$ENCODER').meta
assert (m.embed_dim, m.num_prefix_tokens) == ($EXPECT), m
print('$ENCODER ok:', m)"
}

step1() {  # SCORPION feature cache (train/val/test, CLS and GAP), ~15-30 min
    local c
    if [ -n "$SFX" ]; then  # the default run's cache holds every pooling
        c=$(MANIFEST=$MANIFEST_DEFAULT get CACHE 2>/dev/null || true)
        if [ -n "$c" ]; then
            log "step 1: reusing the cache of $MANIFEST_DEFAULT"
            record CACHE "$c"
            return
        fi
    fi
    log "step 1: $ENCODER feature cache on GPU $GPU0"
    CUDA_VISIBLE_DEVICES=$GPU0 uv run python -m sipe.features.extract \
        --tag "${P}${BK}_cache" --policy none --splits train val test \
        --poolings $CACHE_POOLINGS --encoder "$ENCODER" --data-root "$SCORPION_ROOT" \
        --workers "$WORKERS" $( [ "$SMOKE" = 1 ] && echo --limit-slides 1 ) \
        >"$LOGS/1_cache.log" 2>&1
    record CACHE "$(newest "${P}${BK}_cache")"
}

step2() {  # G0 (closed form, alpha = 1, K = 1) and the identity map (raw row)
    local cache
    cache=$(get CACHE)
    log "step 2: G0 on $cache"
    CUDA_VISIBLE_DEVICES=$GPU0 uv run python -m sipe.eval.mixture_sweep \
        --tag "${P}${BK}_${POOLING}_g0" --features-from "$cache" --pooling "$POOLING" --ks \
        --alphas 1.0 --reference-map none >"$LOGS/2_g0.log" 2>&1
    record G0_RUN "$(newest "scorpion_${P}${BK}_${POOLING}_g0")"
    uv run python -m sipe.bench.identity_map --encoder "$ENCODER" --pooling "$POOLING" \
        --tag "${P}identity_${BK}_${POOLING}" >"$LOGS/2_identity.log" 2>&1
    record IDENTITY_RUN "$(newest "${P}identity_${BK}_${POOLING}")"
}

step3() {  # CGM V1-S, backbone in the loop, 6000 steps
    local dir
    dir=$(uv run python -c "
from pathlib import Path
from sipe.cli import run_dir_name, write_git_info
d = Path('runs') / run_dir_name()
d.mkdir(parents=True)
write_git_info(d / 'git.txt')
print(d)")
    log "step 3: V1-S fit on GPU $GPU1 in $dir (log $LOGS/3_fit.log)"
    CUDA_VISIBLE_DEVICES=$GPU1 SIPE_RUN_DIR="$dir" uv run sipe fit \
        --config configs/consensus_gen_uni2h.yaml \
        --model.init_args.network.init_args.encoder="$ENCODER" \
        --model.init_args.network.init_args.pooling="$POOLING" \
        --data.init_args.data_root="$SCORPION_ROOT" \
        --data.init_args.num_workers="$WORKERS" \
        $( [ "$WORKERS" -eq 0 ] && echo --data.init_args.persistent_workers=false ) \
        $( [ "$SMOKE" = 1 ] && echo --trainer.max_steps=2 \
            --trainer.limit_train_batches=2 --trainer.limit_val_batches=1 \
            --trainer.val_check_interval=1 --model.init_args.init_batches=2 ) \
        >"$LOGS/3_fit.log" 2>&1
    test -f "$dir/map_cgm.pt" || { log "no $dir/map_cgm.pt"; exit 1; }
    record FIT "$dir"
}

step4() {  # SCORPION val/test: raw, G0 (in-run) and V1-S on the shared protocol
    local cache fit  # resolved first: a failing $(get) inside the arguments would not stop
    cache=$(get CACHE)
    fit=$(get FIT)
    log "step 4: cgm_select"
    CUDA_VISIBLE_DEVICES=$GPU0 uv run python -m sipe.eval.cgm_select \
        --tag "${P}${BK}${SFX}_cgm_select" --features-from "$cache" \
        --runs "$fit" --reference-map none --test all >"$LOGS/4_select.log" 2>&1
    record SELECT_RUN "$(newest "scorpion_${P}${BK}${SFX}_cgm_select")"
}

fit_tag() { local f; f=$(get FIT); echo "${P}cgm_v1s_${BK}${SFX}_${f##*_}"; }

plism() {  # <gpu> <map file> <tag>
    log "PLISM $3 on GPU $1"
    CUDA_VISIBLE_DEVICES=$1 bench/run python -m sipe.bench.plism --posthoc "$2" \
        --tag "$3" --download-dir "$PLISM_DIR" --workers "$WORKERS" \
        $( [ "$SMOKE" = 1 ] && echo --smoke --n-tiles 460 ) \
        >"$LOGS/5_plism_$3.log" 2>&1
    record "PLISM_$3" "$(newest "$3")"
}

hest() {  # <gpu> <map file> <tag>
    log "HEST $3 on GPU $1"
    CUDA_VISIBLE_DEVICES=$1 bench/run python -m sipe.bench.hest --posthoc "$2" \
        --tag "$3" --datasets "$( [ "$SMOKE" = 1 ] && echo SKCM || echo '*' )" \
        --bench-data-root "$HEST_BENCH_DATA" \
        --num-workers "$( [ "$WORKERS" -eq 0 ] && echo 0 || echo 1 )" \
        >"$LOGS/6_hest_$3.log" 2>&1
    record "HEST_$3" "$(newest "$3")"
}

run_bench() {  # <plism|hest>: raw + G0 on GPU0, V1-S on GPU1; fails if any run fails
    local kind=$1 identity g0 fit tag p0 p1 status=0 map
    identity="$(get IDENTITY_RUN)/map_identity.pt"
    g0="$(get G0_RUN)/map_G0_K1.pt"
    fit="$(get FIT)/map_cgm.pt"
    tag=$(fit_tag)
    for map in "$identity" "$g0" "$fit"; do
        [ -f "$map" ] || { log "missing map $map"; exit 1; }
    done
    { "$kind" "$GPU0" "$identity" "${P}${BK}_${POOLING}_ref"
      "$kind" "$GPU0" "$g0" "${P}${BK}_${POOLING}_g0"; } &
    p0=$!
    "$kind" "$GPU1" "$fit" "$tag" &
    p1=$!
    wait "$p0" || status=1
    wait "$p1" || status=1
    [ "$status" = 0 ] || { log "$kind: a run failed (logs in $LOGS)"; exit 1; }
}

step5() { run_bench plism; }
step6() { run_bench hest; }

sync_hint() {
    local dirs="" key dir
    for key in $(cut -d= -f1 "$MANIFEST" | sort -u); do  # latest value per key
        dir=$(get "$key")
        [ -e "$dir" ] && dirs="$dirs $dir"
    done
    log "done. Copy the run dirs (and the manifest) to the other machine:"
    echo "rsync -avR $dirs $MANIFEST $SYNC_TO"
}

if [ "$STEPS" = all ]; then
    step0
    step3 & fit=$!
    # A failing foreground step stops the background fit instead of leaving it on the GPU.
    trap 'kill_tree "$fit"' EXIT
    step1  # separate statements: `step1 && step2` would disable set -e inside step1
    step2
    wait "$fit"
    trap - EXIT
    step4
    step5
    step6
else
    for s in $STEPS; do [ "$s" = sync ] || "step$s"; done
fi
sync_hint
