#!/usr/bin/env bash
# Scanner-count study on H0-mini GAP (docs/experiment-log.md): G0 / eraser / CGM-init
# (closed form) and CGM V0-S / V1-S (trained) fitted on n of the 4 SCORPION scanners
# without P1000 (AT2, DP200, GT450, Philips), n = 2 (6 subsets), 3 (4 subsets), 4 (the
# existing LOSO fits without P1000: V1-S a560, V0-S 4b28). Full-data references:
# V1-S 7b71, V0-S a602.
#
# Fixed image budget per step: batch_size = 512 / n locations x n views (n = 2: 256,
# n = 3: 171 (513 images), n = 4: 128 = the LOSO fits). Everything else is
# configs/consensus_gen.yaml (V0-S: encoder_hidden_dim = 0); nothing is re-selected.
#
#   scripts/scanner_count_run.sh                # every step
#   STEPS="2 3" scripts/scanner_count_run.sh    # some steps (outputs from the manifest)
#
# Steps: 1 fits (20, split over GPU0 / GPU1), 2 sipe.eval.scanner_count on val and
# test, 3 PLISM through sipe.bench.plism_apply (no backbone pass: the map is applied to
# the raw features of the H0-mini GAP reference run), first a reproduction check (full
# G0 map, tag scapply_check_g0_full, to compare with mixture_G0_K1), then the jobs of
# step 2 (plism_jobs.tsv) for the models in PLISM_MODELS, in that order.
# Every step is resumable: finished items are in runs/scanner_count_manifest.env
# (KEY=path) and skipped.
#
# Environment (defaults = lxbelshark):
#   GPU0, GPU1        CUDA indices (0, 1)
#   WORKERS           dataloader workers (8; 0 inside claude-box)
#   SCORPION_ROOT     SCORPION tiles (data/processed/SCORPION_tiles_224px_0p5mpp)
#   PLISM_MODELS      PLISM order and filter ("g0 v1s v0s eraser cgminit"); about
#                     35 min per map on one GPU, so 33 maps for the first three
#   SUBSETS           fit subsets (default: the 10 below)
#   ARMS              fit arms ("V1-S V0-S")
set -euo pipefail
cd "$(dirname "$0")/.."

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
WORKERS="${WORKERS:-8}"
SCORPION_ROOT="${SCORPION_ROOT:-data/processed/SCORPION_tiles_224px_0p5mpp}"
STEPS="${STEPS:-all}"
PLISM_MODELS="${PLISM_MODELS:-g0 v1s v0s eraser cgminit}"
ARMS="${ARMS:-V1-S V0-S}"
SUBSETS="${SUBSETS:-AT2,DP200 AT2,GT450 AT2,Philips DP200,GT450 DP200,Philips GT450,Philips AT2,DP200,GT450 AT2,DP200,Philips AT2,GT450,Philips DP200,GT450,Philips}"
UNIVERSE="AT2 DP200 GT450 Philips"
CACHE=runs/bench/2026-10-13_15-44-35_lxbelshark_d6f7_scorpion_softshrink_h0mini_gap
PLISM_RAW=runs/bench/2026-10-13_10-02-08_lxbelshark_053f_h0_mini_gap_ref
G0_FULL_MAP=runs/bench/2026-10-13_21-47-56_claude-box_6fdf_scorpion_mixture_gap/map_G0_K1.pt
REF_FITS="runs/2026-10-16_17-48-06_lxbelshark_a560 runs/2026-10-16_18-27-40_lxbelshark_4b28
runs/2026-10-14_22-30-38_lxbelshark_7b71 runs/2026-10-15_08-50-48_lxbelshark_a602"
MANIFEST=runs/scanner_count_manifest.env
LOGS="runs/scanner_count_logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$LOGS"
touch "$MANIFEST"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOGS/run.log"; }
record() { echo "$1=$2" >>"$MANIFEST"; }
get() { grep "^$1=" "$MANIFEST" | tail -1 | cut -d= -f2- || true; }

new_run_dir() {
    uv run python -c "
from pathlib import Path
from sipe.cli import run_dir_name, write_git_info
d = Path('runs') / run_dir_name()
d.mkdir(parents=True)
write_git_info(d / 'git.txt')
print(d)"
}

fit_one() {  # <gpu> <arm> <subset a,b[,c]>
    local gpu=$1 arm=$2 subset=$3 key dir n batch hidden
    key="FIT_${arm//-/}_${subset//,/-}"
    if [ -n "$(get "$key")" ]; then log "skip $key (done)"; return; fi
    n=$(echo "$subset" | tr ',' '\n' | wc -l)
    batch=$(( (512 + n / 2) / n ))  # 256, 171, 128
    hidden=$( [ "$arm" = V0-S ] && echo 0 || echo 1024 )
    dir=$(new_run_dir)
    log "fit $arm on [$subset] (n=$n, $batch locations) on GPU $gpu in $dir"
    CUDA_VISIBLE_DEVICES=$gpu SIPE_RUN_DIR="$dir" uv run sipe fit \
        --config configs/consensus_gen.yaml \
        --model.init_args.network.init_args.encoder_hidden_dim="$hidden" \
        --data.init_args.scanners="[$subset]" \
        --data.init_args.views="$n" \
        --data.init_args.batch_size="$batch" \
        --data.init_args.data_root="$SCORPION_ROOT" \
        --data.init_args.num_workers="$WORKERS" \
        $( [ "$WORKERS" -eq 0 ] && echo --data.init_args.persistent_workers=false ) \
        >"$LOGS/fit_${arm}_${subset//,/-}.log" 2>&1 || { log "fit $key failed"; return 1; }
    test -f "$dir/map_cgm.pt" || { log "no $dir/map_cgm.pt"; return 1; }
    record "$key" "$dir"
}

step1() {
    local jobs=() i=0 j
    for arm in $ARMS; do for s in $SUBSETS; do jobs+=("$arm $s"); done; done
    worker() {  # <gpu> <parity>
        local k=0
        for j in "${jobs[@]}"; do
            if [ $((k % 2)) -eq "$2" ]; then fit_one "$1" $j || true; fi
            k=$((k + 1))
        done
    }
    worker "$GPU0" 0 & local p0=$!
    worker "$GPU1" 1 & local p1=$!
    wait $p0 && wait $p1
}

step2() {
    local runs=() key
    for key in $(cut -d= -f1 "$MANIFEST" | grep '^FIT_' | sort -u); do runs+=("$(get "$key")"); done
    for split in val test; do
        log "step 2: scanner_count --split $split (${#runs[@]} trained fits + references)"
        CUDA_VISIBLE_DEVICES=$GPU0 uv run python -m sipe.eval.scanner_count \
            --tag "scanner_count_$split" --features-from "$CACHE" --universe $UNIVERSE \
            --split "$split" --runs "${runs[@]}" $REF_FITS \
            >"$LOGS/2_scanner_count_$split.log" 2>&1
        record "SC_$(echo "$split" | tr a-z A-Z)" \
            "$(ls -d runs/bench/*_scorpion_scanner_count_"$split" | tail -1)"
    done
}

plism_one() {  # <gpu> <tag> <map>
    local key="PLISM_$2"
    if [ -n "$(get "$key")" ]; then log "skip $2 (done)"; return; fi
    log "PLISM $2 on GPU $1"
    CUDA_VISIBLE_DEVICES=$1 bench/run python -m sipe.bench.plism_apply \
        --features-from "$PLISM_RAW" --posthoc "$3" --tag "$2" \
        >"$LOGS/plism_$2.log" 2>&1 || { log "PLISM $2 failed"; return 1; }
    record "$key" "$(ls -d runs/bench/*_"$2" | tail -1)"
}

step3() {
    local jobs_tsv list=() m
    jobs_tsv="$(get SC_VAL)/plism_jobs.tsv"
    test -f "$jobs_tsv" || { log "no $jobs_tsv (run step 2)"; exit 1; }
    list+=("scapply_check_g0_full $G0_FULL_MAP")
    for m in $PLISM_MODELS; do
        while IFS=$'\t' read -r tag _ map; do list+=("$tag $map"); done \
            < <(tail -n +2 "$jobs_tsv" | awk -F'\t' -v m="$m" '$1 ~ "^sc_"m"_"')
    done
    log "step 3: ${#list[@]} PLISM jobs"
    worker() {  # <gpu> <parity>
        local k=0 j
        for j in "${list[@]}"; do
            if [ $((k % 2)) -eq "$2" ]; then plism_one "$1" $j || true; fi
            k=$((k + 1))
        done
    }
    worker "$GPU0" 0 & local p0=$!
    worker "$GPU1" 1 & local p1=$!
    wait $p0 && wait $p1
}

for s in 1 2 3; do
    if [ "$STEPS" = all ] || [[ " $STEPS " == *" $s "* ]]; then "step$s"; fi
done
log "done; manifest $MANIFEST"
