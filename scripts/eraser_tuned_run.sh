#!/usr/bin/env bash
# Soft-shrinkage eraser (vendored vfm-geom code) with lambda re-selected by best SCORPION
# val top-1 (`--select-gain-frac 1.0`, finding 18) -> PLISM + HEST.
# The selected lambda comes from selected.json; tags softshrink_gap_<DELTAS>_lam<lambda>.
#
#   DELTAS=slide scripts/eraser_tuned_run.sh   # slide-level pairs (published eraser)
#   DELTAS=tile  scripts/eraser_tuned_run.sh   # tile-level pairs (pairing unit of G0/CGM)
#   DELTAS=tile STEPS=2 scripts/eraser_tuned_run.sh   # benchmarks of a recorded fit
#
# DELTAS   slide (default) or tile: pairing unit of the scanner deltas (B)
# LAMBDAS  lambda grid (default = the grid of the c54b tile sweep, a superset of db10's)
# Common environment: scripts/tuned_lib.sh.
source "$(dirname "$0")/tuned_lib.sh"

DELTAS="${DELTAS:-slide}"
case "$DELTAS" in slide|tile) ;; *) echo "DELTAS must be slide or tile" >&2; exit 1 ;; esac
LAMBDAS="${LAMBDAS:-0 0.05 0.1 0.2 0.3 0.4 0.5 0.7 1 2 3 10 30 100 300 1000}"
KEY="ERASER_${DELTAS}_valbest_RUN"

step0() { preflight; }

step1() {
    local tag="softshrink_gap_${DELTAS}_valbest"
    log "fit: eraser, $DELTAS deltas, lambdas $LAMBDAS, val argmax"
    # shellcheck disable=SC2086
    run "$LOGS/fit_eraser_${DELTAS}.log" env CUDA_VISIBLE_DEVICES="$GPU0" uv run python -m \
        sipe.eval.soft_shrinkage_sweep --tag "$tag" --features-from "$CACHE" \
        --poolings gap --deltas "$DELTAS" --lambdas $LAMBDAS --select-gain-frac 1.0 \
        --workers "$WORKERS" --device 0
    record "$KEY" "$(newest "scorpion_$tag")"
    [ "$DRY" = 1 ] || log "selected: $(json_get "$(get "$KEY")/selected.json" \
        "s['gap']['lambda'], s['gap']['val_top_1'], 'REGION GUARD' if s['gap']['region_guard_violated'] else '', 'edge' if s['gap']['at_grid_edge'] else ''")"
}

map_and_tag() {  # prints "<map file> <tag>"
    local d lam
    if [ "$DRY" = 1 ] && ! grep -q "^$KEY=" "$MANIFEST"; then echo "<fit>/eraser_gap_lam<lambda>.pt softshrink_gap_${DELTAS}_lam<lambda>"; return; fi
    d=$(get "$KEY")
    lam=$(json_get "$d/selected.json" "format(s['gap']['lambda'], 'g')")
    echo "$d/eraser_gap_lam$lam.pt softshrink_gap_${DELTAS}_lam$lam"
}

step2() { local m t; read -r m t <<<"$(map_and_tag)"; bench_both "$m" "$t"; }
step3() { local m t; read -r m t <<<"$(map_and_tag)"; plism "$GPU0" "$m" "$t"; }
step4() { local m t; read -r m t <<<"$(map_and_tag)"; hest "$GPU1" "$m" "$t"; }

main_steps
