#!/usr/bin/env bash
# G0 (closed-form consensus shrinkage, P = C (C + alpha N)^-1) with alpha re-selected by
# best SCORPION val top-1 (`--select-gain-frac 1.0`, finding 18) -> PLISM + HEST.
# The selected alpha comes from selected.json; tags mixture_G0_K1_a<alpha>.
#
#   scripts/g0_tuned_run.sh                      # val-best alpha over the original grid (8)
#   ALPHAS="1 4" LABEL=a4 scripts/g0_tuned_run.sh  # sensitivity row, alpha = 4
#   STEPS="2" LABEL=valbest scripts/g0_tuned_run.sh  # benchmarks of a recorded fit
#
# ALPHAS  alpha grid (default = the grid of the 6fdf mixture sweep); 0 and 1 are added
#         by the sweep. LABEL names the fit in the manifest (G0_<LABEL>_RUN; default
#         valbest). Common environment: scripts/tuned_lib.sh.
source "$(dirname "$0")/tuned_lib.sh"

ALPHAS="${ALPHAS:-0 0.1 0.25 0.5 1 2 4 8 16 32}"
LABEL="${LABEL:-valbest}"
KEY="G0_${LABEL}_RUN"

step0() { preflight; }

step1() {  # G0 only (`--ks` alone), ridge 1e-4 as 6fdf; reference = eraser lambda 0.5
    local tag="mixture_G0_${LABEL}"
    log "fit: G0, alphas $ALPHAS, val argmax"
    # shellcheck disable=SC2086
    run "$LOGS/fit_g0_${LABEL}.log" env CUDA_VISIBLE_DEVICES="$GPU0" uv run python -m \
        sipe.eval.mixture_sweep --tag "$tag" --features-from "$CACHE" --pooling gap \
        --ks --alphas $ALPHAS --select-gain-frac 1.0
    record "$KEY" "$(newest "scorpion_$tag")"
    [ "$DRY" = 1 ] || log "selected: $(json_get "$(get "$KEY")/selected.json" \
        "s['G0_K1']['alpha'], s['G0_K1']['val_top_1'], 'edge' if s['G0_K1']['at_grid_edge'] else ''")"
}

map_and_tag() {  # prints "<map file> <tag>"
    local d alpha
    if [ "$DRY" = 1 ] && ! grep -q "^$KEY=" "$MANIFEST"; then echo "<fit>/map_G0_K1.pt mixture_G0_K1_a<alpha>"; return; fi
    d=$(get "$KEY")
    alpha=$(json_get "$d/selected.json" "format(s['G0_K1']['alpha'], 'g')")
    echo "$d/map_G0_K1.pt mixture_G0_K1_a$alpha"
}

step2() { local m t; read -r m t <<<"$(map_and_tag)"; bench_both "$m" "$t"; }
step3() { local m t; read -r m t <<<"$(map_and_tag)"; plism "$GPU0" "$m" "$t"; }
step4() { local m t; read -r m t <<<"$(map_and_tag)"; hest "$GPU1" "$m" "$t"; }

main_steps
