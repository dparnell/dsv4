#!/bin/sh
# Compare pinned expert subsets against the full router on one prompt.
# usage: experiments/subset.sh "prompt" N [N...]
PROMPT=$1; shift
BIN=./target/release/dsv4.exe
for n in "$@"; do
  echo "=== subset $n ==="
  $BIN run -p "$PROMPT" --max-tokens 200 --temperature 0 --ram-cache-gb 2 \
    --expert-subset "$n" --expert-profile experiments/expert_profile.json 2>&1 \
    | grep -vE "^\s*$|INFO dsv4::stats: (model|GPU|dense|activations|free VRAM|request)"
done
