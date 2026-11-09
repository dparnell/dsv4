#!/bin/sh
# Build an expert usage profile from a few varied prompts.
# usage: experiments/calibrate.sh [profile.json]
PROFILE=${1:-experiments/expert_profile.json}
BIN=./target/release/dsv4.exe
for p in \
  "Write a Python function that merges two sorted lists, with a short docstring." \
  "Solve step by step: a train travels 120 km in 1.5 hours, then 80 km in 1 hour. What is its average speed?" \
  "Write the opening paragraph of a short story about a lighthouse keeper who finds a message in a bottle." \
  "Explain why the sky is blue and why sunsets are red." ; do
  $BIN run -p "$p" --max-tokens 150 --temperature 0 --save-expert-profile "$PROFILE" 2>&1 | grep -E "^\[(stop|length)\]"
done
