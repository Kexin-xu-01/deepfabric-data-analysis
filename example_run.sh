#!/bin/bash
python prune_topic_graph.py \
  --input data/seo-graph-10tools-single-5depth.json \
  --out-filtered data/seo-graph-10tools-single-5depth-filt.json \
  --out-report data/report-seo-graph-10tools-single-5depth-filt.json \
  --model all-MiniLM-L6-v2 \
  --seed 0 \
  --parent_coherence 0.25 \
  --sibling_coherence_lower 0.2 \
  --sibling_coherence_upper 0.68
