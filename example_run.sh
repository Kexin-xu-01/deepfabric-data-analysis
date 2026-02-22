#!/bin/bash
python prune_topic_graph.py \
  --input seo-graph-10tools-single-5depth.json \
  --out-filtered seo-graph-10tools-single-5depth-filt.json \
  --out-report report-seo-graph-10tools-single-5depth-filt.json \
  --model all-MiniLM-L6-v2 \
  --seed 0 \
  --parent_coherence 0.25 \
  --sibling_coherence_lower 0.2 \
  --sibling_coherence_upper 0.68
