#!/bin/bash
python prune_topic_graph.py \
  --input seo-graph-10tools-single-5depth.json \
  --out-filtered seo-graph-10tools-single-5depth-3filt.json \
  --out-report report-seo-graph-10tools-single-5depth-3filt.json \
  --model all-MiniLM-L6-v2 \
  --seed 0 \
  --depth1-gtd 0.25 --gtd-neg 0.0 --ltd 0.25
