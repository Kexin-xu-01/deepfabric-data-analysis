# Prune Topic Graph

**Coherence-Based Hierarchical Graph Pruning**

This project prunes hierarchical topic graphs using three coherence
metrics:

-   **Global Coherence** --- cosine similarity to the root topic
    embedding\
-   **Parent Coherence** --- maximum cosine similarity to parent
    node(s)\
-   **Sibling Coherence** --- mean cosine similarity to sibling nodes

------------------------------------------------------------------------

## Workflow Overview

The pipeline:

1.  Loads a topic graph from JSON\
2.  Computes embeddings (if missing)\
3.  Computes coherence metrics per node:
    -   Global Coherence
    -   Parent Coherence
    -   Sibling Coherence
4.  Flags low-quality nodes\
5.  Applies a **4-step cascading pruning pipeline**\
6.  Saves:
    -   A **filtered graph JSON** (preserving original fields, no
        embeddings)
    -   A **report JSON** with full metrics and pruning details

------------------------------------------------------------------------

# Installation

## 1️⃣ Clone the repository

``` bash
git clone https://github.com/Kexin-xu-01/deepfabric-data-analysis.git
cd deepfabric-data-analysis
```

## 2️⃣ Create a virtual environment (recommended)

``` bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
```

## 3️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

If no `requirements.txt` exists:

``` bash
pip install networkx numpy sentence-transformers scikit-learn
```

------------------------------------------------------------------------

# Input JSON Format

``` json
{
  "root_id": "0",
  "nodes": {
    "0": {
      "topic": "Root topic",
      "metadata": {},
      "parents": [],
      "children": ["1", "2"]
    },
    "1": {
      "topic": "Child topic",
      "metadata": {},
      "parents": ["0"],
      "children": []
    }
  }
}
```

Embeddings are computed automatically if missing.

------------------------------------------------------------------------

# Running the Script

Main script:

    prune_topic_graph.py

## macOS / Linux

``` bash
python prune_topic_graph.py \
  --input seo-graph-10tools-single-5depth.json \
  --out-filtered seo-graph-10tools-single-5depth-filtered.json \
  --out-report report-seo-graph-10tools-single-5depth.json \
  --model all-MiniLM-L6-v2 \
  --seed 0 \
  --parent_coherence 0.25 \
  --sibling_coherence_lower 0.2 \
  --sibling_coherence_upper 0.68
```

------------------------------------------------------------------------

# Pruning Pipeline

Nodes that fail a threshold are removed **along with all their
descendants**.

### Step 1 --- Remove Opposite-Direction Nodes

Global Coherence \< 0

### Step 2 --- Remove Off-Topic Children

Parent Coherence \< threshold (default 0.25)

### Step 3 --- Remove Outlier Siblings

Sibling Coherence \< lower threshold (default 0.2)

### Step 4 --- Remove Redundant Siblings

Sibling Coherence \> upper threshold (default 0.68)

------------------------------------------------------------------------

# Output Files

## 1️⃣ Filtered Graph JSON

-   Same structure as input\
-   Preserves original node fields\
-   Removes pruned nodes\
-   Does **not** include embeddings

## 2️⃣ Report JSON

Contains:

-   Original and filtered node counts\
-   Global / Parent / Sibling coherence statistics\
-   Node count per depth (before and after filtering)\
-   Per-node coherence values\
-   Flagging reasons\
-   Removal breakdown by stage

------------------------------------------------------------------------

# Metric Definitions

### Global Coherence

cosine(embedding(node), embedding(root))

### Parent Coherence

max cosine(embedding(node), embedding(parent))

### Sibling Coherence

mean cosine(embedding(node), embedding(siblings))

------------------------------------------------------------------------

# Troubleshooting

### No Nodes Removed?

Increase thresholds:

``` bash
--parent_coherence 0.4
--sibling_coherence_lower 0.3
```

### Too Many Nodes Removed?

Lower thresholds:

``` bash
--parent_coherence 0.15
--sibling_coherence_lower 0.1
```

------------------------------------------------------------------------

This framework enforces global alignment, local consistency, sibling
cohesion, and redundancy control in hierarchical topic graphs.
