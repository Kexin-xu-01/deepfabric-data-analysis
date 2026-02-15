# Prune Topic Graph

**GTD + LTD Based Graph Pruning**

This project prunes hierarchical topic graphs using:

-   **GTD (Global Topic Drift)** --- cosine similarity to the root topic
-   **LTD (Local Topic Drift)** --- maximum cosine similarity to parent
    node(s)

The workflow:

1.  Loads a topic graph from JSON\
2.  Computes embeddings (if missing)\
3.  Computes GTD and LTD per node\
4.  Flags low-quality nodes\
5.  Removes nodes using configurable thresholds\
6.  Saves:
    -   A **filtered graph JSON** (preserving original fields, no
        embeddings)
    -   A **report JSON** with metrics and pruning details

------------------------------------------------------------------------

# 📦 Installation

## 1️⃣ Clone the repository

``` bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
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

If you don't have a `requirements.txt`, install manually:

``` bash
pip install networkx numpy sentence-transformers scikit-learn
```

------------------------------------------------------------------------

# 📂 Input JSON Format

The input graph must be structured like:

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

Embeddings are computed automatically if not present.

------------------------------------------------------------------------

# 🚀 Running the Script

Main script:

    prune_topic_graph.py

## macOS / Linux

``` bash
python prune_topic_graph.py \
  --input seo-graph-10tools-single-5depth.json \
  --out-filtered filtered_graph.json \
  --out-report report.json \
  --model all-MiniLM-L6-v2 \
  --seed 0 \
  --depth1-gtd 0.25 \
  --gtd-neg 0.0 \
  --ltd 0.25
```

## Windows PowerShell

``` powershell
python prune_topic_graph.py --input seo-graph-10tools-single-5depth.json --out-filtered filtered_graph.json --out-report report.json --model all-MiniLM-L6-v2 --seed 0 --depth1-gtd 0.25 --gtd-neg 0.0 --ltd 0.25
```

------------------------------------------------------------------------

# 📊 Output Files

## 1️⃣ Filtered Graph JSON

-   Same structure as input\
-   Preserves original node fields\
-   Removes pruned nodes\
-   Does **not** include embeddings

## 2️⃣ Report JSON

Contains:

-   Summary statistics\
-   GTD and LTD per node\
-   Global aggregates\
-   Flagged nodes\
-   Nodes removed at each pruning stage

------------------------------------------------------------------------

# 🧠 Methodology

### Global Topic Drift (GTD)

GTD(n) = cosine(embedding(n), embedding(root))

### Local Topic Drift (LTD)

LTD(n) = max cosine(embedding(n), embedding(parent))

------------------------------------------------------------------------

# 🛠 Troubleshooting

### First run downloads embedding model

The SentenceTransformer model downloads automatically.

### No nodes removed?

Try increasing thresholds:

``` bash
--depth1-gtd 0.5
--ltd 0.5
```

------------------------------------------------------------------------

# 📜 License

MIT (or specify your license)
