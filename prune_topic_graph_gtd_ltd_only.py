# Full minimal workflow with preservation & separate report
import json
import copy
import math
from typing import Any, Dict, Tuple, List
from collections import deque, defaultdict

import numpy as np
import networkx as nx
from textwrap import shorten

# --------------------
# Utility functions (unchanged)
# --------------------
def safe_cosine_sim(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def gather_descendants_local(Gd: nx.DiGraph, start) -> set:
    """Return set of descendants (including start). Defensive: returns empty set if start not in Gd."""
    if start not in Gd:
        return set()
    seen = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        if u not in Gd:
            continue
        seen.add(u)
        children = Gd.nodes[u].get('children') or (list(Gd.successors(u)) if Gd.is_directed() else list(Gd.neighbors(u)))
        for c in children:
            if c not in seen:
                stack.append(c)
    return seen

def gather_descendants_local_defensive(Gd, start):
    """Safer alias used in several places."""
    if start not in Gd:
        return set()
    seen = set(); stack=[start]
    while stack:
        u = stack.pop()
        if u in seen: continue
        if u not in Gd: continue
        seen.add(u)
        children = Gd.nodes[u].get('children') or list(Gd.successors(u))
        for c in children:
            if c not in seen:
                stack.append(c)
    return seen

# --------------------
# Loader (your version) - returns G and raw JSON
# --------------------
def load_graph_from_json(path):
    """
    Load JSON (mapping node_id -> node_data) into a networkx.DiGraph.
    Ensures nodes have 'parents' and 'children' attrs, edges are added,
    and 'depth' is computed by BFS from roots.
    Returns (G, raw_json).
    """
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    nodes = data.get('nodes', {})
    G = nx.DiGraph()

    # --- add nodes with provided attrs (coerce ids to strings) ---
    for node_id, nd in nodes.items():
        key = str(node_id)
        attrs = {
            'topic': (nd.get('topic', '') or '').strip(),
            'metadata': nd.get('metadata', {}) or {},
            'parents': [str(p) for p in nd.get('parents', [])] if nd.get('parents') is not None else [],
            'children': [str(c) for c in nd.get('children', [])] if nd.get('children') is not None else []
        }
        # preserve embedding if present
        if 'embedding' in nd:
            attrs['embedding'] = nd['embedding']
        G.add_node(key, **attrs)

    # --- Add edges based on children lists (prefer children as source of truth) ---
    for n in list(G.nodes):
        for c in G.nodes[n].get('children', []):
            if c not in G:
                # create minimal node if child missing in JSON
                G.add_node(c, topic='', metadata={}, parents=[], children=[])
            G.add_edge(n, c)

    # --- If some nodes list parents but no corresponding edge exists, add them ---
    for n in list(G.nodes):
        for p in G.nodes[n].get('parents', []):
            if p not in G:
                G.add_node(p, topic='', metadata={}, parents=[], children=[])
            if not G.has_edge(p, n):
                G.add_edge(p, n)

    # --- Recompute consistent parents/children attributes from edges (authoritative) ---
    for n in list(G.nodes):
        G.nodes[n]['children'] = [str(c) for c in G.successors(n)]
        G.nodes[n]['parents']  = [str(p) for p in G.predecessors(n)]

    # --- Compute depth via BFS from all roots (in_degree == 0) ---
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        # fallback: if JSON contains a root_id, try it; otherwise use first node
        rid = data.get('root_id')
        if rid is not None and str(rid) in G.nodes:
            roots = [str(rid)]
        else:
            first = next(iter(G.nodes), None)
            roots = [first] if first is not None else []

    # initialize depths
    for n in G.nodes:
        G.nodes[n].pop('depth', None)

    for root in roots:
        G.nodes[root]['depth'] = 0
        queue = deque([root])
        while queue:
            cur = queue.popleft()
            cur_depth = G.nodes[cur].get('depth', 0)
            for child in G.successors(cur):
                # only set depth once (first encounter)
                if 'depth' not in G.nodes[child]:
                    G.nodes[child]['depth'] = cur_depth + 1
                    queue.append(child)

    return G, data

# --------------------
# Embedding computation (unchanged)
# --------------------
def compute_embeddings(G: nx.DiGraph, model_name: str = "all-MiniLM-L6-v2"):
    """
    Compute embeddings for nodes missing 'embedding' using SentenceTransformer.
    Stores embeddings in G.nodes[n]['embedding'] as numpy arrays (unit-normalized).
    Returns (G, model_instance).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers is required for compute_embeddings. "
                           "Install with `pip install sentence-transformers`. Original error: " + str(e))

    node_list = list(G.nodes())
    # build texts: prefer topic, then metadata.title, then uuid or node id
    texts = []
    for n in node_list:
        topic = (G.nodes[n].get('topic') or "").strip()
        if topic:
            texts.append(topic)
            continue
        title = G.nodes[n].get('metadata', {}).get('title', '')
        if title:
            texts.append(title)
            continue
        uid = G.nodes[n].get('metadata', {}).get('uuid')
        texts.append(uid if uid is not None else str(n))

    # detect which nodes need embeddings
    need_idx = [i for i, n in enumerate(node_list) if 'embedding' not in G.nodes[n] or G.nodes[n]['embedding'] is None]

    if not need_idx:
        # nothing to do; ensure embeddings are numpy arrays
        for n in node_list:
            emb = G.nodes[n].get('embedding')
            if emb is not None and not isinstance(emb, np.ndarray):
                G.nodes[n]['embedding'] = np.asarray(emb, dtype=float)
        return G, None

    model = SentenceTransformer(model_name)
    # only encode texts for nodes that need embeddings (faster)
    texts_to_encode = [texts[i] for i in need_idx]
    embeddings = model.encode(texts_to_encode, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # insert embeddings back to graph
    for j, i in enumerate(need_idx):
        n = node_list[i]
        emb = embeddings[j]
        # ensure python float array
        G.nodes[n]['embedding'] = np.asarray(emb, dtype=float)

    # ensure any existing embeddings are numpy arrays
    for n in node_list:
        emb = G.nodes[n].get('embedding')
        if emb is not None and not isinstance(emb, np.ndarray):
            G.nodes[n]['embedding'] = np.asarray(emb, dtype=float)

    return G, model

# --------------------
# Metrics (only GTD and LTD)
# --------------------
def compute_gtd_ltd(G: nx.DiGraph, seed_centroid: Any = None, raw_json: Dict = None) -> Dict[str, Dict]:
    """
    Compute per-node GTD and LTD.
    GTD: cosine similarity to seed_centroid (seed can be a node id in G or a raw vector).
    LTD: max cosine similarity to any parent (NaN if no parents).
    Returns dict: {'gtd': {node:val}, 'ltd': {node:val}}
    """
    # resolve seed vector
    resolved_seed = None
    if seed_centroid is None:
        # prefer raw_json root_id if present
        if raw_json is not None and raw_json.get('root_id') is not None:
            rid = str(raw_json['root_id'])
            if rid in G.nodes and 'embedding' in G.nodes[rid]:
                resolved_seed = G.nodes[rid].get('embedding')
        if resolved_seed is None and '0' in G.nodes:
            resolved_seed = G.nodes['0'].get('embedding')
        else:
            first = next(iter(G.nodes), None)
            if first is not None:
                resolved_seed = G.nodes[first].get('embedding')
    else:
        if (isinstance(seed_centroid, (str, int))) and (seed_centroid in G.nodes):
            resolved_seed = G.nodes[seed_centroid].get('embedding')
        else:
            try:
                resolved_seed = np.asarray(seed_centroid, dtype=float)
            except Exception:
                resolved_seed = None

    node_list = list(G.nodes)
    emb_map = {n: (np.asarray(G.nodes[n].get('embedding'), dtype=float) if G.nodes[n].get('embedding') is not None else None)
               for n in node_list}

    gtd = {}
    ltd = {}
    for n in node_list:
        # GTD
        if resolved_seed is not None and emb_map[n] is not None:
            gtd[n] = safe_cosine_sim(emb_map[n], resolved_seed)
        else:
            gtd[n] = float('nan')

        # LTD = max similarity to parents (no 1 - ...)
        parents = G.nodes[n].get('parents', []) or []
        if not parents:
            # try predecessors if parents missing
            parents = [str(p) for p in G.predecessors(n)]
        if not parents:
            ltd[n] = float('nan')
        else:
            parent_sims = []
            for p in parents:
                if p in emb_map and emb_map[p] is not None and emb_map[n] is not None:
                    parent_sims.append(safe_cosine_sim(emb_map[n], emb_map[p]))
            if parent_sims:
                ltd[n] = float(np.max(parent_sims))
            else:
                ltd[n] = float('nan')
    return {"gtd": gtd, "ltd": ltd}

# --------------------
# Flagging (enhanced)
# --------------------
DEFAULT_THRESHOLDS = {
    "depth1_gtd": 0.25,   # remove depth-1 nodes with GTD < this
    "gtd_neg": 0.0,       # remove nodes with GTD < this
    "ltd": 0.25           # remove nodes with LTD < this (insufficient parent similarity)
}

def flag_nodes_simple(
    G: nx.DiGraph,
    metrics: Dict[str, Dict],
    thresholds: Dict = None,
    max_parents: int = 3,
    max_children: int = 3,
    text_width: int = 80
) -> List[Dict]:
    """
    Return list of flagged nodes with:
    - GTD/LTD values
    - reasons
    - node topic
    - parent info
    - sample child info
    """

    thresholds = thresholds or DEFAULT_THRESHOLDS
    flagged = []

    for n in G.nodes():
        reasons = []
        g = metrics['gtd'].get(n)
        l = metrics['ltd'].get(n)

        # --- threshold logic ---
        if (G.nodes[n].get('depth') == 1) and (g is not None) and (not math.isnan(g)) and g < thresholds['depth1_gtd']:
            reasons.append("DEPTH1_LOW_GTD")

        if (g is not None) and (not math.isnan(g)) and g < thresholds['gtd_neg']:
            reasons.append("GTD_NEGATIVE")

        if (l is not None) and (not math.isnan(l)) and l < thresholds['ltd']:
            reasons.append("LOW_LTD")

        if not reasons:
            continue

        # --- content info ---
        topic = (G.nodes[n].get('topic') or "").strip()
        topic_short = shorten(topic, width=text_width) if topic else ""

        parents = list(G.predecessors(n))
        children = list(G.successors(n))

        parent_ids = parents[:max_parents]
        child_ids = children[:max_children]

        parent_topics = []
        for p in parent_ids:
            pt = (G.nodes[p].get('topic') or "").strip()
            parent_topics.append(shorten(pt, width=text_width))

        child_topics = []
        for c in child_ids:
            ct = (G.nodes[c].get('topic') or "").strip()
            child_topics.append(shorten(ct, width=text_width))

        flagged.append({
            "node_id": n,
            "depth": G.nodes[n].get('depth'),
            "reasons": reasons,
            "gtd": None if g is None or math.isnan(g) else float(g),
            "ltd": None if l is None or math.isnan(l) else float(l),
            "topic": topic_short,
            "parent_ids": parent_ids,
            "parent_topics": parent_topics,
            "child_ids": child_ids,
            "child_topics": child_topics,
        })

    return flagged

# --------------------
# Filter pipeline (unchanged)
# --------------------
def filter_graph_pipeline_simple(G_in: nx.DiGraph, metrics: Dict[str, Dict], thresholds: Dict = None, verbose: bool = True) -> Tuple[nx.DiGraph, Dict]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    G0 = copy.deepcopy(G_in)
    stats = {}

    # Step 1: depth-1 GTD
    G1 = copy.deepcopy(G0)
    depth1_bad = [n for n in list(G1.nodes)
                  if G1.nodes[n].get('depth') == 1
                  and (metrics['gtd'].get(n) is not None)
                  and (not math.isnan(metrics['gtd'][n]))
                  and metrics['gtd'][n] < thresholds['depth1_gtd']]
    rem1 = set()
    for n in depth1_bad:
        if n in G1:
            rem1.update(gather_descendants_local(G1, n))
    for u in list(rem1):
        if u in G1:
            G1.remove_node(u)
    stats['step1_removed'] = len(rem1)

    # Step 2: GTD < gtd_neg (on current graph)
    G2 = copy.deepcopy(G1)
    gtd_neg_nodes = [n for n in list(G2.nodes)
                     if (metrics['gtd'].get(n) is not None)
                     and (not math.isnan(metrics['gtd'][n]))
                     and metrics['gtd'][n] < thresholds['gtd_neg']]
    rem2 = set()
    for n in gtd_neg_nodes:
        if n in G2:
            rem2.update(gather_descendants_local(G2, n))
    for u in list(rem2):
        if u in G2:
            G2.remove_node(u)
    stats['step2_removed'] = len(rem2)

    # Step 3: LTD < ltd_thresh (on current graph) — LTD is similarity to parent
    G3 = copy.deepcopy(G2)
    ltd_nodes = [n for n in list(G3.nodes)
                 if (metrics['ltd'].get(n) is not None)
                 and (not math.isnan(metrics['ltd'][n]))
                 and metrics['ltd'][n] < thresholds['ltd']]
    rem3 = set()
    for n in ltd_nodes:
        if n in G3:
            rem3.update(gather_descendants_local(G3, n))
    for u in list(rem3):
        if u in G3:
            G3.remove_node(u)
    stats['step3_removed'] = len(rem3)

    stats['counts'] = {
        "orig": G0.number_of_nodes(),
        "after_step1": G1.number_of_nodes(),
        "after_step2": G2.number_of_nodes(),
        "after_step3": G3.number_of_nodes()
    }
    if verbose:
        print("Filter pipeline summary:", stats)
    return G3, stats

# --------------------
# New: preserve original node fields & save report
# --------------------
def save_filtered_graph_preserve_original(original_raw_json: Dict,
                                          G_filtered: nx.DiGraph,
                                          out_path: str):
    """
    Save filtered graph preserving original node fields (no embeddings).
    original_raw_json: the raw JSON object loaded from input file (so we know original node keys).
    G_filtered: filtered networkx.DiGraph with node ids as strings.
    """
    orig_nodes_map = original_raw_json.get('nodes', {})

    out_nodes = {}
    # The input JSON used mapping node_id -> node_data; preserve that shape
    for original_node_id, original_node_data in orig_nodes_map.items():
        sid = str(original_node_id)
        if sid not in G_filtered.nodes:
            continue
        preserved = {}
        for k, v in original_node_data.items():
            if k == 'embedding':
                continue
            if k == 'children':
                # reflect filtered children
                preserved['children'] = [c for c in G_filtered.nodes[sid].get('children', []) if c in G_filtered.nodes]
            elif k == 'parents':
                preserved['parents'] = [p for p in G_filtered.nodes[sid].get('parents', []) if p in G_filtered.nodes]
            else:
                # if filtered graph has an updated value, prefer it (but not embedding)
                if k in G_filtered.nodes[sid]:
                    if k == 'embedding':
                        continue
                    preserved[k] = G_filtered.nodes[sid][k]
                else:
                    preserved[k] = v
        out_nodes[sid] = preserved

    out = {
        **({'root_id': original_raw_json.get('root_id')} if original_raw_json.get('root_id') is not None else {}),
        'nodes': out_nodes
    }

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"Saved filtered graph (preserving original fields, no embeddings) to: {out_path}")

def compute_simple_global_metrics(metrics: Dict[str, Dict], G: nx.DiGraph) -> Dict:
    """
    Compute a few simple global aggregates for the report:
    - counts, mean/median GTD/LTD, number NaNs, nodes per depth distribution.
    """
    node_list = list(G.nodes())
    gtd_vals_all = np.array([metrics['gtd'].get(n) for n in node_list], dtype=float)
    ltd_vals_all = np.array([metrics['ltd'].get(n) for n in node_list], dtype=float)

    def stats(arr):
        mask = ~np.isnan(arr)
        if mask.sum() == 0:
            return {'count': 0}
        return {
            'count': int(mask.sum()),
            'min': float(np.nanmin(arr)),
            '25%': float(np.nanpercentile(arr[mask], 25)),
            'median': float(np.nanpercentile(arr[mask], 50)),
            '75%': float(np.nanpercentile(arr[mask], 75)),
            'max': float(np.nanmax(arr))
        }

    depth_counts = defaultdict(int)
    for n in node_list:
        d = G.nodes[n].get('depth')
        depth_counts[int(d) if d is not None else -1] += 1

    return {
        'gtd_stats': stats(gtd_vals_all),
        'ltd_stats': stats(ltd_vals_all),
        'total_nodes': len(node_list),
        'depth_counts': dict(depth_counts),
        'nodes_missing_embedding': sum(1 for n in node_list if 'embedding' not in G.nodes[n])
    }

def save_report_json(original_raw_json: Dict,
                     metrics: Dict[str, Dict],
                     global_metrics: Dict,
                     flagged: List[Dict],
                     removals: Dict[str, List[str]],
                     out_path: str):
    """
    Write a report JSON containing:
    - summary: counts and global_metrics
    - metrics_per_node: mapping node_id -> {gtd, ltd}
    - flagged_nodes: flagged list (with reasons)
    - removals: which node ids removed at each step
    """
    metrics_out = {}
    all_node_ids = list(original_raw_json.get('nodes', {}).keys())
    for nid in all_node_ids:
        sid = str(nid)
        g = metrics['gtd'].get(sid)
        l = metrics['ltd'].get(sid)
        metrics_out[sid] = {
            'gtd': None if g is None or (isinstance(g, float) and math.isnan(g)) else float(g),
            'ltd': None if l is None or (isinstance(l, float) and math.isnan(l)) else float(l)
        }

    report = {
        'summary': {
            'original_node_count': len(all_node_ids),
            'filtered_node_count': global_metrics.get('total_nodes'),
            **global_metrics
        },
        'metrics_per_node': metrics_out,
        'flagged_nodes': flagged,
        'removals': removals
    }

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"Saved report to: {out_path}")

# --------------------
# Run wrapper (integrated save)
# --------------------
def run_simple_workflow_and_save(input_json_path: str,
                                 output_filtered_json_path: str,
                                 output_report_path: str,
                                 model_name: str = "all-MiniLM-L6-v2",
                                 seed_centroid: Any = "0",
                                 thresholds: Dict = None,
                                 save_filtered_graph: bool = True) -> Dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS

    # load original graph & raw json
    G, raw_json = load_graph_from_json(input_json_path)

    # compute embeddings if needed
    G, model = compute_embeddings(G, model_name=model_name)

    # compute metrics
    metrics = compute_gtd_ltd(G, seed_centroid=seed_centroid, raw_json=raw_json)

    # flag nodes (for reporting)
    flagged = flag_nodes_simple(G, metrics, thresholds=thresholds)

    # run filter pipeline but capture which nodes are removed at each step
    G0 = copy.deepcopy(G)
    removals = {'step1': [], 'step2': [], 'step3': []}

    # Step 1: depth1 GTD
    G1 = copy.deepcopy(G0)
    depth1_bad = [n for n in list(G1.nodes)
                  if G1.nodes[n].get('depth') == 1
                  and (metrics['gtd'].get(n) is not None)
                  and (not math.isnan(metrics['gtd'][n]))
                  and metrics['gtd'][n] < thresholds['depth1_gtd']]
    rem1 = set()
    for n in depth1_bad:
        if n in G1:
            desc = gather_descendants_local_defensive(G1, n)
            rem1.update(desc)
    removals['step1'] = sorted(list(rem1))
    for u in list(rem1):
        if u in G1: G1.remove_node(u)

    # Step 2: GTD < gtd_neg
    G2 = copy.deepcopy(G1)
    gtd_neg_nodes = [n for n in list(G2.nodes)
                     if (metrics['gtd'].get(n) is not None)
                     and (not math.isnan(metrics['gtd'][n]))
                     and metrics['gtd'][n] < thresholds['gtd_neg']]
    rem2 = set()
    for n in gtd_neg_nodes:
        if n in G2:
            desc = gather_descendants_local_defensive(G2, n)
            rem2.update(desc)
    removals['step2'] = sorted(list(rem2))
    for u in list(rem2):
        if u in G2: G2.remove_node(u)

    # Step 3: LTD < ltd_thresh
    G3 = copy.deepcopy(G2)
    ltd_nodes = [n for n in list(G3.nodes)
                 if (metrics['ltd'].get(n) is not None)
                 and (not math.isnan(metrics['ltd'][n]))
                 and metrics['ltd'][n] < thresholds['ltd']]
    rem3 = set()
    for n in ltd_nodes:
        if n in G3:
            desc = gather_descendants_local_defensive(G3, n)
            rem3.update(desc)
    removals['step3'] = sorted(list(rem3))
    for u in list(rem3):
        if u in G3: G3.remove_node(u)

    G_filtered = G3

    # compute global metrics on filtered graph for report
    global_metrics = compute_simple_global_metrics(metrics, G_filtered)

    # save filtered graph (preserve original fields; remove embedding)
    if save_filtered_graph:
        save_filtered_graph_preserve_original(raw_json, G_filtered, output_filtered_json_path)

    # save report
    save_report_json(raw_json, metrics, global_metrics, flagged, removals, output_report_path)

    return {
        "G": G,
        #"metrics": metrics,
        "flagged": flagged,
        "G_filtered": G_filtered,
        "removals": removals,
        "global_metrics": global_metrics,
        "stats": {
            "orig": G0.number_of_nodes(),
            "after_step1": G1.number_of_nodes(),
            "after_step2": G2.number_of_nodes(),
            "after_step3": G3.number_of_nodes()
        }
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run minimal GTD/LTD workflow and save results.")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON (mapping node_id -> node_data).")
    parser.add_argument("--out-filtered", "-o", required=True, help="Path to write filtered JSON (preserve original fields).")
    parser.add_argument("--out-report", "-r", required=True, help="Path to write report JSON.")
    parser.add_argument("--model", "-m", default="all-MiniLM-L6-v2", help="SentenceTransformers model name.")
    parser.add_argument("--seed", "-s", default="0", help="Seed centroid node id or provide a vector string (optional).")
    parser.add_argument("--depth1-gtd", type=float, default=0.25)
    parser.add_argument("--gtd-neg", type=float, default=0.0)
    parser.add_argument("--ltd", type=float, default=0.25)
    args = parser.parse_args()

    thresholds = {"depth1_gtd": args.depth1_gtd, "gtd_neg": args.gtd_neg, "ltd": args.ltd}

    # run workflow
    res = run_simple_workflow_and_save(
        input_json_path=args.input,
        output_filtered_json_path=args.out_filtered,
        output_report_path=args.out_report,
        model_name=args.model,
        seed_centroid=args.seed,
        thresholds=thresholds,
        save_filtered_graph=True
    )

    print("Done. Stats:", res["stats"])