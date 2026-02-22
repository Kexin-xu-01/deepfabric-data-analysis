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
def _resolve_seed_vector(G: nx.DiGraph, seed_centroid: Any = None, raw_json: Dict = None):
    """
    Internal helper to resolve seed_centroid to a numpy array (or None).
    Behavior preserved from previous implementation:
      - if seed_centroid is None: prefer raw_json['root_id'] -> node '0' -> first node fallback
      - if seed_centroid is a node id in G, use that node's embedding
      - otherwise try to convert seed_centroid to a numpy array
    Returns: numpy array or None
    """
    resolved_seed = None
    if seed_centroid is None:
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
    # normalize to numpy array where possible
    if resolved_seed is not None and not isinstance(resolved_seed, np.ndarray):
        try:
            resolved_seed = np.asarray(resolved_seed, dtype=float)
        except Exception:
            resolved_seed = None
    return resolved_seed

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
# Metrics (renamed outputs)
# --------------------
def compute_global_coherence(G: nx.DiGraph, seed_centroid: Any = None, raw_json: Dict = None) -> Dict[str, float]:
    """
    Compute per-node global_coherence: cosine similarity to seed_centroid.
    seed_centroid may be:
      - None (use raw_json['root_id'] -> '0' -> first node fallback)
      - a node id in G
      - a raw vector (list/np.array)
    Returns dict: {node_id: coherence_value} with float('nan') when not computable.
    """
    resolved_seed = _resolve_seed_vector(G, seed_centroid=seed_centroid, raw_json=raw_json)

    node_list = list(G.nodes)
    emb_map = {
        n: (np.asarray(G.nodes[n].get('embedding'), dtype=float) if G.nodes[n].get('embedding') is not None else None)
        for n in node_list
    }

    global_coherence = {}
    for n in node_list:
        if resolved_seed is not None and emb_map[n] is not None:
            try:
                global_coherence[n] = safe_cosine_sim(emb_map[n], resolved_seed)
            except Exception:
                global_coherence[n] = float('nan')
        else:
            global_coherence[n] = float('nan')
    return global_coherence


def compute_parent_coherence(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute per-node parent_coherence: the maximum cosine similarity to any parent.
    If a node has no parents (or embeddings missing) returns float('nan') for that node.
    Returns dict: {node_id: coherence_value}
    """
    node_list = list(G.nodes)
    emb_map = {
        n: (np.asarray(G.nodes[n].get('embedding'), dtype=float) if G.nodes[n].get('embedding') is not None else None)
        for n in node_list
    }

    parent_coherence = {}
    for n in node_list:
        parents = G.nodes[n].get('parents', []) or []
        if not parents:
            # fallback to predecessors if parents attr missing
            parents = [str(p) for p in G.predecessors(n)]
        if not parents:
            parent_coherence[n] = float('nan')
            continue

        sims = []
        emb_n = emb_map.get(n)
        if emb_n is None:
            parent_coherence[n] = float('nan')
            continue

        for p in parents:
            # allow numeric parent ids stored as ints in JSON
            pstr = p if (p in emb_map) else str(p)
            if pstr in emb_map and emb_map[pstr] is not None:
                try:
                    sims.append(safe_cosine_sim(emb_n, emb_map[pstr]))
                except Exception:
                    continue

        if not sims:
            parent_coherence[n] = float('nan')
        else:
            parent_coherence[n] = float(np.max(sims))
    return parent_coherence

def compute_sibling_coherence(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute per-node local cosine similarity to siblings (nodes that share a parent).
    Returns dict: node_id -> sibling_coherence_mean (float) or NaN.
    """
    node_list = list(G.nodes)

    # build embedding map (None if missing)
    emb_map = {
        n: (np.asarray(G.nodes[n].get('embedding'), dtype=float)
            if G.nodes[n].get('embedding') is not None else None)
        for n in node_list
    }

    sibling_coherence = {}

    for n in node_list:
        # gather parents
        parents = G.nodes[n].get('parents', []) or []
        if not parents:
            # fallback to graph predecessors if node attribute missing
            parents = [str(p) for p in G.predecessors(n)]

        # collect sibling node ids (children of each parent), exclude the node itself
        sibling_ids = set()
        for p in parents:
            pstr = p
            if pstr not in G.nodes and isinstance(p, (int, float)):
                pstr = str(p)
            if pstr in G.nodes:
                try:
                    children = list(G.successors(pstr))
                except Exception:
                    children = []
                for c in children:
                    if str(c) != str(n):
                        sibling_ids.add(str(c))

        if not sibling_ids:
            sibling_coherence[n] = float('nan')
            continue

        sims = []
        emb_n = emb_map[n]
        if emb_n is None:
            sibling_coherence[n] = float('nan')
            continue

        for s in sibling_ids:
            if s in emb_map and emb_map[s] is not None:
                try:
                    sims.append(safe_cosine_sim(emb_n, emb_map[s]))
                except Exception:
                    continue

        if not sims:
            sibling_coherence[n] = float('nan')
        else:
            sibling_coherence[n] = float(np.mean(sims))

    return sibling_coherence



def compute_coherences(G: nx.DiGraph, seed_centroid: Any = None, raw_json: Dict = None) -> Dict[str, Dict]:
    """
    Compatibility wrapper for older callsites: returns
      {'global_coherence': {...}, 'parent_coherence': {...}, 'sibling_coherence': {...}}
    """
    return {
        "global_coherence": compute_global_coherence(G, seed_centroid=seed_centroid, raw_json=raw_json),
        "parent_coherence": compute_parent_coherence(G),
        "sibling_coherence": compute_sibling_coherence(G)
    }

# --------------------
# Flagging (updated names & thresholds)
# --------------------
DEFAULT_THRESHOLDS = {
    # "global_coherence": 0.0,    
    "parent_coherence": 0.25,
    "sibling_coherence_lower": 0.2,
    "sibling_coherence_upper": 0.68
}

def flag_nodes(
    G: nx.DiGraph,
    metrics: Dict[str, Dict],
    thresholds: Dict = None,
    max_parents: int = 3,
    max_children: int = 3,
    text_width: int = 80
) -> List[Dict]:
    """
    Return list of flagged nodes with:
    - global_coherence / parent_coherence values
    - reasons
    - node topic
    - parent info
    - sample child info
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    flagged = []

    # metrics keys expected: metrics['global_coherence'], metrics['parent_coherence'], metrics['sibling_coherence']
    for n in G.nodes():
        reasons = []
        g = metrics.get('global_coherence', {}).get(n)
        p = metrics.get('parent_coherence', {}).get(n)
        s = metrics.get('sibling_coherence', {}).get(n)

        # --- threshold logic ---
        if (g is not None) and (not math.isnan(g)) and g < 0:
            reasons.append("NEGATIVE_GLOBAL_COHERENCE")

        # if (g is not None) and (not math.isnan(g)) and g < thresholds['global_coherence']:
        #     reasons.append("LOW_GLOBAL_COHERENCE")

        if (p is not None) and (not math.isnan(p)) and p < thresholds['parent_coherence']:
            reasons.append("LOW_PARENT_COHERENCE")

        if (s is not None) and (not math.isnan(s)):
            if s < thresholds['sibling_coherence_lower']:
                reasons.append("LOW_SIBLING_COHERENCE")
            if s > thresholds['sibling_coherence_upper']:
                reasons.append("HIGH_SIBLING_COHERENCE")

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
        for par in parent_ids:
            pt = (G.nodes[par].get('topic') or "").strip()
            parent_topics.append(shorten(pt, width=text_width))

        child_topics = []
        for c in child_ids:
            ct = (G.nodes[c].get('topic') or "").strip()
            child_topics.append(shorten(ct, width=text_width))

        flagged.append({
            "node_id": n,
            "depth": G.nodes[n].get('depth'),
            "reasons": reasons,
            "global_coherence": None if g is None or math.isnan(g) else float(g),
            "parent_coherence": None if p is None or math.isnan(p) else float(p),
            "sibling_coherence": None if s is None or math.isnan(s) else float(s),
            "topic": topic_short,
            "parent_ids": parent_ids,
            "parent_topics": parent_topics,
            "child_ids": child_ids,
            "child_topics": child_topics,
        })

    return flagged

# --------------------
# Filter pipeline (now with four cascading removal steps)
# --------------------
def filter_graph_pipeline(G_in: nx.DiGraph, metrics: Dict[str, Dict], thresholds: Dict = None, verbose: bool = True) -> Tuple[nx.DiGraph, Dict]:
    """
    Pipeline with four cascading removal steps:
      Step 1: Remove nodes with global_coherence < 0 (and their descendants)
      Step 2: Remove nodes with parent_coherence < thresholds['parent_coherence'] (and their descendants)
      Step 3: Remove nodes with sibling_coherence_mean < thresholds['sibling_coherence_lower'] (and their descendants)
      Step 4: Remove nodes with sibling_coherence_mean > thresholds['sibling_coherence_upper'] (and their descendants)
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    G0 = copy.deepcopy(G_in)
    stats = {}

    # Step 1: global_coherence < 0 (all depths)
    G1 = copy.deepcopy(G0)
    global_bad = [
        n for n in list(G1.nodes)
        if (metrics.get('global_coherence', {}).get(n) is not None)
        and (not math.isnan(metrics['global_coherence'][n]))
        and metrics['global_coherence'][n] < 0.0
    ]
    rem1 = set()
    for n in global_bad:
        if n in G1:
            rem1.update(gather_descendants_local(G1, n))
    for u in list(rem1):
        if u in G1:
            G1.remove_node(u)
    stats['step1_removed'] = len(rem1)

    # Step 2: parent_coherence < threshold (on remaining graph)
    G2 = copy.deepcopy(G1)
    parent_bad = [
        n for n in list(G2.nodes)
        if (metrics.get('parent_coherence', {}).get(n) is not None)
        and (not math.isnan(metrics['parent_coherence'][n]))
        and metrics['parent_coherence'][n] < thresholds['parent_coherence']
    ]
    rem2 = set()
    for n in parent_bad:
        if n in G2:
            rem2.update(gather_descendants_local(G2, n))
    for u in list(rem2):
        if u in G2:
            G2.remove_node(u)
    stats['step2_removed'] = len(rem2)

    # Step 3: sibling_coherence_mean < lower threshold
    G3 = copy.deepcopy(G2)
    sib_map = metrics.get('sibling_coherence', {})
    # compute mapping: sib_map might be {"sibling_coherence": {...}} or direct dict
    if isinstance(sib_map, dict) and 'sibling_coherence' in sib_map and isinstance(sib_map['sibling_coherence'], dict):
        sib_vals = sib_map['sibling_coherence']
    else:
        sib_vals = sib_map if isinstance(sib_map, dict) else {}
    sibling_low_bad = [
        n for n in list(G3.nodes)
        if (sib_vals.get(n) is not None)
        and (not math.isnan(sib_vals[n]))
        and sib_vals[n] < thresholds['sibling_coherence_lower']
    ]
    rem3 = set()
    for n in sibling_low_bad:
        if n in G3:
            rem3.update(gather_descendants_local(G3, n))
    for u in list(rem3):
        if u in G3:
            G3.remove_node(u)
    stats['step3_removed'] = len(rem3)

    # Step 4: sibling_coherence_mean > upper threshold (too similar/repetitive)
    G4 = copy.deepcopy(G3)
    sibling_high_bad = [
        n for n in list(G4.nodes)
        if (sib_vals.get(n) is not None)
        and (not math.isnan(sib_vals[n]))
        and sib_vals[n] > thresholds['sibling_coherence_upper']
    ]
    rem4 = set()
    for n in sibling_high_bad:
        if n in G4:
            rem4.update(gather_descendants_local(G4, n))
    for u in list(rem4):
        if u in G4:
            G4.remove_node(u)
    stats['step4_removed'] = len(rem4)

    stats['counts'] = {
        "orig": G0.number_of_nodes(),
        "after_step1": G1.number_of_nodes(),
        "after_step2": G2.number_of_nodes(),
        "after_step3": G3.number_of_nodes(),
        "after_step4": G4.number_of_nodes()
    }
    if verbose:
        print("Filter pipeline summary:", stats)
    return G4, stats

# --------------------
# Global metrics & report helpers (updated metric keys)
# --------------------
def compute_simple_global_metrics(metrics: Dict[str, Dict], G: nx.DiGraph) -> Dict:
    """
    Compute a few simple global aggregates for the report:
    - counts, mean/median global_coherence/parent_coherence/sibling_coherence, number NaNs, nodes per depth distribution.
    """
    node_list = list(G.nodes())
    g_vals_all = np.array([metrics.get('global_coherence', {}).get(n) for n in node_list], dtype=float)
    p_vals_all = np.array([metrics.get('parent_coherence', {}).get(n) for n in node_list], dtype=float)
    s_vals_all = np.array([metrics.get('sibling_coherence', {}).get(n) for n in node_list], dtype=float)

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
        'global_coherence_stats': stats(g_vals_all),
        'parent_coherence_stats': stats(p_vals_all),
        'sibling_coherence_stats': stats(s_vals_all),
        'total_nodes': len(node_list),
        'depth_counts': dict(depth_counts),
        'nodes_missing_embedding': sum(1 for n in node_list if 'embedding' not in G.nodes[n])
    }

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

    print(f"Saved filtered graph to: {out_path}")



def save_report_json(original_raw_json: Dict,
                     metrics: Dict[str, Dict],
                     original_global_metrics: Dict,
                     filtered_global_metrics: Dict,
                     flagged: List[Dict],
                     removals: Dict[str, List[str]],
                     out_path: str):
    """
    Write a report JSON containing:
    - summary: both original and filtered global metrics (counts, stats, depth distributions)
    - metrics_per_node: mapping node_id -> {global_coherence, parent_coherence, sibling_coherence} (for original nodes)
    - flagged_nodes: flagged list (with reasons)
    - removals: which node ids removed at each step
    """
    metrics_out = {}
    all_node_ids = list(original_raw_json.get('nodes', {}).keys())
    for nid in all_node_ids:
        sid = str(nid)
        g = metrics.get('global_coherence', {}).get(sid)
        p = metrics.get('parent_coherence', {}).get(sid)
        s = metrics.get('sibling_coherence', {}).get(sid)

        metrics_out[sid] = {
            'global_coherence': None if g is None or (isinstance(g, float) and math.isnan(g)) else float(g),
            'parent_coherence': None if p is None or (isinstance(p, float) and math.isnan(p)) else float(p),
            'sibling_coherence': None if s is None or (isinstance(s, float) and math.isnan(s)) else float(s)
        }

    report = {
        'summary': {
            # original (pre-filter) snapshot
            'original_node_count': len(all_node_ids),
            'original_global_metrics': original_global_metrics,
            # filtered snapshot (post-filter)
            'filtered_node_count': filtered_global_metrics.get('total_nodes'),
            'filtered_global_metrics': filtered_global_metrics
        },
        'metrics_per_node': metrics_out,
        'flagged_nodes': flagged,
        'removals': removals
    }

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"Saved report to: {out_path}")

# --------------------
# Run wrapper (integrated save) — updated removal steps to match the requested logic
# --------------------
def run_workflow_and_save(input_json_path: str,
                          output_filtered_json_path: str,
                          output_report_path: str,
                          model_name: str = "all-MiniLM-L6-v2",
                          seed_centroid: Any = "0",
                          thresholds: Dict = None,
                          save_filtered_graph: bool = True) -> Dict:
    """
    Run full pipeline:
      - load graph
      - compute embeddings if needed
      - compute metrics (global_coherence, parent_coherence, sibling_coherence)
      - flag nodes (for reporting)
      - remove nodes failing each step (and their descendants) in order:
          1) global_coherence < 0
          2) parent_coherence < parent threshold
          3) sibling_coherence_mean < sibling lower threshold
          4) sibling_coherence_mean > sibling upper threshold
      - save filtered graph and report
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS

    # load original graph & raw json
    G, raw_json = load_graph_from_json(input_json_path)

    # compute embeddings if needed
    G, model = compute_embeddings(G, model_name=model_name)

    # compute metrics
    metrics = compute_coherences(G, seed_centroid=seed_centroid, raw_json=raw_json)

    # flag nodes (for reporting)
    flagged = flag_nodes(G, metrics, thresholds=thresholds)

    # run filter pipeline but capture which nodes are removed at each step
    G0 = copy.deepcopy(G)
    removals = {'step1': [], 'step2': [], 'step3': [], 'step4': []}

    # Step 1: remove nodes with global_coherence < 0 (all depths)
    G1 = copy.deepcopy(G0)
    global_bad = [
        n for n in list(G1.nodes)
        if (metrics.get('global_coherence', {}).get(n) is not None)
        and (not math.isnan(metrics['global_coherence'][n]))
        and metrics['global_coherence'][n] < 0.0
    ]
    rem1 = set()
    for n in global_bad:
        if n in G1:
            desc = gather_descendants_local_defensive(G1, n)
            rem1.update(desc)
    removals['step1'] = sorted(list(rem1))
    for u in list(rem1):
        if u in G1: G1.remove_node(u)

    # Step 2: remove nodes with parent_coherence < threshold (on current graph)
    G2 = copy.deepcopy(G1)
    parent_bad = [
        n for n in list(G2.nodes)
        if (metrics.get('parent_coherence', {}).get(n) is not None)
        and (not math.isnan(metrics['parent_coherence'][n]))
        and metrics['parent_coherence'][n] < thresholds['parent_coherence']
    ]
    rem2 = set()
    for n in parent_bad:
        if n in G2:
            desc = gather_descendants_local_defensive(G2, n)
            rem2.update(desc)
    removals['step2'] = sorted(list(rem2))
    for u in list(rem2):
        if u in G2: G2.remove_node(u)

    # Step 3: remove nodes with sibling_coherence_mean < lower threshold
    G3 = copy.deepcopy(G2)
    sib_map = metrics.get('sibling_coherence', {})
    if isinstance(sib_map, dict) and 'sibling_coherence' in sib_map and isinstance(sib_map['sibling_coherence'], dict):
        sib_vals = sib_map['sibling_coherence']
    else:
        sib_vals = sib_map if isinstance(sib_map, dict) else {}
    sibling_low_bad = [
        n for n in list(G3.nodes)
        if (sib_vals.get(n) is not None)
        and (not math.isnan(sib_vals[n]))
        and sib_vals[n] < thresholds['sibling_coherence_lower']
    ]
    rem3 = set()
    for n in sibling_low_bad:
        if n in G3:
            desc = gather_descendants_local_defensive(G3, n)
            rem3.update(desc)
    removals['step3'] = sorted(list(rem3))
    for u in list(rem3):
        if u in G3: G3.remove_node(u)

    # Step 4: remove nodes with sibling_coherence_mean > upper threshold (repetitive)
    G4 = copy.deepcopy(G3)
    sibling_high_bad = [
        n for n in list(G4.nodes)
        if (sib_vals.get(n) is not None)
        and (not math.isnan(sib_vals[n]))
        and sib_vals[n] > thresholds['sibling_coherence_upper']
    ]
    rem4 = set()
    for n in sibling_high_bad:
        if n in G4:
            desc = gather_descendants_local_defensive(G4, n)
            rem4.update(desc)
    removals['step4'] = sorted(list(rem4))
    for u in list(rem4):
        if u in G4: G4.remove_node(u)

    G_filtered = G4

    # compute global metrics on original graph (before removals) and on filtered graph for report
    original_global_metrics = compute_simple_global_metrics(metrics, G)
    filtered_global_metrics = compute_simple_global_metrics(metrics, G_filtered)

    # save filtered graph (preserve original fields; remove embedding)
    if save_filtered_graph:
        save_filtered_graph_preserve_original(raw_json, G_filtered, output_filtered_json_path)

    # save report (now includes both original and filtered metrics)
    save_report_json(raw_json, metrics, original_global_metrics, filtered_global_metrics, flagged, removals, output_report_path)

    return {
        "G": G,
        "flagged": flagged,
        "G_filtered": G_filtered,
        "removals": removals,
        "original_global_metrics": original_global_metrics,
        "filtered_global_metrics": filtered_global_metrics,
        "stats": {
            "orig": G0.number_of_nodes(),
            "after_global_coherence_filtering": G1.number_of_nodes(),
            "after_parent_coherence_filtering": G2.number_of_nodes(),
            "after_sibling_coherence_lower_filtering": G3.number_of_nodes(),
            "after_sibling_coherence_upper_filtering": G4.number_of_nodes()
        }
    }


if __name__ == "__main__":
    import sys

    # Detect if running inside IPython / Jupyter (ipykernel)
    in_ipykernel = any("ipykernel" in m for m in sys.modules)

    if in_ipykernel:
        print("Detected IPython / ipykernel. Skipping CLI argument parsing.")
        print("To run interactively, import this module and call run_workflow_and_save(...).")
        # Do not call sys.exit() — just stop the __main__ block so notebooks won't raise SystemExit.
    else:
        # Normal CLI flow (terminal)
        import argparse
        parser = argparse.ArgumentParser(description="Prune topic graph by coherence thresholds")
        parser.add_argument("--input", "-i", required=True, help="Input graph JSON path")
        parser.add_argument("--out-filtered", "-o", required=True, help="Output filtered JSON path")
        parser.add_argument("--out-report", "-r", required=True, help="Output report JSON path")
        parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
        parser.add_argument("--seed", default="0", help="Seed centroid (node id or vector JSON)")
        parser.add_argument("--parent_coherence", type=float, default=DEFAULT_THRESHOLDS['parent_coherence'],
                            help="Threshold for parent_coherence (remove nodes below this)")
        parser.add_argument("--sibling_coherence_lower", type=float, default=DEFAULT_THRESHOLDS['sibling_coherence_lower'],
                            help="Lower threshold for sibling coherence (remove nodes below this)")
        parser.add_argument("--sibling_coherence_upper", type=float, default=DEFAULT_THRESHOLDS['sibling_coherence_upper'],
                            help="Upper threshold for sibling coherence (remove nodes above this)")
        args = parser.parse_args()

        thresholds = {
            "parent_coherence": args.parent_coherence,
            "sibling_coherence_lower": args.sibling_coherence_lower,
            "sibling_coherence_upper": args.sibling_coherence_upper
        }

        res = run_workflow_and_save(
            input_json_path=args.input,
            output_filtered_json_path=args.out_filtered,
            output_report_path=args.out_report,
            model_name=args.model,
            seed_centroid=args.seed,
            thresholds=thresholds,
            save_filtered_graph=True
        )

        print("Done. Stats:", res.get("stats"))