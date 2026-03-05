import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats


# =========================
# 0) Repro + Paths
# =========================
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PATH_NODES = "data/02_kg/kg_nodes_v2.csv"
PATH_EDGES = "data/02_kg/kg_edges_v2.csv"
PATH_DRUGBANK_CLEAN = "data/01_clean/drugbank_pralsetinib_proteins_cleaned.csv"

OUT_DIR = "models/mechanistic/model_results"
OUT_METRICS = os.path.join(OUT_DIR, "pralsetinib_baseline_vs_gnn_fair_metrics.csv")
OUT_FIG = os.path.join(OUT_DIR, "pralsetinib_baseline_vs_gnn_fair.png")
OUT_GNN_SCORES = os.path.join(OUT_DIR, "pralsetinib_gnn_scores_fair.csv")
OUT_BASELINE_SCORES = os.path.join(OUT_DIR, "pralsetinib_baseline_scores_fair.csv")
OUT_REASONING = os.path.join(OUT_DIR, "pralsetinib_mechanistic_reasoning_example.txt")

DRUG_NODE = "drug:Pralsetinib"

for p in [PATH_NODES, PATH_EDGES, PATH_DRUGBANK_CLEAN]:
    assert os.path.exists(p), f"Missing file: {p}"


# =========================
# 1) Load KG + targets
# =========================
nodes = pd.read_csv(PATH_NODES)
edges = pd.read_csv(PATH_EDGES)
db = pd.read_csv(PATH_DRUGBANK_CLEAN)

nodes["node_id"] = nodes["node_id"].astype(str)
edges["source"] = edges["source"].astype(str)
edges["target"] = edges["target"].astype(str)
edges["edge_type"] = edges["edge_type"].astype(str)

if DRUG_NODE not in set(nodes["node_id"]):
    raise ValueError(f"Could not find {DRUG_NODE} in kg_nodes_v2.csv")

protein_nodes = nodes[nodes["node_type"].astype(str).str.lower() == "protein"].copy()
protein_nodes["node_id"] = protein_nodes["node_id"].astype(str)

# ground truth: DrugBank role == target
true_uniprots = set(db.loc[db["role"].astype(str).str.lower() == "target", "uniprot_id"]
                    .dropna().astype(str).tolist())

# map UniProt -> protein node_id
uniprot_to_node = dict(
    protein_nodes.dropna(subset=["uniprot_id"])
    .assign(uniprot_id=lambda d: d["uniprot_id"].astype(str))
    .set_index("uniprot_id")["node_id"]
    .astype(str).to_dict()
)

true_targets: Set[str] = set()
for u in true_uniprots:
    if u in uniprot_to_node:
        true_targets.add(uniprot_to_node[u])

if len(true_targets) < 4:
    raise ValueError(f"Too few matched true targets ({len(true_targets)}). Check UniProt mapping in kg_nodes_v2.csv.")

all_proteins = protein_nodes["node_id"].tolist()
all_proteins_set = set(all_proteins)

print(f"KG nodes={len(nodes):,}, edges={len(edges):,}, proteins={len(all_proteins):,}")
print(f"True targets (DrugBank role=target): {len(true_targets)} matched protein nodes")


# =========================
# 2) Define DIRECT drug→protein evidence edges for baseline
# =========================
# Baseline ONLY uses direct edges from Pralsetinib to proteins (no GO/CTD reasoning).
DIRECT_EDGE_TYPES = {"binds_to", "inhibits", "targets"}  # keep flexible

direct_dp = edges[
    (edges["source"] == DRUG_NODE) &
    (edges["target"].isin(all_proteins_set)) &
    (edges["edge_type"].isin(DIRECT_EDGE_TYPES))
].copy()

# If your KG doesn't have those edge types, fallback to ANY drug->protein edge:
if len(direct_dp) == 0:
    direct_dp = edges[
        (edges["source"] == DRUG_NODE) &
        (edges["target"].isin(all_proteins_set))
    ].copy()
    print("⚠️ No binds_to/inhibits/targets edges found; using ANY Pralsetinib->protein edges as 'direct evidence baseline'.")

direct_pos_nodes = sorted(set(direct_dp["target"].tolist()))
print(f"Direct evidence edges from Pralsetinib to proteins: {len(direct_pos_nodes)}")


# =========================
# 3) FAIR split: hold out true targets as unseen links
# =========================
# This is the key change vs your previous plot.
# We hold out a subset of TRUE TARGETS as test links, remove them from the KG for training.
true_targets_list = sorted(list(true_targets))
random.shuffle(true_targets_list)

test_n = max(2, len(true_targets_list) // 3)   # ~1/3 held out
test_targets = set(true_targets_list[:test_n])
train_targets = set(true_targets_list[test_n:])

print(f"Train targets={len(train_targets)}, Test targets={len(test_targets)}")

# We'll remove any edges (Pralsetinib -> test_target) from the KG before training/scoring.
# Also remove them from baseline evidence.
edges_train = edges[
    ~((edges["source"] == DRUG_NODE) & (edges["target"].isin(test_targets)))
].copy()

direct_dp_train = direct_dp[
    ~direct_dp["target"].isin(test_targets)
].copy()

print("Removed held-out edges from training KG:",
      (len(edges) - len(edges_train)))


# =========================
# 4) Build BASELINE score (direct evidence only)
# =========================
# Baseline score:
#   1 if there's any remaining direct Pralsetinib->protein evidence edge in training data, else 0.
# That means it CANNOT recover held-out targets (score=0 for them).
baseline_evidence = set(direct_dp_train["target"].tolist())

baseline_df = pd.DataFrame({
    "protein_node_id": all_proteins,
})
baseline_df["baseline_score"] = baseline_df["protein_node_id"].isin(baseline_evidence).astype(float)
baseline_df.to_csv(OUT_BASELINE_SCORES, index=False)


# =========================
# 5) GNN: 2-layer GCN link predictor on FULL neuro-symbolic KG (minus heldout edges)
# =========================
node_ids = nodes["node_id"].tolist()
node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
num_nodes = len(node_ids)
drug_idx = node_to_idx[DRUG_NODE]

# Convert training edges to undirected adjacency
src_idx = edges_train["source"].map(node_to_idx).values
tgt_idx = edges_train["target"].map(node_to_idx).values
mask = (~pd.isna(src_idx)) & (~pd.isna(tgt_idx))
src_idx = src_idx[mask].astype(int)
tgt_idx = tgt_idx[mask].astype(int)

row = np.concatenate([src_idx, tgt_idx])
col = np.concatenate([tgt_idx, src_idx])

# add self loops
self_idx = np.arange(num_nodes)
row = np.concatenate([row, self_idx])
col = np.concatenate([col, self_idx])

indices = torch.tensor(np.vstack([row, col]), dtype=torch.long)
values = torch.ones(indices.shape[1], dtype=torch.float32)
A = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    r, c = A.indices()
    v = A.values() * deg_inv_sqrt[r] * deg_inv_sqrt[c]
    return torch.sparse_coo_tensor(A.indices(), v, A.shape).coalesce()

A_norm = normalize_adj(A)

# Training positives = (Pralsetinib -> train_targets)
train_pos = sorted(list(train_targets))
train_pos_idx = torch.tensor([node_to_idx[p] for p in train_pos], dtype=torch.long)

# Candidate proteins for scoring
all_prot_idx = torch.tensor([node_to_idx[p] for p in all_proteins], dtype=torch.long)

# Negative pool: proteins that are NOT train positives (we can include test targets here; that's okay)
neg_pool = sorted(list(all_proteins_set - set(train_pos)))
neg_pool_idx = torch.tensor([node_to_idx[p] for p in neg_pool], dtype=torch.long)

@dataclass
class GCNConfig:
    hidden_dim: int = 64
    epochs: int = 500
    lr: float = 1e-2
    weight_decay: float = 1e-4
    neg_ratio: int = 10
    print_every: int = 50

class GCNLinkPredictor(nn.Module):
    def __init__(self, n_nodes: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)

    def encode(self, A_norm: torch.Tensor) -> torch.Tensor:
        X = self.emb.weight
        Z = torch.sparse.mm(A_norm, X)
        Z = F.relu(self.W1(Z))
        Z = torch.sparse.mm(A_norm, Z)
        Z = self.W2(Z)
        Z = F.normalize(Z, p=2, dim=1)
        return Z

    def score(self, Z: torch.Tensor, src: int, dst: torch.Tensor) -> torch.Tensor:
        s = Z[src].unsqueeze(0)
        d = Z[dst]
        return (s * d).sum(dim=1)

def train_gnn(cfg: GCNConfig) -> GCNLinkPredictor:
    model = GCNLinkPredictor(num_nodes, cfg.hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        Z = model.encode(A_norm)

        n_pos = len(train_pos_idx)
        n_neg = cfg.neg_ratio * n_pos
        neg_sample = neg_pool_idx[torch.randint(0, len(neg_pool_idx), (n_neg,))]

        pos_logits = model.score(Z, drug_idx, train_pos_idx)
        neg_logits = model.score(Z, drug_idx, neg_sample)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 1 or ep % cfg.print_every == 0:
            with torch.no_grad():
                pos_p = torch.sigmoid(pos_logits).mean().item()
                neg_p = torch.sigmoid(neg_logits).mean().item()
            print(f"epoch {ep:4d} | loss={loss.item():.4f} | mean(sigmoid pos)={pos_p:.3f} | mean(sigmoid neg)={neg_p:.3f}")

    return model

cfg = GCNConfig()
model = train_gnn(cfg)

# Score all proteins
model.eval()
with torch.no_grad():
    Z = model.encode(A_norm)
    logits = model.score(Z, drug_idx, all_prot_idx)
    probs = torch.sigmoid(logits).cpu().numpy()

gnn_df = pd.DataFrame({"protein_node_id": all_proteins, "gnn_score": probs})
gnn_df.to_csv(OUT_GNN_SCORES, index=False)


# =========================
# 6) Evaluate Baseline vs GNN on HELD-OUT targets
# =========================
def recall_at_k(ranked: List[str], truth: Set[str], k: int) -> float:
    if len(truth) == 0:
        return 0.0
    return len(set(ranked[:k]) & truth) / len(truth)

# Rankings
baseline_rank = baseline_df.sort_values("baseline_score", ascending=False)["protein_node_id"].tolist()
gnn_rank = gnn_df.sort_values("gnn_score", ascending=False)["protein_node_id"].tolist()

Ks = [5, 10, 20, 50, 100]
rows = []
for k in Ks:
    rows.append({
        "k": k,
        "baseline_recall": recall_at_k(baseline_rank, test_targets, k),
        "gnn_recall": recall_at_k(gnn_rank, test_targets, k),
        "n_test_targets": len(test_targets),
        "n_train_targets": len(train_targets),
    })

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT_METRICS, index=False)
print("\nMetrics:\n", metrics)

# Plot
fig, ax = plt.subplots(figsize=(7.6, 4.3))
x = np.arange(len(Ks))
w = 0.35
ax.bar(x - w/2, metrics["baseline_recall"], width=w, label="Baseline (direct evidence only)")
ax.bar(x + w/2, metrics["gnn_recall"], width=w, label="Neuro-symbolic GNN (KG reasoning)")
ax.set_xticks(x)
ax.set_xticklabels([f"Top-{k}" for k in Ks])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Recall (held-out targets)")
ax.set_title("Pralsetinib held-out target prediction: Baseline vs Neuro-symbolic GNN")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
print("Saved figure:", OUT_FIG)


# =========================
# 7) Mechanistic reasoning example (path explanation)
# =========================
# We’ll pick a TEST TARGET that the GNN ranks highly, and print a shortest path in the TRAINING KG.
# This is the “reasoning chain” you can put on a slide.

# Build adjacency list WITH edge types from edges_train for path explanation
adj = defaultdict(list)  # node -> list[(neighbor, edge_type)]
for r in edges_train.itertuples(index=False):
    s = r.source
    t = r.target
    et = r.edge_type
    adj[s].append((t, et))
    adj[t].append((s, et))  # undirected for explanation

def shortest_path_with_edges(start: str, goal: str, max_depth: int = 6) -> Optional[List[Tuple[str, str]]]:
    """
    Returns path as [(node0, ""), (node1, edge_type), (node2, edge_type), ...]
    edge_type is the edge used to arrive at that node.
    """
    q = deque([(start, 0)])
    parent = {start: None}
    parent_edge = {start: ""}

    while q:
        cur, d = q.popleft()
        if cur == goal:
            break
        if d >= max_depth:
            continue
        for nb, et in adj.get(cur, []):
            if nb not in parent:
                parent[nb] = cur
                parent_edge[nb] = et
                q.append((nb, d + 1))

    if goal not in parent:
        return None

    # reconstruct
    path_nodes = []
    cur = goal
    while cur is not None:
        path_nodes.append(cur)
        cur = parent[cur]
    path_nodes.reverse()

    out = [(path_nodes[0], "")]
    for i in range(1, len(path_nodes)):
        out.append((path_nodes[i], parent_edge[path_nodes[i]]))
    return out

# pick a good test target found by GNN in top-K
gnn_ranked = gnn_df.sort_values("gnn_score", ascending=False)
rank_pos = {pid: i+1 for i, pid in enumerate(gnn_ranked["protein_node_id"].tolist())}

# Find the highest-ranked held-out target
best_test = sorted(list(test_targets), key=lambda p: rank_pos.get(p, 10**9))[0]
best_rank = rank_pos.get(best_test, None)

path = shortest_path_with_edges(DRUG_NODE, best_test, max_depth=7)

lines = []
lines.append("=== Mechanistic reasoning example (from training KG; held-out link removed) ===\n")
lines.append(f"Chosen held-out true target: {best_test}\n")
lines.append(f"GNN rank among all proteins: {best_rank}\n\n")

if path is None:
    lines.append("No path found within max_depth. Increase max_depth or check KG connectivity.\n")
else:
    lines.append("Shortest reasoning path (node --edge_type--> node):\n")
    for i in range(1, len(path)):
        node, et = path[i]
        prev = path[i-1][0]
        lines.append(f"  {prev} --{et}--> {node}\n")

# Add readable labels if present
label_map = nodes.set_index("node_id")["label"].astype(str).to_dict() if "label" in nodes.columns else {}
def lab(n): 
    return f"{n} ({label_map.get(n,'')})" if n in label_map else n

if path is not None:
    lines.append("\nSame path with labels (if available):\n")
    for i in range(1, len(path)):
        node, et = path[i]
        prev = path[i-1][0]
        lines.append(f"  {lab(prev)} --{et}--> {lab(node)}\n")

with open(OUT_REASONING, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Saved reasoning example:", OUT_REASONING)
print("Done ✅")