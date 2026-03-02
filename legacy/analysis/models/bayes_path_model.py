#!/usr/bin/env python3
"""
Bayesian Path Scoring for KG-based toxicity theme ranking.

KG schema expected:
  drug --binds_to--> protein
  protein --involved_in--> go
  go --maps_to--> tox_theme        (optional but strongly recommended)
  drug --reported_with--> tox_theme (FAERS counts, used for priors)

If you pass --go_theme_map, the model will add GO->Theme mappings from that CSV
even if maps_to edges are missing in kg_edges.csv.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict
import math
import pandas as pd


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def load_go_theme_map(path: Path) -> dict[str, set[str]]:
    """
    Load a GO -> ToxicityTheme
    """
    df = pd.read_csv(path)

    cols = {c.lower().strip(): c for c in df.columns}

    go_col = (
        cols.get("go_id") or cols.get("go") or cols.get("go_term") or
        cols.get("go_identifier") or cols.get("goid")
    )
    theme_col = (
        cols.get("toxicity_theme") or cols.get("theme") or cols.get("toxicity") or
        cols.get("category") or cols.get("group")
    )

    if go_col is None or theme_col is None:
        raise ValueError(
            f"Can't find GO/theme columns in {path}. Columns: {df.columns.tolist()}"
        )

    m: dict[str, set[str]] = defaultdict(set)
    tmp = df[[go_col, theme_col]].dropna()

    for go_id, theme in tmp.itertuples(index=False):
        go_id = str(go_id).strip()
        theme = str(theme).strip()

        if not theme.startswith(("tox:", "theme:", "toxicity:")):
            theme = "tox:" + theme

        m[go_id].add(theme)

    return m


class BayesPathModel:
    def __init__(
        self,
        nodes_path: Path,
        edges_path: Path,
        alpha_prior: float = 1.0,
        base_path_prob: float = 0.15,
        role_weights: dict | None = None,
        go_specificity_power: float = 1.0,
        top_paths_per_theme: int = 5,
    ):
        self.nodes_path = nodes_path
        self.edges_path = edges_path
        self.alpha_prior = alpha_prior
        self.base_path_prob = base_path_prob
        self.go_specificity_power = go_specificity_power
        self.top_paths_per_theme = top_paths_per_theme

        self.role_weights = role_weights or {
            "target": 1.00,
            "enzyme": 0.70,
            "transporter": 0.70,
            "carrier": 0.70,
            "other": 0.60,
            "": 0.60,
        }

        self.nodes = pd.read_csv(self.nodes_path)
        self.edges = pd.read_csv(self.edges_path)

        self._index_graph()

    def _index_graph(self):
        #NODE LOOK UP
        self.node_type = {}
        if "node_id" in self.nodes.columns and "node_type" in self.nodes.columns:
            self.node_type = dict(zip(self.nodes["node_id"], self.nodes["node_type"]))

        e = self.edges

        self.drug_to_protein = defaultdict(list)       # drug -> [(protein, role)]
        self.protein_to_go = defaultdict(set)          # protein -> {go}
        self.go_to_theme = defaultdict(set)            # go -> {theme}
        self.drug_to_theme_counts = defaultdict(dict)  # drug -> {theme: count}
        self.go_in_degree = defaultdict(int)           # go -> degree (for specificity)

        for _, r in e.iterrows():
            src = str(r["source"])
            et = str(r["edge_type"])
            tgt = str(r["target"])

            if et == "binds_to":
                role = str(r.get("interaction_role", "")).strip().lower()
                self.drug_to_protein[src].append((tgt, role))

            elif et == "involved_in":
                self.protein_to_go[src].add(tgt)
                self.go_in_degree[tgt] += 1

            elif et == "maps_to":
                self.go_to_theme[src].add(tgt)

            elif et == "reported_with":
                cnt = r.get("count", 0)
                try:
                    cnt = int(cnt)
                except Exception:
                    cnt = 0
                self.drug_to_theme_counts[src][tgt] = self.drug_to_theme_counts[src].get(tgt, 0) + cnt

        # Collect all themes seen anywhere
        self.all_themes = set()
        for _, m in self.drug_to_theme_counts.items():
            self.all_themes |= set(m.keys())
        for _, themes in self.go_to_theme.items():
            self.all_themes |= set(themes)

        if not self.all_themes:
            raise ValueError(
                "No toxicity themes found. "
                "Do you have Drug->Theme (reported_with) edges and/or GO->Theme (maps_to) edges?"
            )

    def _prior_from_faers(self, drug_id: str) -> dict[str, float]:
        """
        prior(theme) = (count(theme)+alpha) / sum(count+alpha)
        and iff no FAERS counts exist, use uniform prior.
        """
        counts = self.drug_to_theme_counts.get(drug_id, {})
        if not counts:
            uni = 1.0 / len(self.all_themes)
            return {t: uni for t in self.all_themes}

        prior = {}
        total = 0.0
        for t in self.all_themes:
            val = float(counts.get(t, 0)) + self.alpha_prior
            prior[t] = val
            total += val

        for t in prior:
            prior[t] /= total
        return prior

    def _go_specificity_weight(self, go_id: str) -> float:
        """
          w = 1 / (1 + degree)^power
        """
        deg = float(self.go_in_degree.get(go_id, 0))
        return 1.0 / ((1.0 + deg) ** self.go_specificity_power)

    def _role_weight(self, role: str) -> float:
        role = (role or "").strip().lower()
        if "target" in role:
            role = "target"
        elif "enzyme" in role:
            role = "enzyme"
        elif "transporter" in role:
            role = "transporter"
        return float(self.role_weights.get(role, self.role_weights.get("other", 0.6)))

    def rank_themes(self, drug_id: str, topk: int = 10) -> list[dict]:
        if drug_id not in self.drug_to_protein:
            raise KeyError(
                f"Drug {drug_id} has no binds_to edges in KG. "
                f"Available drugs: {list(self.drug_to_protein.keys())[:5]} ..."
            )

        prior = self._prior_from_faers(drug_id)

        path_probs = defaultdict(list)  # theme -> [(p_path, protein, go)]

        for protein_id, role in self.drug_to_protein[drug_id]:
            w_role = self._role_weight(role)
            gos = self.protein_to_go.get(protein_id, set())
            if not gos:
                continue

            for go_id in gos:
                w_go = self._go_specificity_weight(go_id)
                themes = self.go_to_theme.get(go_id, set())
                if not themes:
                    continue

                for theme_id in themes:
                    p_path = clamp(self.base_path_prob * w_role * w_go)
                    if p_path > 0:
                        path_probs[theme_id].append((p_path, protein_id, go_id))

        evidence = {}
        top_paths = {}
        for theme_id in self.all_themes:
            paths = path_probs.get(theme_id, [])
            if not paths:
                evidence[theme_id] = 0.0
                top_paths[theme_id] = []
                continue

            prod = 1.0
            for p, _, _ in paths:
                prod *= (1.0 - clamp(p))
            evidence[theme_id] = clamp(1.0 - prod)

            top_paths[theme_id] = sorted(paths, key=lambda x: x[0], reverse=True)[: self.top_paths_per_theme]

        unnorm = {t: prior[t] * (evidence[t] + 1e-12) for t in self.all_themes}
        Z = sum(unnorm.values())

        results = []
        for t in self.all_themes:
            post = unnorm[t] / Z if Z > 0 else 0.0
            results.append({
                "theme": t,
                "posterior": post,
                "prior": prior[t],
                "evidence": evidence[t],
                "top_paths": top_paths[t],
            })

        results.sort(key=lambda d: d["posterior"], reverse=True)
        return results[:topk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=Path, required=True, help="KG nodes CSV")
    ap.add_argument("--edges", type=Path, required=True, help="KG edges CSV")
    ap.add_argument("--drug", type=str, required=True, help="Drug node_id (e.g., drug:CHEMBL4582651)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--alpha_prior", type=float, default=1.0)
    ap.add_argument("--base_path_prob", type=float, default=0.15)
    ap.add_argument("--go_specificity_power", type=float, default=1.0)
    ap.add_argument("--top_paths_per_theme", type=int, default=5)

    ap.add_argument("--go_theme_map", type=Path, default=None,
                    help="Optional GO->Theme mapping CSV (adds maps_to links if KG lacks them)")
    ap.add_argument("--emit_edges_with_maps_to", type=Path, default=None,
                    help="Optional output path to write a new KG edges CSV with maps_to edges appended")
    args = ap.parse_args()

    model = BayesPathModel(
        nodes_path=args.nodes,
        edges_path=args.edges,
        alpha_prior=args.alpha_prior,
        base_path_prob=args.base_path_prob,
        go_specificity_power=args.go_specificity_power,
        top_paths_per_theme=args.top_paths_per_theme,
    )

    # Merge GO->Theme mappings from CSV (optional)
    extra = None
    if args.go_theme_map is not None:
        extra = load_go_theme_map(args.go_theme_map)
        added = 0
        for go_id, themes in extra.items():
            before = len(model.go_to_theme.get(go_id, set()))
            model.go_to_theme[go_id] |= set(themes)
            after = len(model.go_to_theme[go_id])
            added += max(0, after - before)

        for themes in extra.values():
            model.all_themes |= set(themes)

        print(f"[INFO] Added {added} GO->Theme mappings from {args.go_theme_map}")

    # Optionally emit a new KG edges file with maps_to appended
    if args.emit_edges_with_maps_to is not None:
        if extra is None:
            raise ValueError("--emit_edges_with_maps_to requires --go_theme_map")

        out_edges = model.edges.copy()
        existing = set(zip(
            out_edges["source"].astype(str),
            out_edges["edge_type"].astype(str),
            out_edges["target"].astype(str),
        ))

        new_rows = []
        for go_id, themes in extra.items():
            for theme in themes:
                key = (go_id, "maps_to", theme)
                if key in existing:
                    continue
                new_rows.append({
                    "source": go_id,
                    "target": theme,
                    "edge_type": "maps_to",
                    "source_type": "go",
                    "target_type": "toxicity_theme",
                    "provenance": f"go_theme_map:{args.go_theme_map.name}",
                })

        if new_rows:
            out_edges = pd.concat([out_edges, pd.DataFrame(new_rows)], ignore_index=True)

        out_edges.to_csv(args.emit_edges_with_maps_to, index=False)
        print(f"[INFO] Wrote edges with maps_to to: {args.emit_edges_with_maps_to} (added {len(new_rows)} edges)")

    ranked = model.rank_themes(args.drug, topk=args.topk)

    print("\n=== Bayes-Path Ranked Toxicity Themes ===")
    for i, r in enumerate(ranked, 1):
        print(f"\n{i}. {r['theme']}")
        print(f"   posterior={r['posterior']:.4f}  prior(FAERS)={r['prior']:.4f}  evidence(paths)={r['evidence']:.4f}")

        if r["top_paths"]:
            print("   top mechanistic paths:")
            for p, prot, go in r["top_paths"]:
                print(f"     p_path={p:.4f}  {args.drug} -> {prot} -> {go} -> {r['theme']}")
        else:
            print("   (no complete Drug->Protein->GO->Theme paths; check GO->Theme mapping)")


if __name__ == "__main__":
    main()
