import pandas as pd
import networkx as nx
from pyvis.network import Network
from pathlib import Path

# Paths
NODES_PATH = Path("kg_files/kg_nodes.csv")
EDGES_PATH = Path("kg_files/kg_edges.csv")
OUT_HTML = Path("kg_files/kg_interactive.html")

nodes_df = pd.read_csv(NODES_PATH)
edges_df = pd.read_csv(EDGES_PATH)

# Graph
G = nx.DiGraph()

for _, r in nodes_df.iterrows():
    G.add_node(
        r["node_id"],
        label=r.get("label", r["node_id"]),
        node_type=r["node_type"],
        faers_count=r.get("faers_count", 1)
    )

for _, r in edges_df.iterrows():
    G.add_edge(
        r["source"],
        r["target"],
        label=r["edge_type"],
        provenance=r.get("provenance", "")
    )

net = Network(
    height="800px",
    width="100%",
    bgcolor="#ffffff",
    font_color="black",
    directed=True,
    notebook=False
)

net.from_nx(G)

# Style
COLOR_MAP = {
    "Drug": "#ff6b6b",
    "Target": "#4dabf7",
    "Disease": "#51cf66",
    "AdverseEvent": "#ffa94d",
    "GO": "#9775fa"
}

for node in net.nodes:
    ntype = G.nodes[node["id"]].get("node_type", "")
    node["color"] = COLOR_MAP.get(ntype, "#ced4da")

    if ntype == "Drug":
        node["size"] = 35
    elif ntype == "Target":
        node["size"] = 25
    elif ntype == "Disease":
        node["size"] = 18
    elif ntype == "AdverseEvent":
        node["size"] = min(
            10 + int(G.nodes[node["id"]].get("faers_count", 1) / 5),
            40
        )
    else:
        node["size"] = 12

    node["title"] = (
        f"<b>{node['label']}</b><br>"
        f"Type: {ntype}"
    )

for edge in net.edges:
    edge["title"] = (
        f"Relation: {edge.get('label','')}<br>"
        f"Source: {edge.get('from')}<br>"
        f"Target: {edge.get('to')}"
    )
    edge["arrows"] = "to"

net.set_options("""
var options = {
  "physics": {
    "enabled": true,
    "stabilization": {
      "iterations": 200
    },
    "barnesHut": {
      "gravitationalConstant": -8000,
      "springLength": 200,
      "springConstant": 0.04
    }
  }
}
""")

net.write_html(str(OUT_HTML))
print(f"Interactive KG written to {OUT_HTML}")
