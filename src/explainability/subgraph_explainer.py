"""
src/explainability/subgraph_explainer.py

GNN Subgraph Explainability Engine.
Identifies the critical communication paths, malicious subgraphs, and pivot routes
responsible for triggering a forecasted network attack state.

Computes edge saliency masks via gradient backpropagation:
Saliency(e_ij) = || d(ThreatScore) / d(e_ij) ||
and extracts the induced malicious communication subnetwork.
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np

from src.models.gnn_encoder import GNNEncoder
from src.models.prediction_head import ThreatHead
from src.preprocessing.normalization import CLASS_INDEX_TO_NAME

class SubgraphExplainer:
    """
    Computes communication edge importance and extracts the critical attacker subgraph.
    """

    def __init__(
        self,
        gnn_encoder: GNNEncoder,
        threat_head: ThreatHead,
        top_k_edges: int = 15
    ):
        self.gnn = gnn_encoder
        self.threat_head = threat_head
        self.top_k_edges = top_k_edges

    def explain_graph(
        self,
        pyg_data,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes edge attribution scores for the given PyG graph window.

        Returns:
            Dictionary containing:
            - top_edges: list of (u_ip, v_ip, attribution_score)
            - critical_nodes: unique set of IPs participating in top critical edges
            - edge_attributions: normalized saliency scores
        """
        self.gnn.eval()
        self.threat_head.eval()

        device = next(self.gnn.parameters()).device
        x = pyg_data.x.clone().detach().to(device)
        edge_index = pyg_data.edge_index.clone().detach().to(device)
        edge_attr = pyg_data.edge_attr.clone().detach().to(device)
        num_edges = edge_attr.size(0)

        if num_edges == 0:
            return {
                "top_edges": [],
                "critical_nodes": [],
                "edge_saliency": [],
                "primary_attacker_ip": "N/A",
                "primary_victim_ip": "N/A"
            }

        # Learnable / differentiable edge mask initialized to ones
        edge_mask = torch.nn.Parameter(torch.ones(num_edges, 1, device=device))
        masked_edge_attr = edge_attr * edge_mask

        # Forward pass through GNN Encoder
        z = self.gnn(x=x, edge_index=edge_index, edge_attr=masked_edge_attr)
        logits = self.threat_head(z)  # [1, num_classes]

        if target_class is None:
            # If no target class specified, explain the highest non-benign threat score
            target_class = int(torch.argmax(logits[:, 1:], dim=-1).item() + 1)

        target_logit = logits[0, target_class]
        target_logit.backward()

        # Compute gradient magnitude per edge
        if edge_mask.grad is not None:
            saliency = torch.norm(edge_mask.grad, p=2, dim=-1).cpu().detach().numpy()
        else:
            saliency = np.ones(num_edges, dtype=np.float32)

        # Normalize saliency to sum to 1.0 (or percentage)
        total_sal = np.sum(saliency)
        norm_saliency = saliency / (total_sal + 1e-9)

        # Rank edges by attribution
        top_indices = np.argsort(norm_saliency)[::-1][:self.top_k_edges]

        node_ips = getattr(pyg_data, "node_ips", None)
        edge_idx_np = edge_index.cpu().numpy()

        critical_edges = []
        critical_nodes_set = set()
        node_out_sal = {}
        node_in_sal = {}

        for idx in top_indices:
            score = float(norm_saliency[idx])
            u_idx = int(edge_idx_np[0, idx])
            v_idx = int(edge_idx_np[1, idx])

            u_name = node_ips[u_idx] if (node_ips and u_idx < len(node_ips)) else f"Host_{u_idx}"
            v_name = node_ips[v_idx] if (node_ips and v_idx < len(node_ips)) else f"Host_{v_idx}"

            critical_edges.append({
                "source_ip": u_name,
                "target_ip": v_name,
                "edge_index": int(idx),
                "attribution_score": round(score, 4),
                "percentage": f"{round(score * 100, 1)}%"
            })
            critical_nodes_set.add(u_name)
            critical_nodes_set.add(v_name)

            node_out_sal[u_name] = node_out_sal.get(u_name, 0.0) + score
            node_in_sal[v_name] = node_in_sal.get(v_name, 0.0) + score

        # Determine primary suspected attacker (highest outbound attribution) and victim (highest inbound)
        primary_attacker = max(node_out_sal.items(), key=lambda x: x[1])[0] if node_out_sal else "N/A"
        primary_victim = max(node_in_sal.items(), key=lambda x: x[1])[0] if node_in_sal else "N/A"

        return {
            "explained_class_index": target_class,
            "explained_class_name": CLASS_INDEX_TO_NAME.get(target_class, "UNKNOWN"),
            "primary_attacker_ip": primary_attacker,
            "primary_victim_ip": primary_victim,
            "critical_nodes": sorted(list(critical_nodes_set)),
            "top_edges": critical_edges,
            "subgraph_edge_count": len(critical_edges),
            "subgraph_node_count": len(critical_nodes_set)
        }
