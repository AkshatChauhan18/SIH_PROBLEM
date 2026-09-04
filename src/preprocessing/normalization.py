"""
src/preprocessing/normalization.py
Zero-leakage normalization and operational label mapping.
Fitted exclusively on training data (Days 1–3).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler

# Mapping 15 raw CIC-IDS2017 classes to 7 operational threat classes
EVENT_CLASS_MAPPING = {
    "BENIGN": "BENIGN",
    "PortScan": "RECONNAISSANCE",
    "FTP-Patator": "BRUTE_FORCE",
    "SSH-Patator": "BRUTE_FORCE",
    "Web Attack - Brute Force": "BRUTE_FORCE",
    "Web Attack - XSS": "WEB_EXPLOIT",
    "Web Attack - Sql Injection": "WEB_EXPLOIT",
    "DoS Hulk": "DENIAL_OF_SERVICE",
    "DoS GoldenEye": "DENIAL_OF_SERVICE",
    "DoS slowloris": "DENIAL_OF_SERVICE",
    "DoS Slowhttptest": "DENIAL_OF_SERVICE",
    "DDoS": "DENIAL_OF_SERVICE",
    "Heartbleed": "DENIAL_OF_SERVICE",
    "Bot": "BOTNET",
    "Infiltration": "INFILTRATION",
}

CLASS_NAME_TO_INDEX = {
    "BENIGN": 0,
    "RECONNAISSANCE": 1,
    "BRUTE_FORCE": 2,
    "WEB_EXPLOIT": 3,
    "DENIAL_OF_SERVICE": 4,
    "BOTNET": 5,
    "INFILTRATION": 6,
}

CLASS_INDEX_TO_NAME = {v: k for k, v in CLASS_NAME_TO_INDEX.items()}

# Hierarchical priority when assigning a window-level target label
# A single Infiltration or Exploit flow in a window must not be drowned out by benign traffic
SEVERITY_HIERARCHY = [
    "INFILTRATION",
    "WEB_EXPLOIT",
    "BRUTE_FORCE",
    "BOTNET",
    "RECONNAISSANCE",
    "DENIAL_OF_SERVICE",
    "BENIGN",
]

def map_raw_label_to_category(raw_label: str) -> str:
    """Maps raw CIC label string to one of the 7 operational categories."""
    clean = raw_label.replace("\ufffd", "-").replace("–", "-").strip()
    return EVENT_CLASS_MAPPING.get(clean, "BENIGN")

def map_category_to_index(cat_name: str) -> int:
    """Returns integer target index (0..6)."""
    return CLASS_NAME_TO_INDEX.get(cat_name, 0)

class PreprocessorRegistry:
    """
    Holds fitted imputation medians and feature scalers.
    ZERO LEAKAGE: Must only be fitted on training partition.
    """

    def __init__(self):
        self.flow_medians: Dict[str, float] = {}
        self.node_scaler: Optional[RobustScaler] = None
        self.edge_scaler: Optional[RobustScaler] = None
        self.is_fitted: bool = False

    def fit_flow_medians(self, train_df):
        """Fits median values for numeric flow features on training set."""
        numeric_cols = [
            "flow_duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
            "flow_bytes_s", "flow_pkts_s", "flow_iat_mean", "flow_iat_std",
            "pkt_len_mean", "pkt_len_std", "init_win_fwd", "init_win_bwd",
            "syn_count", "rst_count", "ack_count", "fin_count", "psh_count"
        ]
        medians = {}
        for col in numeric_cols:
            if col in train_df.columns:
                val = train_df[col].drop_nulls().median()
                medians[col] = float(val) if val is not None else 0.0
        self.flow_medians = medians

    def fit_graph_scalers(self, train_node_feats: np.ndarray, train_edge_feats: np.ndarray):
        """Fits RobustScalers for node and edge features on training graphs."""
        self.node_scaler = RobustScaler()
        if len(train_node_feats) > 0:
            self.node_scaler.fit(train_node_feats)

        self.edge_scaler = RobustScaler()
        if len(train_edge_feats) > 0:
            self.edge_scaler.fit(train_edge_feats)

        self.is_fitted = True

    def transform_node_features(self, node_feats: np.ndarray) -> np.ndarray:
        if self.node_scaler and len(node_feats) > 0:
            return self.node_scaler.transform(node_feats)
        return node_feats

    def transform_edge_features(self, edge_feats: np.ndarray) -> np.ndarray:
        if self.edge_scaler and len(edge_feats) > 0:
            return self.edge_scaler.transform(edge_feats)
        return edge_feats

    def save(self, filepath: Union[str, Path]):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PreprocessorRegistry":
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessor file not found: {path.resolve()}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
