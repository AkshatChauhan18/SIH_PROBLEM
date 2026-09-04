"""
src/adapters/base_adapter.py
Strongly typed FlowRecord abstraction and abstract base adapter interface.
Ensures unified schema across CSV and PCAP data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Union, Optional
import polars as pl

@dataclass(slots=True)
class FlowRecord:
    """
    Standardized network flow record representing a single bidirectional flow.
    Labels are target-only and NEVER used as graph/model input features.
    """
    # Network ID
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    # Temporal
    timestamp: datetime
    flow_duration: float

    # Packet / Byte statistics
    fwd_pkts: float
    bwd_pkts: float
    fwd_bytes: float
    bwd_bytes: float
    flow_bytes_s: Optional[float]
    flow_pkts_s: Optional[float]

    # Flow dynamics
    flow_iat_mean: float
    flow_iat_std: float
    pkt_len_mean: float
    pkt_len_std: float
    init_win_fwd: float
    init_win_bwd: float

    # TCP flags
    syn_count: float
    rst_count: float
    ack_count: float
    fin_count: float
    psh_count: float

    # Target (Target-only ground truth)
    label: str

class BaseAdapter(ABC):
    """Abstract base adapter for loading raw network telemetry into standardized FlowRecords."""

    @abstractmethod
    def load_flows(self, source_path: Union[str, Path]) -> pl.LazyFrame:
        """
        Reads raw data source lazily and returns a standardized Polars LazyFrame
        whose columns match FlowRecord field names.
        """
        pass
