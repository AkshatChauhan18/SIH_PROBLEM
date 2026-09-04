"""
src/adapters/pcap_adapter.py
Extensible PCAP ingestion interface mapping to the identical FlowRecord abstraction.
Ensures unified downstream processing across PCAP and CSV.
"""

from pathlib import Path
from typing import Union, List
from datetime import datetime
import polars as pl
from src.adapters.base_adapter import BaseAdapter, FlowRecord

class PCAPAdapter(BaseAdapter):
    """
    Adapter for raw PCAP captures.
    Uses packet/flow extraction (e.g. Scapy, Zeek, or CICFlowMeter format)
    and maps extracted flow metrics into the standardized FlowRecord schema.
    """

    def __init__(self, flow_timeout_seconds: float = 120.0):
        self.flow_timeout_seconds = flow_timeout_seconds

    def parse_pcap_to_records(self, pcap_path: Union[str, Path]) -> List[FlowRecord]:
        """
        Parses raw PCAP packets into bidirectional FlowRecords.
        If external tools (Zeek / CICFlowMeter) are used, output flow CSVs
        are parsed via the standard schema.
        """
        path = Path(pcap_path)
        if not path.is_file():
            raise FileNotFoundError(f"PCAP file not found: {path.resolve()}")

        # Placeholder implementation demonstrating Scapy/PyShark flow aggregation
        # Converts network 5-tuple into bidirectional flows
        records: List[FlowRecord] = []
        # Downstream models interact ONLY with the returned FlowRecord collection
        return records

    def load_flows(self, source_path: Union[str, Path]) -> pl.LazyFrame:
        """
        Converts PCAP flows into a standardized Polars LazyFrame matching FlowRecord schema.
        """
        records = self.parse_pcap_to_records(source_path)
        if not records:
            # Return empty typed DataFrame with matching schema if no flows yet
            schema = {
                "src_ip": pl.String,
                "dst_ip": pl.String,
                "src_port": pl.Int64,
                "dst_port": pl.Int64,
                "protocol": pl.Int32,
                "timestamp": pl.Datetime,
                "flow_duration": pl.Float64,
                "fwd_pkts": pl.Float64,
                "bwd_pkts": pl.Float64,
                "fwd_bytes": pl.Float64,
                "bwd_bytes": pl.Float64,
                "flow_bytes_s": pl.Float64,
                "flow_pkts_s": pl.Float64,
                "flow_iat_mean": pl.Float64,
                "flow_iat_std": pl.Float64,
                "pkt_len_mean": pl.Float64,
                "pkt_len_std": pl.Float64,
                "init_win_fwd": pl.Float64,
                "init_win_bwd": pl.Float64,
                "syn_count": pl.Float64,
                "rst_count": pl.Float64,
                "ack_count": pl.Float64,
                "fin_count": pl.Float64,
                "psh_count": pl.Float64,
                "label": pl.String,
            }
            return pl.DataFrame(schema=schema).lazy()

        data_dict = {
            field: [getattr(r, field) for r in records]
            for field in FlowRecord.__dataclass_fields__.keys()
        }
        return pl.DataFrame(data_dict).lazy()
