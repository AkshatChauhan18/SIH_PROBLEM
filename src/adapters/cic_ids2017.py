"""
src/adapters/cic_ids2017.py
Concrete adapter for CIC-IDS2017 flow CSV files using Polars LazyFrames.
"""

from pathlib import Path
from typing import Union
import polars as pl
from src.adapters.base_adapter import BaseAdapter

class CICIDS2017Adapter(BaseAdapter):
    """
    High-performance lazy adapter for CIC-IDS2017 dataset.
    Handles dirty headers, 12-to-24 hour timestamp correction, and null/inf conversion.
    """

    def load_flows(self, source_path: Union[str, Path]) -> pl.LazyFrame:
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"CIC-IDS2017 file not found: {path.resolve()}")

        # Read raw header to construct schema overrides for numeric columns
        # Handles scientific notation (e.g. 1.23E+07, 6.36E+07) without integer overflow
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_header_line = f.readline().rstrip("\r\n")
        raw_cols = raw_header_line.split(",")
        string_col_names = {"flow id", "source ip", "destination ip", "timestamp", "label"}
        schema_overrides = {
            col: pl.Float64
            for col in raw_cols
            if col.strip().lower() not in string_col_names
        }

        # Scan lazily with utf8-lossy to handle any non-ASCII characters
        lf = pl.scan_csv(
            path,
            encoding="utf8-lossy",
            schema_overrides=schema_overrides,
            truncate_ragged_lines=True,
            null_values=["Infinity", "-Infinity", "+Infinity", "NaN", "nan", "null", ""],
        )

        schema = lf.collect_schema()
        names = schema.names()

        # Find raw column by stripped name
        def get_col(name_stripped: str) -> str:
            for c in names:
                if c.strip() == name_stripped:
                    return c
            raise KeyError(f"Column '{name_stripped}' not found in {path.name}")

        c_flow_id = get_col("Flow ID")
        c_src_ip = get_col("Source IP")
        c_dst_ip = get_col("Destination IP")
        c_src_port = get_col("Source Port")
        c_dst_port = get_col("Destination Port")
        c_protocol = get_col("Protocol")
        c_timestamp = get_col("Timestamp")
        c_duration = get_col("Flow Duration")

        c_fwd_pkts = get_col("Total Fwd Packets")
        c_bwd_pkts = get_col("Total Backward Packets")
        c_fwd_bytes = get_col("Total Length of Fwd Packets")
        c_bwd_bytes = get_col("Total Length of Bwd Packets")
        c_flow_bytes_s = get_col("Flow Bytes/s")
        c_flow_pkts_s = get_col("Flow Packets/s")

        c_flow_iat_mean = get_col("Flow IAT Mean")
        c_flow_iat_std = get_col("Flow IAT Std")
        c_pkt_len_mean = get_col("Packet Length Mean")
        c_pkt_len_std = get_col("Packet Length Std")
        c_init_win_fwd = get_col("Init_Win_bytes_forward")
        c_init_win_bwd = get_col("Init_Win_bytes_backward")

        c_syn = get_col("SYN Flag Count")
        c_rst = get_col("RST Flag Count")
        c_ack = get_col("ACK Flag Count")
        c_fin = get_col("FIN Flag Count")
        c_psh = get_col("PSH Flag Count")
        c_label = get_col("Label")

        # 1. Filter out empty / corrupt padding lines
        is_valid_row = (
            pl.col(c_flow_id).is_not_null()
            & (pl.col(c_flow_id).str.strip_chars() != "")
            & pl.col(c_src_ip).is_not_null()
            & (pl.col(c_src_ip).str.strip_chars() != "")
            & pl.col(c_timestamp).is_not_null()
            & (pl.col(c_timestamp).str.strip_chars() != "")
        )
        lf = lf.filter(is_valid_row)

        # 2. Robust 12-to-24 hour timestamp parsing
        ts_raw = pl.col(c_timestamp).str.strip_chars()
        parsed_dt = pl.coalesce([
            ts_raw.str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S", strict=False),
            ts_raw.str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False),
            ts_raw.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
            ts_raw.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M", strict=False),
        ])
        # Afternoon correction for working hours captures
        dt_24h = (
            pl.when(parsed_dt.dt.hour().is_between(1, 7))
            .then(parsed_dt + pl.duration(hours=12))
            .otherwise(parsed_dt)
        )

        # 3. Sanitize label strings (normalize non-ASCII dashes)
        clean_label = (
            pl.col(c_label)
            .cast(pl.String)
            .str.replace_all("\ufffd", "-")
            .str.replace_all("–", "-")
            .str.replace_all("—", "-")
            .str.strip_chars()
        )

        # 4. Standardize to FlowRecord schema
        standardized_lf = lf.select([
            pl.col(c_src_ip).cast(pl.String).str.strip_chars().alias("src_ip"),
            pl.col(c_dst_ip).cast(pl.String).str.strip_chars().alias("dst_ip"),
            pl.col(c_src_port).cast(pl.Int64, strict=False).fill_null(0).alias("src_port"),
            pl.col(c_dst_port).cast(pl.Int64, strict=False).fill_null(0).alias("dst_port"),
            pl.col(c_protocol).cast(pl.Int32, strict=False).fill_null(6).alias("protocol"),

            dt_24h.alias("timestamp"),
            pl.col(c_duration).cast(pl.Float64, strict=False).fill_null(0.0).alias("flow_duration"),

            pl.col(c_fwd_pkts).cast(pl.Float64, strict=False).fill_null(0.0).alias("fwd_pkts"),
            pl.col(c_bwd_pkts).cast(pl.Float64, strict=False).fill_null(0.0).alias("bwd_pkts"),
            pl.col(c_fwd_bytes).cast(pl.Float64, strict=False).fill_null(0.0).alias("fwd_bytes"),
            pl.col(c_bwd_bytes).cast(pl.Float64, strict=False).fill_null(0.0).alias("bwd_bytes"),
            pl.col(c_flow_bytes_s).cast(pl.Float64, strict=False).alias("flow_bytes_s"),
            pl.col(c_flow_pkts_s).cast(pl.Float64, strict=False).alias("flow_pkts_s"),

            pl.col(c_flow_iat_mean).cast(pl.Float64, strict=False).fill_null(0.0).alias("flow_iat_mean"),
            pl.col(c_flow_iat_std).cast(pl.Float64, strict=False).fill_null(0.0).alias("flow_iat_std"),
            pl.col(c_pkt_len_mean).cast(pl.Float64, strict=False).fill_null(0.0).alias("pkt_len_mean"),
            pl.col(c_pkt_len_std).cast(pl.Float64, strict=False).fill_null(0.0).alias("pkt_len_std"),
            pl.col(c_init_win_fwd).cast(pl.Float64, strict=False).fill_null(0.0).alias("init_win_fwd"),
            pl.col(c_init_win_bwd).cast(pl.Float64, strict=False).fill_null(0.0).alias("init_win_bwd"),

            pl.col(c_syn).cast(pl.Float64, strict=False).fill_null(0.0).alias("syn_count"),
            pl.col(c_rst).cast(pl.Float64, strict=False).fill_null(0.0).alias("rst_count"),
            pl.col(c_ack).cast(pl.Float64, strict=False).fill_null(0.0).alias("ack_count"),
            pl.col(c_fin).cast(pl.Float64, strict=False).fill_null(0.0).alias("fin_count"),
            pl.col(c_psh).cast(pl.Float64, strict=False).fill_null(0.0).alias("psh_count"),

            clean_label.alias("label"),
        ])

        return standardized_lf
