"""
src/forecasting/kill_chain.py

Multi-Stage Cyber Kill-Chain & MITRE ATT&CK Trajectory Pathfinding Engine.
Analyzes temporal trajectories [S[t-N] ... S[t] ... S[t+K]] to identify coherent
multi-stage attack campaigns:
Reconnaissance -> Initial Access -> Lateral Movement / Infiltration -> Impact / Exfiltration.

Computes Kill-Chain Progression Stage, Campaign Advancement Velocity,
and Next Tactical Objective.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.preprocessing.normalization import CLASS_INDEX_TO_NAME

class MultiStageKillChainEngine:
    """
    Correlates continuous multi-step forecasting trajectories with the Lockheed Martin Cyber Kill Chain
    and MITRE ATT&CK Enterprise Matrix tactical phases.
    """

    # Kill-Chain stages ordered by tactical progression
    KILL_CHAIN_STAGES = [
        {
            "stage_idx": 1,
            "stage_name": "Reconnaissance & Probing",
            "tactic_id": "TA0043",
            "associated_classes": ["RECONNAISSANCE"],
            "severity": "LOW / PROBING",
            "next_likely_tactics": ["Initial Access (TA0001)", "Credential Access (TA0006)"]
        },
        {
            "stage_idx": 2,
            "stage_name": "Initial Access & Web Exploitation",
            "tactic_id": "TA0001",
            "associated_classes": ["WEB_EXPLOIT"],
            "severity": "HIGH / BREACH ATTEMPT",
            "next_likely_tactics": ["Credential Access / Brute Force (TA0006)", "Lateral Movement (TA0008)"]
        },
        {
            "stage_idx": 3,
            "stage_name": "Credential Access & Authentication Brute Force",
            "tactic_id": "TA0006",
            "associated_classes": ["BRUTE_FORCE"],
            "severity": "HIGH / CREDENTIAL COMPROMISE",
            "next_likely_tactics": ["Lateral Movement (TA0008)", "Command and Control (TA0011)"]
        },
        {
            "stage_idx": 4,
            "stage_name": "Internal Lateral Movement & Infiltration",
            "tactic_id": "TA0008",
            "associated_classes": ["INFILTRATION"],
            "severity": "CRITICAL / ACTIVE PIVOT",
            "next_likely_tactics": ["Command and Control (TA0011)", "Impact (TA0040)"]
        },
        {
            "stage_idx": 5,
            "stage_name": "Command & Control (C2) / Botnet Recruitment",
            "tactic_id": "TA0011",
            "associated_classes": ["BOTNET"],
            "severity": "CRITICAL / C2 ESTABLISHED",
            "next_likely_tactics": ["Impact / Flooding (TA0040)", "Exfiltration (TA0010)"]
        },
        {
            "stage_idx": 6,
            "stage_name": "Operational Impact & Service Denial",
            "tactic_id": "TA0040",
            "associated_classes": ["DENIAL_OF_SERVICE"],
            "severity": "CRITICAL / DISRUPTION",
            "next_likely_tactics": ["Campaign Exhaustion", "Secondary Extortion"]
        }
    ]

    @classmethod
    def analyze_trajectory(
        cls,
        history_labels: List[str],
        forecast_labels: List[str],
        threat_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Analyzes the combined historical and forecasted trajectory.
        """
        full_trajectory = history_labels + forecast_labels
        attack_events = [e for e in full_trajectory if e != "BENIGN"]

        if not attack_events:
            return {
                "campaign_detected": False,
                "current_stage_name": "Normal Enterprise Operations",
                "current_stage_idx": 0,
                "max_stage_idx": 0,
                "progression_velocity": "STABLE / BASELINE",
                "advancement_summary": "Network telemetry remains within nominal baseline parameters.",
                "next_tactical_objective": "None (Passive Monitoring)",
                "observed_kill_chain_path": ["Normal Operations (TA0000)"]
            }

        # Map observed events to stages
        observed_stages = []
        for event in full_trajectory:
            stage_found = None
            for s in cls.KILL_CHAIN_STAGES:
                if event in s["associated_classes"]:
                    stage_found = s
                    break
            if stage_found:
                observed_stages.append(stage_found)

        if not observed_stages:
            stage_indices = [0]
            current_stage = cls.KILL_CHAIN_STAGES[0]
        else:
            stage_indices = [s["stage_idx"] for s in observed_stages]
            current_stage = observed_stages[-1]

        max_stage = max(stage_indices)
        is_advancing = len(stage_indices) >= 2 and (stage_indices[-1] > stage_indices[0])

        if is_advancing:
            velocity = "ACCELERATING ADVANCEMENT"
        elif any(s >= 4 for s in stage_indices):
            velocity = "CRITICAL / SUSTAINED LATE-STAGE CAMPAIGN"
        else:
            velocity = "EARLY STAGE PROBING"

        path_names = []
        for s in observed_stages:
            name = f"{s['stage_name']} ({s['tactic_id']})"
            if name not in path_names:
                path_names.append(name)

        return {
            "campaign_detected": True,
            "current_stage_name": current_stage["stage_name"],
            "current_stage_idx": current_stage["stage_idx"],
            "max_stage_idx": max_stage,
            "severity": current_stage["severity"],
            "progression_velocity": velocity,
            "advancement_summary": (
                f"Adversary trajectory has progressed to {current_stage['stage_name']} "
                f"({current_stage['tactic_id']}). Target systems are under active {current_stage['severity'].lower()}."
            ),
            "next_tactical_objective": current_stage["next_likely_tactics"][0] if current_stage["next_likely_tactics"] else "Campaign Mission Goal",
            "observed_kill_chain_path": path_names
        }
