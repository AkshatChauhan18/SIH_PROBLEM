"""
src/forecasting/mitre_mapping.py
Separate interpretation layer mapping predicted operational behaviors to MITRE ATT&CK.
Explicitly distinguishes raw MODEL PREDICTION from MITRE ATT&CK INTERPRETATION.
"""

from typing import Dict, Any, List

# Documented rule-based mapping from predicted operational category to MITRE ATT&CK framework
MITRE_KNOWLEDGE_BASE = {
    "BENIGN": {
        "tactic": "Normal Operations",
        "tactic_id": "TA0000",
        "technique": "Authorized Enterprise Traffic",
        "technique_id": "T0000",
        "description": "Baseline background traffic consistent with standard enterprise workflows.",
        "recommended_action": "Maintain continuous passive monitoring.",
    },
    "RECONNAISSANCE": {
        "tactic": "Reconnaissance",
        "tactic_id": "TA0043",
        "technique": "Network Service Discovery",
        "technique_id": "T1046",
        "description": "Adversary attempting to discover active hosts, open ports, and vulnerable services.",
        "recommended_action": "Rate-limit probing IPs, inspect firewall logs for horizontal port sweeps, enforce IDS drop rules.",
    },
    "BRUTE_FORCE": {
        "tactic": "Credential Access",
        "tactic_id": "TA0006",
        "technique": "Brute Force: Password Guessing / Password Spraying",
        "technique_id": "T1110",
        "description": "Systematic guessing of credentials against remote services (SSH, FTP, Web authentication endpoints).",
        "recommended_action": "Enforce account lockouts, deploy fail2ban / rate-limiting, enforce multi-factor authentication (MFA).",
    },
    "WEB_EXPLOIT": {
        "tactic": "Initial Access",
        "tactic_id": "TA0001",
        "technique": "Exploit Public-Facing Application",
        "technique_id": "T1190",
        "description": "Targeting flaws in web applications (SQL Injection, Cross-Site Scripting) to execute malicious inputs.",
        "recommended_action": "Inspect WAF alerts, sanitize application queries, validate input encoding, isolate target web server.",
    },
    "DENIAL_OF_SERVICE": {
        "tactic": "Impact",
        "tactic_id": "TA0040",
        "technique": "Network Denial of Service: Direct Network Flood",
        "technique_id": "T1498",
        "description": "Flooding network boundaries with voluminous traffic or resource-exhaustion requests (SYN floods, slowloris).",
        "recommended_action": "Engage upstream DDoS scrubbing, enable SYN cookies, rate-limit ingress UDP/TCP connections.",
    },
    "BOTNET": {
        "tactic": "Command and Control",
        "tactic_id": "TA0011",
        "technique": "Application Layer Protocol: Web Protocols",
        "technique_id": "T1071.001",
        "description": "Internal host communicating with external command and control (C2) server or participating in coordinated botnet activity.",
        "recommended_action": "Quarantine infected endpoint, sinkhole C2 destination IP, perform memory forensics on affected host.",
    },
    "INFILTRATION": {
        "tactic": "Lateral Movement / Exfiltration",
        "tactic_id": "TA0008",
        "technique": "Remote Services / Lateral Tool Transfer",
        "technique_id": "T1021 / T1570",
        "description": "Post-compromise pivot into internal network segments, lateral credential reuse, or stealthy data movement.",
        "recommended_action": "Sever internal segmentation bridge, revoke active Kerberos/NTLM sessions, audit internal pivot paths.",
    },
}

def interpret_prediction_as_mitre(
    predicted_class: str,
    confidence: float,
    threat_score: float,
) -> Dict[str, Any]:
    """
    Translates model output into structured MITRE ATT&CK interpretation.
    Strictly separates model prediction outputs from expert security mapping.
    """
    clean_cat = predicted_class.strip().upper()
    info = MITRE_KNOWLEDGE_BASE.get(clean_cat, MITRE_KNOWLEDGE_BASE["BENIGN"])

    return {
        "model_prediction": {
            "predicted_category": predicted_class,
            "confidence": round(confidence, 4),
            "threat_score": round(threat_score, 4),
        },
        "mitre_attack_interpretation": {
            "tactic": info["tactic"],
            "tactic_id": info["tactic_id"],
            "technique": info["technique"],
            "technique_id": info["technique_id"],
            "security_description": info["description"],
            "recommended_soc_action": info["recommended_action"],
        },
    }
