"""
src/forecasting/soar_playbooks.py

Automated SOAR (Security Orchestration, Automation, and Response) Playbook Engine.
Translates multi-step World Model attack forecasts and MITRE ATT&CK techniques into
actionable, machine-executable defensive rules and playbooks:
- Linux iptables / nftables firewall rules
- Snort & Suricata intrusion detection signatures
- Host / VLAN network containment playbooks (EDR API commands)
- Identity & Credential revocation steps
- Executive DFIR Incident Briefs
"""

from typing import Dict, List, Any, Optional
import datetime

class SOARPlaybookEngine:
    """
    Automated Defensive Response & Orchestration Playbook Generator.
    Maps forecasted threat scenarios to concrete, multi-layered defensive playbooks.
    """

    TACTIC_PLAYBOOKS = {
        "DENIAL_OF_SERVICE": {
            "title": "High-Volume Denial of Service (DoS/DDoS) Mitigation Playbook",
            "priority": "P1 - CRITICAL",
            "mitre_technique": "Network Denial of Service: Direct Network Flood (T1498)",
            "containment_steps": [
                "Activate upstream BGP Flowspec or Cloud DDoS scrubbing (Cloudflare / AWS Shield).",
                "Enable TCP SYN cookies on perimeter Linux firewalls (`sysctl -w net.ipv4.tcp_syncookies=1`).",
                "Implement ingress rate-limiting on gateway routers (committed access rate: 100 Mbps).",
                "Drop malformed packets and invalid TCP state combinations."
            ],
            "forensic_actions": [
                "Trigger automated rolling PCAP ring-buffer snapshot on boundary interfaces.",
                "Extract top 50 source IPs and autonomous system numbers (ASNs) sending SYN/UDP bursts.",
                "Capture netflow telemetry for bandwidth exhaustion volume metrics."
            ]
        },
        "BRUTE_FORCE": {
            "title": "Credential Guessing & Authentication Brute Force Playbook",
            "priority": "P2 - HIGH",
            "mitre_technique": "Password Guessing / Password Spraying (T1110.001 / T1110.003)",
            "containment_steps": [
                "Enforce temporary IP lockout at perimeter proxy/firewall (threshold: 10 failures in 60s).",
                "Force multi-factor authentication (MFA) re-challenge on targeted enterprise accounts.",
                "Revoke active Kerberos TGT and NTLM sessions for targeted service principals.",
                "Block inbound RDP (port 3389) and SSH (port 22) from external untrusted subnets."
            ],
            "forensic_actions": [
                "Query Windows Event Logs for Event ID 4625 (Logon Failure) and 4624 Type 3 (Network Logon).",
                "Audit internal SSH auth.log / secure log for rapid dictionary iteration patterns.",
                "Cross-reference source IPs against known Tor exit nodes and commercial VPN egress pools."
            ]
        },
        "BOTNET": {
            "title": "Botnet Command and Control (C2) Communication Playbook",
            "priority": "P1 - CRITICAL",
            "mitre_technique": "Application Layer Protocol / Dynamic Resolution (T1071 / T1568)",
            "containment_steps": [
                "Isolate compromised host workstation from internal subnet via EDR network quarantine API.",
                "Blackhole destination C2 domains via DNS sinkholing (RPZ / CoreDNS response policy zone).",
                "Null-route C2 destination IP addresses at core boundary switch.",
                "Terminate unauthorized outbound beaconing processes via EDR agent."
            ],
            "forensic_actions": [
                "Dump volatile memory of suspect host processes using WinPmem / LiME for malware analysis.",
                "Collect persistence artifacts (Scheduled Tasks, Run keys, systemd services).",
                "Extract JA3/JA4 TLS client fingerprints from outbound encrypted channels."
            ]
        },
        "WEB_EXPLOIT": {
            "title": "Web Application Exploitation & Injection Defense Playbook",
            "priority": "P2 - HIGH",
            "mitre_technique": "Exploit Public-Facing Application (T1190)",
            "containment_steps": [
                "Enable strict blocking mode on Web Application Firewall (WAF) for SQLi/XSS/RCE inspection rules.",
                "Temporarily place vulnerable URI endpoint into maintenance/sandbox diversion mode.",
                "Rate limit HTTP POST request frequency to 5 req/sec per client session.",
                "Sanitize and validate database parameterized query inputs at gateway reverse proxy."
            ],
            "forensic_actions": [
                "Collect WAF access logs and correlate client session tokens with HTTP User-Agent strings.",
                "Review database query audit logs for anomalous `UNION SELECT` or `xp_cmdshell` executions.",
                "Inspect web application directory for newly dropped webshells (`.php`, `.jsp`, `.aspx`)."
            ]
        },
        "INFILTRATION": {
            "title": "Stealth Infiltration & Post-Compromise Pivot Playbook",
            "priority": "P1 - CRITICAL",
            "mitre_technique": "Remote Services / Lateral Tool Transfer (T1021 / T1570)",
            "containment_steps": [
                "Sever internal segmentation bridges connecting corporate VLAN to database/server VLAN.",
                "Invalidate domain administrator credentials and initiate tier-0 privileged credential rotation.",
                "Disable compromised internal host switchport via 802.1X dynamic VLAN reassignment.",
                "Block internal SMB (445), RPC (135), and WinRM (5985) between disparate workstation pools."
            ],
            "forensic_actions": [
                "Collect Sysmon Event ID 1 (Process Creation) and Event ID 3 (Network Connection) on pivot hosts.",
                "Triage LSASS memory access events (Event ID 10) to detect Mimikatz-style credential dumping.",
                "Scan all endpoint hosts for newly executed staged payloads in `%TEMP%` and `%APPDATA%`."
            ]
        },
        "RECONNAISSANCE": {
            "title": "Pre-Attack Network Probing & Port Scan Containment Playbook",
            "priority": "P3 - MEDIUM",
            "mitre_technique": "Active Scanning: Network Service Scanning (T1595.002)",
            "containment_steps": [
                "Blacklist probing source IP at external edge firewall for 24 hours.",
                "Enable port-knocking or honeypot port redirection for unallocated internal ports.",
                "Suppress ICMP unreachable and TCP RST responses to deny attacker topology mapping.",
                "Review public-facing service banners for version disclosure leaks."
            ],
            "forensic_actions": [
                "Correlate scanned port range to identify whether scanning was targeted or indiscriminate.",
                "Check perimeter IDS logs for vulnerability scanning tool fingerprints (Nmap, Masscan, Shodan)."
            ]
        },
        "BENIGN": {
            "title": "Routine Enterprise Telemetry Monitoring Routine",
            "priority": "P5 - INFORMATIONAL",
            "mitre_technique": "Authorized Enterprise Traffic (T0000)",
            "containment_steps": [
                "No containment required.",
                "Maintain baseline network flow logging and normal SOC telemetry."
            ],
            "forensic_actions": [
                "Continue passive behavioral baseline aggregation."
            ]
        }
    }

    @classmethod
    def generate_iptables_rules(
        cls,
        threat_class: str,
        src_ip: Optional[str] = None,
        dst_ip: Optional[str] = None,
        dst_port: Optional[int] = None
    ) -> str:
        """Generates ready-to-run Linux iptables defense commands."""
        rules = [
            f"# ==================================================================",
            f"# AUTOMATED MITIGATION RULES — THREAT: {threat_class}",
            f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"# =================================================================="
        ]

        src_spec = f"-s {src_ip}" if src_ip else ""
        dst_spec = f"-d {dst_ip}" if dst_ip else ""
        port_spec = f"--dport {dst_port}" if dst_port else ""

        if threat_class == "DENIAL_OF_SERVICE":
            rules.extend([
                "# 1. Drop invalid packets and enforce SYN flood thresholding",
                "iptables -A INPUT -m conntrack --ctstate INVALID -j DROP",
                "iptables -A INPUT -p tcp --syn -m limit --limit 100/s --limit-burst 150 -j ACCEPT",
                "iptables -A INPUT -p tcp --syn -j DROP",
                "# 2. Block attack source IP with logging",
                f"iptables -I INPUT 1 {src_spec} -m limit --limit 5/m --log-prefix '[SOC-DOS-BLOCKED] ' -j LOG",
                f"iptables -I INPUT 2 {src_spec} -j DROP" if src_ip else "# No single IP isolated"
            ])
        elif threat_class in ["BRUTE_FORCE", "WEB_EXPLOIT"]:
            rules.extend([
                f"# 1. Block attacker IP targeting port {dst_port or 'all'}",
                f"iptables -I INPUT 1 {src_spec} -p tcp {port_spec} -j DROP" if src_ip else "iptables -I INPUT 1 -p tcp --dport 22 -m state --state NEW -m recent --set",
                "# 2. Rate-limit authentication connection attempts (max 5 per minute)",
                f"iptables -A INPUT -p tcp {port_spec} -m state --state NEW -m recent --update --seconds 60 --hitcount 5 -j DROP"
            ])
        elif threat_class in ["BOTNET", "INFILTRATION"]:
            rules.extend([
                f"# 1. Immediate bidirectional quarantine of compromised node",
                f"iptables -I FORWARD 1 -s {src_ip} -j DROP" if src_ip else "# Host quarantine rule",
                f"iptables -I FORWARD 2 -d {src_ip} -j DROP" if src_ip else "# Host quarantine rule",
                f"iptables -I OUTPUT 1 -d {dst_ip} -j DROP" if dst_ip else "# C2 blackhole rule"
            ])
        elif threat_class == "RECONNAISSANCE":
            rules.extend([
                f"# 1. Block reconnaissance source IP from port scanning",
                f"iptables -I INPUT 1 {src_spec} -j DROP" if src_ip else "iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP",
                "iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP"
            ])
        else:
            rules.append("# Network traffic within normal baseline parameters. No drop rules generated.")

        return "\n".join(rules)

    @classmethod
    def generate_suricata_signature(
        cls,
        threat_class: str,
        sid_start: int = 1000001,
        src_ip: Optional[str] = None
    ) -> str:
        """Generates custom Suricata/Snort detection signatures."""
        ip_field = src_ip if src_ip else "$EXTERNAL_NET"
        if threat_class == "DENIAL_OF_SERVICE":
            return (
                f'alert tcp {ip_field} any -> $HOME_NET any (msg:"SIH-SOC Forecasted DoS SYN Flood Volume Burst"; '
                f'flags:S; threshold:type both, track by_src, count 200, seconds 5; '
                f'classtype:denial-of-service; sid:{sid_start}; rev:1;)'
            )
        elif threat_class == "BRUTE_FORCE":
            return (
                f'alert tcp {ip_field} any -> $HOME_NET [22,3389,21] (msg:"SIH-SOC Forecasted Multi-Stage Brute Force Attempt"; '
                f'flags:S; threshold:type both, track by_src, count 10, seconds 30; '
                f'classtype:attempted-admin; sid:{sid_start+1}; rev:1;)'
            )
        elif threat_class == "BOTNET":
            return (
                f'alert tcp $HOME_NET any -> {ip_field} any (msg:"SIH-SOC Forecasted Botnet C2 Beaconing Channel"; '
                f'flow:established,to_server; threshold:type both, track by_src, count 15, seconds 60; '
                f'classtype:trojan-activity; sid:{sid_start+2}; rev:1;)'
            )
        elif threat_class == "INFILTRATION":
            return (
                f'alert tcp $HOME_NET any -> $HOME_NET [445,135,5985] (msg:"SIH-SOC Forecasted Internal Lateral Infiltration Pivot"; '
                f'flow:to_server,established; threshold:type both, track by_src, count 5, seconds 20; '
                f'classtype:successful-recon-limited; sid:{sid_start+3}; rev:1;)'
            )
        elif threat_class == "WEB_EXPLOIT":
            return (
                f'alert http {ip_field} any -> $HTTP_SERVERS any (msg:"SIH-SOC Forecasted Web Exploit Injection Burst"; '
                f'flow:established,to_server; content:"POST"; http_method; threshold:type both, track by_src, count 8, seconds 10; '
                f'classtype:web-application-attack; sid:{sid_start+4}; rev:1;)'
            )
        elif threat_class == "RECONNAISSANCE":
            return (
                f'alert tcp {ip_field} any -> $HOME_NET any (msg:"SIH-SOC Forecasted Ingress Reconnaissance Port Sweep"; '
                f'flags:S; threshold:type both, track by_src, count 30, seconds 10; '
                f'classtype:attempted-recon; sid:{sid_start+5}; rev:1;)'
            )
        return '# Normal traffic profile. No alert signature generated.'

    @classmethod
    def get_playbook(
        cls,
        threat_class: str,
        lead_time_seconds: int = 30,
        threat_probability: float = 0.85,
        src_ip: Optional[str] = None,
        dst_ip: Optional[str] = None,
        dst_port: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compiles the full SOAR defensive response package for the forecasted event.
        """
        base = cls.TACTIC_PLAYBOOKS.get(threat_class, cls.TACTIC_PLAYBOOKS["BENIGN"])

        iptables_script = cls.generate_iptables_rules(threat_class, src_ip, dst_ip, dst_port)
        suricata_sig = cls.generate_suricata_signature(threat_class, src_ip=src_ip)

        advance_notice = f"+{lead_time_seconds}s advance lead time"
        urgency = "IMMEDIATE AUTOMATED EXECUTION" if threat_probability > 0.70 and threat_class != "BENIGN" else "SOC ANALYST REVIEW RECOMMENDED"

        return {
            "threat_class": threat_class,
            "title": base["title"],
            "priority": base["priority"],
            "mitre_technique": base["mitre_technique"],
            "advance_lead_time": advance_notice,
            "urgency": urgency,
            "threat_probability": round(threat_probability * 100, 1),
            "containment_steps": base["containment_steps"],
            "forensic_actions": base["forensic_actions"],
            "iptables_script": iptables_script,
            "suricata_signature": suricata_sig,
            "incident_summary": (
                f"Automated World Model telemetry has detected an emerging {threat_class} attack pattern with "
                f"{round(threat_probability*100, 1)}% confidence, giving SOC operators {advance_notice} "
                f"to isolate assets before post-compromise damage materializes."
            )
        }
