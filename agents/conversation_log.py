"""
Immutable Conversation Log — append-only debate transcript.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal


@dataclass(frozen=True)
class LogEntry:
    timestamp: str
    agent_name: str
    role: Literal["advocate", "critic", "fairness", "orchestrator"]
    round_number: int
    content: str
    entry_type: Literal["argument", "rebuttal", "fact_check", "verdict", "meta"]


class ConversationLog:
    def __init__(self) -> None:
        self._entries: list[LogEntry] = []

    def add_entry(
        self,
        agent_name: str,
        role: Literal["advocate", "critic", "fairness", "orchestrator"],
        round_number: int,
        content: str,
        entry_type: Literal["argument", "rebuttal", "fact_check", "verdict", "meta"] = "argument",
    ) -> LogEntry:
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=agent_name,
            role=role,
            round_number=round_number,
            content=content,
            entry_type=entry_type,
        )
        self._entries.append(entry)
        return entry

    def get_full_transcript(self) -> list[LogEntry]:
        return copy.deepcopy(self._entries)

    def get_round(self, round_number: int) -> list[LogEntry]:
        return [e for e in self._entries if e.round_number == round_number]

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    def export_markdown(self) -> str:
        if not self._entries:
            return "_No debate entries recorded._"
        lines = ["# 🗂 Debate Transcript", ""]
        current_round = -1
        for entry in self._entries:
            if entry.round_number != current_round:
                current_round = entry.round_number
                lines.append(f"## Round {current_round}")
                lines.append("")
            role_emoji = {"advocate": "🟢", "critic": "🔴", "fairness": "⚖️", "orchestrator": "🔷"}.get(entry.role, "🔹")
            lines.append(f"### {role_emoji} {entry.agent_name} ({entry.entry_type})")
            lines.append(f"_{entry.timestamp}_")
            lines.append("")
            lines.append(entry.content)
            lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines)

    def export_dict(self) -> list[dict]:
        return [
            {
                "timestamp": e.timestamp,
                "agent_name": e.agent_name,
                "role": e.role,
                "round_number": e.round_number,
                "content": e.content,
                "entry_type": e.entry_type,
            }
            for e in self._entries
        ]
