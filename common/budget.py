# common/budget.py
"""Token/成本预算追踪器：超限抛 BudgetExceededError 干净终止。"""
from __future__ import annotations

from dataclasses import dataclass


class BudgetExceededError(Exception):
    """预算超限。"""


@dataclass
class BudgetTracker:
    max_tokens: int | None = None
    max_cost: float | None = None
    _used_tokens: int = 0
    _used_cost: float = 0.0

    def consume(self, tokens: int = 0, cost: float = 0.0) -> None:
        self._used_tokens += int(tokens)
        self._used_cost += float(cost)
        if self.exceeded():
            raise BudgetExceededError(self.to_summary())

    def check(self) -> bool:
        return not self.exceeded()

    def exceeded(self) -> bool:
        if self.max_tokens is not None and self._used_tokens >= self.max_tokens:
            return True
        if self.max_cost is not None and self._used_cost >= self.max_cost:
            return True
        return False

    def usage(self) -> dict:
        return {"used_tokens": self._used_tokens,
                "used_cost": round(self._used_cost, 4),
                "max_tokens": self.max_tokens,
                "max_cost": self.max_cost}

    def to_summary(self) -> str:
        return (f"预算追踪：tokens={self._used_tokens}/{self.max_tokens}，"
                f"cost={self._used_cost:.4f}/{self.max_cost}")

    def merge(self, other: "BudgetTracker") -> None:
        self._used_tokens += other._used_tokens
        self._used_cost += other._used_cost
