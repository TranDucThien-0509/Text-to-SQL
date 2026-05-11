"""
Evaluator – Spider-style metrics.

Metrics:
  1. Exact Match (EM)         – normalized SQL string equality
  2. Execution Accuracy (EX)  – result-set equality via SQLite
  3. Component Matching       – per-clause precision (SELECT / WHERE / GROUP BY …)
  4. Skeleton Accuracy        – skeleton string equality
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from text2sql.retrieval.sql_normalizer import SQLNormalizer
from text2sql.sql.sql_executor import SQLExecutor, ExecStatus

logger = logging.getLogger(__name__)


@dataclass
class SampleScore:
    index: int
    db_id: str
    question: str
    predicted: str
    gold: str
    exact_match: bool = False
    exec_match: bool = False
    skeleton_match: bool = False
    component_scores: Dict[str, bool] = field(default_factory=dict)
    complexity: str = ""
    error_msg: str = ""


@dataclass
class AggregateMetrics:
    total: int = 0
    exact_match_acc: float = 0.0
    exec_acc: float = 0.0
    skeleton_acc: float = 0.0
    component_acc: Dict[str, float] = field(default_factory=dict)
    by_complexity: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"  Total samples : {self.total}",
            f"  Exact Match   : {self.exact_match_acc:.1%}",
            f"  Exec Accuracy : {self.exec_acc:.1%}",
            f"  Skeleton Acc  : {self.skeleton_acc:.1%}",
        ]
        if self.component_acc:
            lines.append("  Component accuracy:")
            for clause, acc in sorted(self.component_acc.items()):
                lines.append(f"    {clause:12s}: {acc:.1%}")
        if self.by_complexity:
            lines.append("  By complexity (EM / EX):")
            for lvl in ("easy", "medium", "hard", "extra"):
                if lvl in self.by_complexity:
                    em = self.by_complexity[lvl].get("em", 0)
                    ex = self.by_complexity[lvl].get("ex", 0)
                    lines.append(f"    {lvl:8s}: EM={em:.1%}  EX={ex:.1%}")
        return "\n".join(lines)


class Evaluator:
    """
    Computes evaluation metrics for a list of (predicted, gold, db_id) triples.

    Args:
        executor: SQLExecutor instance (for execution accuracy).
                  If None, execution accuracy is skipped.
    """

    _COMPONENTS = ["SELECT", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "JOIN"]

    def __init__(self, executor: Optional[SQLExecutor] = None) -> None:
        self._executor = executor

    # ── Public API ─────────────────────────────────────────────────────────

    def score_sample(
        self,
        index: int,
        db_id: str,
        question: str,
        predicted: str,
        gold: str,
    ) -> SampleScore:
        sample = SampleScore(
            index=index,
            db_id=db_id,
            question=question,
            predicted=predicted,
            gold=gold,
            complexity=SQLNormalizer.complexity_label(gold),
        )

        pred_norm = SQLNormalizer.normalize(predicted)
        gold_norm = SQLNormalizer.normalize(gold)

        # 1. Exact Match
        sample.exact_match = pred_norm == gold_norm

        # 2. Skeleton Match
        sample.skeleton_match = (
            SQLNormalizer.skeleton(predicted) == SQLNormalizer.skeleton(gold)
        )

        # 3. Component Matching
        sample.component_scores = self._component_match(pred_norm, gold_norm)

        # 4. Execution Accuracy
        if self._executor is not None:
            sample.exec_match = self._exec_match(predicted, gold, db_id)

        return sample

    def evaluate(
        self,
        samples: List[Tuple[int, str, str, str, str]],
        # List of (index, db_id, question, predicted, gold)
    ) -> Tuple[List[SampleScore], AggregateMetrics]:
        scores: List[SampleScore] = []
        for args in samples:
            scores.append(self.score_sample(*args))

        agg = self._aggregate(scores)
        logger.info("Evaluation complete:\n%s", agg.summary())
        return scores, agg

    # ── Internals ─────────────────────────────────────────────────────────

    def _exec_match(self, predicted: str, gold: str, db_id: str) -> bool:
        if self._executor is None:
            return False
        pred_res = self._executor.execute(predicted, db_id)
        gold_res = self._executor.execute(gold, db_id)
        if not pred_res.success or not gold_res.success:
            return False
        return self._result_sets_equal(pred_res.rows, gold_res.rows)

    @staticmethod
    def _result_sets_equal(
        rows_a: list, rows_b: list, ordered: bool = False
    ) -> bool:
        """Compare two result sets (as bags of rows)."""
        if len(rows_a) != len(rows_b):
            return False
        norm = lambda rows: [tuple(str(c).strip().lower() for c in r) for r in rows]
        na, nb = norm(rows_a), norm(rows_b)
        if ordered:
            return na == nb
        return sorted(na) == sorted(nb)

    def _component_match(
        self, pred: str, gold: str
    ) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for clause in self._COMPONENTS:
            pred_part = self._extract_clause(pred, clause)
            gold_part = self._extract_clause(gold, clause)
            results[clause] = (pred_part == gold_part)
        return results

    @staticmethod
    def _extract_clause(sql: str, clause: str) -> str:
        """Extract the value portion of a specific SQL clause."""
        pattern = rf"\b{re.escape(clause)}\b(.*?)(?:\b(?:FROM|WHERE|GROUP|HAVING|ORDER|LIMIT|JOIN|UNION|$)\b|$)"
        m = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip().upper()
        return ""

    @staticmethod
    def _aggregate(scores: List[SampleScore]) -> AggregateMetrics:
        n = len(scores)
        if n == 0:
            return AggregateMetrics()

        agg = AggregateMetrics(total=n)
        agg.exact_match_acc = sum(s.exact_match for s in scores) / n
        agg.exec_acc = sum(s.exec_match for s in scores) / n
        agg.skeleton_acc = sum(s.skeleton_match for s in scores) / n

        # Component accuracy
        all_clauses = set(k for s in scores for k in s.component_scores)
        for clause in all_clauses:
            hits = sum(s.component_scores.get(clause, False) for s in scores)
            agg.component_acc[clause] = hits / n

        # By complexity
        from collections import defaultdict
        buckets: Dict[str, List[SampleScore]] = defaultdict(list)
        for s in scores:
            buckets[s.complexity].append(s)
        for lvl, bucket in buckets.items():
            bn = len(bucket)
            agg.by_complexity[lvl] = {
                "em": sum(s.exact_match for s in bucket) / bn,
                "ex": sum(s.exec_match for s in bucket) / bn,
            }

        return agg