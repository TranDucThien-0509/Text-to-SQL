"""
SQLRepairer – self-correction loop for failed SQL queries.

Flow:
  1. Execute generated SQL
  2. If failed → build repair prompt with error feedback
  3. Re-call LLM → re-execute
  4. Repeat up to max_attempts times
  5. Return best result (last successful or original)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from text2sql.sql.sql_executor import ExecutionResult, ExecStatus, SQLExecutor

if TYPE_CHECKING:
    from text2sql.core.llm_client import OpenRouterClient

logger = logging.getLogger(__name__)


@dataclass
class RepairAttempt:
    attempt_num: int
    sql: str
    exec_result: ExecutionResult
    repair_prompt_snippet: str = ""


@dataclass
class RepairOutcome:
    final_sql: str
    success: bool
    attempts: List[RepairAttempt] = field(default_factory=list)

    @property
    def num_repairs(self) -> int:
        return max(0, len(self.attempts) - 1)


class SQLRepairer:
    """
    Wraps the generate → execute → repair loop.

    Args:
        executor:     SQLExecutor instance.
        llm:          OpenRouterClient instance.
        max_attempts: Total attempts including the first generation.
    """

    _REPAIR_TEMPLATE = """\
The following SQLite query failed with this error:
Error: {error}

Failed SQL:
{failed_sql}

Database schema:
{schema}

Original question: {question}

Fix the SQL query. Output ONLY the corrected SQL with no explanation.
SELECT """

    def __init__(
        self,
        executor: SQLExecutor,
        llm: "OpenRouterClient",
        max_attempts: int = 3,
    ) -> None:
        self._executor = executor
        self._llm = llm
        self._max_attempts = max_attempts

    # ── Public API ─────────────────────────────────────────────────────────

    def repair(
        self,
        initial_sql: str,
        db_id: str,
        question: str,
        schema_text: str,
    ) -> RepairOutcome:
        """
        Try to execute *initial_sql*; repair if it fails.

        Args:
            initial_sql:  First-pass generated SQL.
            db_id:        Target database identifier.
            question:     Original NL question (for repair prompt context).
            schema_text:  DDL string (for repair prompt context).

        Returns:
            RepairOutcome with the best SQL found and all attempt records.
        """
        attempts: List[RepairAttempt] = []
        current_sql = initial_sql

        for attempt_num in range(1, self._max_attempts + 1):
            exec_result = self._executor.execute(current_sql, db_id)
            attempt = RepairAttempt(
                attempt_num=attempt_num,
                sql=current_sql,
                exec_result=exec_result,
            )
            attempts.append(attempt)

            if exec_result.success:
                logger.debug(
                    "[SQLRepairer] %s: success on attempt %d/%d",
                    db_id, attempt_num, self._max_attempts
                )
                return RepairOutcome(final_sql=current_sql, success=True, attempts=attempts)

            if attempt_num == self._max_attempts:
                break

            error_msg = exec_result.short_error()
            logger.info(
                "[SQLRepairer] %s: attempt %d failed (%s). Repairing...",
                db_id, attempt_num, error_msg
            )

            repair_prompt = self._build_repair_prompt(
                failed_sql=current_sql,
                error=error_msg,
                question=question,
                schema=schema_text,
            )
            attempt.repair_prompt_snippet = repair_prompt[:300]

            repaired = self._llm.generate(repair_prompt)
            current_sql = repaired

        # All attempts failed – return last generated SQL
        logger.warning(
            "[SQLRepairer] %s: all %d attempts failed. Returning last SQL.",
            db_id, self._max_attempts
        )
        return RepairOutcome(final_sql=current_sql, success=False, attempts=attempts)

    # ── Internals ─────────────────────────────────────────────────────────

    def _build_repair_prompt(
        self,
        failed_sql: str,
        error: str,
        question: str,
        schema: str,
    ) -> str:
        return self._REPAIR_TEMPLATE.format(
            error=error,
            failed_sql=failed_sql,
            schema=schema,
            question=question,
        )