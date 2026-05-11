"""
SQLExecutor – executes SQLite queries with timeout and error classification.

Returns an ExecutionResult that distinguishes:
  - success with results
  - syntax error
  - runtime error (no such column / table)
  - timeout
  - empty result
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExecStatus(Enum):
    SUCCESS = auto()
    SYNTAX_ERROR = auto()
    RUNTIME_ERROR = auto()
    TIMEOUT = auto()
    EMPTY = auto()
    DB_NOT_FOUND = auto()


@dataclass
class ExecutionResult:
    status: ExecStatus
    rows: List[Tuple[Any, ...]] = field(default_factory=list)
    error_msg: str = ""
    sql: str = ""

    @property
    def success(self) -> bool:
        return self.status in (ExecStatus.SUCCESS, ExecStatus.EMPTY)

    @property
    def has_data(self) -> bool:
        return self.status == ExecStatus.SUCCESS and bool(self.rows)

    def short_error(self) -> str:
        """Return a concise error description suitable for a repair prompt."""
        if self.status == ExecStatus.TIMEOUT:
            return "Query timed out."
        if self.status == ExecStatus.DB_NOT_FOUND:
            return "Database file not found."
        if self.error_msg:
            # Strip verbose SQLite prefix
            msg = self.error_msg
            for prefix in ("OperationalError: ", "sqlite3.OperationalError: "):
                msg = msg.replace(prefix, "")
            return msg[:200]
        return ""


class SQLExecutor:
    """
    Executes a SQL string against a SQLite database file.

    Args:
        db_dir:          Root directory for database files.
        timeout_seconds: Max execution time before aborting.
        max_rows:        Cap on returned rows (avoids memory blowup).
    """

    def __init__(
        self,
        db_dir: Path,
        timeout_seconds: float = 5.0,
        max_rows: int = 1000,
    ) -> None:
        self._db_dir = Path(db_dir)
        self._timeout = timeout_seconds
        self._max_rows = max_rows

    # ── Public API ─────────────────────────────────────────────────────────

    def execute(self, sql: str, db_id: str) -> ExecutionResult:
        """Execute *sql* against *db_id* and return an ExecutionResult."""
        db_path = self._resolve_db(db_id)
        if db_path is None:
            return ExecutionResult(
                status=ExecStatus.DB_NOT_FOUND,
                sql=sql,
                error_msg=f"No SQLite file found for db_id='{db_id}'",
            )

        result: List[ExecutionResult] = []
        exc_holder: List[Exception] = []

        def _run() -> None:
            try:
                con = sqlite3.connect(str(db_path), timeout=self._timeout)
                con.execute("PRAGMA query_only = ON")
                cur = con.cursor()
                cur.execute(sql)
                rows = cur.fetchmany(self._max_rows)
                con.close()
                if rows:
                    result.append(ExecutionResult(status=ExecStatus.SUCCESS, rows=rows, sql=sql))
                else:
                    result.append(ExecutionResult(status=ExecStatus.EMPTY, sql=sql))
            except sqlite3.OperationalError as e:
                msg = str(e)
                status = (
                    ExecStatus.SYNTAX_ERROR
                    if "syntax error" in msg.lower()
                    else ExecStatus.RUNTIME_ERROR
                )
                result.append(
                    ExecutionResult(status=status, sql=sql, error_msg=msg)
                )
            except Exception as e:
                exc_holder.append(e)
                result.append(
                    ExecutionResult(
                        status=ExecStatus.RUNTIME_ERROR,
                        sql=sql,
                        error_msg=str(e),
                    )
                )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self._timeout + 1)

        if t.is_alive():
            logger.warning("[SQLExecutor] Query timed out for db_id='%s'", db_id)
            return ExecutionResult(status=ExecStatus.TIMEOUT, sql=sql, error_msg="Execution timeout")

        if result:
            res = result[0]
            if not res.success:
                logger.debug(
                    "[SQLExecutor] %s | %s | %s", db_id, res.status.name, res.short_error()
                )
            return res

        return ExecutionResult(
            status=ExecStatus.RUNTIME_ERROR,
            sql=sql,
            error_msg="Unknown execution error",
        )

    def validate_syntax(self, sql: str, db_id: str) -> Tuple[bool, str]:
        """
        Lightweight syntax check using EXPLAIN.
        Returns (is_valid, error_message).
        """
        db_path = self._resolve_db(db_id)
        if db_path is None:
            return False, f"DB not found: {db_id}"
        try:
            con = sqlite3.connect(str(db_path))
            con.execute(f"EXPLAIN {sql}")
            con.close()
            return True, ""
        except sqlite3.OperationalError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    # ── Internals ─────────────────────────────────────────────────────────

    def _resolve_db(self, db_id: str) -> Optional[Path]:
        nested = self._db_dir / db_id / f"{db_id}.sqlite"
        if nested.exists():
            return nested
        flat = self._db_dir / f"{db_id}.sqlite"
        if flat.exists():
            return flat
        return None