"""
ConfigManager – YAML + CLI + environment variable config system.
Supports experiment-level config profiles for reproducibility.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class PipelineConfig:
    # ── Paths ────────────────────────────────────────────────────────────────
    # Vietnamese dataset (input cho pipeline)
    base_dir: Path = Path("C:\\Users\\Admin\\Documents\\Uni\\DS319\\Text-to-SQL\\data\\word-level")
    tables_file: str = "tables.json"
    train_file: str = "train.json"
    dev_file: str = "dev.json"
    test_file: str = "test.json"

    # English Spider (gold SQL + SQLite database để execute)
    en_base_dir: Path = Path("C:\\Users\\Admin\\Documents\\Uni\\DS319\\Text-to-SQL\\data\\spider_data")
    en_dev_file: str = "test.json"
    db_dir: Path = Path("C:\\Users\\Admin\\Documents\\Uni\\DS319\\Text-to-SQL\\data\\spider_data\\database")

    # Pre-built schema mapping VI↔EN
    schema_map_path: Path = Path("C:\\Users\\Admin\\Documents\\Uni\\DS319\\Text-to-SQL\\pretrain\\schema_matching_map_word.json")

    output_dir: Path = Path("outputs")

    # ── Model ────────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "qwen/qwen3-8b"
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "sk-or-v1-")
    )

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_examples: int = 10
    retrieval_alpha: float = 1.0       # weight for question similarity
    retrieval_beta: float = 0.0        # weight for sql skeleton similarity
    retrieval_gamma: float = 0.0       # weight for schema similarity
    use_mmr: bool = False
    mmr_lambda: float = 0.5            # diversity weight for MMR
    token_budget: int = 3000           # max prompt tokens (approx)

    # ── Schema features ──────────────────────────────────────────────────────
    use_schema_linking: bool = True
    use_schema_pruning: bool = True
    use_cell_value: bool = True
    max_cell_values: int = 5           # values per column to retrieve
    cell_value_fuzzy_threshold: float = 0.7

    # ── Execution & Repair ───────────────────────────────────────────────────
    use_execution_guided: bool = True  # execute predicted SQL and repair on failure
    use_self_repair: bool = True
    max_repair_attempts: int = 3
    sql_timeout: float = 5.0           # seconds

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_temperature: float = 0.0
    api_retries: int = 3
    api_retry_delay: float = 2.0
    api_timeout: int = 60

    # ── Runtime ──────────────────────────────────────────────────────────────
    test_limit: Optional[int] = None
    max_workers: int = 5
    log_level: str = "INFO"
    experiment_name: str = "default"

    # ── Output filenames ─────────────────────────────────────────────────────
    output_json_file: str = "results.json"
    output_sql_file: str = "predicted.sql"
    experiment_log_file: str = "experiment_log.jsonl"

    # ── Derived paths (properties) ────────────────────────────────────────────
    @property
    def tables_path(self) -> Path:
        return self.base_dir / self.tables_file

    @property
    def train_path(self) -> Path:
        return self.base_dir / self.train_file

    @property
    def dev_path(self) -> Path:
        return self.base_dir / self.dev_file

    @property
    def test_path(self) -> Path:
        return self.base_dir / self.test_file

    @property
    def en_dev_path(self) -> Path:
        return self.en_base_dir / self.en_dev_file

    @property
    def output_json_path(self) -> Path:
        return self.output_dir / self.output_json_file

    @property
    def output_sql_path(self) -> Path:
        return self.output_dir / self.output_sql_file

    @property
    def experiment_log_path(self) -> Path:
        return self.output_dir / self.experiment_log_file


class ConfigManager:
    """
    Loads config from YAML file, then overrides from env vars and CLI kwargs.

    Priority (highest → lowest):
        CLI kwargs  >  environment variables  >  YAML file  >  dataclass defaults
    """

    _ENV_PREFIX = "T2SQL_"

    @classmethod
    def load(
        cls,
        yaml_path: Optional[str | Path] = None,
        **overrides: Any,
    ) -> PipelineConfig:
        """
        Build a PipelineConfig from layered sources.

        Args:
            yaml_path: Path to a YAML config file (optional).
            **overrides: Key-value pairs that override everything else.

        Returns:
            Fully resolved PipelineConfig instance.
        """
        cfg: Dict[str, Any] = {}

        # 1. YAML file
        if yaml_path and HAS_YAML:
            cfg.update(cls._load_yaml(Path(yaml_path)))

        # 2. Environment variables (T2SQL_<UPPER_FIELD>=value)
        cfg.update(cls._load_env())

        # 3. CLI overrides
        cfg.update({k: v for k, v in overrides.items() if v is not None})

        # Cast Path fields
        path_fields = {f.name for f in fields(PipelineConfig) if f.type in ("Path", Path)}
        for k in path_fields:
            if k in cfg and not isinstance(cfg[k], Path):
                cfg[k] = Path(cfg[k])

        return PipelineConfig(**{k: v for k, v in cfg.items() if hasattr(PipelineConfig, k)})

    @classmethod
    def _load_yaml(cls, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    @classmethod
    def _load_env(cls) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        field_names = {f.name for f in fields(PipelineConfig)}
        for key, val in os.environ.items():
            if key.startswith(cls._ENV_PREFIX):
                field_name = key[len(cls._ENV_PREFIX):].lower()
                if field_name in field_names:
                    result[field_name] = val
        return result

    @staticmethod
    def to_yaml(config: PipelineConfig, path: str | Path) -> None:
        """Serialize config to YAML for reproducibility."""
        if not HAS_YAML:
            raise ImportError("Install PyYAML: pip install pyyaml")
        data = {
            f.name: str(getattr(config, f.name))
            if isinstance(getattr(config, f.name), Path)
            else getattr(config, f.name)
            for f in fields(config)
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False)