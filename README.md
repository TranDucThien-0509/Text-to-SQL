# DAIL-SQL — Vietnamese Text-to-SQL Pipeline

Một pipeline **Text-to-SQL** xây dựng trên nền tảng [DAIL-SQL](https://arxiv.org/abs/2308.15363), được thiết kế để dịch câu hỏi tiếng Việt sang SQL trên dataset Spider. Pipeline tích hợp DAIL Selection, DAIL Organization, schema linking tiếng Việt, cell-value retrieval và post-processing tự động.

---

## Mục lục

- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu cài đặt](#yêu-cầu-cài-đặt)
- [Cấu hình](#cấu-hình)
- [Dữ liệu](#dữ-liệu)
- [Cách chạy](#cách-chạy)
- [Chi tiết từng module](#chi-tiết-từng-module)
- [Kết quả đầu ra](#kết-quả-đầu-ra)
- [Thực nghiệm & Reproducibility](#thực-nghiệm--reproducibility)

---
## Kiến trúc tổng quan

![Kiến trúc tổng quan]
(https://raw.githubusercontent.com/TranDucThien-0509/Text-to-SQL/main/architecture.png)

---

## Cấu trúc thư mục

```
dailsql/
└── text2sql/
    ├── core/
    │   ├── config.py            # ConfigManager — YAML + env + CLI config
    │   ├── llm_client.py        # OpenRouter API wrapper
    │   └── prompt_builder.py    # Assembles final LLM prompt (DAIL style)
    │
    ├── retrieval/
    │   ├── few_shot_retriever.py  # DAIL_S + DAIL_O + MMR retrieval
    │   └── sql_normalizer.py      # SQL skeleton extraction for similarity
    │
    ├── schema/
    │   ├── cell_value_retriever.py  # Fuzzy match câu hỏi với cell values
    │   ├── schema_linker.py         # 4-way schema linking (tiếng Việt)
    │   ├── schema_processor.py      # Parse tables.json → DatabaseSchema
    │   └── schema_pruner.py         # Loại bỏ bảng/cột không liên quan
    │
    ├── sql/
    │   ├── sql_executor.py        # Chạy SQL trên SQLite + tính EX
    │   ├── sql_postprocessor.py   # Clean raw LLM output
    │   └── sql_repairer.py        # Self-repair khi SQL lỗi (optional)
    │
    ├── evaluation/                # Metrics & reporting
    └── utils/
        ├── embedding_cache.py     # Cache sentence embeddings xuống disk
        └── experiment_logger.py   # Log từng sample ra JSONL
```

---

## Yêu cầu cài đặt

**Python 3.9+**

```bash
pip install sentence-transformers torch pyvi pyyaml tqdm
```

| Package | Vai trò |
|---|---|
| `sentence-transformers` | Encode câu hỏi & SQL skeleton cho DAIL Selection |
| `torch` | Backend tensor cho similarity computation |
| `pyvi` | Tokenizer tiếng Việt (dùng trong SchemaLinker) |
| `pyyaml` | Đọc file config YAML |
| `tqdm` | Progress bar khi encode embeddings |

---

## Cấu hình

Pipeline dùng `ConfigManager` với **3 lớp ưu tiên** (cao → thấp):

```
CLI / code kwargs  >  Biến môi trường (T2SQL_*)  >  YAML file  >  Default
```

### Tạo file config YAML

Tạo file `configs/spider_qwen.yaml` (hoặc tên tùy ý):

```yaml
# configs/spider_qwen.yaml

# ── Paths ──────────────────────────────────────────────────
base_dir: C:\Users\Admin\Documents\Uni\DS319\Text-to-SQL\data\spider_data
tables_file: tables_spider.json
train_file: train_spider.json
dev_file: dev_spider.json

en_base_dir: C:\Users\Admin\Documents\Uni\DS319\Text-to-SQL\data\spider_data
en_dev_file: test.json
db_dir: C:\Users\Admin\Documents\Uni\DS319\Text-to-SQL\data\spider_data\database

schema_map_path: C:\Users\Admin\Documents\Uni\DS319\Text-to-SQL\pretrain\schema_matching_map.json
output_dir: outputs/spider_qwen

# ── Model ──────────────────────────────────────────────────
embedding_model: BAAI/bge-m3
llm_model: qwen/qwen3-14b          # hoặc openai/gpt-4o-mini

# ── Retrieval ──────────────────────────────────────────────
top_k_examples: 10
retrieval_alpha: 1.0               # weight question similarity
retrieval_beta: 0.0                # weight sql skeleton similarity (0 = tắt)
retrieval_gamma: 0.0
use_mmr: false
token_budget: 3000

# ── Schema ─────────────────────────────────────────────────
use_schema_linking: true
use_schema_pruning: true
use_cell_value: true
max_cell_values: 5
cell_value_fuzzy_threshold: 0.7

# ── Execution & Repair ─────────────────────────────────────
use_execution_guided: false
use_self_repair: false
max_repair_attempts: 2
sql_timeout: 5.0

# ── LLM ────────────────────────────────────────────────────
llm_temperature: 0.0
api_retries: 3
api_timeout: 60

# ── Runtime ────────────────────────────────────────────────
test_limit: null                   # null = chạy toàn bộ dev set
max_workers: 5
experiment_name: spider_qwen
log_level: INFO
```

### Load config trong code

```python
from text2sql.core.config import ConfigManager

cfg = ConfigManager.load("configs/spider_qwen.yaml")

# Override từ code (ưu tiên cao nhất)
cfg = ConfigManager.load(
    "configs/spider_qwen.yaml",
    test_limit=50,
    llm_model="openai/gpt-4o-mini",
)
```

### Biến môi trường

```bash
# Thay thế/override bất kỳ field nào với prefix T2SQL_
export OPENROUTER_API_KEY="sk-or-v1-..."
export T2SQL_LLM_MODEL="qwen/qwen3-14b"
export T2SQL_TEST_LIMIT="100"
```

---

## Dữ liệu

### Cấu trúc thư mục data

```
data/spider_data/
├── tables_spider.json       # Schema định nghĩa (tên tiếng Việt)
├── train_spider.json        # Training examples (question VI + SQL EN)
├── dev_spider.json          # Dev set để evaluate
├── test.json                # Gold SQL (English Spider format)
└── database/
    ├── concert_singer/
    │   └── concert_singer.sqlite
    ├── pets_1/
    │   └── pets_1.sqlite
    └── ...

pretrain/
└── schema_matching_map.json  # Map tên cột VI ↔ EN
```

### Format `train_spider.json`

```json
[
  {
    "db_id": "concert_singer",
    "question": "Có bao nhiêu ca sĩ?",
    "query": "select count ( * ) from singer"
  },
  ...
]
```

---

## Cách chạy

### Chạy trực tiếp từ Python

```python
from text2sql.core.config import ConfigManager
from text2sql.retrieval.few_shot_retriever import FewShotRetriever
from text2sql.schema.schema_processor import SchemaProcessor
from text2sql.schema.schema_linker import SchemaLinker
from text2sql.schema.schema_pruner import SchemaPruner
from text2sql.schema.cell_value_retriever import CellValueRetriever
from text2sql.core.prompt_builder import PromptBuilder
from text2sql.core.llm_client import LLMClient
from text2sql.sql.sql_postprocessor import post_process_sql

# 1. Load config
cfg = ConfigManager.load("configs/spider_qwen.yaml")

# 2. Khởi tạo các module
retriever = FewShotRetriever(cfg).load()
schema_proc = SchemaProcessor(cfg)
linker = SchemaLinker()
pruner = SchemaPruner()
cell_retriever = CellValueRetriever(cfg)
prompt_builder = PromptBuilder(token_budget=cfg.token_budget)
llm = LLMClient(cfg)

# 3. Chạy một câu hỏi
question = "Liệt kê tên và quốc tịch của tất cả ca sĩ nữ."
db_id = "concert_singer"

schema = schema_proc.get_schema(db_id)
linking = linker.link(question, schema, cell_retriever)
pruned_schema = pruner.prune(schema, linking)
few_shot = retriever.retrieve(question)
prompt = prompt_builder.build(pruned_schema, few_shot, question, linking)

raw_sql = llm.complete(prompt)
clean_sql = post_process_sql(raw_sql)
print(clean_sql)
```

### Chạy evaluation toàn bộ dev set

```bash
python run_pipeline.py --config configs/spider_qwen.yaml
```

### Giới hạn số sample (debug nhanh)

```bash
python run_pipeline.py --config configs/spider_qwen.yaml --test_limit 50
```

---

## Chi tiết từng module

### `core/config.py` — ConfigManager

Quản lý toàn bộ hyperparameter qua `PipelineConfig` dataclass. Hỗ trợ serialize ra YAML để tái tạo thực nghiệm:

```python
ConfigManager.to_yaml(cfg, "outputs/spider_qwen/run_config.yaml")
```

### `retrieval/few_shot_retriever.py` — FewShotRetriever

Implement **DAIL_S + DAIL_O** với các tính năng mở rộng:

| Feature | Config key | Mô tả |
|---|---|---|
| Question similarity | `retrieval_alpha` | Cosine sim của masked question embedding (BAAI/bge-m3) |
| SQL skeleton sim | `retrieval_beta` | Jaccard-style skeleton similarity |
| MMR diversity | `use_mmr`, `mmr_lambda` | Tránh chọn examples quá giống nhau |
| Duplicate filter | — | Loại examples có SQL skeleton trùng |
| Embedding cache | tự động | Lưu embeddings xuống `.cache/` để tránh encode lại |

**Output format (DAIL_O):**
```
/* Some example questions and corresponding SQL queries are provided based on similar problems: */
/* Answer the following: Có bao nhiêu ca sĩ? */
select count ( * ) from singer

/* Answer the following: Liệt kê tên tất cả ca sĩ. */
select tên from ca_sĩ
```

### `schema/schema_linker.py` — SchemaLinker

Schema linking 4 loại, hỗ trợ tiếng Việt (dùng `pyvi.ViTokenizer`):

| Match type | Mô tả | Ví dụ |
|---|---|---|
| `q_col_match` | Tên cột khớp token trong câu hỏi | "quốc tịch" → cột `quốc_tịch` |
| `q_tab_match` | Tên bảng khớp token trong câu hỏi | "ca sĩ" → bảng `ca_sĩ` |
| `cell_match` | Giá trị trong DB khớp câu hỏi | "nữ" → `giới_tính = "female"` |
| `num_date_match` | Số/ngày tháng trong câu hỏi | "năm 2023" → cột kiểu DATE |

Matching hierarchy: **exact > partial > n-gram** (n=1,2,3).

### `core/prompt_builder.py` — PromptBuilder

Assembles prompt theo cấu trúc DAIL-SQL:

```
[System instruction (15 quy tắc tiếng Việt)]

/* Given the following database schema: */
CREATE TABLE ca_sĩ (id INTEGER, tên TEXT, quốc_tịch TEXT, giới_tính TEXT)

/* Relevant tables: ca_sĩ */
/* Relevant columns: ca_sĩ.quốc_tịch, ca_sĩ.giới_tính */
/* Cell value hints: ca_sĩ.giới_tính = "female" */

/* Some example questions ... */
/* Answer the following: ... */
select
```

Nếu prompt vượt `token_budget`, tự động drop few-shot examples từ cuối (least similar) cho đến khi vừa budget.

### `sql/sql_postprocessor.py` — SQLPostProcessor

6 bước xử lý theo thứ tự cố định:

| Bước | Xử lý | Ví dụ |
|---|---|---|
| 1. Strip markdown | Xóa ` ```sql ... ``` ` | → clean SQL |
| 2. Cut repetition | Phát hiện vòng lặp `AND ... IN (SELECT ...)` ≥ 3 lần | Cắt + đóng ngoặc |
| 3. Normalize quotes | `'female'` → `"female"` | Single → double quote |
| 4. Fix bare values | `= female` → `= "female"` | Thêm quote cho unquoted string |
| 5. Comma spacing | `a,b` → `a , b` | Chuẩn hóa theo gold SQL format |
| 6. Normalize whitespace | Collapse multiple spaces | |

---

## Kết quả đầu ra

Sau khi chạy, tất cả output nằm trong `output_dir` (mặc định: `outputs/`):

```
outputs/spider_qwen/
├── results.json          # Chi tiết từng sample (question, pred_sql, gold_sql, correct)
├── predicted.sql         # Chỉ predicted SQL, mỗi dòng một câu (cho evaluator)
├── experiment_log.jsonl  # Log từng bước (linking, retrieval, prompt, raw output)
└── run_config.yaml       # Config đã dùng (để reproduce)
```

### Format `results.json`

```json
[
  {
    "idx": 0,
    "db_id": "concert_singer",
    "question": "Liệt kê tên và quốc tịch của tất cả ca sĩ nữ.",
    "gold_sql": "select tên , quốc_tịch from ca_sĩ where giới_tính = \"female\"",
    "pred_sql": "select tên , quốc_tịch from ca_sĩ where giới_tính = \"female\"",
    "correct": true,
    "linking": { "q_col_match": [...], "cell_match": [...] }
  }
]
```

---

## Thực nghiệm & Reproducibility

### So sánh config profiles

| Config | LLM | Embedding | top_k | alpha | beta | MMR |
|---|---|---|---|---|---|---|
| `spider_qwen` | qwen/qwen3-14b | BAAI/bge-m3 | 10 | 1.0 | 0.0 | ✗ |
| `spider_gpt4o` | openai/gpt-4o-mini | BAAI/bge-m3 | 10 | 1.0 | 0.0 | ✗ |
| `spider_mmr` | qwen/qwen3-14b | BAAI/bge-m3 | 10 | 0.7 | 0.3 | ✓ |

### Reproduce một experiment

```python
# Lưu config sau khi chạy
ConfigManager.to_yaml(cfg, "outputs/spider_qwen/run_config.yaml")

# Load lại để reproduce
cfg = ConfigManager.load("outputs/spider_qwen/run_config.yaml")
```

---

## Tham khảo

- **DAIL-SQL**: [Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation](https://arxiv.org/abs/2308.15363) — Gao et al., VLDB 2024
- **BAAI/bge-m3**: Multilingual embedding model hỗ trợ tốt tiếng Việt
- **Spider Dataset**: [Yale Semantic Parsing and Text-to-SQL Challenge](https://yale-lily.github.io/spider)
- **pyvi**: Vietnamese tokenizer
