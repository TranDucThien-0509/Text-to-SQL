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

![Kiến trúc tổng quan](https://raw.githubusercontent.com/TranDucThien-0509/Text-to-SQL/main/architecture.png)

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
pip install -r requirements.txt
```
---

## Cấu hình

Pipeline dùng `ConfigManager` với **3 lớp ưu tiên** (cao → thấp):

```
CLI / code kwargs  >  Biến môi trường (T2SQL_*)  >  YAML file  >  Default
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
├── train_spider.json        # Training examples
├── dev_spider.json          # Dev set 
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
### Chạy evaluation toàn bộ dev set

```bash
python run_eval.py --config configs/spider_qwen.yaml
```

### Giới hạn số sample (debug nhanh)

```bash
python run_pipeline.py --config configs/spider_qwen.yaml --test_limit 50
```

## Kết quả đầu ra

Sau khi chạy, tất cả output nằm trong `output_dir` (mặc định: `outputs/`):

```
outputs/spider_qwen/
├── results.json          # Chi tiết từng sample (question, pred_sql, gold_sql, correct)
└── translated_results.yaml       # Result đã được execute trên database
```

---
## Tham khảo

- **DAIL-SQL**: [Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation](https://arxiv.org/abs/2308.15363) — Gao et al., VLDB 2024
- **BAAI/bge-m3**: Multilingual embedding model hỗ trợ tốt tiếng Việt
- **Vietnamese Dataset**: [A Pilot Study of Text-to-SQL Semantic Parsing for Vietnamese](https://yale-lily.github.io/spider)
- **Spider Dataset**: [Yale Semantic Parsing and Text-to-SQL Challenge]([https://yale-lily.github.io/spider](https://github.com/VinAIResearch/ViText2SQL))
- **pyvi**: Vietnamese tokenizer
