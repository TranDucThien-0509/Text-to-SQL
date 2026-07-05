"""
Text2SQLPipeline – Trình điều phối toàn bộ luồng xử lý.
"""
from __future__ import annotations
import logging
from pathlib import Path
from unittest import result
from text2sql.core.config import PipelineConfig
from text2sql.core.llm_client import OpenRouterClient
from text2sql.core.prompt_builder import PromptBuilder
from text2sql.schema.schema_processor import SchemaProcessor
from text2sql.schema.schema_linker import SchemaLinker
from text2sql.schema.schema_pruner import SchemaPruner
from text2sql.retrieval.few_shot_retriever import FewShotRetriever
from text2sql.schema.cell_value_retriever import CellValueRetriever
from text2sql.utils.question_translator import QuestionTranslator
from text2sql.schema.schema_graph_linker import SchemaGraphLinker 
from text2sql.schema.masker import mask_question
from text2sql.sql.sql_executor import SQLExecutor
from text2sql.sql.sql_repairer import SQLRepairer
from text2sql.sql.sql_postprocessor import post_process_sql
from pretrain.schema_matching import translate_sql, create_schema_matching_dict



logger = logging.getLogger(__name__)

class Text2SQLPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Khởi tạo các thành phần
        self.schema_processor = SchemaProcessor(config.tables_path).load()
        self.linker = SchemaLinker()
        # self.graph_linker = SchemaGraphLinker(force_union=True, fallback_hops=1)
        self.pruner = SchemaPruner()
        self.builder = PromptBuilder(token_budget=config.token_budget)
        self.llm = OpenRouterClient(config)
        
        # Thành phần tùy chọn (Retrieval & Database)
        self.retriever = FewShotRetriever(config).load()
        
        self.cell_retriever = None
        if config.use_cell_value:
            translator = None
            if getattr(config, "use_value_translation", False):
                translator = QuestionTranslator(
                    model_name=getattr(config, "translator_model", "google/mt5-large")
                )
                logger.info("QuestionTranslator loaded for cell-value matching.")
            self.cell_retriever = CellValueRetriever(config.db_dir, translator=translator)
            
        # ── Cấu hình truy xuất DB (SQLExecutor / SQLRepairer) ──
        # Dùng getattr với default an toàn vì PipelineConfig có thể chưa có
        # các field này — nếu bạn thêm vào PipelineConfig thật, những dòng
        # getattr này vẫn hoạt động đúng, chỉ là không còn cần fallback.
        sql_timeout_seconds = getattr(config, "sql_timeout_seconds", 5.0)
        sql_max_rows = getattr(config, "sql_max_rows", 1000)
        repair_max_values_per_col = getattr(config, "repair_max_values_per_col", 10)
        repair_max_value_dump_chars = getattr(config, "repair_max_value_dump_chars", 2000)

        self.executor = SQLExecutor(
            config.db_dir,
            timeout_seconds=sql_timeout_seconds,
            max_rows=sql_max_rows,
        )

        # Load schema matching map để SQLRepairer có thể translate VI→EN trước execute
        self._matching_dict: dict = {}
        if hasattr(config, "schema_map_path") and config.schema_map_path.exists():
            import json
            with open(config.schema_map_path, encoding="utf-8") as f:
                self._matching_dict = json.load(f)
            logger.info("Schema matching map loaded: %d databases", len(self._matching_dict))
        else:
            logger.warning(
                "schema_map_path không tìm thấy — SQLRepairer sẽ execute VI SQL trực tiếp (có thể fail)."
            )

        translate_fn = (
            (lambda vi_sql, db_id: translate_sql(vi_sql, db_id, self._matching_dict))
            if self._matching_dict else None
        )
        # QUAN TRỌNG: phải truyền db_dir=... — thiếu nó thì toàn bộ bước
        # value-matching + full-table value-dump trong SQLRepairer bị vô hiệu
        # hóa hoàn toàn (self._db_dir sẽ là None), dù use_self_repair=True.
        self.repairer = SQLRepairer(
            self.executor,
            self.llm,
            max_attempts=config.max_repair_attempts,
            translate_fn=translate_fn,
            db_dir=str(config.db_dir),
            value_dump_max_values_per_col=repair_max_values_per_col,
            value_dump_max_chars=repair_max_value_dump_chars,
        )

    def run(self, question: str, db_id: str) -> str:
        """Thực hiện toàn bộ luồng từ câu hỏi đến SQL."""
        
        # 1. Lấy thông tin Schema
        schema = self.schema_processor.get_schema(db_id)
        
        # 2. Schema Linking (Nối câu hỏi với bảng/cột)
        # result = self.graph_linker.link(question, schema, self.cell_retriever)
        result = self.linker.link(question, schema, self.cell_retriever)
        masked = mask_question(question, result)
        
        # 3. Schema Pruning (Cắt gọn DDL)
        pruned_ddl = schema.ddl
        if self.config.use_schema_pruning:
            pruned_ddl = self.pruner.prune(schema, result, self.schema_processor)
            
        # 4. Few-shot Retrieval (Tìm ví dụ tương tự)
        few_shot_block = self.retriever.retrieve(masked)
        
        # 5. Build Prompt
        prompt = self.builder.build(
            schema_text=pruned_ddl,
            few_shot_block=few_shot_block,
            question=question,
            linking_result=result,
            cell_retriever=self.cell_retriever,
            db_id=db_id,
        )
        
        # 6. LLM Generation (Lần 1)
        initial_sql = self.llm.generate(prompt)
        initial_sql = post_process_sql(initial_sql) 
        
        # 7. Self-Repair (Nếu được bật và có DB)
        if self.config.use_self_repair:
            outcome = self.repairer.repair(initial_sql, db_id, question, pruned_ddl)
            return outcome.final_sql
            
        return initial_sql
    
if __name__ == "__main__":
    print("File Text2SQLPipeline đã load thành công!")
    print("Mọi module đã được kết nối. Sẵn sàng nhận nhiệm vụ!")