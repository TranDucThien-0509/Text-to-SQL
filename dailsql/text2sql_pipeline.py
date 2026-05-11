"""
Text2SQLPipeline – Trình điều phối toàn bộ luồng xử lý.
"""
from __future__ import annotations
import logging
from pathlib import Path
from text2sql.core.config import PipelineConfig
from text2sql.core.llm_client import OpenRouterClient
from text2sql.core.prompt_builder import PromptBuilder
from text2sql.schema.schema_processor import SchemaProcessor
from text2sql.schema.schema_linker import SchemaLinker
from text2sql.schema.schema_pruner import SchemaPruner
from text2sql.retrieval.few_shot_retriever import FewShotRetriever
from text2sql.schema.cell_value_retriever import CellValueRetriever
from text2sql.sql.sql_executor import SQLExecutor
from text2sql.sql.sql_repairer import SQLRepairer

logger = logging.getLogger(__name__)

class Text2SQLPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Khởi tạo các thành phần
        self.schema_processor = SchemaProcessor(config.tables_path).load()
        self.linker = SchemaLinker()
        self.pruner = SchemaPruner()
        self.builder = PromptBuilder(token_budget=config.token_budget)
        self.llm = OpenRouterClient(config)
        
        # Thành phần tùy chọn (Retrieval & Database)
        self.retriever = FewShotRetriever(config).load()
        
        self.cell_retriever = None
        if config.use_cell_value:
            self.cell_retriever = CellValueRetriever(config.db_dir)
            
        self.executor = SQLExecutor(config.db_dir)
        self.repairer = SQLRepairer(self.executor, self.llm, config.max_repair_attempts)

    def run(self, question: str, db_id: str) -> str:
        """Thực hiện toàn bộ luồng từ câu hỏi đến SQL."""
        
        # 1. Lấy thông tin Schema
        schema = self.schema_processor.get_schema(db_id)
        
        # 2. Schema Linking (Nối câu hỏi với bảng/cột)
        linking_res = self.linker.link(question, schema, self.cell_retriever)
        
        # 3. Schema Pruning (Cắt gọn DDL)
        pruned_ddl = schema.ddl
        if self.config.use_schema_pruning:
            pruned_ddl = self.pruner.prune(schema, linking_res, self.schema_processor)
            
        # 4. Few-shot Retrieval (Tìm ví dụ tương tự)
        few_shot_block = self.retriever.retrieve(question)
        
        # 5. Build Prompt
        prompt = self.builder.build(
            schema_text=pruned_ddl,
            few_shot_block=few_shot_block,
            question=question,
            linking_result=linking_res
        )
        
        # 6. LLM Generation (Lần 1)
        initial_sql = self.llm.generate(prompt)
        
        # 7. Self-Repair (Nếu được bật và có DB)
        if self.config.use_self_repair:
            outcome = self.repairer.repair(initial_sql, db_id, question, pruned_ddl)
            return outcome.final_sql
            
        return initial_sql
    
if __name__ == "__main__":
    print("File Text2SQLPipeline đã load thành công!")
    print("Mọi module đã được kết nối. Sẵn sàng nhận nhiệm vụ!")