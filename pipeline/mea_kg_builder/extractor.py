# mea_kg_builder/extractor.py
# 核心提取引擎：实现“提取-修正”双重管道 (集成 Schema RAG 版)

import logging
import json
import torch
from typing import List, Dict
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util  # 新增依赖
from .llm_client import global_client
from .ontology_loader import global_ontology
from . import prompt_templates as prompts

logger = logging.getLogger(__name__)


# --- 1. Pydantic 数据模型 ---
class Triplet(BaseModel):
    head_entity: str = Field(..., description="The exact name of the head entity as it appears in the text.")
    head_type: str = Field(..., description="The type of the head entity, chosen from the candidate list.")
    relation: str = Field(..., description="The relation type, chosen from the candidate list.")
    tail_entity: str = Field(..., description="The exact name of the tail entity as it appears in the text.")
    tail_type: str = Field(..., description="The type of the tail entity, chosen from the candidate list.")


class ExtractionResult(BaseModel):
    triplets: List[Triplet] = Field(default_factory=list, description="A list of valid extracted triplets.")


# --- 2. 双重管道提取器类 (集成 Schema RAG) ---
class DualPipelineExtractor:
    def __init__(self):
        self.llm = global_client
        self.ontology = global_ontology

        # 1. 加载完整本体列表
        self.all_entities = self.ontology.get_entity_type_list()
        self.all_relations = self.ontology.get_relation_type_list()
        logger.info(f"提取器初始化: 全局加载 {len(self.all_entities)} 实体, {len(self.all_relations)} 关系。")

        # 2. 初始化 Schema RAG 向量模型 (使用轻量级模型以免爆显存)
        logger.info("正在初始化 Schema RAG 向量模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 注意：如果您的显存非常紧张，可以强制使用 "cpu"
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        # 3. 预编码本体 (创建索引)
        logger.info("正在为本体构建向量索引...")
        self.entity_index = self.sbert.encode(self.all_entities, convert_to_tensor=True, show_progress_bar=False)
        self.relation_index = self.sbert.encode(self.all_relations, convert_to_tensor=True, show_progress_bar=False)
        logger.info("Schema RAG 索引构建完毕。")

    def _get_dynamic_constraints(self, text: str, top_k_ent=50, top_k_rel=20) -> Dict[str, str]:
        """
        核心升级：根据输入文本，动态检索最相关的本体约束。
        """
        # 1. 编码输入文本
        text_emb = self.sbert.encode(text, convert_to_tensor=True, device=self.device)

        # 2. 检索最相关的实体类型 (Top-K)
        ent_hits = util.cos_sim(text_emb, self.entity_index)[0]
        top_ent_indices = torch.topk(ent_hits, k=min(top_k_ent, len(self.all_entities))).indices.tolist()
        relevant_entities = [self.all_entities[i] for i in top_ent_indices]

        # 3. 检索最相关的关系类型 (Top-K)
        rel_hits = util.cos_sim(text_emb, self.relation_index)[0]
        top_rel_indices = torch.topk(rel_hits, k=min(top_k_rel, len(self.all_relations))).indices.tolist()
        relevant_relations = [self.all_relations[i] for i in top_rel_indices]

        return {
            "entity_types": json.dumps(relevant_entities),
            "relation_types": json.dumps(relevant_relations)
        }

    def extract(self, text: str) -> List[Triplet]:
        if not text or len(text.strip()) < 10: return []

        # --- 关键：为当前文本获取动态约束 ---
        # 将 4000+ 个概念缩减为最相关的 50 个，极大降低 Prompt 长度
        current_constraints = self._get_dynamic_constraints(text, top_k_ent=50, top_k_rel=20)

        # --- Phase 1 ---
        logger.info(">>> Phase 1: 动态约束提取...")
        try:
            initial = self._phase1_extraction(text, current_constraints)
            if not initial.triplets:
                logger.info("Phase 1 未提取到内容。")
                return []
        except Exception as e:
            logger.error(f"Phase 1 失败: {e}")
            return []

        # --- Phase 2 ---
        logger.info(f">>> Phase 2: 自我修正 ({len(initial.triplets)} 个候选)...")
        try:
            final = self._phase2_correction(text, initial.triplets, current_constraints)
            logger.info(f"Phase 2 完成，得到 {len(final.triplets)} 个最终三元组。")
            return final.triplets
        except Exception as e:
            logger.error(f"Phase 2 失败: {e}，返回 Phase 1 结果。")
            return initial.triplets

    def _phase1_extraction(self, text: str, constraints: Dict) -> ExtractionResult:
        # 使用动态约束构建 Prompt
        sys_prompt = prompts.PHASE1_SYSTEM.format(**constraints)
        user_prompt = prompts.PHASE1_USER.format(text=text)

        # 调试日志：查看现在的 Prompt 长度
        logger.info(f"Phase 1 动态 Prompt 长度: 约 {len(sys_prompt) + len(user_prompt)} 字符")

        return self.llm.create_structured_completion(
            ExtractionResult,
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        )

    def _phase2_correction(self, text: str, triplets: List[Triplet], constraints: Dict) -> ExtractionResult:
        triplets_json = json.dumps([t.model_dump() for t in triplets], indent=2)
        sys_prompt = prompts.PHASE2_SYSTEM.format(**constraints)
        user_prompt = prompts.PHASE2_USER.format(text=text, initial_json=triplets_json)

        return self.llm.create_structured_completion(
            ExtractionResult,
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        )


global_extractor = DualPipelineExtractor()
