# mea_kg_builder/ontology_loader.py
# 本体加载器 (OWL版)：负责读取 .owl 文件并提供全局约束

import logging
import os
from typing import List, Set, Dict, Any
from dataclasses import dataclass
import owlready2
from . import config

logger = logging.getLogger(__name__)


# --- 1. 数据结构定义 ---
# 我们使用 dataclass 来定义内存中的本体结构，方便下游模块使用，
# 而无需直接操作复杂的 owlready2 对象。

@dataclass
class Concept:
    """表示本体中的一个类 (Class)"""
    name: str
    description: str
    parent_names: List[str]  # 存储父类名称列表


@dataclass
class Relationship:
    """表示本体中的一个对象属性 (Object Property)"""
    name: str
    description: str
    domain: List[str]  # 定义域：关系的主语必须是这些类型之一
    range: List[str]  # 值域：关系的宾语必须是这些类型之一


# --- 2. 本体管理器类 ---

class Ontology:
    _instance = None

    def __new__(cls):
        """单例模式，确保本体只被加载一次"""
        if cls._instance is None:
            cls._instance = super(Ontology, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("正在初始化本体管理器 (OWL版)...")
        # 核心存储字典
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[str, Relationship] = {}
        # 快速查找集合 (用于验证)
        self.valid_entity_types: Set[str] = set()
        self.valid_relation_types: Set[str] = set()

        # 执行加载
        self._load_owl_file()

        self._initialized = True
        logger.info("本体管理器初始化完成。")
        logger.info(f"共加载: {len(self.concepts)} 个概念类, {len(self.relations)} 种关系。")

    def _load_owl_file(self):
        """使用 owlready2 加载 OWL 文件并解析内容"""
        path = config.ONTOLOGY_OWL_PATH
        if not os.path.exists(path):
            logger.error(f"严重错误: OWL 文件未找到: {path}")
            raise FileNotFoundError(f"OWL file not found at {path}")

        try:
            # 加载本体
            # 使用 file:// 前缀是 owlready2 的要求
            onto = owlready2.get_ontology(f"file://{path}").load()

            # --- 1. 解析类 (Concepts) ---
            for cls in onto.classes():
                # 跳过 OWL 的顶层基类 Thing
                if cls == owlready2.Thing:
                    continue

                # 获取类名 (例如 "GeologicFeature")
                name = cls.name

                # 获取描述 (通常存储在 rdfs:comment 中)
                # comment 返回的是一个列表，我们取第一个元素
                description = cls.comment[0] if cls.comment else "No description available."

                # 获取父类名称列表
                # is_a 属性包含了所有父类约束，我们只过滤出具名的类
                parents = [p.name for p in cls.is_a if isinstance(p, owlready2.ThingClass)]

                # 存入字典
                self.concepts[name] = Concept(
                    name=name,
                    description=description,
                    parent_names=parents
                )
                self.valid_entity_types.add(name)

            # --- 2. 解析对象属性 (Relationships) ---
            for prop in onto.object_properties():
                name = prop.name
                description = prop.comment[0] if prop.comment else "No description available."

                # 解析定义域 (Domain)
                # domain 可能包含多个类，也可能是复杂的逻辑组合
                domain_list = self._parse_domain_range(prop.domain)

                # 解析值域 (Range)
                range_list = self._parse_domain_range(prop.range)

                # 存入字典
                self.relations[name] = Relationship(
                    name=name,
                    description=description,
                    domain=domain_list,
                    range=range_list
                )
                self.valid_relation_types.add(name)

        except Exception as e:
            logger.error(f"解析 OWL 文件时发生错误: {e}")
            raise

    def _parse_domain_range(self, dr_objects) -> List[str]:
        """辅助函数：解析 domain/range 对象列表为字符串列表"""
        result = []
        if not dr_objects:
            return result
        for obj in dr_objects:
            # 如果是具名类，直接取名字
            if hasattr(obj, 'name'):
                result.append(obj.name)
            else:
                # 对于复杂的逻辑组合 (如 Union)，简化为字符串表示
                # 在实际应用中，这里可能需要更复杂的递归解析
                result.append(str(obj))
        return result

    # --- 3. 公共 API (供其他模块调用) ---

    def get_entity_type_list(self) -> List[str]:
        """获取所有合法的实体类型名称列表 (已排序)"""
        return sorted(list(self.valid_entity_types))

    def get_relation_type_list(self) -> List[str]:
        """获取所有合法的关系类型名称列表 (已排序)"""
        return sorted(list(self.valid_relation_types))

    def is_valid_entity(self, name: str) -> bool:
        """检查一个实体类型名是否在本体中定义"""
        return name in self.valid_entity_types

    def get_relation_info(self, name: str) -> Dict[str, Any]:
        """获取某个关系的详细信息 (描述、定义域、值域)"""
        rel = self.relations.get(name)
        if rel:
            return {
                "description": rel.description,
                "domain": rel.domain,
                "range": rel.range
            }
        return {}


# 创建全局单例实例
global_ontology = Ontology()
