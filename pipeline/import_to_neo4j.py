# pipeline/import_to_neo4j.py
# 知识图谱入库脚本：将 JSONL 导入 Neo4j 并挂载本体

import os
import json
import logging
from neo4j import GraphDatabase
from tqdm import tqdm

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 配置部分 ---
# 优先从环境变量读取，否则使用默认值 (请在生产环境中设置环境变量)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j") # Replace with your password or env var

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')

# 输入文件路径 (extraction_results.jsonl)
INPUT_FILE = os.path.join(DATA_DIR, 'knowledge_graph_build', 'extraction_results.jsonl')

# 批量大小 (每 1000 条提交一次，提高性能)
BATCH_SIZE = 1000

class KnowledgeGraphImporter:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("✅ 成功连接到 Neo4j 数据库")
        except Exception as e:
            logger.error(f"❌ 无法连接到 Neo4j: {e}")
            raise

    def close(self):
        self.driver.close()

    def import_batch(self, tx, triplets):
        """
        执行 Cypher 语句进行批量导入和本体挂载。
        核心逻辑：
        1. 创建/合并 Instance 节点。
        2. 根据 type 属性，查找对应的 Ontology Class 节点。
        3. 创建 INSTANCE_OF 关系 (挂载)。
        4. 创建实体间的 EXTRACTED_RELATION 关系。
        """
        query = """
        UNWIND $batch AS row

        // --- 1. 处理头实体 (Head Entity) ---
        // 使用 MERGE 避免重复创建
        MERGE (h:Instance {name: row.head_entity})
        ON CREATE SET h.type = row.head_type, h.source = 'LLM_Extraction'
        ON MATCH SET h.type = row.head_type // 更新类型以防变更

        // [核心修正] 连接头实体与本体 (Schema Mounting)
        // 使用 n4sch__name (Neosemantics 标准) 来查找对应的本体类
        WITH h, row
        OPTIONAL MATCH (classH:n4sch__Class {n4sch__name: row.head_type})
        FOREACH (_ IN CASE WHEN classH IS NOT NULL THEN [1] ELSE [] END |
            MERGE (h)-[:INSTANCE_OF]->(classH)
        )

        // --- 2. 处理尾实体 (Tail Entity) ---
        MERGE (t:Instance {name: row.tail_entity})
        ON CREATE SET t.type = row.tail_type, t.source = 'LLM_Extraction'
        ON MATCH SET t.type = row.tail_type

        // [核心修正] 连接尾实体与本体
        WITH h, t, row
        OPTIONAL MATCH (classT:n4sch__Class {n4sch__name: row.tail_type})
        FOREACH (_ IN CASE WHEN classT IS NOT NULL THEN [1] ELSE [] END |
            MERGE (t)-[:INSTANCE_OF]->(classT)
        )

        // --- 3. 处理实体间关系 ---
        // 关系类型作为属性存储在通用关系 EXTRACTED_RELATION 中
        // 这样做是为了避免动态 Cypher 类型带来的复杂性和注入风险
        MERGE (h)-[r:EXTRACTED_RELATION {original_type: row.relation}]->(t)
        """
        tx.run(query, batch=triplets)

    def run_import(self, file_path):
        if not os.path.exists(file_path):
            logger.error(f"输入文件未找到: {file_path}")
            logger.error("请先运行 pipeline/main_extraction.py 生成数据。")
            return

        triplets_buffer = []
        total_triplets = 0

        logger.info(f"开始读取数据文件: {file_path}")
        logger.info("正在执行全量导入并自动挂载本体...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 计算总行数用于进度条 (可选，如果文件很大可能会慢)
                # total_lines = sum(1 for _ in f)
                # f.seek(0)
                
                for line in tqdm(f, desc="Processing & Linking"):
                    try:
                        if not line.strip(): continue
                        record = json.loads(line)
                        
                        if "triplets" in record and record["triplets"]:
                            for t in record["triplets"]:
                                triplet_data = {
                                    "head_entity": t["head_entity"],
                                    "head_type": t["head_type"],
                                    "relation": t["relation"],
                                    "tail_entity": t["tail_entity"],
                                    "tail_type": t["tail_type"]
                                }
                                triplets_buffer.append(triplet_data)

                                # 缓冲区满，执行批量插入
                                if len(triplets_buffer) >= BATCH_SIZE:
                                    with self.driver.session() as session:
                                        session.execute_write(self.import_batch, triplets_buffer)
                                    total_triplets += len(triplets_buffer)
                                    triplets_buffer = [] # 清空缓冲区
                                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"行处理错误: {e}")
                        continue

                # 处理剩余数据
                if triplets_buffer:
                    with self.driver.session() as session:
                        session.execute_write(self.import_batch, triplets_buffer)
                    total_triplets += len(triplets_buffer)

            logger.info(f"✅ 导入完成！共处理三元组: {total_triplets}")
            logger.info("图谱构建完毕！您现在可以使用 application/app.py 进行可视化了。")

        except Exception as e:
            logger.error(f"文件读取或导入失败: {e}")

if __name__ == "__main__":
    # 简单的安全检查
    if "123456" in NEO4J_PASSWORD or "password" in NEO4J_PASSWORD:
         logger.warning("⚠️ 您正在使用默认或弱密码，请确保在生产环境中修改 config。")

    importer = KnowledgeGraphImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        importer.run_import(INPUT_FILE)
    finally:
        importer.close()
