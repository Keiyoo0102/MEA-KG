# pipeline/main_extraction.py
# 主程序：全量语料库并行提取执行器
# 功能：扫描所有语料 -> 检查已处理列表 -> 多线程调用提取器 -> 实时保存结果

import os
# Ensure HF mirror is set if needed, though usually handled by env vars
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

import json
import logging
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional

# 导入我们的工程模块 (作为包导入)
# 确保在 pipeline/ 目录下有 __init__.py (虽然 Python 3.3+ 不需要，但推荐加上)
try:
    from mea_kg_builder import config
    from mea_kg_builder.extractor import global_extractor
except ImportError:
    # Fallback for running directly without package context
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mea_kg_builder import config
    from mea_kg_builder.extractor import global_extractor

# --- 配置日志 ---
# 设置为 INFO 级别，既能看到进度，又不会被太多细节淹没
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extraction.log", mode='a', encoding='utf-8'),  # 保存日志到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)


# --- 辅助函数 ---

def get_processed_files() -> set:
    """
    读取现有的结果文件，获取所有已成功处理的文件名集合。
    用于实现断点续传。
    """
    processed = set()
    output_path = config.EXTRACTION_RESULTS_PATH
    
    if os.path.exists(output_path):
        logger.info(f"发现现有结果文件: {output_path}，正在扫描已处理记录...")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        if not line.strip(): continue
                        record = json.loads(line)
                        # 假设每个记录都有 'filename' 字段
                        if 'filename' in record:
                            processed.add(record['filename'])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"读取现有结果文件失败: {e}")
            
    logger.info(f"已找到 {len(processed)} 个已处理文件。")
    return processed


def process_single_file(file_info: Dict[str, Any]) -> Optional[str]:
    """
    处理单个文件的核心工作函数。
    会被线程池调用。
    """
    file_path = file_info['path']
    filename = file_info['filename']
    source_type = file_info['source_type']

    try:
        # 1. 读取文本
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 2. 调用核心引擎进行提取 (包含 Phase 1 & Phase 2)
        # global_extractor 内部已经封装了重试机制
        triplets = global_extractor.extract(text)

        # 3. 格式化结果记录
        if triplets:
            result_record = {
                "filename": filename,
                "source_type": source_type,
                # 将 Pydantic 对象列表转换为普通字典列表
                "triplets": [t.model_dump() for t in triplets]
            }
            # 返回 JSON 字符串以便写入
            return json.dumps(result_record, ensure_ascii=False)
        else:
            # 即使没有提取到三元组，也记录一条空结果，避免重复处理
            # (可选) 也可以选择不记录，但这会导致下次运行重新处理该文件
            return json.dumps({"filename": filename, "source_type": source_type, "triplets": []}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"处理文件失败 [{filename}]: {e}")
        return None


# --- 主执行流程 ---

def main():
    logger.info("=== 开始全量语料库知识提取任务 ===")

    # 1. 扫描所有待处理文件
    corpus_root = Path(config.CORPUS_DIR)
    all_files = []
    
    # 遍历 academic, news, web 三个子目录
    # 这些目录名必须与 corpus/preprocess.py 中生成的一致
    for sub_dir in ['academic', 'news', 'web']:
        dir_path = corpus_root / sub_dir
        if dir_path.exists():
            # 递归查找所有 .txt 文件
            files_in_dir = list(dir_path.glob('*.txt'))
            for file_path in files_in_dir:
                all_files.append({
                    'path': str(file_path),
                    'filename': file_path.name,
                    'source_type': sub_dir
                })
            logger.info(f"目录 '{sub_dir}' 发现 {len(files_in_dir)} 个文件。")
        else:
            logger.warning(f"警告: 语料库子目录不存在: {dir_path}")

    logger.info(f"扫描到语料库文件总数: {len(all_files)}")

    if not all_files:
        logger.error("错误: 未找到任何语料文件。请先运行 corpus/preprocess.py。")
        return

    # 2. 过滤掉已处理的文件 (断点续传)
    processed_files = get_processed_files()
    tasks_to_do = [f for f in all_files if f['filename'] not in processed_files]
    
    logger.info(f"剩余待处理文件数: {len(tasks_to_do)}")

    if not tasks_to_do:
        logger.info("所有文件均已处理完毕！🎉")
        return

    # 3. 并行执行提取任务
    # 使用 ThreadPoolExecutor 因为主要瓶颈是 I/O (网络/API调用)
    # 如果是本地模型，MAX_WORKERS 建议设为 1，否则显存容易爆
    max_workers = config.MAX_WORKERS
    logger.info(f"启动线程池，并发数: {max_workers}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(config.EXTRACTION_RESULTS_PATH), exist_ok=True)

    # 打开输出文件 (追加模式 'a')
    with open(config.EXTRACTION_RESULTS_PATH, 'a', encoding='utf-8') as f_out:

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            # 使用 future_to_file 字典来追踪每个任务对应的文件
            future_to_file = {
                executor.submit(process_single_file, file_info): file_info['filename'] 
                for file_info in tasks_to_do
            }

            # 使用 tqdm 显示进度条
            progress_bar = tqdm(total=len(tasks_to_do), desc="Processing Corpus", dynamic_ncols=True)

            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result_json_str = future.result()
                    if result_json_str:
                        # 实时写入结果到文件
                        f_out.write(result_json_str + '\n')
                        f_out.flush()  # 确保立即写入硬盘，防止数据丢失

                except Exception as e:
                    logger.error(f"任务执行异常 [{filename}]: {e}")

                finally:
                    progress_bar.update(1)

            progress_bar.close()

    logger.info("=== 全量提取任务完成 ===")
    logger.info(f"结果已保存至: {config.EXTRACTION_RESULTS_PATH}")


if __name__ == '__main__':
    # 再次提醒：如果在本地跑，务必确认显存足够
    print(f"当前配置并发数: {config.MAX_WORKERS}")
    print("如果使用本地 Ollama 且显存较小(<24G)，请确保 mea_kg_builder/config.py 中 MAX_WORKERS = 1")
    
    # 开始运行
    main()
