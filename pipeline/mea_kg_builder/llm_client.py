# mea_kg_builder/llm_client.py
# LLM 客户端封装 (集成 LoRA 模型验证、instructor 和 tenacity)

import logging
import sys
from openai import OpenAI
import instructor
import tenacity
from . import config  # 导入同级目录下的 config 模块

logger = logging.getLogger(__name__)

class LLMClient:
    _instance = None

    def __new__(cls):
        """实现单例模式，确保全局只创建一个客户端连接"""
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        """初始化连接并验证 LoRA 模型是否存在"""
        logger.info(f"正在初始化 Ollama 客户端 (指向 {config.OLLAMA_BASE_URL})...")
        
        try:
            # 1. 基础 OpenAI 客户端连接
            base_client = OpenAI(
                base_url=config.OLLAMA_BASE_URL,
                api_key=config.OLLAMA_API_KEY
            )
            
            # 2. [LoRA 关键步骤] 验证微调后的模型是否存在
            target_model = config.OLLAMA_MODEL_NAME
            try:
                available_models = [m.id for m in base_client.models.list().data]
                if target_model not in available_models:
                    logger.warning(f"⚠️ 警告: 目标模型 '{target_model}' 未在 Ollama 中找到！")
                    logger.warning(f"当前可用模型: {available_models}")
                    logger.error(f"请确保您已使用 Modelfile 创建了微调模型: 'ollama create {target_model} -f ../Modelfile'")
                    # 这里我们可以选择报错退出，或者让用户自行承担风险
                    # raise RuntimeError(f"Model '{target_model}' not found in Ollama.") 
                else:
                    logger.info(f"✅ 成功检测到微调模型: {target_model}")
            except Exception as e:
                logger.warning(f"模型列表获取失败 (非致命错误): {e}")

            # 3. 使用 instructor.patch 增强客户端 (用于结构化提取)
            self.client = instructor.patch(
                base_client,
                mode=instructor.Mode.JSON
            )
            logger.info("LLM 客户端初始化完成。")

        except Exception as e:
            logger.error(f"Ollama 连接严重失败: {e}")
            raise RuntimeError("无法连接到 Ollama，请确保服务已启动 (ollama serve)。")

    # 定义通用的重试策略 (借鉴 AutoSchemaKG)
    # 停止：最多重试 5 次 或 总耗时超过 60 秒
    # 等待：指数退避，从 1 秒开始，最长等待 10 秒
    retry_policy = tenacity.retry(
        stop=(tenacity.stop_after_attempt(5) | tenacity.stop_after_delay(60)),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING)
    )

    @retry_policy
    def create_structured_completion(self, response_model, messages, temperature=0.0):
        """
        通用的结构化生成方法，带有自动重试功能。
        会自动使用 config.py 中配置的 (LoRA微调后) 模型名称。

        Args:
            response_model: Pydantic 模型类，定义了期望的输出结构。
            messages: 发送给模型的对话历史列表。
            temperature: 采样温度，默认为 0.0 以获得确定性输出。

        Returns:
            符合 response_model 结构的实例化对象。
        """
        return self.client.chat.completions.create(
            model=config.OLLAMA_MODEL_NAME, # 这里会自动使用 "mea-kg-model"
            response_model=response_model,
            messages=messages,
            temperature=temperature
        )

# 创建一个全局可访问的客户端实例
global_client = LLMClient()
