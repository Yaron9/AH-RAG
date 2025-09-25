#!/usr/bin/env python3
"""
统一LLM客户端管理器
集中管理所有模块的LLM配置和客户端初始化，避免重复代码
"""
from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, Optional, Union
from enum import Enum

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


class LLMModule(Enum):
    """LLM使用模块枚举"""
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    SEMANTIC_AGGREGATION = "semantic_aggregation"
    AGENT_DECISION = "agent_decision"
    ANSWER_GENERATION = "answer_generation"
    EVALUATION_JUDGE = "evaluation_judge"


class LLMClientManager:
    """
    统一LLM客户端管理器

    负责：
    1. 从配置读取每个模块的LLM设置
    2. 统一初始化OpenAI兼容客户端
    3. 提供模块级开关控制
    4. 避免重复的API key和base_url配置
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.llm_config = config.get("llm", {})
        self.global_enabled = self.llm_config.get("enabled", True)
        self.modules_config = self.llm_config.get("modules", {})

        # 缓存已初始化的客户端
        self._clients: Dict[str, Optional[OpenAI]] = {}

    def is_enabled(self, module: Union[LLMModule, str]) -> bool:
        """检查指定模块是否启用LLM"""
        if not self.global_enabled:
            return False

        module_name = module.value if isinstance(module, LLMModule) else module
        module_config = self.modules_config.get(module_name, {})
        return module_config.get("enabled", False)

    def get_client(self, module: Union[LLMModule, str]) -> Optional[OpenAI]:
        """获取指定模块的LLM客户端"""
        if not self.is_enabled(module):
            return None

        if OpenAI is None:
            return None

        module_name = module.value if isinstance(module, LLMModule) else module

        # 检查缓存
        if module_name in self._clients:
            return self._clients[module_name]

        # 初始化新客户端
        client = self._init_client_for_module(module_name)
        self._clients[module_name] = client
        return client

    def get_model_config(self, module: Union[LLMModule, str]) -> Dict[str, Any]:
        """获取指定模块的模型配置"""
        module_name = module.value if isinstance(module, LLMModule) else module
        module_config = self.modules_config.get(module_name, {})

        # 合并默认配置和模块特定配置
        config = {
            "model": module_config.get("model", self.llm_config.get("default_model", "deepseek-chat")),
            "temperature": module_config.get("temperature", self.llm_config.get("default_temperature", 0.1)),
            "max_retries": module_config.get("max_retries", self.llm_config.get("default_max_retries", 2)),
        }

        # 添加模块特定配置
        for key, value in module_config.items():
            if key not in ["enabled", "model", "temperature", "max_retries"]:
                config[key] = value

        return config

    def _init_client_for_module(self, module_name: str) -> Optional[OpenAI]:
        """为指定模块初始化OpenAI客户端"""
        module_config = self.modules_config.get(module_name, {})
        model = module_config.get("model", self.llm_config.get("default_model", "deepseek-chat"))

        # 根据模型名称选择API配置
        api_key = None
        base_url = None

        if model in ["kimi", "moonshot-v1-8k"]:
            api_key = os.getenv("KIMI_API_KEY")
            base_url = os.getenv("KIMI_BASE_URL") or "https://api.moonshot.cn/v1"
        elif model in ["deepseek-chat", "deepseek-coder"]:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        elif model.startswith("gpt-"):
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
        elif model.startswith("ollama/"):
            # Ollama本地模型，不需要API key
            return None
        else:
            # 默认尝试DeepSeek
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"

        if not api_key:
            return None

        try:
            return OpenAI(api_key=api_key, base_url=base_url)
        except Exception:
            return None

    def create_chat_completion(
        self,
        module: Union[LLMModule, str],
        messages: list,
        **kwargs
    ) -> Any:
        """统一的聊天补全接口"""
        client = self.get_client(module)
        if client is None:
            raise RuntimeError(f"LLM client not available for module {module}")

        config = self.get_model_config(module)

        # 合并配置和传入参数
        # Extract retry-related overrides before passing to API client
        max_retries_cfg = config.get("max_retries", self.llm_config.get("default_max_retries", 2))
        max_retries_override = kwargs.pop("_max_retries", None)
        max_attempts = max(0, int(max_retries_override if max_retries_override is not None else max_retries_cfg)) + 1

        rate_limit_wait = kwargs.pop(
            "rate_limit_wait",
            config.get("rate_limit_wait", self.llm_config.get("default_rate_limit_wait", 5)),
        )
        retry_jitter = kwargs.pop(
            "retry_jitter",
            config.get("retry_jitter", self.llm_config.get("default_retry_jitter", 0.0)),
        )
        non_rate_wait = kwargs.pop(
            "retry_wait",
            config.get("retry_wait", self.llm_config.get("default_retry_wait", 2.0)),
        )

        call_params = {
            "model": config["model"],
            "messages": messages,
            "temperature": config["temperature"],
            "max_tokens": kwargs.get("max_tokens", 400),
        }
        call_params.update(kwargs)

        def _is_rate_limit_error(err: Exception) -> bool:
            text = str(err).lower()
            return "rate limit" in text or "max rpm" in text or "too many requests" in text

        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return client.chat.completions.create(**call_params)
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt >= max_attempts:
                    break
                wait = rate_limit_wait if _is_rate_limit_error(exc) else non_rate_wait
                wait = max(0.0, float(wait))
                # simple progressive backoff
                wait *= attempt
                if retry_jitter:
                    wait += random.uniform(0, float(retry_jitter))
                if wait > 0:
                    time.sleep(wait)
        # Exhausted retries
        if last_error:
            raise last_error
        raise RuntimeError("LLM call failed without exception context")


# 全局单例管理器
_global_manager: Optional[LLMClientManager] = None


def get_llm_manager(config: Optional[Dict[str, Any]] = None) -> LLMClientManager:
    """获取全局LLM管理器实例"""
    global _global_manager

    if _global_manager is None or config is not None:
        if config is None:
            from ah_rag.utils.config import load_config
            config = load_config()
        _global_manager = LLMClientManager(config)

    return _global_manager


def is_llm_enabled(module: Union[LLMModule, str]) -> bool:
    """检查指定模块是否启用LLM（便捷函数）"""
    return get_llm_manager().is_enabled(module)


def get_llm_client(module: Union[LLMModule, str]) -> Optional[OpenAI]:
    """获取指定模块的LLM客户端（便捷函数）"""
    return get_llm_manager().get_client(module)


def create_chat_completion(
    module: Union[LLMModule, str],
    messages: list,
    **kwargs
) -> Any:
    """创建聊天补全（便捷函数）"""
    return get_llm_manager().create_chat_completion(module, messages, **kwargs)
