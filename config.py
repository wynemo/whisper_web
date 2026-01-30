import secrets

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # CUDA 配置
    USE_CUDA: bool = False

    # JWT 认证配置
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 小时

    # Whisper 引擎选择: "openai" (原版) 或 "faster" (faster-whisper, 仅 Linux)
    WHISPER_ENGINE: Literal["openai", "faster"] = "openai"

    # Whisper 模型配置
    WHISPER_MODEL: str = "turbo"  # tiny, base, small, medium, large-v2, large-v3, turbo

    # faster-whisper 计算类型 (仅在 WHISPER_ENGINE="faster" 时生效)
    # float16: 最快, int8: 省显存, int8_float16: 平衡
    FASTER_WHISPER_COMPUTE_TYPE: str = "float16"

    # 火山引擎配置
    DOUBAO_APPID: str = ""
    DOUBAO_ACCESS_TOKEN: str = ""
    DOUBAO_RESOURCE_ID: str = "volc.megatts.default"
    DOUBAO_DEFAULT_VOICE_TYPE: str = "zh_female_vv_uranus_bigtts"

    # LLM 配置 (OpenAI 兼容 API)
    LLM_API_BASE_URL: str = ""
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""

    # 数据库配置
    DATABASE_URL: str = "sqlite:///data.db"


settings = Settings()
