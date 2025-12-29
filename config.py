from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 火山引擎配置
    DOUBAO_APPID: str = ""
    DOUBAO_ACCESS_TOKEN: str = ""
    DOUBAO_RESOURCE_ID: str = "volc.megatts.default"
    DOUBAO_DEFAULT_VOICE_TYPE: str = "zh_female_vv_uranus_bigtts"


settings = Settings()
