from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings that can be configured via environment variables
    or .env file with CODEVIZ prefix.
    """
    openai_api_key: str | None = None
    log_level: str = "INFO"
    # Cache settings
    cache_dir: Path = Path.home() / ".cache" / "codeviz"

    # Configure the settings with env vars prefixed with CODEVIZ_
    model_config = SettingsConfigDict(
        envprefix="CODEVIZ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)


settings = Settings()