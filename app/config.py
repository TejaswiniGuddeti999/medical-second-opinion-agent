from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # LLM
    openai_api_key:str
    openai_model: str = "gpt-4o"

    # External APIs
    ncbi_api_key: str
    ncbi_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    openfda_base_url : str = "https://api.fda.gov/drug"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/medical_agent"
    database_url_sync: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/medical_agent"
 
    # App
    app_name: str = "Medical Second Opinion Agent"
    app_version: str = "0.1.0"
    debug:bool = False
    log_level: str = "INFO"

    #LangSmith observability
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "medical-second-opinion"

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False,
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()

    