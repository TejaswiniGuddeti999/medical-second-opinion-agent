from openai import AsyncOpenAI
from app.config import get_settings

settings = get_settings()

client = AsyncOpenAI(
    api_key = settings.groq_api_key,
    base_url = settings.groq_base_url,
    max_retries = 5,  # Groq free tier = 8K tokens/min; on 429 the SDK waits retry-after and tries again
)


MODEL = settings.groq_model
