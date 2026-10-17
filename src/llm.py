"""The chat model. Swap providers here without touching the rest of the code."""
from langchain_anthropic import ChatAnthropic

from . import config


def get_llm(model: str = config.LLM_MODEL) -> ChatAnthropic:
    # No temperature argument: newer Claude models reject it.
    return ChatAnthropic(model=model, max_tokens=config.LLM_MAX_TOKENS)
