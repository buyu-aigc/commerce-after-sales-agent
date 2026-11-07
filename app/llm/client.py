import os

from langchain_openai import ChatOpenAI

from app.llm.offline import OfflineChatModel


class LlmClient:
    """OpenAI-compatible chat client wrapper.

    Created once in di.py and injected into LLM components.
    """

    def __init__(self, api_key: str = "", model: str = "gpt-4o-mini", base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        if not api_key or api_key in {"offline", "test-key", "sk-your-key-here"}:
            self.mode = "offline_workflow"
            self.chat_openai = OfflineChatModel()
        else:
            self.mode = "live_llm_agent"
            self.chat_openai = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.3,
                # The workflow has deterministic fallbacks. A long gateway wait
                # hurts interactive support more than it helps a single turn.
                timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "12")),
                max_retries=0,
                max_tokens=900,
                extra_body={"thinking": {"type": "disabled"}}
                if os.environ.get("LLM_DISABLE_THINKING") == "1" else None,
            )
