"""LLM client wrapper — provider-agnostic via OpenAI-compatible API.

Works identically with:
    - Ollama on laptop (default): http://localhost:11434/v1
    - vLLM on UVA Rivanna:        http://localhost:8247/v1
    - Any OpenAI-compatible API
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("agent.llm_client")


class LLMClient:
    """Wrapper around an OpenAI-compatible LLM endpoint.

    Uses langchain-openai's ChatOpenAI, which works with any server
    that implements the /v1/chat/completions endpoint (vLLM, Ollama, etc.).

    No API key is needed for local models — a dummy value is used.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """Initialize the LLM client.

        Args:
            base_url: Server URL. Defaults to env LLM_BASE_URL or http://localhost:11434/v1 (Ollama)
            model: Model name. Defaults to env LLM_MODEL or qwen2.5:7b
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum response length
        """
        self.base_url = base_url or os.environ.get(
            "LLM_BASE_URL", "http://localhost:11434/v1"
        )
        self.model = model or os.environ.get("LLM_MODEL", "qwen3:8b")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    def _get_llm(self):
        """Lazy-initialize the LangChain ChatOpenAI client."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai is required. Install with: pip install langchain-openai"
                )

            self._llm = ChatOpenAI(
                base_url=self.base_url,
                api_key="not-needed",  # Local models don't check this
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._llm

    def invoke(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the text response.

        Args:
            prompt: The full prompt string

        Returns:
            The LLM's response text

        Raises:
            ConnectionError: If the LLM server is unreachable
        """
        llm = self._get_llm()
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "refused" in error_msg or "timeout" in error_msg.lower():
                raise ConnectionError(
                    f"Could not connect to LLM server at {self.base_url}. "
                    f"Make sure the server is running.\n"
                    f"  Ollama: ollama serve & ollama pull {self.model}\n"
                    f"  vLLM:   python -m vllm.entrypoints.openai.api_server --model {self.model}\n"
                    f"Original error: {e}"
                ) from e
            raise

    def is_available(self) -> bool:
        """Check if the LLM server is reachable AND the configured model is installed.

        Returns:
            True if the server responds AND the model can be invoked, False otherwise.
            Use diagnose() for a more detailed error message.
        """
        try:
            llm = self._get_llm()
            llm.invoke("Hi")
            return True
        except Exception:
            return False

    def diagnose(self) -> Optional[str]:
        """Return None if everything is OK, or a human-readable error string.

        Distinguishes between three failure modes:
            - server unreachable          (Ollama not running)
            - model not found             (need to `ollama pull`)
            - other (e.g. langchain-openai missing)
        """
        try:
            llm = self._get_llm()
            llm.invoke("Hi")
            return None
        except ImportError as e:
            return f"Missing dependency: {e}"
        except Exception as e:
            msg = str(e)
            if "Connection" in msg or "refused" in msg or "ConnectError" in msg:
                return (
                    f"LLM server not reachable at {self.base_url}.\n"
                    f"  Start Ollama: brew services start ollama"
                )
            if "model" in msg.lower() and ("not found" in msg.lower() or "404" in msg):
                return (
                    f"Model {self.model!r} not installed on the server.\n"
                    f"  Pull it: ollama pull {self.model}\n"
                    f"  Or pass --model <name> with a model that is installed."
                )
            return f"LLM error: {msg}"
