"""
Tree Base - LLM Client
Провайдер-агностичный клиент для работы с LLM.

Поддерживаемые провайдеры:
- openai (OpenAI-совместимые API: DeepSeek, GPT, Groq, Together)
- gemini (Google Generative AI)
- anthropic (Claude)

Расширение: добавить новый провайдер = добавить метод _call_<provider>.
"""

import logging
import aiohttp
import json
from typing import Optional

from ..config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Унифицированный LLM-клиент.

    Не используется для генерации на лету при навигации (ТЗ запрещает).
    Используется только при:
    - Индексации: генерация summary для узлов (батчами)
    - Семантическом парсинге: извлечение концепций из документов
    """

    def __init__(
        self,
        provider: str = None,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
    ):
        settings = get_settings()
        self.provider = provider or settings.llm_provider
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.llm_base_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate(self, prompt: str, system: str = "") -> str:
        """
        Генерация текста через LLM.

        Args:
            prompt: Пользовательский промпт
            system: Системный промпт (опционально)

        Returns:
            Текст ответа от модели
        """
        if not self.api_key:
            logger.warning("LLM API key not configured, returning empty")
            return ""

        dispatch = {
            "gemini": self._call_gemini,
            "openai": self._call_openai,
            "anthropic": self._call_anthropic,
        }

        handler = dispatch.get(self.provider)
        if not handler:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        return await handler(prompt, system)

    async def generate_with_metrics(self, prompt: str, system: str = "") -> dict:
        """
        Генерация с возвратом метрик (токены, время).
        Returns: {"text": str, "input_tokens": int, "output_tokens": int, "total_tokens": int}
        """
        import time
        start = time.time()

        if not self.api_key:
            return {"text": "", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "time_s": 0}

        if self.provider == "gemini":
            result = await self._call_gemini_with_metrics(prompt, system)
        elif self.provider == "openai":
            result = await self._call_openai_with_metrics(prompt, system)
        else:
            text = await self.generate(prompt, system)
            result = {
                "text": text,
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(text) // 4,
                "total_tokens": (len(prompt) + len(text)) // 4,
            }

        result["time_s"] = round(time.time() - start, 2)
        return result

    async def _call_gemini_with_metrics(self, prompt: str, system: str = "") -> dict:
        """Gemini API с возвратом usage metadata"""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": system}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 8192},
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

        session = await self._get_session()
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"Gemini API error {resp.status}: {error}")
                return {"text": "", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            data = await resp.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                # Log why extraction failed
                finish_reason = "unknown"
                try:
                    finish_reason = data["candidates"][0].get("finishReason", "unknown")
                except (KeyError, IndexError):
                    pass
                block_reason = ""
                try:
                    block_reason = data.get("promptFeedback", {}).get("blockReason", "")
                except:
                    pass
                logger.warning(f"Gemini empty: finishReason={finish_reason}, blockReason={block_reason}")
                text = ""

            usage = data.get("usageMetadata", {})
            return {
                "text": text,
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            }

    async def _call_gemini(self, prompt: str, system: str = "") -> str:
        """Google Gemini API (REST, без SDK)"""
        result = await self._call_gemini_with_metrics(prompt, system)
        return result["text"]

    async def _call_openai_with_metrics(self, prompt: str, system: str = "") -> dict:
        """OpenAI-совместимый API с метриками (DeepSeek, GPT, Groq, etc.)"""
        url = self.base_url or "https://api.openai.com/v1/chat/completions"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "max_tokens": 8192}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        session = await self._get_session()
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"OpenAI API error {resp.status}: {error}")
                return {"text": "", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {
                "text": text,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

    async def _call_openai(self, prompt: str, system: str = "") -> str:
        """OpenAI-совместимый API (DeepSeek, GPT, Groq, Together, etc.)"""
        result = await self._call_openai_with_metrics(prompt, system)
        return result["text"]

    async def _call_anthropic(self, prompt: str, system: str = "") -> str:
        """Anthropic Claude API"""
        url = "https://api.anthropic.com/v1/messages"

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        session = await self._get_session()
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error = await resp.text()
                logger.error(f"Anthropic API error {resp.status}: {error}")
                return ""
            data = await resp.json()
            return data["content"][0]["text"]


    async def batch_summarize(
        self,
        nodes: list,
        batch_size: int = 30,
    ) -> dict:
        """
        Батч-генерация summary для множества узлов.

        Вместо N API-вызовов (1 на узел) делаем ceil(N/batch_size) вызовов.
        Один промпт содержит до batch_size узлов, LLM возвращает
        JSON-массив summary для всех сразу.

        100K узлов: вместо 100K вызовов → ~3.3K вызовов (30x экономия).

        Returns:
            {node_id: summary_text, ...}
        """
        results = {}

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]

            # Формируем блок для LLM
            entries = []
            for idx, node in enumerate(batch):
                code = (node.source_code or "")[:300]
                entries.append(
                    f"[{idx}] type={node.type} name={node.name}\n"
                    f"code: {code}"
                )

            prompt = (
                "For each code element below, write a ONE-sentence summary (max 80 chars).\n"
                "Return ONLY a JSON array of strings, one per element, in order.\n"
                "Example: [\"Manages physics simulation loop\", \"Checks collision between two bodies\"]\n\n"
                + "\n---\n".join(entries)
            )

            raw = await self.generate(prompt, system="Return valid JSON array only. No markdown.")
            if not raw:
                continue

            # Парсим JSON-массив из ответа
            try:
                # Убираем markdown code block если LLM обернул
                clean = raw.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1]
                    clean = clean.rsplit("```", 1)[0]
                summaries = json.loads(clean)
                if isinstance(summaries, list):
                    for idx, node in enumerate(batch):
                        if idx < len(summaries) and summaries[idx]:
                            results[node.node_id] = str(summaries[idx])[:200]
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Batch summary parse error: {e}, falling back")
                # Fallback: пытаемся построчно
                lines = [l.strip().strip('"').strip(',') for l in raw.strip().split('\n') if l.strip()]
                for idx, node in enumerate(batch):
                    if idx < len(lines):
                        results[node.node_id] = lines[idx][:200]

        return results


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
