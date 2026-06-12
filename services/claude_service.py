import logging

import anthropic

from config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a contract analysis assistant.

Answer questions ONLY from the provided contract context.

If the answer is not found in the context, reply:
"The requested information is not available in the uploaded documents."

Provide a clear and concise answer.
DO NOT include a list of source documents or page references at the end of your response, as the user interface already displays the sources automatically.

Do not hallucinate information."""


class ClaudeService:
    def __init__(self, settings: Settings):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment variables.")

        self.settings = settings
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate_answer(self, question: str, context_blocks: list[str]) -> str:
        context = "\n\n---\n\n".join(context_blocks)
        user_message = f"""Context:
{context}

Question:
{question}

Answer:"""

        response = self.client.messages.create(
            model=self.settings.claude_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text.strip()
