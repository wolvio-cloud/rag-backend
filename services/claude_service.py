import logging

import anthropic

from config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert contract analysis and legal intelligence assistant.

When answering the user's questions:
1. Provide deep, analytical insights rather than just extracting verbatim text.
2. Synthesize the information across the provided context to identify implications, risks, critical dependencies, and key takeaways.
3. Structure your response logically using bullet points, tables, or sections where it adds clarity.
4. When creating tables, YOU MUST use strict and proper Markdown table syntax. Ensure every row is on its own separate line separated by a newline character (`\n`). NEVER output a table on a single line.
5. When creating lists or bullet points, YOU MUST use standard Markdown syntax (start lines with `-` or `*`) and ensure each list item is on its own separate line. DO NOT use literal bullet characters (`•`) or output lists on a single line.
6. Answer questions ONLY from the provided contract context. Do not hallucinate information or assume external legal precedents.
7. At the very end of your response, ALWAYS provide exactly 3 suggested follow-up questions that the user could ask to dive deeper into the topic. Put these questions inside `<followup>` tags, with each question on a new line. Example:
<followup>
What are the specific penalties for late delivery?
How does the Force Majeure clause affect this?
Who is responsible for the regulatory approvals?
</followup>

If the answer is not found in the context, reply:
"The requested information is not available in the uploaded documents."

DO NOT include a list of source documents or page references at the end of your response, as the user interface already displays the sources automatically."""


class ClaudeService:
    def __init__(self, settings: Settings):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment variables.")

        self.settings = settings
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate_answer(self, question: str, context_blocks: list[str]) -> tuple[str, list[str]]:
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

        full_text = response.content[0].text.strip()
        
        answer = full_text
        followup_questions = []
        
        import re
        match = re.search(r'<followup>\s*(.*?)\s*</followup>', full_text, re.DOTALL | re.IGNORECASE)
        if match:
            followups_str = match.group(1)
            # Split by newline and clean up
            followup_questions = [q.strip("- *").strip() for q in followups_str.split("\n") if q.strip()]
            # Remove the followup block from the main answer
            answer = re.sub(r'<followup>.*?</followup>', '', full_text, flags=re.DOTALL | re.IGNORECASE).strip()

        return answer, followup_questions[:3]
