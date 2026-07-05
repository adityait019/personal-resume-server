import json

from parser.models import ParsedResume
from parser.prompts import RESUME_SYSTEM_PROMPT


class ResumeParser:

    def __init__(self, llm):
        self.llm = llm

    def parse(self, resume_text: str) -> ParsedResume:

        response = self.llm.generate(
            system_prompt=RESUME_SYSTEM_PROMPT,
            user_prompt=resume_text,
            response_model=ParsedResume
        )
        print("LLM Response:", response)  # Debugging line to print the raw response
        return response