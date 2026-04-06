from __future__ import annotations

import os
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

MIME_TEXT = "text/plain"


# -------------------------
# Skills
# -------------------------

def build_model_skill() -> AgentSkill:
    return AgentSkill(
        id="aditya_personal_assistant",
        name="model",
        description=(
            "I am Aditya's personal assistant. I can answer questions about "
            "Aditya's professional and personal life. Always provide accurate responses."
        ),
        tags=["llm"],
        input_modes=[MIME_TEXT],
        output_modes=[MIME_TEXT],
        examples=[
            "Tell me about Aditya",
            "What does Aditya do professionally?",
            "Who is Aditya?"
        ]
    )


def build_professional_tool_skill() -> AgentSkill:
    return AgentSkill(
        id="aditya_personal_assistant-about_professional_life",
        name="about_professional_life",
        description="Fetch information about Aditya's professional life.",
        tags=["llm", "tools"],
        input_modes=[MIME_TEXT],
        output_modes=[MIME_TEXT],
        examples=[
            "What is Aditya's work experience?",
            "Tell me about Aditya's professional background"
        ]
    )


def build_personal_tool_skill() -> AgentSkill:
    return AgentSkill(
        id="aditya_personal_assistant-about_personal_life",
        name="about_personal_life",
        description="Fetch information about Aditya's personal life.",
        tags=["llm", "tools"],
        input_modes=[MIME_TEXT],
        output_modes=[MIME_TEXT],
        examples=[
            "What are Aditya's hobbies?",
            "Tell me about Aditya's personal life"
        ]
    )


# -------------------------
# Capabilities
# -------------------------

def build_capabilities() -> AgentCapabilities:
    return AgentCapabilities(
        streaming=True,
        push_notifications=None,
        state_transition_history=None,
        extensions=None,
    )


# -------------------------
# Agent Card
# -------------------------

def build_agent_card(base_url: str | None = None) -> AgentCard:
    rpc_root = (
        base_url
        or os.environ.get("PUBLIC_BASE_URL")
        or "http://192.168.1.5:8005"
    ).rstrip("/") + "/"

    return AgentCard(
        name="aditya_personal_assistant",
        description="An assistant designed to help with various tasks.",
        url=rpc_root,
        version="0.0.1",
        preferred_transport="JSONRPC",
        protocol_version="0.3.0",
        default_input_modes=[MIME_TEXT],
        default_output_modes=[MIME_TEXT],
        capabilities=build_capabilities(),
        skills=[
            build_model_skill(),
            build_professional_tool_skill(),
            build_personal_tool_skill(),
        ],
        supports_authenticated_extended_card=False,
    )