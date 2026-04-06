# agents/agent_executor.py
from __future__ import annotations

import inspect
import logging
import json
import re
from collections.abc import AsyncGenerator
from typing import Awaitable, Callable, Optional, Union, List, cast
import uuid

# A2A SDK (0.3.22+) imports
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart,
    FileWithBytes,
    FileWithUri,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
    Message,
    Role,
)
from a2a.utils.errors import ServerError

# Google ADK + GenAI
from google.adk.runners import Runner
from google.adk.events import Event
from google.genai import types as genai_types

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# A factory that returns a Runner (sync or async)
RunnerFactory = Callable[[], Union[Runner, Awaitable[Runner]]]


class PersonalAgentExecutor(AgentExecutor):
    """
    Runs an ADK Runner for A2A requests and enforces URI-only FilePart policy.

    - Rejects inline FileWithBytes on inbound messages.
    - Converts only file URIs into GenAI-readable signals for the LLM (as text markers).
    - Emits only URI-based artifacts on responses (rejects inline bytes).
    """

    def __init__(self, runner_or_factory: Union[Runner, RunnerFactory]) -> None:
        self._runner_or_factory = runner_or_factory
        self._runner: Optional[Runner] = None

    async def _resolve_runner(self) -> Runner:
        # Direct Runner instance
        if isinstance(self._runner_or_factory, Runner):
            return self._runner_or_factory

        # Cached result
        if self._runner is not None:
            return self._runner

        # Call the factory
        candidate = self._runner_or_factory
        if callable(candidate):
            out = candidate()
            if inspect.isawaitable(out):
                resolved = await cast(Awaitable[Runner], out)
            else:
                resolved = cast(Runner, out)
            self._runner = resolved
            return resolved

        raise TypeError(
            "runner_or_factory must be Runner or Callable[[], Runner | Awaitable[Runner]], "
            f"got: {type(candidate)}"
        )

    def _run_agent(
        self,
        runner: Runner,
        session_id: str,
        new_message: genai_types.Content,
    ) -> AsyncGenerator[Event, None]:
        """Delegate to ADK Runner async generator."""
        return runner.run_async(
            session_id=session_id,
            user_id=runner.app_name or "adk_agent",
            new_message=new_message,
        )

    async def _upsert_session(self, runner: Runner, session_id: str):
        session = await runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=runner.app_name or "adk_agent",
            session_id=session_id,
        )
        if session is None:
            session = await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id=runner.app_name or "adk_agent",
                session_id=session_id,
            )
        if session is None:
            raise RuntimeError(f"Failed to get or create session: {session_id}")
        return session

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Process a task request from A2A.
        Emits streaming updates via TaskUpdater and calls `failed()` with a proper A2A Message
        (with message_id) on any error so the client can display it to the user.
        """
        # Validate critical fields early; if these fail we can't even create a task safely
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        try:
            if not context.message:
                raise ValueError("RequestContext must have a message")

            # Enforce URI-only on inbound parts (can raise ValueError)
            context.message.parts = sanitize_a2a_parts_uri_only(context.message.parts)

            runner = await self._resolve_runner()

            # Submit/start task (server-side task lifecycle)
            if not context.current_task:
                await updater.submit()
            await updater.start_work()

            # Build GenAI content from A2A message parts (can raise on conversion)
            user_content = genai_types.UserContent(
                parts=convert_a2a_parts_to_genai(context.message.parts),
            )

            # Upsert session (use context_id as session id)
            session_obj = await self._upsert_session(runner, context.context_id)
            session_id = session_obj.id

            # ---- Controls & Accumulators ----
            tool_output_emitted = False
            url_like_re = re.compile(r"https?://", re.IGNORECASE)

            # Track “best seen” usage; emit once, merged in the final message
            usage_totals = {"input": 0, "output": 0, "total": 0}

            def _format_usage_text(inp: int, out: int, tot: int) -> str:
                return f"Token usage — input: {inp:,} • output: {out:,} • total: {tot:,}"

            def _compose_final_text(status_lines: List[str]) -> str:
                """
                Build a single final message by merging:
                - status text(s) (no URLs)
                - token usage summary (if available)
                """
                base = "\n".join([line for line in status_lines if line.strip()]) if status_lines else ""
                if not base:
                    base = "Completed successfully. Files attached."
                # if any(usage_totals.values()):
                #     base = base + "\n" + _format_usage_text(
                #         usage_totals["input"], usage_totals["output"], usage_totals["total"]
                #     )
                return base

            # 🔁 Stream events from ADK -> back to A2A
            async for event in self._run_agent(runner, session_id, user_content):
                try:
                    genai_parts: List[genai_types.Part] = (
                        event.content.parts if (event.content and event.content.parts) else []
                    )

                    # Helpful debug: see what the agent actually produced
                    kind_list = []
                    for gp in genai_parts:
                        if gp.file_data:
                            kind_list.append("file_data")
                        elif gp.inline_data:
                            kind_list.append("inline_data")
                        elif gp.text:
                            kind_list.append("text")
                        elif gp.function_call:
                            kind_list.append("function_call")
                        elif gp.function_response:
                            kind_list.append("function_response")
                        else:
                            kind_list.append("other")
                    logger.debug("Server Event parts (genai kinds): %s", kind_list)

                    # Convert outbound parts to A2A (can raise)
                    a2a_parts: List[Part] = convert_genai_parts_to_a2a(genai_parts)

                    # Separate file parts and text parts
                    file_parts: List[Part] = [p for p in a2a_parts if isinstance(p.root, FilePart)]
                    text_parts: List[Part] = [p for p in a2a_parts if isinstance(p.root, TextPart)]

                    # ---- ACCUMULATE TOKEN USAGE (defensive) ----
                    um = getattr(event, "usage_metadata", None)
                    if um:
                        inp = getattr(um, "prompt_token_count", None)
                        out = getattr(um, "candidates_token_count", None)
                        tot = getattr(um, "total_token_count", None)
                        if tot is None:
                            tot = (inp or 0) + (out or 0)

                        if isinstance(inp, int):
                            usage_totals["input"] = max(usage_totals["input"], inp)
                        if isinstance(out, int):
                            usage_totals["output"] = max(usage_totals["output"], out)
                        if isinstance(tot, int):
                            usage_totals["total"] = max(usage_totals["total"], tot)

                    
                    is_tool_response = ("function_response" in kind_list) or (file_parts and not event.is_final_response())
                    if is_tool_response:
                        if file_parts:
                            logger.debug("Emitting artifacts from tool response immediately: %s", file_parts)
                            await updater.add_artifact(file_parts)
                            tool_output_emitted = True

                        # Build final merged message
                        status_lines: List[str] = []
                        if text_parts:
                            await updater.update_status(
                                TaskState.working,
                                message=updater.new_agent_message(parts=text_parts,metadata={"tool_response": True}),
                            )
                            continue  # Emit text immediately, do not merge into final message


                        # Build a machine-readable JSON meta line
                        usage_meta_json = {
                            "type": "token_usage",
                            "input": usage_totals["input"],
                            "output": usage_totals["output"],
                            "total": usage_totals["total"],
                        }
                        meta_line = f"[META:TOKEN_USAGE] {json.dumps(usage_meta_json, separators=(',',':'))}"


                        await updater.update_status(
                            TaskState.working,
                            message=updater.new_agent_message(
                                parts=[Part(root=TextPart(text=meta_line))],
                                metadata={"token_usage": usage_meta_json}
                            )

                        )
                        continue  # Emit tool response immediately, do not merge into final message
                    
                    if event.is_final_response():
                        if file_parts:
                            logger.debug("Final artifacts to emit: %s", file_parts)
                            await updater.add_artifact(file_parts)

                        # Build final merged message
                        status_lines: List[str] = []
                        if text_parts:
                            for tp in text_parts:
                                t = tp.root.text if hasattr(tp.root, "text") else ""
                                if tool_output_emitted and url_like_re.search(t or ""):
                                    continue
                                if t:
                                    status_lines.append(t)
                        final_text = _compose_final_text(status_lines)

                        await updater.update_status(
                            TaskState.working,
                            message=updater.new_agent_message(
                                [Part(root=TextPart(text=final_text))]
                            ),
                        )
                        await updater.complete()
                        break

                    # Interim updates (if any)
                    if file_parts:
                        logger.debug("Interim artifacts to emit: %s", file_parts)
                        await updater.add_artifact(file_parts)
                        tool_output_emitted = True

                    if text_parts and not event.get_function_calls():
                        # After tool outputs, suppress any URL-like text
                        filtered: List[Part] = []
                        for tp in text_parts:
                            t = tp.root.text if hasattr(tp.root, "text") else ""
                            if tool_output_emitted and url_like_re.search(t or ""):
                                continue
                            filtered.append(tp)
                        if filtered:
                            await updater.update_status(
                                TaskState.working,
                                message=updater.new_agent_message(filtered),
                            )

                except Exception as inner_exc:
                    # Any per-event conversion/emit failure -> mark task failed with a message
                    logger.exception(
                        "Error while handling streamed event for task %s", context.task_id
                    )
                    await updater.failed(
                        message=Message(
                            message_id=str(uuid.uuid4()),  # ✅ required for client schema
                            role=Role("agent"),
                            parts=[Part(root=TextPart(text=f"Server Error while streaming: {inner_exc}"))],
                        )
                    )
                    return  # Stop processing this task

        except Exception as exc:
            # Catch-all safety net -> user-friendly failure
            logger.exception("Unexpected server error in executor for task %s", context.task_id)
            await updater.failed(
                message=Message(
                    message_id=str(uuid.uuid4()),  # ✅ required
                    role=Role("agent"),
                    parts=[Part(root=TextPart(text=f"Server Error: {exc}"))],
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())


# ---------------------------
# Converters & URI-only policy
# ---------------------------

def sanitize_a2a_parts_uri_only(parts: List[Part]) -> List[Part]:
    """
    Ensure that all FileParts are URI-based (FileWithUri). Reject inline bytes (FileWithBytes).
    """
    cleaned: List[Part] = []
    for p in parts:
        root = p.root
        if isinstance(root, FilePart):
            if isinstance(root.file, FileWithUri):
                cleaned.append(p)
            elif isinstance(root.file, FileWithBytes):
                raise ValueError(
                    "Inline file bytes are not allowed in FilePart. Provide a URI instead."
                )
            else:
                raise ValueError(f"Unsupported FilePart inner type: {type(root.file)}")
        else:
            cleaned.append(p)
    return cleaned


def convert_a2a_parts_to_genai(parts: List[Part]) -> List[genai_types.Part]:
    """
    Inbound (A2A -> LLM input):
    - TextPart -> GenAI text
    - FilePart(FileWithUri) -> GenAI *text marker* (do not feed file bytes to LLM)
      We intentionally do NOT attach file_data to the model prompt.
    """
    out: List[genai_types.Part] = []
    latest_file_uri: str | None = None
    for part in parts:
        root = part.root

        if isinstance(root, TextPart):
            out.append(genai_types.Part(text=root.text))
            continue

        if isinstance(root, FilePart):
            f = root.file

            if isinstance(f, FileWithUri):
                latest_file_uri = f.uri
                # Keep only a single marker for the model (policy)
                continue

            if isinstance(f, FileWithBytes):
                raise ValueError("Inline file bytes are not allowed. Send a URI instead.")

        raise ValueError(f"Unsupported part type: {type(part)}")

    if latest_file_uri:
        out.append(genai_types.Part(text=f"FILE_URL: {latest_file_uri}"))

    return out


# ----- Helpers to parse tool JSON payloads -----

def _convert_file_message_obj(obj: dict) -> List[Part]:
    """
    Convert a dict payload like:
      {"file_data":[{"file_uri":"...","mime_type":"..."}], "message":"..."}
    (and nested {"result": {...}}) into A2A parts.
    Also supports older schema:
      {"tool_response":[{"fileUri":"...","mimeType":"..."}], "agent_response":[...]}
    """
    out: List[Part] = []

    # Direct file_data + message
    file_data = obj.get("file_data")
    if isinstance(file_data, list):
        for item in file_data:
            if not isinstance(item, dict):
                continue
            uri = item.get("file_uri")
            mime = item.get("mime_type") or "application/octet-stream"
            if isinstance(uri, str) and uri:
                out.append(Part(root=FilePart(file=FileWithUri(uri=uri, mime_type=mime))))

    msg = obj.get("message")
    if isinstance(msg, str) and msg.strip():
        out.append(Part(root=TextPart(text=msg)))

    # Nested: {"result": {...}}
    result = obj.get("result")
    if isinstance(result, dict):
        out.extend(_convert_file_message_obj(result))

    # Legacy/newer alt schema: {"tool_response":[...], "agent_response":[...]}
    tool_resp = obj.get("tool_response")
    if isinstance(tool_resp, list):
        for item in tool_resp:
            if not isinstance(item, dict):
                continue
            uri = item.get("fileUri") or item.get("file_uri")
            mime = item.get("mimeType") or item.get("mime_type") or "application/octet-stream"
            if isinstance(uri, str) and uri:
                out.append(Part(root=FilePart(file=FileWithUri(uri=uri, mime_type=mime))))
    agent_resp = obj.get("agent_response")
    if isinstance(agent_resp, list):
        for item in agent_resp:
            if isinstance(item, dict) and item.get("kind") == "text" and isinstance(item.get("text"), str):
                out.append(Part(root=TextPart(text=item["text"])))
            elif isinstance(item, str):
                out.append(Part(root=TextPart(text=item)))

    return out


def _maybe_extract_fileparts_from_json_text(text: str) -> List[Part]:
    """
    If the model/tool returned a JSON string like
    {"file_data":[{"file_uri":"...","mime_type":"..."}], "message":"..."}
    convert that into A2A FilePart(FileWithUri) entries (and optional TextPart).
    """
    try:
        obj = json.loads(text)
    except Exception:
        return []

    if not isinstance(obj, dict):
        return []

    return _convert_file_message_obj(obj)


def _extract_parts_from_function_response(fr_obj) -> List[Part]:
    """
    Try to pull a payload out of a function_response object (string or dict),
    then convert to FilePart/TextPart via the helpers above.
    """
    # 0) Dict-like function_response (some SDKs deliver dict directly)
    if isinstance(fr_obj, dict):
        logger.debug("function_response payload is dict; converting directly.")
        parts = _convert_file_message_obj(fr_obj)
        if parts:
            logger.debug("Parsed %d part(s) from function_response dict.", len(parts))
        return parts

    # 1) Common attribute names across SDKs that may hold dict or string
    candidate_attrs = ("response", "content", "result", "output", "output_text", "text")
    for attr in candidate_attrs:
        try:
            val = getattr(fr_obj, attr, None)
        except Exception:
            val = None
        if not val:
            continue

        # dict payload
        if isinstance(val, dict):
            logger.debug("function_response.%s is dict; converting.", attr)
            parts = _convert_file_message_obj(val)
            if parts:
                logger.debug("Parsed %d part(s) from function_response.%s dict.", len(parts), attr)
            return parts

        # string payload (JSON)
        if isinstance(val, str) and val.strip():
            logger.debug("function_response.%s is text; attempting JSON parse.", attr)
            parts = _maybe_extract_fileparts_from_json_text(val)
            if parts:
                logger.debug("Parsed %d part(s) from function_response.%s JSON.", len(parts), attr)
            return parts

    # 2) Rare: function_response is a JSON string itself
    if isinstance(fr_obj, str) and fr_obj.strip():
        logger.debug("function_response object is text; attempting JSON parse.")
        parts = _maybe_extract_fileparts_from_json_text(fr_obj)
        if parts:
            logger.debug("Parsed %d part(s) from function_response text.", len(parts))
        return parts

    logger.debug("No usable payload found in function_response.")
    return []


def convert_genai_parts_to_a2a(parts: List[genai_types.Part]) -> List[Part]:
    """
    Outbound (LLM/tool output -> A2A):
    - text -> A2A TextPart (unless it is a JSON wrapper for file_data, which we convert to FileParts)
    - file_data -> A2A FilePart(FileWithUri) so the orchestrator receives proper file_data again
    - function_response -> parse JSON/dict payload from tool and convert to FileParts/TextPart
    - inline_data -> rejected (URI-only policy)
    """
    out: List[Part] = []
    for p in parts:
        if p.text:
            parts_from_json = _maybe_extract_fileparts_from_json_text(p.text)
            if parts_from_json:
                out.extend(parts_from_json)
            else:
                out.append(Part(root=TextPart(text=p.text)))

        elif p.file_data:
            if not p.file_data.file_uri:
                raise ValueError("File URI is missing in file_data")
            out.append(
                Part(
                    root=FilePart(
                        file=FileWithUri(
                            uri=p.file_data.file_uri,
                            mime_type=p.file_data.mime_type,
                        )
                    )
                )
            )

        elif p.function_response:
            # ✅ Parse the function_response payload (dict or JSON text)
            fr_parts = _extract_parts_from_function_response(p.function_response)
            out.extend(fr_parts)

        elif p.inline_data:
            # Enforce URI-only policy strictly
            raise ValueError("Agent produced inline bytes; URI-only policy enforced.")

        # Optional: handle function_call if needed.

    return out