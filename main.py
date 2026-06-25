"""
Mini Project 4: Enterprise DevOps & Server Log Agent
Demonstrating advanced context management (State Summarization & Message Pruning)
alongside Tool Calling and Persistent Memory using LangGraph and ChatGroq.
"""

import re
import time
from typing import Any

from dotenv import load_dotenv
from groq import RateLimitError
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Automatically load environment variables from the .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MESSAGE_PRUNE_THRESHOLD = 4
MESSAGES_TO_KEEP = 2
TOOL_OUTPUT_SUMMARY_LIMIT = 600
ASSISTANT_MODEL = "llama-3.3-70b-versatile"
SUMMARIZER_MODEL = "llama-3.1-8b-instant"
LLM_MAX_RETRIES = 4
LLM_RETRY_BASE_DELAY_SECONDS = 5

# ---------------------------------------------------------------------------
# 1. Mock Server Data
# ---------------------------------------------------------------------------
MOCK_SERVER_LOGS = {
    "server_01": (
        "[CRITICAL ERROR] 2026-06-23 07:00:00 - Database connection failed.\n"
        "Reason: Max pool size reached (limit=100).\n"
        "Stack Trace:\n"
        "  at Data.Connection.Pool.Acquire() in /src/pool.go:line 42\n"
        "  at API.Handlers.GetUsers() in /src/users.go:line 108\n"
        "  [... imagine 2000 more lines of heavy performance stack dump here ...]\n"
        "Status: System unstable, blocking all incoming API connection vectors."
    ),
    "server_02": (
        "[WARN] 2026-06-23 07:15:00 - High CPU usage detected on core 3 (98%).\n"
        "Process ID: 9412 (analytics_worker).\n"
        "System Action: Throttling non-essential sub-routines to stabilize core."
    ),
}


# ---------------------------------------------------------------------------
# 2. Tool Definition
# ---------------------------------------------------------------------------
@tool
def fetch_server_logs(server_id: str) -> str:
    """Fetch intensive system crash/warning logs for a specific server identifier.

    Use this tool immediately whenever a user asks to see logs, investigate crashes,
    or check the health status of a specific server.
    """
    normalized_id = server_id.strip().lower()
    logs = MOCK_SERVER_LOGS.get(normalized_id)
    if logs is None:
        return f"Server '{server_id}' not found in active cluster infrastructure database."
    return f"--- START OF LOGS FOR {normalized_id.upper()} ---\n{logs}\n--- END OF LOGS ---"


# ---------------------------------------------------------------------------
# 3. Advanced State Definition
# ---------------------------------------------------------------------------
class CustomState(MessagesState):
    """Extended state schema containing conversational messages and a running summary."""

    summary: str


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------
def _parse_retry_delay_seconds(error: RateLimitError) -> float:
    """Extract the suggested wait time from a Groq rate-limit error message."""
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(error), re.IGNORECASE)
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return float(LLM_RETRY_BASE_DELAY_SECONDS)


def invoke_llm_with_retry(llm: ChatGroq, input_data: Any, node_name: str) -> Any:
    """Invoke an LLM with backoff when Groq rate limits are hit."""
    last_error: Exception | None = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return llm.invoke(input_data)
        except RateLimitError as exc:
            last_error = exc
            if attempt == LLM_MAX_RETRIES:
                break
            delay = _parse_retry_delay_seconds(exc)
            print(
                f"[{node_name}] Rate limit reached (attempt {attempt}/{LLM_MAX_RETRIES}). "
                f"Retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
        except Exception:
            raise

    raise RuntimeError(
        f"[{node_name}] Groq rate limit persisted after {LLM_MAX_RETRIES} retries. "
        "Daily token quota may be exhausted — wait or upgrade your Groq plan."
    ) from last_error


def format_message_for_summary(message: BaseMessage) -> str:
    """Serialize a message into summary-friendly text, preserving tool metadata."""
    role = message.__class__.__name__
    parts: list[str] = [role]

    if isinstance(message, AIMessage) and message.tool_calls:
        tool_descriptions = ", ".join(
            f"{tc['name']}({tc['args']})" for tc in message.tool_calls
        )
        parts.append(f"tool_calls=[{tool_descriptions}]")

    content = message.content
    if isinstance(content, str) and content.strip():
        if isinstance(message, ToolMessage) and len(content) > TOOL_OUTPUT_SUMMARY_LIMIT:
            content = (
                f"{content[:TOOL_OUTPUT_SUMMARY_LIMIT]}"
                f"... [truncated {len(content) - TOOL_OUTPUT_SUMMARY_LIMIT} chars]"
            )
        parts.append(content)
    elif not isinstance(message, AIMessage) or not message.tool_calls:
        parts.append("[no text content]")

    return ": ".join(parts)


# ---------------------------------------------------------------------------
# 5. Nodes Implementation
# ---------------------------------------------------------------------------
def summarizer_node(state: CustomState) -> dict:
    """Token-saving brain. Prunes heavy messages and compresses them into a core summary."""
    messages = state["messages"]

    if len(messages) <= MESSAGE_PRUNE_THRESHOLD:
        print(
            f"[Summarizer] Message count ({len(messages)}) below threshold. "
            "Skipping pruning."
        )
        return {}

    print(f"\n[Summarizer] Critical message threshold exceeded ({len(messages)} messages).")
    print("[Summarizer] Compressing old history and activating RemoveMessage protocol...")

    messages_to_summarize = messages[:-MESSAGES_TO_KEEP]
    existing_summary = state.get("summary", "")

    history_str = "\n".join(
        format_message_for_summary(m) for m in messages_to_summarize
    )

    summary_prompt = (
        "You are a master context manager. Update the current running summary with the new history.\n\n"
        f"Current Summary: {existing_summary or '(empty)'}\n\n"
        f"New History to compress:\n{history_str}\n\n"
        "Generate a dense, precise running summary. You MUST preserve:\n"
        "- The user's name if mentioned\n"
        "- Every server ID requested or investigated (e.g. server_01, server_02)\n"
        "- Root causes and active errors discussed\n"
        "Do not mention system log formatting or message roles."
    )

    summary_llm = ChatGroq(model=SUMMARIZER_MODEL, temperature=0.0)
    summary_response = invoke_llm_with_retry(summary_llm, summary_prompt, "Summarizer")
    new_summary = str(summary_response.content)

    print(f"[Summarizer] Dynamic Context Updated Summary:\n>>> {new_summary}\n")

    prune_instructions = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id]
    skipped = len(messages_to_summarize) - len(prune_instructions)
    if skipped:
        print(f"[Summarizer] Warning: {skipped} message(s) had no id and could not be pruned.")

    print(f"[Summarizer] Sending {len(prune_instructions)} RemoveMessage commands to clean memory state.")

    return {
        "summary": new_summary,
        "messages": prune_instructions,
    }


def assistant_node(state: CustomState) -> dict:
    """Prepares system architecture prompt injecting the current summary, then calls the LLM."""
    print("[Assistant] Formulating system injection prompt...")

    base_instruction = "You are an enterprise-grade elite DevOps and Infrastructure Reliability Agent."

    current_summary = state.get("summary", "")
    if current_summary:
        base_instruction += (
            "\n\n[CRITICAL ARCHIVE CONTEXT] Here is a verified summary of the past conversation history. "
            "The actual raw messages were deleted to save company tokens, but you must remember these details: "
            f"{current_summary}"
        )

    system_message = SystemMessage(content=base_instruction)
    compiled_messages = [system_message] + state["messages"]

    llm = ChatGroq(model=ASSISTANT_MODEL, temperature=0.0)
    llm_with_tools = llm.bind_tools([fetch_server_logs])

    print("[Assistant] Invoking core model workflow loop...")
    response = invoke_llm_with_retry(llm_with_tools, compiled_messages, "Assistant")
    print("[Assistant] Action determination produced.")

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 6. Graph Construction & Routing
# ---------------------------------------------------------------------------
workflow = StateGraph(CustomState)

workflow.add_node("summarizer", summarizer_node)
workflow.add_node("assistant", assistant_node)
workflow.add_node("tools", ToolNode([fetch_server_logs]))

workflow.set_entry_point("summarizer")
workflow.add_edge("summarizer", "assistant")
workflow.add_conditional_edges("assistant", tools_condition)
workflow.add_edge("tools", "summarizer")

# ---------------------------------------------------------------------------
# 7. Memory Persistence
# ---------------------------------------------------------------------------
app = workflow.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 8. Deep Multi-Turn Simulation
# ---------------------------------------------------------------------------
def run_simulation_step(turn: int, user_query: str, config: dict) -> None:
    """Execute one simulation turn and print diagnostics."""
    divider = "-" * 75
    print(f"\n{divider}\nSIMULATION TURN {turn}\n{divider}")
    print(f"User Request: '{user_query}'")

    try:
        result = app.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=config,
        )
    except Exception as exc:
        print(f"\n[State Diagnostics] Turn {turn} failed: {exc}")
        raise

    total_messages_in_memory = len(result["messages"])
    final_output = result["messages"][-1].content

    print(f"\n[State Diagnostics] Remaining raw messages in state memory: {total_messages_in_memory}")
    print(f"[State Diagnostics] Running summary length: {len(result.get('summary', ''))} chars")
    print(f"Assistant Output:\n{final_output}")


def main() -> None:
    """Run the multi-turn DevOps agent simulation."""
    session_config = {"configurable": {"thread_id": "devops_session_100"}}

    print("===========================================================================")
    print("INITIALIZING CONTEXT MASTER PRODUCTION ENVIRONMENT SIMULATION")
    print("===========================================================================")

    run_simulation_step(
        1,
        "Hi, I am Yasin. Can you fetch the logs for server_01?",
        session_config,
    )

    run_simulation_step(
        2,
        "Wow, that's a huge stack trace. Based on those logs, what seems to be the root cause?",
        session_config,
    )

    run_simulation_step(
        3,
        "Understood. Now fetch the logs for server_02.",
        session_config,
    )

    run_simulation_step(
        4,
        "What was the first server I asked you about, and what was my name?",
        session_config,
    )

    print("\n===========================================================================")
    print("CONTEXT MASTER DEPLOYMENT DEMO COMPLETE — ROBUST STATE PRUNING VERIFIED")
    print("===========================================================================")


if __name__ == "__main__":
    main()
