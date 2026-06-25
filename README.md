🚀 Enterprise DevOps & Server Log Agent (LangGraph)

An autonomous, stateful AI assistant built using LangGraph, LangChain Core, and ChatGroq (llama-3.3-70b-versatile). This project is designed to process high-volume server logs, featuring smart message pruning, state summarization, and safe tool-calling to optimize token consumption and prevent context window exhaustion.

🧠 Architecture Overview

The system operates on a stateful graph structure using a custom state subclass (CustomState) that retains the message history and a dynamically updated global conversation summary.

graph TD
    START([START]) --> summarizer[summarizer_node]
    summarizer --> assistant[assistant_node]
    assistant -->|Tool Call Needed| tools[tools_node: fetch_server_logs]
    tools -->|Direct Return / No Pruning| assistant
    assistant -->|Final Response| END([END])
    
    classDef nodeStyle fill:#1e293b,stroke:#475569,stroke-width:1.5px,color:#94a3b8;
    classDef activeStyle fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#dbeafe;
    class START,END nodeStyle;
    class summarizer,assistant,tools activeStyle;


Key Graph Components

summarizer_node: Triggered only at the start of a turn when the message count exceeds the threshold ($N > 4$). It condenses the older history into a single string, updates the state's summary variable, and triggers the RemoveMessage protocol to purge raw messages from memory.

assistant_node: Formulates a system message containing the condensed summary, analyzes the user's query, and determines whether to answer immediately or invoke external diagnostics tools.

tools: Executes Python-native queries (such as fetch_server_logs) to retrieve server diagnostic reports based on a requested ID.

📈 Memory & Token Optimization Model

In vanilla conversational agents, token consumption increases quadratically as the session progresses. By introducing automatic message pruning and state summarization, we bound the memory overhead to $O(1)$ instead of $O(N^2)$ context growth:

$$\text{Without Pruning: } T(n) = \sum_{i=1}^{n} m_i = O(n^2)$$

$$\text{With LangGraph Pruning: } T(n) \le C \quad (\text{where } C \text{ is a constant token ceiling})$$

This model guarantees that the agent can run indefinitely without hitting LLM context limits or causing extreme API billing spikes.

🛠️ The "Mid-Turn Pruning" Deadlock & Engineering Solution

The Bug (System Deadlock)

Originally, after the tools node fetched the log files, the workflow transitioned back to the start of the graph:

# DEPRECATED INSECURE PATHway
workflow.add_edge("tools", "summarizer")


This caused a critical race condition. When a tool returned a long server log, the sudden addition of messages triggered the summarizer mid-turn (before the assistant could formulate the final reply). The summarizer aggressively purged the initial HumanMessage to fit the memory threshold. Consequently, the assistant lost the original user intent, fell into confusion, re-called the tool, and trapped the system in an infinite looping deadlock.

The Solution (Isolated Topology)

We isolated the summarizer to run only at the beginning of a user turn, routing the tool outputs directly back to the assistant:

# PRODUCTION-GRADE SECURE PATHWAY
workflow.add_edge("tools", "assistant")


Additionally, robust timeout=10 parameters were introduced to all LLM invocations to prevent hanging API connections in active servers.

🚀 Getting Started

Prerequisites

Python 3.11+ (Optimized for Ubuntu Environment)

A valid Groq API Key

Installation

Clone the repository:

git clone https://github.com/yasinyagmur01/langgraph-devops-log-agent.git
cd langgraph-devops-log-agent


Set up a virtual environment:

python3 -m venv venv
source venv/bin/activate


Install Dependencies:

pip install langgraph langchain-core langchain-groq python-dotenv


Environment Setup:
Create a .env file in the root directory and append your API credentials:

GROQ_API_KEY=your_actual_api_key_here


Run the Agent Simulation:

python main.py


🗺️ Roadmap

[ ] Semantic Routing: Inspect log errors dynamically and route queries to specialized prompt environments.

[ ] Multi-Agent Orchestration: Separate roles into a dedicated "Log Analyst Agent" and a "DevOps Engineer Agent".

[ ] Human-in-the-Loop (HITL): Implement interrupt_before=["tools"] to prompt for admin approval before executing potential server hotfixes.

Developed as part of the Enterprise DevOps Agent Architecture Analysis.