import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "gte-large:latest"
LLM_MODEL = "llama-3.2-3b-it:latest"
OUTPUT_FOLDER = "./resolutions"

llm = ChatOllama(model=LLM_MODEL, temperature=0)
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

incident_vc = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=embedding, 
    collection_name="incident_collection"
)
sop_vc = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=embedding, 
    collection_name="sop_collection"
)

@tool
def fetch_incidents(query: str) -> str:
    """
    Use this tool ONLY when the user provides a specific error message or incident description. 
    DO NOT use this tool for greetings like "Hi" or "Hello".
    Retrieves 3 similar incidents. Returns: Timestamp, Root Cause, Relevance %, and Solutions.
    """
    print(f"\n[Tool] Querying Vector DB for 3 similar incidents...")
    results = incident_vc.similarity_search_with_relevance_scores(query, k=3)

    context = "### TOP 3 RELEVANT HISTORICAL INCIDENTS ###\n"
    for doc, score in results:
        relevance_pct = round(score * 100, 2)
        context += f"--- Incident ID: {doc.metadata.get('id', 'N/A')} ---\n"
        context += f"Timestamp: {doc.metadata.get('timestamp', 'N/A')}\n"
        context += f"Relevance Score: {relevance_pct}%\n"
        # Truncating content slightly to prevent context overflow on small models
        context += f"Full Log: {doc.page_content[:2000]}\n\n"
    return context

@tool
def fetch_sops(query: str) -> str:
    """
    Use this tool ONLY when the user provides a specific error message or incident description. 
    DO NOT use this tool for greetings like "Hi" or "Hello".
    Retrieves 3 similar SOPs for synthesis.
    """
    print(f"\n[Tool] Querying Vector DB for 3 similar SOPs...")
    results = sop_vc.similarity_search(query, k=3)

    context = "### TOP 3 RELEVANT SOP DOCUMENTS ###\n"
    for i, doc in enumerate(results, 1):
        context += f"--- SOP Document {i} ---\nContent: {doc.page_content[:2000]}\n\n"
    return context

@tool
def save_resolution_draft(incident_id: str, final_content: str):
    """
    Saves the final resolution and synthesized SOP draft to a text file.
    Use this once the user is happy with the generated analysis.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Sanitize filename
    safe_id = incident_id.replace(' ', '_').replace('/', '-')
    filename = f"{OUTPUT_FOLDER}/Resolution_{safe_id}.txt"
    
    with open(filename, "w", encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"\n[Tool] Draft saved to: {filename}")
    return f"Successfully saved draft to {filename}"

tools = [fetch_incidents, fetch_sops, save_resolution_draft]

llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contain tool calls."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

system_prompt = """
You are an AI SRE Assistant.

### INSTRUCTIONS ###
1. **GREETING PHASE:** If the user says "Hi", "Hello", or asks a general question, respond conversationally. **DO NOT USE ANY TOOLS.** Simply ask them to provide the incident logs.
2. **ANALYSIS PHASE:** ONLY when the user provides specific error logs, stack traces, or an incident description, you must:
   - Call 'fetch_incidents' to find similar past cases.
   - Call 'fetch_sops' to find relevant procedures.
3. **SYNTHESIS:** Analyze the tool outputs and present a solution.
4. **FINALIZATION:** Ask if the user wants to save the draft. If yes, call 'save_resolution_draft'.
   - **CRITICAL:** The 'final_content' argument MUST include:
     1. The Incident ID.
     2. The Full list of Similar Incidents found.
     3. The Full list of SOPs found.
     4. Your final synthesized analysis/resolution.

REMEMBER: Do not call 'fetch_incidents' until you actually see an incident description.
"""

def call_llm(state: AgentState):
    """Function to call the LLM with the current state."""
    messages = state["messages"]
    
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("assistant", call_llm)
workflow.add_node("tools", ToolNode(tools))

workflow.add_conditional_edges("assistant", should_continue)
workflow.add_edge("tools", "assistant")
workflow.set_entry_point("assistant")

checkpointer = MemorySaver()
agent = workflow.compile(checkpointer=checkpointer)

def running_agent():
    print("\n---- AI SRE AGENT (Type 'exit' to quit) ----")
    config = {"configurable": {"thread_id": "1"}}

    while True:
        try:
            user_input = input('User: ')
        except EOFError:
            break

        if user_input.lower() in ["exit", "quit"]:
            break

        result = agent.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)

        print("\n--- Assistant ---")
        print(result["messages"][-1].content)

if __name__ == "__main__":
    running_agent()