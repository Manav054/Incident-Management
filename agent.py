import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# --- Configuration ---
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "gte-large:latest"
LLM_MODEL = "llama-3.2-3b-it:latest"
OUTPUT_FOLDER = "./final_drafts"

# --- Initialize Components ---
llm = ChatOllama(model=LLM_MODEL, temperature=0)
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

# --- Initialize Vector Collections ---
# Note: This assumes you have already ingested data into these collections.
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

# --- Tool 1: Incident Retrieval ---
@tool
def fetch_incidents(query: str):
    """Retrieves 3 similar incidents. Returns: Timestamp, Root Cause, Relevance %, and Solutions."""
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
    return context, results

# --- Tool 2: SOP Retrieval ---
@tool
def fetch_sops(query: str):
    """Retrieves 3 similar SOPs for synthesis."""
    print(f"\n[Tool] Querying Vector DB for 3 similar SOPs...")
    results = sop_vc.similarity_search(query, k=3)

    context = "### TOP 3 RELEVANT SOP DOCUMENTS ###\n"
    for i, doc in enumerate(results, 1):
        context += f"--- SOP Document {i} ---\nContent: {doc.page_content[:2000]}\n\n"
    return context, results

# --- Tool 3: Save Final Draft ---
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

# --- LangGraph Orchestration ---

def assistant_node(state: MessagesState):
    system_prompt = SystemMessage(content=(
        "You are an AI SRE Assistant. Start by greeting the user and asking for incident logs.\n\n"
        "Once logs are provided:\n"
        "1. CALL 'fetch_incidents' to find 3 similar past cases. Report their timestamps, root causes, and relevance %.\n"
        "2. CALL 'fetch_sops' to find 3 relevant procedures. SYNTHESIZE these 3 into a single unified SOP.\n"
        "3. Present the findings clearly to the user.\n"
        "4. ASK the user if they want to save this as a final draft. If they say yes, use 'save_resolution_draft'."
    ))
    
    tools = [fetch_incidents, fetch_sops, save_resolution_draft]
    # We bind tools and invoke. We prepend the system prompt only for this call (stateless for graph).
    return {"messages": [llm.bind_tools(tools).invoke([system_prompt] + state["messages"])]}

# Graph Construction
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode([fetch_incidents, fetch_sops, save_resolution_draft]))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile(checkpointer=MemorySaver())

# --- Execution Logic ---

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "sre_session_001"}}
    print("--- SRE Assistant Active ---")
    print("Type 'exit' or 'quit' to stop.")
    
    # 1. Start the conversation (System greeting)
    # We send an empty message to trigger the Assistant node to run its logic (greeting)
    initial_events = graph.stream(
        {"messages": [HumanMessage(content="Hi, I am ready to start.")]}, 
        config, 
        stream_mode="updates"
    )
    
    for event in initial_events:
        for value in event.values():
            if "messages" in value:
                # Print the AI's greeting
                value["messages"][-1].pretty_print()

    # 2. Interactive Loop
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]: 
                break
            
            # Stream updates (this prints tools and responses as they happen)
            for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="updates"):
                for node_name, value in event.items():
                    if "messages" in value:
                        last_msg = value["messages"][-1]
                        
                        # Pretty print AI responses
                        if isinstance(last_msg, AIMessage):
                            # If it has tool calls, print them specifically
                            if last_msg.tool_calls:
                                print(f"\n[Assistant] Calling Tools: {[tc['name'] for tc in last_msg.tool_calls]}")
                            else:
                                last_msg.pretty_print()
                        
                        # Optional: Print Tool Outputs if you want to debug
                        # if node_name == "tools":
                        #     print(f"\n[Tool Output] {last_msg.content[:200]}...")

        except Exception as e:
            print(f"An error occurred: {e}")
            break