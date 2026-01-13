import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# --- Configuration ---
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "gte-large:latest"
LLM_MODEL = "llama-3.2-3b-it:latest"

print("--- Setting up Vector Store ---")
embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Ensure DB directory exists to avoid initial errors
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_function
)

@tool(response_format="content_and_artifact")
def retrieve_similar_incidents(query: str, category_filter: str):
    """
    Call this tool to find a similar past incidents and their solutions.

    Args:
        query: A description of the problem or error log.
        category_filter: (Optional) Filter by specific metadata category to narrow search.
                        valid options usually include: 'Infrastructure', 'Techinal Debt', 'CI/CD Pipeline', 'Frontend / UX', 'Performance'.
    """
    print(f"\n[Tool] Querying database for: '{query}' | Filter: {category_filter}")

    search_kwargs = {"k": 2}

    if category_filter:
        search_kwargs["filter"] = {"category": category_filter}

    results = vector_store.similarity_search(query, **search_kwargs)

    if not results: 
        return "No similar incidents found in the database.", []

    context_text = "\n\n".join([
        f"--- Past Incident ({doc.metadata.get('id', 'Unknown ID')}) ---\n"
        f"Category: {doc.metadata.get('category', 'N/A')}\n"
        f"Service: {doc.metadata.get('service', 'N/A')}\n"
        f"Impact: {doc.metadata.get('impact', 'N/A')}\n"
        f"Content:\n{doc.page_content}"
        for doc in results
    ])

    return context_text, results

tools = [retrieve_similar_incidents]
llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url="http://localhost:11434")

agent_llm = llm.bind_tools(tools)

def agent_node(state: MessagesState):
    """
    The main decision node.
    """
    messages = state["messages"]
    response = agent_llm.invoke(messages)
    
    # FIX 2: Return key must be "messages" to match State schema
    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)

workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge('tools', "agent")

app = workflow.compile()

def filter_critical_logs(full_log):
    """
    Extracts only ERROR and CRITICAL lines to reduce noise for the embedding model.
    """
    lines = full_log.strip().split('\n')
    # Filter for high severity lines
    critical_lines = [
        line for line in lines 
        if "ERROR" in line or "CRITICAL" in line
    ]
    
    if not critical_lines:
        return "No Critical Errors found in logs."
    
    return "\n".join(critical_lines)

def solve_incident(new_incident_log):
    print("\nProcessing New Incident Log...\n")

    filtered_query = filter_critical_logs(new_incident_log)
    print(f"--- Filtered Query for RAG ---\n{filtered_query}\n------------------------------")

    system_prompt = (
        "You are a Senior Site Reliability Engineer (SRE). "
        "You have access to a database of past incidents. "
        "When given a log snippet, ALWAYS categorise the incident first and search the database to see if a similar solution exists. "
        "Focus on the 'ERROR' and 'CRITICAL' lines provided."
        "Then, provide a Root Cause and a Recommended Solution based on the retrieved data."
        "Do not generate any examples except the root cause and solution."
    )

    initial_state = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze these specific errors:\n{filtered_query}")
        ]
    }

    final_response = None
    for event in app.stream(initial_state, stream_mode="values"):
        message = event["messages"][-1]

        if hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
            print("Agent decided to search knowledge base...")
        elif message.type == "ai" and not message.tool_calls:
            final_response = message.content
    
    return final_response

if __name__ == "__main__":
    sample_incident = """
[2026-01-07 10:15:02] [INFO]  [Source] Triggered by commit 'a8f3b21' (Author: dev_user)
[2026-01-07 10:15:10] [INFO]  [Build] Starting container build: docker-image-v2.4.1
[2026-01-07 10:17:45] [INFO]  [Build] Docker image built successfully.
[2026-01-07 10:17:50] [INFO]  [Test] Initiating Unit Tests...
[2026-01-07 10:18:30] [INFO]  [Test] Unit Tests passed (142/142).
[2026-01-07 10:18:35] [INFO]  [Test] Initiating Integration Tests...
[2026-01-07 10:19:12] [ERROR] [Test] Integration Test Failure: 'DatabaseConnectionError'
[2026-01-07 10:19:12] [ERROR] [Test] Details: Connection timeout to staging-db-01.
[2026-01-07 10:19:15] [WARN]  [Pipeline] Retrying Integration Tests (Attempt 1/3)...
[2026-01-07 10:19:40] [ERROR] [Test] Integration Test Failure: 'DatabaseConnectionError'
[2026-01-07 10:19:45] [CRITICAL] [Pipeline] STAGE FAILED: Test phase unsuccessful.
[2026-01-07 10:19:46] [INFO]  [Pipeline] Status changed to 'FAILED'.
[2026-01-07 10:19:47] [INFO]  [Notification] Alert sent to Slack channel #ops-alerts.
[2026-01-07 10:19:48] [INFO]  [Cleanup] Terminating ephemeral test environments.
[2026-01-07 10:25:00] [INFO]  [Incident] Manual investigation started by On-Call Engineer.
[2026-01-07 10:32:15] [DEBUG] [Incident] Root Cause: Staging DB credentials expired in the Vault.
[2026-01-07 10:40:00] [INFO]  [Incident] Credentials updated. Manual trigger initiated.
    """

    solution = solve_incident(sample_incident)

    print("\n" + "="*40)
    print("Final Agent Report")
    print("="*40)
    print(solution)