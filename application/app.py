import os
import re
import json
import tempfile
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network

# --- LangChain Imports ---
try:
    from langchain_community.graphs import Neo4jGraph
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    st.error(f"❌ Library Import Error: {e}")
    st.info("Please run: `pip install langchain langchain-community langchain-openai langchain-core`")
    st.stop()

# --- 1. Global Configuration ---
st.set_page_config(
    page_title="MEA-KG Explorer",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 8px; }
    .chat-message { padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex }
    .chat-message.user { background-color: #e6f3ff }
    .chat-message.bot { background-color: #f0f2f6 }
</style>
""", unsafe_allow_html=True)

# --- Security: Load Credentials from Environment ---
# In production (Streamlit Cloud), set these in "Secrets"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j") # Replace default or set env var

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.gptsapi.net/v1")

AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# --- 2. Robust Database Functions ---
@st.cache_resource
def get_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"❌ Database Connection Error: {e}")
        return None

def run_query(query, params=None):
    driver = get_driver()
    if not driver: return []
    try:
        with driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]
    except Exception as e:
        st.error(f"Query Execution Error: {e}")
        return []

# --- Custom Graph Wrapper ---
class SimpleNeo4jGraph(Neo4jGraph):
    def __init__(self, url, username, password):
        # Initialize connection manually to handle custom drivers if needed
        self._driver = GraphDatabase.driver(url, auth=(username, password))

        # 1. String Schema (Simplified for LLM context)
        self.schema = """
        Node properties:
        - **Instance**:
          - name: STRING
          - type: STRING
          - source: STRING
        - **n4sch__Class**:
          - n4sch__name: STRING

        Relationship properties:
        - **EXTRACTED_RELATION**:
          - original_type: STRING

        The relationships:
        (:Instance)-[:INSTANCE_OF]->(:n4sch__Class)
        (:Instance)-[:EXTRACTED_RELATION]->(:Instance)
        """

    def query(self, query, params=None):
        with self._driver.session() as session:
            try:
                result = session.run(query, params)
                return [r.data() for r in result]
            except Exception as e:
                return []

    def refresh_schema(self):
        pass

# --- NEW: KBQA Backend Engines (API Version) ---
@st.cache_resource
def get_llm():
    """Initialize Cloud API LLM (gpt-4o-mini)"""
    if not OPENAI_API_KEY:
        st.error("⚠️ OPENAI_API_KEY not found in environment variables.")
        return None
        
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY
    )

@st.cache_resource
def get_graph():
    """Initialize Graph Connection"""
    return SimpleNeo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

# --- 3. Visualization Engine ---
def render_graph(nodes, edges, height="750px"):
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black")
    # Physics optimized for stability
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)
    
    for n in nodes:
        net.add_node(n['id'], label=n['label'], title=n.get('title', n['label']), color=n['color'], size=n['size'], borderWidth=1)
    
    for e in edges:
        net.add_edge(e['source'], e['target'], title=e['label'], label=e['label'], color="#bdc3c7", width=1, arrows="to")
    
    try:
        path = os.path.join(tempfile.gettempdir(), "mea_kg_graph.html")
        net.save_graph(path)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Graph Rendering Error: {e}")
        return None

# --- 4. UI Structure ---
# Sidebar
st.sidebar.title("🚀 MEA-KG Navigator")
mode = st.sidebar.radio(
    "Select Module:",
    ["📊 System Overview", "🔍 Semantic Search", "🕸️ Exploratory Query", "❓ Knowledge QA"]
)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: The QA module uses advanced LLMs to translate natural language into Graph Queries.")
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 MEA-KG Project\nBridging Earth & Mars")

# Main Header
st.title("🪐 MEA-KG: Comparative Planetology Knowledge Graph")
st.markdown("### An Automated Knowledge Graph Construction Framework for Comparative Planetology via Ontology-Guided Large Language Models")

# ==================================================
# MODULE 1: SYSTEM OVERVIEW
# ==================================================
if mode == "📊 System Overview":
    st.markdown("---")
    st.subheader("📈 Topological Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ontology Classes", "4,367", "Schema Layer")
    col2.metric("Entity Instances", "12,362", "Data Layer")
    col3.metric("Semantic Relations", "10,799", "Knowledge Links")

    st.success("""
    **Statistical Summary:** After full processing, MEA-KG contains **4,367** ontology class nodes, **12,362** entity nodes, and **10,799** semantic relations. 
    Through the automated mounting algorithm, the physical connection between the **Instance Layer** and the **Schema Layer** is successfully realized.
    """)

    st.subheader("🌌 Global Graph Visualization (Skeleton + Flesh)")
    st.markdown("Visualizing the hierarchical organization: **Red (Schema)** vs **Blue (Data)**.")

    if st.button("Load Global Graph (Max 2000 Nodes)"):
        with st.spinner("Fetching global topology data..."):
            # Combined efficient query
            cypher_schema = "MATCH (n:n4sch__Class) RETURN n.n4sch__name as id, 'Schema' as type LIMIT 1000"
            cypher_instance = "MATCH (n:Instance) RETURN n.name as id, 'Instance' as type LIMIT 1000"
            cypher_rels = """
                MATCH (i:Instance)-[r:INSTANCE_OF]->(c:n4sch__Class)
                RETURN i.name as source, c.n4sch__name as target, 'INSTANCE_OF' as label
                LIMIT 1500
            """

            nodes_data = []
            edges_data = []
            seen_ids = set()

            # Schema
            for r in run_query(cypher_schema):
                nid = r['id']
                if nid and nid not in seen_ids:
                    nodes_data.append({'id': nid, 'label': nid, 'color': '#D32F2F', 'size': 25, 'title': f"Class: {nid}", 'group': 'Schema'})
                    seen_ids.add(nid)
            
            # Instances
            for r in run_query(cypher_instance):
                nid = r['id']
                if nid and nid not in seen_ids:
                    nodes_data.append({'id': nid, 'label': nid, 'color': '#1976D2', 'size': 10, 'title': f"Instance: {nid}", 'group': 'Instance'})
                    seen_ids.add(nid)

            # Relations
            for r in run_query(cypher_rels):
                if r['source'] in seen_ids and r['target'] in seen_ids:
                    edges_data.append(r)

            html = render_graph(nodes_data, edges_data, height="800px")
            if html: st.components.v1.html(html, height=810, scrolling=False)

# ==================================================
# MODULE 2: SEMANTIC SEARCH
# ==================================================
elif mode == "🔍 Semantic Search":
    st.subheader("Scientific Entity Search")
    query_term = st.text_input("Enter keyword (e.g., Gale Crater, Hematite, Water):", "Gale Crater")

    if query_term:
        with st.spinner(f"Searching for '{query_term}'..."):
            # Find target
            cypher_target = f"""
            MATCH (n:Instance) WHERE toLower(n.name) CONTAINS toLower('{query_term}')
            RETURN n.name as id, labels(n) as labels, properties(n) as props LIMIT 1
            """
            target_res = run_query(cypher_target)

            if not target_res:
                st.warning(f"No entity found matching '{query_term}'.")
            else:
                target_node = target_res[0]
                center_id = target_node['id']
                st.success(f"Found Entity: **{center_id}**")

                # Find 1-hop neighbors
                cypher_neighbors = f"""
                MATCH (center:Instance {{name: '{center_id}'}})-[r]-(neighbor)
                RETURN neighbor.name as n_id, labels(neighbor) as n_labels, neighbor.n4sch__name as class_name, type(r) as r_type, r.original_type as r_original_type
                LIMIT 50
                """
                neighbors = run_query(cypher_neighbors)

                # Build Graph
                nodes = [{'id': center_id, 'label': center_id, 'color': '#FBC02D', 'size': 30, 'title': 'Target', 'group': 'Target'}]
                edges = []
                seen = {center_id}

                for row in neighbors:
                    n_id = row['n_id'] if row['n_id'] else row['class_name']
                    if not n_id: continue
                    
                    n_type = 'Schema' if 'n4sch__Class' in row['n_labels'] else 'Instance'
                    color = '#D32F2F' if n_type == 'Schema' else '#1976D2'

                    if n_id not in seen:
                        nodes.append({'id': n_id, 'label': n_id, 'color': color, 'size': 15 if n_type == 'Instance' else 20, 'title': f"{n_type}: {n_id}", 'group': n_type})
                        seen.add(n_id)

                    edge_label = row['r_original_type'] if row['r_original_type'] else row['r_type']
                    edges.append({'source': center_id, 'target': n_id, 'label': edge_label})

                col1, col2 = st.columns([3, 1])
                with col1:
                    html = render_graph(nodes, edges)
                    if html: st.components.v1.html(html, height=600)
                with col2:
                    st.markdown("### 📄 Metadata")
                    st.json(target_node['props'])
                    st.markdown(f"**{len(neighbors)}** connections.")

# ==================================================
# MODULE 3: EXPLORATORY QUERY
# ==================================================
elif mode == "🕸️ Exploratory Query":
    st.subheader("Filter by Entity Type")

    with st.spinner("Loading entity types..."):
        type_res = run_query("MATCH (n:Instance) RETURN DISTINCT n.type as t LIMIT 100")
        all_types = sorted([r['t'] for r in type_res if r['t']])

    if not all_types:
        st.error("No entity types found.")
    else:
        selected_type = st.selectbox("Select an Entity Type:", all_types)

        if st.button(f"Visualize '{selected_type}' Network"):
            cypher = f"""
            MATCH (n:Instance {{type: '{selected_type}'}})
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n.name as n_id, m.name as m_id, m.n4sch__name as m_class, r.original_type as r_orig, type(r) as r_type
            LIMIT 50
            """
            data = run_query(cypher)

            if not data:
                st.warning("No connections found.")
            else:
                nodes = []
                edges = []
                seen = set()

                for row in data:
                    n_id = row['n_id']
                    if n_id and n_id not in seen:
                        nodes.append({'id': n_id, 'label': n_id, 'color': '#009688', 'size': 20, 'group': selected_type})
                        seen.add(n_id)
                    
                    m_id = row['m_id'] if row['m_id'] else row['m_class']
                    if m_id:
                        if m_id not in seen:
                            nodes.append({'id': m_id, 'label': m_id, 'color': '#9E9E9E', 'size': 10, 'group': 'Related'})
                            seen.add(m_id)
                        
                        r_label = row['r_orig'] if row['r_orig'] else row['r_type']
                        if not r_label: r_label = "RELATED"
                        edges.append({'source': n_id, 'target': m_id, 'label': r_label})

                html = render_graph(nodes, edges)
                if html:
                    st.components.v1.html(html, height=600)
                    st.success(f"Visualizing network for **{selected_type}** ({len(nodes)} nodes).")

# ==================================================
# MODULE 4: KNOWLEDGE QA
# ==================================================
elif mode == "❓ Knowledge QA":
    st.subheader("🤖 Knowledge-Based Question Answering (KBQA)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "thinking" in message:
                with st.expander("📜 View Thinking Process"):
                    st.markdown(message["thinking"])

    def is_greeting_or_simple(text):
        simple_keywords = {
            "hello": "Hello! How can I help you with Mars or Earth geology today?",
            "hi": "Hi there! I'm your planetary science assistant.",
            "你好": "你好！我是您的地火类比知识助手。",
            "谢谢": "不客气！"
        }
        cleaned = re.sub(r'[^\w\s]', '', text.lower()).strip()
        return simple_keywords.get(cleaned, None)

    if prompt := st.chat_input("Ask me anything about Mars, Earth analogs, or geology..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            fast_response = is_greeting_or_simple(prompt)
            if fast_response:
                st.write(fast_response)
                st.session_state.messages.append({"role": "assistant", "content": fast_response, "thinking": "**Router:** `Level 1` - Instant Reply"})
            else:
                status_container = st.status("🤔 Thinking...", expanded=True)
                try:
                    llm = get_llm()
                    graph = get_graph()
                    
                    if not llm or not graph:
                        raise Exception("LLM or Graph connection failed. Check API keys.")

                    # 1. Router
                    status_container.write("🧠 Analyzing intent...")
                    router_prompt = PromptTemplate.from_template(
                        """Determine if the question requires database search. 
                        Question: {question}
                        Return 'SEARCH' if it asks for facts. Return 'ANSWER' if general chat. No explanations."""
                    )
                    router_chain = router_prompt | llm | StrOutputParser()
                    intent = router_chain.invoke({"question": prompt}).strip()
                    thinking_log = f"**Intent:** `{intent}`\n\n"

                    if "ANSWER" in intent:
                        status_container.update(label="✅ General Chat", state="complete", expanded=False)
                        response_chain = PromptTemplate.from_template("{question}") | llm | StrOutputParser()
                        answer = response_chain.invoke({"question": prompt})
                    else:
                        # 2. Cypher Gen
                        status_container.write("🔍 Generating Robust Cypher...")
                        schema_str = graph.schema
                        cypher_prompt = PromptTemplate.from_template(
                            """Task: Generate a FLEXIBLE Cypher query for Neo4j.
                            Schema:
                            {schema}
                            
                            **CRITICAL INSTRUCTIONS:**
                            1. **Fuzzy Matching:** NEVER use exact match (`=`). ALWAYS use `toLower(n.name) CONTAINS toLower('keyword')`.
                            2. **Broad Search:** If the user asks about "Jarosite", search for nodes where name contains "jarosite".
                            3. **Return Data:** Return `n.name`, `labels(n)`, and relationships.
                            4. **Limit:** Limit results to 10.

                            Question: {question}
                            Cypher Query:"""
                        )
                        cypher_chain = cypher_prompt | llm | StrOutputParser()
                        cypher_query = cypher_chain.invoke({"question": prompt, "schema": schema_str})
                        cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
                        thinking_log += f"**Generated Cypher:**\n```cypher\n{cypher_query}\n```\n\n"

                        # 3. Execute
                        status_container.write("⚙️ Executing query...")
                        context_data = graph.query(cypher_query)
                        thinking_log += f"**Database Result:**\n`{str(context_data)[:500]}...`\n\n"

                        # 4. Synthesize
                        status_container.write("💡 Synthesizing...")
                        qa_prompt = PromptTemplate.from_template(
                            """Use the Context to answer. 
                            If context is not empty, explicitly cite the entities found in the graph.
                            If context is empty, say "Graph data not found, but generally..." and give a scientific answer.

                            Context: {context}
                            Question: {question}
                            Answer:"""
                        )
                        qa_chain = qa_prompt | llm | StrOutputParser()
                        answer = qa_chain.invoke({"question": prompt, "context": context_data})
                        status_container.update(label="✅ Search Complete", state="complete", expanded=False)

                    message_placeholder = st.empty()
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "thinking": thinking_log})

                except Exception as e:
                    status_container.update(label="❌ Error", state="error")
                    st.error(f"Error: {e}")
