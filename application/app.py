"""
app.py — MEA-KG Interactive Streamlit Dashboard

Modules
-------
1. System Overview   : Topological statistics and global graph visualisation.
2. Semantic Search   : Fuzzy entity search with 1-hop neighbourhood graph.
3. Exploratory Query : Filter and visualise nodes by entity type.
4. Knowledge QA      : LangChain-powered KBQA with intent routing.
5. MEA-Reasoner      : White-box Graph Chain-of-Thought (Graph-CoT) reasoning.

Credentials are loaded exclusively from environment variables.
See ``.env.example`` for the required keys.

Run
---
    cd application
    streamlit run app.py
"""

import os
import re
import sys
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network

# ---------------------------------------------------------------------------
# Load .env if present (safe no-op if python-dotenv is not installed)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Add project root to sys.path so that ``reasoner`` package is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# LangChain Imports
# ---------------------------------------------------------------------------
try:
    from langchain_community.graphs import Neo4jGraph
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    _LANGCHAIN_OK = True
except ImportError as _lc_err:
    _LANGCHAIN_OK = False
    _LANGCHAIN_ERR = str(_lc_err)

# ---------------------------------------------------------------------------
# MEA-Reasoner Import
# ---------------------------------------------------------------------------
try:
    from reasoner import GraphCoT, GraphCoTResult
    _REASONER_OK = True
except ImportError as _rsn_err:
    _REASONER_OK = False
    _REASONER_ERR = str(_rsn_err)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MEA-KG Explorer",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 8px; }
    .chat-message { padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem;
                    display: flex; }
    .chat-message.user { background-color: #e6f3ff; }
    .chat-message.bot  { background-color: #f0f2f6; }
    .cot-stage { background: #f0f7ff; border-left: 4px solid #1976D2;
                 padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Credentials (from environment only — never hardcoded)
# ---------------------------------------------------------------------------
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL= os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1")
AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------
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
    if not driver:
        return []
    try:
        with driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]
    except Exception as e:
        st.error(f"Query Execution Error: {e}")
        return []


class SimpleNeo4jGraph(Neo4jGraph):
    """Minimal Neo4j graph wrapper compatible with LangChain chains."""

    def __init__(self, url, username, password):
        self._driver = GraphDatabase.driver(url, auth=(username, password))
        self.schema = """
        Node properties:
        - **Instance**: name: STRING, type: STRING, source: STRING
        - **n4sch__Class**: n4sch__name: STRING
        Relationship properties:
        - **EXTRACTED_RELATION**: original_type: STRING
        The relationships:
        (:Instance)-[:INSTANCE_OF]->(:n4sch__Class)
        (:Instance)-[:EXTRACTED_RELATION]->(:Instance)
        """

    def query(self, query, params=None):
        with self._driver.session() as session:
            try:
                result = session.run(query, params)
                return [r.data() for r in result]
            except Exception:
                return []

    def refresh_schema(self):
        pass


# ---------------------------------------------------------------------------
# LLM / Graph Cache
# ---------------------------------------------------------------------------
@st.cache_resource
def get_llm():
    if not OPENAI_API_KEY:
        st.error("⚠️ OPENAI_API_KEY not set in environment.")
        return None
    if not _LANGCHAIN_OK:
        st.error(f"❌ LangChain import error: {_LANGCHAIN_ERR}")
        return None
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )


@st.cache_resource
def get_graph():
    return SimpleNeo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)


# ---------------------------------------------------------------------------
# Graph Renderer
# ---------------------------------------------------------------------------
def render_graph(nodes, edges, height="750px"):
    net = Network(
        height=height, width="100%", bgcolor="#ffffff", font_color="black"
    )
    net.force_atlas_2based(
        gravity=-50, central_gravity=0.01,
        spring_length=100, spring_strength=0.08,
        damping=0.4, overlap=0,
    )
    for n in nodes:
        net.add_node(
            n["id"], label=n["label"],
            title=n.get("title", n["label"]),
            color=n["color"], size=n["size"], borderWidth=1,
        )
    for e in edges:
        net.add_edge(
            e["source"], e["target"],
            title=e["label"], label=e["label"],
            color="#bdc3c7", width=1, arrows="to",
        )
    try:
        path = os.path.join(tempfile.gettempdir(), "mea_kg_graph.html")
        net.save_graph(path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Graph Rendering Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🚀 MEA-KG Navigator")
mode = st.sidebar.radio(
    "Select Module:",
    [
        "📊 System Overview",
        "🔍 Semantic Search",
        "🕸️ Exploratory Query",
        "❓ Knowledge QA",
        "🧠 MEA-Reasoner (Graph-CoT)",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip**: The *MEA-Reasoner* module demonstrates the Graph-CoT "
    "mechanism from Section 2.3 of the paper — entity extraction → "
    "subgraph retrieval → grounded inference."
)
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 MEA-KG Project\nBridging Earth & Mars")

# Main Header
st.title("🪐 MEA-KG: Comparative Planetology Knowledge Graph")
st.markdown(
    "### An Automated Knowledge Graph Construction Framework for Comparative "
    "Planetology via Ontology-Guided Large Language Models"
)

# ===========================================================================
# MODULE 1: SYSTEM OVERVIEW
# ===========================================================================
if mode == "📊 System Overview":
    st.markdown("---")
    st.subheader("📈 Topological Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ontology Classes",    "4,367",  "Schema Layer")
    col2.metric("Entity Instances",    "12,362", "Data Layer")
    col3.metric("Semantic Relations",  "10,799", "Knowledge Links")

    st.success(
        "**Statistical Summary:** After full processing, MEA-KG contains "
        "**4,367** ontology class nodes, **12,362** entity nodes, and "
        "**10,799** semantic relations.  Through the automated mounting algorithm, "
        "the physical connection between the **Instance Layer** and the "
        "**Schema Layer** is successfully realised."
    )

    st.subheader("🌌 Global Graph Visualisation (Skeleton + Flesh)")
    st.markdown("Visualising the hierarchical organisation: **Red (Schema)** vs **Blue (Data)**.")

    if st.button("Load Global Graph (Max 2000 Nodes)"):
        with st.spinner("Fetching global topology data..."):
            cypher_schema   = "MATCH (n:n4sch__Class) RETURN n.n4sch__name AS id, 'Schema' AS type LIMIT 1000"
            cypher_instance = "MATCH (n:Instance) RETURN n.name AS id, 'Instance' AS type LIMIT 1000"
            cypher_rels     = """
                MATCH (i:Instance)-[r:INSTANCE_OF]->(c:n4sch__Class)
                RETURN i.name AS source, c.n4sch__name AS target, 'INSTANCE_OF' AS label
                LIMIT 1500
            """
            nodes_data, edges_data, seen_ids = [], [], set()
            for r in run_query(cypher_schema):
                nid = r["id"]
                if nid and nid not in seen_ids:
                    nodes_data.append({"id": nid, "label": nid, "color": "#D32F2F",
                                       "size": 25, "title": f"Class: {nid}", "group": "Schema"})
                    seen_ids.add(nid)
            for r in run_query(cypher_instance):
                nid = r["id"]
                if nid and nid not in seen_ids:
                    nodes_data.append({"id": nid, "label": nid, "color": "#1976D2",
                                       "size": 10, "title": f"Instance: {nid}", "group": "Instance"})
                    seen_ids.add(nid)
            for r in run_query(cypher_rels):
                if r["source"] in seen_ids and r["target"] in seen_ids:
                    edges_data.append(r)
            html = render_graph(nodes_data, edges_data, height="800px")
            if html:
                st.components.v1.html(html, height=810, scrolling=False)

# ===========================================================================
# MODULE 2: SEMANTIC SEARCH
# ===========================================================================
elif mode == "🔍 Semantic Search":
    st.subheader("Scientific Entity Search")
    query_term = st.text_input(
        "Enter keyword (e.g., Gale Crater, Hematite, Water):", "Gale Crater"
    )
    if query_term:
        with st.spinner(f"Searching for '{query_term}'..."):
            cypher_target = f"""
            MATCH (n:Instance) WHERE toLower(n.name) CONTAINS toLower('{query_term}')
            RETURN n.name AS id, labels(n) AS labels, properties(n) AS props LIMIT 1
            """
            target_res = run_query(cypher_target)
            if not target_res:
                st.warning(f"No entity found matching '{query_term}'.")
            else:
                target_node = target_res[0]
                center_id = target_node["id"]
                st.success(f"Found Entity: **{center_id}**")
                cypher_neighbors = f"""
                MATCH (center:Instance {{name: '{center_id}'}})-[r]-(neighbor)
                RETURN neighbor.name AS n_id, labels(neighbor) AS n_labels,
                       neighbor.n4sch__name AS class_name, type(r) AS r_type,
                       r.original_type AS r_original_type
                LIMIT 50
                """
                neighbors = run_query(cypher_neighbors)
                nodes = [{"id": center_id, "label": center_id, "color": "#FBC02D",
                          "size": 30, "title": "Target", "group": "Target"}]
                edges = []
                seen = {center_id}
                for row in neighbors:
                    n_id = row["n_id"] if row["n_id"] else row["class_name"]
                    if not n_id:
                        continue
                    n_type = "Schema" if "n4sch__Class" in row["n_labels"] else "Instance"
                    color  = "#D32F2F" if n_type == "Schema" else "#1976D2"
                    if n_id not in seen:
                        nodes.append({"id": n_id, "label": n_id, "color": color,
                                      "size": 15 if n_type == "Instance" else 20,
                                      "title": f"{n_type}: {n_id}", "group": n_type})
                        seen.add(n_id)
                    edge_label = row["r_original_type"] if row["r_original_type"] else row["r_type"]
                    edges.append({"source": center_id, "target": n_id, "label": edge_label})
                col1, col2 = st.columns([3, 1])
                with col1:
                    html = render_graph(nodes, edges)
                    if html:
                        st.components.v1.html(html, height=600)
                with col2:
                    st.markdown("### 📄 Metadata")
                    st.json(target_node["props"])
                    st.markdown(f"**{len(neighbors)}** connections.")

# ===========================================================================
# MODULE 3: EXPLORATORY QUERY
# ===========================================================================
elif mode == "🕸️ Exploratory Query":
    st.subheader("Filter by Entity Type")
    with st.spinner("Loading entity types..."):
        type_res  = run_query("MATCH (n:Instance) RETURN DISTINCT n.type AS t LIMIT 100")
        all_types = sorted([r["t"] for r in type_res if r["t"]])
    if not all_types:
        st.error("No entity types found.")
    else:
        selected_type = st.selectbox("Select an Entity Type:", all_types)
        if st.button(f"Visualise '{selected_type}' Network"):
            cypher = f"""
            MATCH (n:Instance {{type: '{selected_type}'}})
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n.name AS n_id, m.name AS m_id, m.n4sch__name AS m_class,
                   r.original_type AS r_orig, type(r) AS r_type
            LIMIT 50
            """
            data = run_query(cypher)
            if not data:
                st.warning("No connections found.")
            else:
                nodes, edges, seen = [], [], set()
                for row in data:
                    n_id = row["n_id"]
                    if n_id and n_id not in seen:
                        nodes.append({"id": n_id, "label": n_id, "color": "#009688",
                                      "size": 20, "group": selected_type})
                        seen.add(n_id)
                    m_id = row["m_id"] if row["m_id"] else row["m_class"]
                    if m_id:
                        if m_id not in seen:
                            nodes.append({"id": m_id, "label": m_id, "color": "#9E9E9E",
                                          "size": 10, "group": "Related"})
                            seen.add(m_id)
                        r_label = row["r_orig"] if row["r_orig"] else row["r_type"]
                        if not r_label:
                            r_label = "RELATED"
                        edges.append({"source": n_id, "target": m_id, "label": r_label})
                html = render_graph(nodes, edges)
                if html:
                    st.components.v1.html(html, height=600)
                    st.success(f"Visualising network for **{selected_type}** ({len(nodes)} nodes).")

# ===========================================================================
# MODULE 4: KNOWLEDGE QA
# ===========================================================================
elif mode == "❓ Knowledge QA":
    st.subheader("🤖 Knowledge-Based Question Answering (KBQA)")

    if not _LANGCHAIN_OK:
        st.error(
            f"LangChain is not available: {_LANGCHAIN_ERR}\n\n"
            "Install with: `pip install langchain langchain-community langchain-openai langchain-core`"
        )
        st.stop()

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
        }
        cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
        return simple_keywords.get(cleaned, None)

    if prompt := st.chat_input("Ask me anything about Mars, Earth analogs, or geology..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            fast_response = is_greeting_or_simple(prompt)
            if fast_response:
                st.write(fast_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": fast_response,
                     "thinking": "**Router:** `Level 1` - Instant Reply"}
                )
            else:
                status_container = st.status("🤔 Thinking...", expanded=True)
                try:
                    llm   = get_llm()
                    graph = get_graph()
                    if not llm or not graph:
                        raise Exception("LLM or Graph connection failed. Check API keys.")

                    status_container.write("🧠 Analysing intent...")
                    router_prompt = PromptTemplate.from_template(
                        "Determine if the question requires database search.\n"
                        "Question: {question}\n"
                        "Return 'SEARCH' if it asks for facts. Return 'ANSWER' if general chat. "
                        "No explanations."
                    )
                    router_chain = router_prompt | llm | StrOutputParser()
                    intent = router_chain.invoke({"question": prompt}).strip()
                    thinking_log = f"**Intent:** `{intent}`\n\n"

                    if "ANSWER" in intent:
                        status_container.update(label="✅ General Chat", state="complete", expanded=False)
                        response_chain = PromptTemplate.from_template("{question}") | llm | StrOutputParser()
                        answer = response_chain.invoke({"question": prompt})
                    else:
                        status_container.write("🔍 Generating Robust Cypher...")
                        schema_str = graph.schema
                        cypher_prompt = PromptTemplate.from_template(
                            "Task: Generate a FLEXIBLE Cypher query for Neo4j.\n"
                            "Schema:\n{schema}\n\n"
                            "**CRITICAL INSTRUCTIONS:**\n"
                            "1. **Fuzzy Matching:** ALWAYS use `toLower(n.name) CONTAINS toLower('keyword')`.\n"
                            "2. **Return Data:** Return `n.name`, `labels(n)`, and relationships.\n"
                            "3. **Limit:** Limit results to 10.\n\n"
                            "Question: {question}\n"
                            "Cypher Query:"
                        )
                        cypher_chain  = cypher_prompt | llm | StrOutputParser()
                        cypher_query  = cypher_chain.invoke({"question": prompt, "schema": schema_str})
                        cypher_query  = cypher_query.replace("```cypher", "").replace("```", "").strip()
                        thinking_log += f"**Generated Cypher:**\n```cypher\n{cypher_query}\n```\n\n"

                        status_container.write("⚙️ Executing query...")
                        context_data = graph.query(cypher_query)
                        thinking_log += f"**Database Result:**\n`{str(context_data)[:500]}...`\n\n"

                        status_container.write("💡 Synthesising...")
                        qa_prompt = PromptTemplate.from_template(
                            "Use the Context to answer.\n"
                            "If context is not empty, explicitly cite the entities found in the graph.\n"
                            "If context is empty, say 'Graph data not found, but generally...' and give "
                            "a scientific answer.\n\n"
                            "Context: {context}\nQuestion: {question}\nAnswer:"
                        )
                        qa_chain = qa_prompt | llm | StrOutputParser()
                        answer   = qa_chain.invoke({"question": prompt, "context": context_data})
                        status_container.update(label="✅ Search Complete", state="complete", expanded=False)

                    message_placeholder = st.empty()
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "thinking": thinking_log}
                    )

                except Exception as e:
                    status_container.update(label="❌ Error", state="error")
                    st.error(f"Error: {e}")

# ===========================================================================
# MODULE 5: MEA-REASONER (GRAPH-CoT)  — NEW
# ===========================================================================
elif mode == "🧠 MEA-Reasoner (Graph-CoT)":
    st.subheader("🧠 MEA-Reasoner: Graph Chain-of-Thought Inference")
    st.markdown(
        """
        This module demonstrates the **Graph-CoT** mechanism described in
        **Section 2.3** of the manuscript.  Unlike the standard KBQA module,
        Graph-CoT provides full *white-box* transparency:

        | Stage | Description |
        |-------|-------------|
        | **① Entity Extraction** | Identify domain entities from the query using an LLM |
        | **② Subgraph Retrieval** | Fuzzy-match entities against NEO4J; build evidence subgraph $G_K$ |
        | **③ Augmented Inference** | Synthesise a grounded, hallucination-free hypothesis |

        Enable the **"Ablation: w/o Analogy Rel"** toggle to reproduce the
        ablation study result from Section 3.3 (all Mars–Earth analogy
        relationships are suppressed during retrieval).
        """
    )
    st.markdown("---")

    if not _REASONER_OK:
        st.error(
            f"MEA-Reasoner could not be imported: {_REASONER_ERR}\n\n"
            "Ensure the `reasoner/` package is in the project root and all "
            "dependencies are installed."
        )
        st.stop()

    if not OPENAI_API_KEY:
        st.warning(
            "⚠️ `OPENAI_API_KEY` is not set.  The reasoner requires an "
            "OpenAI-compatible LLM.  Set the key in your `.env` file and "
            "restart the app."
        )

    # --- Controls ---
    col_q, col_opts = st.columns([3, 1])
    with col_q:
        user_query = st.text_area(
            "Enter your scientific question:",
            value=(
                "What polygonal terrain patterns exist in Utopia Planitia, "
                "and what Earth analogs help explain their formation?"
            ),
            height=100,
        )
    with col_opts:
        mask_analogy = st.toggle(
            "Ablation: w/o Analogy Rel",
            value=False,
            help=(
                "When enabled, Mars–Earth analogy relationships are hidden "
                "from the subgraph, reproducing the ablation condition in "
                "Section 3.3."
            ),
        )
        model_choice = st.selectbox(
            "LLM Model:",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )

    run_btn = st.button("▶ Run Graph-CoT", type="primary", use_container_width=True)

    if run_btn:
        if not user_query.strip():
            st.warning("Please enter a query.")
            st.stop()

        with st.spinner("Running Graph-CoT pipeline..."):
            try:
                reasoner = GraphCoT(model=model_choice)
                result: GraphCoTResult = reasoner.run(
                    user_query.strip(), mask_analogy=mask_analogy
                )
                reasoner.close()
            except Exception as exc:
                st.error(f"Graph-CoT Error: {exc}")
                st.stop()

        # ---- Stage 1: Entity Extraction ----
        st.markdown("---")
        st.markdown('<div class="cot-stage">', unsafe_allow_html=True)
        st.markdown("#### ① Entity Extraction")
        st.markdown("**Extracted entities** (Stage 1 LLM output):")
        if result.extracted_entities:
            st.write(", ".join(f"`{e}`" for e in result.extracted_entities))
        else:
            st.info("No entities extracted.")
        with st.expander("Raw Stage 1 LLM Output"):
            st.code(result.stage1_raw, language="json")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Stage 2: Subgraph Retrieval ----
        st.markdown('<div class="cot-stage">', unsafe_allow_html=True)
        st.markdown("#### ② Subgraph Retrieval")
        n_triplets = len(result.subgraph_triplets)
        if mask_analogy:
            st.warning(
                f"🔬 **Ablation mode active** — analogy relationships suppressed.  "
                f"{n_triplets} triplets retrieved."
            )
        else:
            st.success(f"{n_triplets} triplet(s) retrieved from MEA-KG.")

        if result.subgraph_triplets:
            # Build mini graph visualisation
            nodes_vis, edges_vis, seen_vis = [], [], set()
            for t in result.subgraph_triplets[:40]:
                for nid, color, size in [
                    (t.subject, "#FBC02D" if not t.is_analogy else "#FF7043", 20),
                    (t.obj,     "#1976D2",                                    15),
                ]:
                    if nid and nid not in seen_vis:
                        nodes_vis.append({"id": nid, "label": nid[:25],
                                          "color": color, "size": size,
                                          "title": nid})
                        seen_vis.add(nid)
                if t.subject and t.obj:
                    edges_vis.append({"source": t.subject, "target": t.obj,
                                      "label": t.relation[:20]})
            html_g = render_graph(nodes_vis, edges_vis, height="400px")
            if html_g:
                st.components.v1.html(html_g, height=415, scrolling=False)

            # Triplet table
            with st.expander(f"View all {n_triplets} triplets"):
                df = pd.DataFrame([
                    {
                        "Subject": t.subject,
                        "Relation": t.relation,
                        "Object": t.obj,
                        "Analogy?": "✅" if t.is_analogy else "",
                    }
                    for t in result.subgraph_triplets
                ])
                st.dataframe(df, use_container_width=True)
        else:
            st.info(
                "No graph triplets retrieved.  Ensure Neo4j is running and "
                "the knowledge graph has been populated."
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Stage 3: Grounded Hypothesis ----
        st.markdown('<div class="cot-stage">', unsafe_allow_html=True)
        st.markdown("#### ③ Grounded Hypothesis (Stage 3 Inference)")
        st.markdown(result.hypothesis)
        with st.expander("View full Stage 3 prompt (white-box)"):
            st.code(result.stage3_prompt, language="text")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Export ----
        st.markdown("---")
        export_data = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Download Result (JSON)",
            data=export_data,
            file_name="graph_cot_result.json",
            mime="application/json",
        )
