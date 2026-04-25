"""
graph_cot.py — Graph Chain-of-Thought (Graph-CoT) Reasoner for MEA-KG

Overview
--------
The MEA-Reasoner is the core inference engine of the MEA-KG project.  It
implements a *three-stage* Graph Chain-of-Thought mechanism:

    Stage 1 – Entity Extraction (E)
        Extract candidate domain entities from the user query K using an LLM.
        This converts a free-form question into a structured set of search terms.

    Stage 2 – Subgraph Retrieval (F)
        For each extracted entity, perform *fuzzy-match* Cypher queries against
        the Neo4j knowledge graph to retrieve relevant triplet paths
        (subject → relation → object).  This forms the evidence subgraph G_K.

    Stage 3 – Augmented Logic Inference
        Feed the original query K together with the evidence subgraph G_K to an
        LLM.  The LLM synthesises a grounded, hallucination-free hypothesis H
        that explicitly cites graph-derived facts.

When the flag ``mask_analogy=True`` is passed to ``GraphCoT.run()``, all
Mars–Earth analogy relationships are hidden from the subgraph before inference.
This reproduces the *"w/o Analogy Rel"* ablation condition described in the
manuscript (Section 3.3).

References
----------
- Paper Section 2.3: MEA-Reasoner / Graph-CoT Mechanism
- Paper Section 3.3: Ablation Study on Analogy Relationships

Usage
-----
    # Standard inference
    from reasoner import GraphCoT
    reasoner = GraphCoT()
    result = reasoner.run("What polygonal terrain patterns exist in Utopia Planitia?")
    print(result.hypothesis)

    # Ablation: mask analogy relationships
    result_no_analogy = reasoner.run(query, mask_analogy=True)

Environment Variables Required
-------------------------------
    NEO4J_URI        - Neo4j Bolt URI   (default: bolt://localhost:7687)
    NEO4J_USER       - Neo4j username   (default: neo4j)
    NEO4J_PASSWORD   - Neo4j password
    OPENAI_API_KEY   - LLM API key
    OPENAI_BASE_URL  - LLM API base URL (default: https://api.openai.com/v1)
"""

from __future__ import annotations

import os
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Analogy relation types that are masked in the ablation experiment.
# Extend this list to match your graph schema.
# ---------------------------------------------------------------------------
_ANALOGY_REL_TYPES: set[str] = {
    "ANALOG_OF",
    "MARS_EARTH_ANALOG",
    "ANALOGOUS_TO",
    "STRUCTURAL_ANALOG",
    "PROCESS_ANALOG",
}

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class KGTriplet:
    """A single subject–relation–object triplet retrieved from the knowledge graph."""
    subject: str
    relation: str
    obj: str
    is_analogy: bool = False

    def to_text(self) -> str:
        """Render the triplet as a natural-language triple string."""
        return f"({self.subject}) --[{self.relation}]--> ({self.obj})"


@dataclass
class GraphCoTResult:
    """Container for the complete Graph-CoT inference result.

    Attributes
    ----------
    query : str
        The original natural-language query.
    extracted_entities : list[str]
        Entities identified in Stage 1.
    subgraph_triplets : list[KGTriplet]
        Triplets retrieved in Stage 2 (analogy triplets excluded if
        ``mask_analogy=True``).
    hypothesis : str
        The grounded hypothesis produced in Stage 3.
    mask_analogy : bool
        Whether the analogy ablation was active.
    stage1_raw : str
        Raw LLM output from Stage 1 (for debugging).
    stage3_prompt : str
        The final prompt submitted to Stage 3 (for white-box inspection).
    """
    query: str
    extracted_entities: list[str] = field(default_factory=list)
    subgraph_triplets: list[KGTriplet] = field(default_factory=list)
    hypothesis: str = ""
    mask_analogy: bool = False
    stage1_raw: str = ""
    stage3_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dictionary (JSON-friendly)."""
        return {
            "query": self.query,
            "extracted_entities": self.extracted_entities,
            "subgraph_triplets": [
                {"subject": t.subject, "relation": t.relation, "object": t.obj,
                 "is_analogy": t.is_analogy}
                for t in self.subgraph_triplets
            ],
            "hypothesis": self.hypothesis,
            "mask_analogy": self.mask_analogy,
        }


# ---------------------------------------------------------------------------
# Core Reasoner
# ---------------------------------------------------------------------------

class GraphCoT:
    """Three-stage Graph Chain-of-Thought reasoner.

    Parameters
    ----------
    neo4j_uri : str, optional
        Bolt URI for the Neo4j instance.  Defaults to ``NEO4J_URI`` env var.
    neo4j_user : str, optional
        Neo4j username.  Defaults to ``NEO4J_USER`` env var.
    neo4j_password : str, optional
        Neo4j password.  Defaults to ``NEO4J_PASSWORD`` env var.
    openai_api_key : str, optional
        OpenAI-compatible API key.  Defaults to ``OPENAI_API_KEY`` env var.
    openai_base_url : str, optional
        OpenAI-compatible base URL.  Defaults to ``OPENAI_BASE_URL`` env var.
    model : str, optional
        LLM model identifier (default ``"gpt-4o-mini"``).
    max_triplets : int, optional
        Maximum number of subgraph triplets to retrieve per entity (default 20).
    """

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        model: str = "gpt-4o-mini",
        max_triplets: int = 20,
    ) -> None:
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = model
        self.max_triplets = max_triplets

        self._driver = None
        self._llm_client = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_driver(self):
        """Lazily initialise and return the Neo4j driver."""
        if not _NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package is not installed. Run: pip install neo4j"
            )
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            self._driver.verify_connectivity()
            logger.info("Neo4j connection established at %s", self.neo4j_uri)
        return self._driver

    def _get_llm(self) -> "OpenAI":
        """Lazily initialise and return the OpenAI-compatible client."""
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is not installed. Run: pip install openai"
            )
        if self._llm_client is None:
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set.  "
                    "Export it in your shell or add it to .env."
                )
            self._llm_client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
            )
        return self._llm_client

    def close(self) -> None:
        """Close open connections (Neo4j driver)."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Stage 1 — Entity Extraction
    # ------------------------------------------------------------------

    def _stage1_extract_entities(self, query: str) -> tuple[list[str], str]:
        """Extract domain entities from *query* using an LLM.

        Returns
        -------
        entities : list[str]
            Extracted entity keywords.
        raw_output : str
            Raw LLM response for logging / debugging.
        """
        system_prompt = (
            "You are a planetary geology expert specialising in Mars–Earth "
            "comparative analysis.  Given a scientific question, extract the key "
            "named entities (geological features, minerals, processes, locations, "
            "morphological patterns) that should be looked up in a knowledge graph.\n\n"
            "Return ONLY a JSON array of strings, e.g.:\n"
            '["Utopia Planitia", "polygonal terrain", "thermal contraction", "permafrost"]\n'
            "No explanation, no markdown, just valid JSON."
        )
        user_message = f"Question: {query}"

        llm = self._get_llm()
        response = llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        logger.debug("Stage 1 raw output: %s", raw)

        # Parse JSON array; fall back to simple comma-split on parse error
        try:
            entities: list[str] = json.loads(raw)
            if not isinstance(entities, list):
                raise ValueError("Expected a JSON array")
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Stage 1: JSON parse failed; falling back to line/comma split."
            )
            # Strip common wrappers and split
            cleaned = re.sub(r"```[a-z]*|```", "", raw)
            entities = [e.strip().strip('"') for e in re.split(r"[,\n]+", cleaned) if e.strip()]

        logger.info("Stage 1 extracted %d entities: %s", len(entities), entities)
        return entities, raw

    # ------------------------------------------------------------------
    # Stage 2 — Subgraph Retrieval
    # ------------------------------------------------------------------

    def _stage2_retrieve_subgraph(
        self, entities: list[str], mask_analogy: bool = False
    ) -> list[KGTriplet]:
        """Retrieve relevant triplets from Neo4j for each extracted entity.

        Parameters
        ----------
        entities : list[str]
            Entity keywords from Stage 1.
        mask_analogy : bool
            If ``True``, filter out all Mars–Earth analogy relationship types.

        Returns
        -------
        list[KGTriplet]
            Deduplicated list of triplets forming the evidence subgraph.
        """
        driver = self._get_driver()

        # Cypher query: fuzzy-match on entity name, return 1-hop paths
        cypher = """
        MATCH (n:Instance)-[r]->(m)
        WHERE toLower(n.name) CONTAINS toLower($keyword)
           OR toLower(m.name) CONTAINS toLower($keyword)
        RETURN
            n.name        AS subject,
            type(r)       AS relation,
            COALESCE(r.original_type, type(r)) AS relation_label,
            COALESCE(m.name, m.n4sch__name, toString(id(m))) AS object
        LIMIT $limit
        """

        seen: set[tuple[str, str, str]] = set()
        triplets: list[KGTriplet] = []

        with driver.session() as session:
            for keyword in entities:
                try:
                    records = session.run(
                        cypher,
                        keyword=keyword,
                        limit=self.max_triplets,
                    )
                    for rec in records:
                        subj = rec["subject"] or ""
                        rel = rec["relation_label"] or rec["relation"] or ""
                        obj = rec["object"] or ""

                        if not (subj and rel and obj):
                            continue

                        is_analogy = rel.upper() in _ANALOGY_REL_TYPES
                        if mask_analogy and is_analogy:
                            continue  # ablation: hide analogy edges

                        key = (subj, rel, obj)
                        if key not in seen:
                            seen.add(key)
                            triplets.append(
                                KGTriplet(
                                    subject=subj,
                                    relation=rel,
                                    obj=obj,
                                    is_analogy=is_analogy,
                                )
                            )
                except Exception as exc:
                    logger.warning(
                        "Subgraph retrieval failed for keyword '%s': %s",
                        keyword, exc,
                    )

        logger.info(
            "Stage 2 retrieved %d triplets (mask_analogy=%s)",
            len(triplets), mask_analogy,
        )
        return triplets

    # ------------------------------------------------------------------
    # Stage 3 — Augmented Logic Inference
    # ------------------------------------------------------------------

    def _stage3_augmented_inference(
        self,
        query: str,
        triplets: list[KGTriplet],
        mask_analogy: bool = False,
    ) -> tuple[str, str]:
        """Generate a grounded hypothesis from the query and evidence subgraph.

        Parameters
        ----------
        query : str
            Original user question.
        triplets : list[KGTriplet]
            Evidence subgraph from Stage 2.
        mask_analogy : bool
            Passed through for prompt transparency.

        Returns
        -------
        hypothesis : str
            The final grounded answer.
        prompt_used : str
            The complete prompt string (for white-box inspection).
        """
        if triplets:
            graph_context = "\n".join(f"  • {t.to_text()}" for t in triplets)
        else:
            graph_context = "  [No relevant subgraph triplets found in MEA-KG]"

        ablation_note = (
            "\n[ABLATION MODE: Mars–Earth analogy relationships have been masked.]\n"
            if mask_analogy
            else ""
        )

        system_prompt = (
            "You are a senior planetary geologist with expertise in Mars–Earth "
            "comparative analysis.  Your answers MUST be grounded in the provided "
            "knowledge graph evidence.  Cite specific entities and relationships "
            "from the graph context.  Do not speculate beyond what the graph supports; "
            "if the graph is silent on a point, say so explicitly."
        )

        user_prompt = f"""Scientific Question:
{query}
{ablation_note}
Knowledge Graph Evidence (MEA-KG subgraph):
{graph_context}

Task: Using the graph evidence above as your primary source, synthesise a concise,
scientifically rigorous hypothesis or answer.  Structure your response as:

**[Grounded Hypothesis]**
<your answer, citing graph entities in the format (EntityName)>

**[Reasoning Chain]**
<step-by-step logical derivation referencing specific graph triplets>

**[Confidence & Limitations]**
<brief note on what the graph confirms vs. what requires additional evidence>
"""

        full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"

        llm = self._get_llm()
        response = llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        hypothesis = response.choices[0].message.content.strip()
        logger.info("Stage 3 inference complete (%d chars).", len(hypothesis))
        return hypothesis, full_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, mask_analogy: bool = False) -> GraphCoTResult:
        """Execute the full three-stage Graph-CoT pipeline.

        Parameters
        ----------
        query : str
            Natural-language question about Mars / Earth geology.
        mask_analogy : bool
            If ``True``, suppress analogy edges during subgraph retrieval
            (reproduces the ablation study condition "w/o Analogy Rel").

        Returns
        -------
        GraphCoTResult
            Structured result containing extracted entities, evidence subgraph,
            and the final grounded hypothesis.
        """
        logger.info("=== Graph-CoT START | mask_analogy=%s ===", mask_analogy)
        logger.info("Query: %s", query)

        # Stage 1
        entities, stage1_raw = self._stage1_extract_entities(query)

        # Stage 2
        triplets = self._stage2_retrieve_subgraph(entities, mask_analogy=mask_analogy)

        # Stage 3
        hypothesis, prompt_used = self._stage3_augmented_inference(
            query, triplets, mask_analogy=mask_analogy
        )

        logger.info("=== Graph-CoT END ===")
        return GraphCoTResult(
            query=query,
            extracted_entities=entities,
            subgraph_triplets=triplets,
            hypothesis=hypothesis,
            mask_analogy=mask_analogy,
            stage1_raw=stage1_raw,
            stage3_prompt=prompt_used,
        )

    # ------------------------------------------------------------------
    # Demo / standalone execution
    # ------------------------------------------------------------------

    @staticmethod
    def demo(query: str | None = None, mask_analogy: bool = False) -> None:
        """Quick smoke-test that prints the Graph-CoT result to stdout.

        Parameters
        ----------
        query : str, optional
            Custom query.  Defaults to a Utopia Planitia polygon question.
        mask_analogy : bool
            Ablation flag.
        """
        q = query or (
            "What are the polygonal terrain patterns in Utopia Planitia, "
            "and what Earth analogs help explain their formation?"
        )
        print(f"\n{'='*70}")
        print("MEA-Reasoner Graph-CoT Demo")
        print(f"Query : {q}")
        print(f"Ablation (mask_analogy): {mask_analogy}")
        print("=" * 70)

        reasoner = GraphCoT()
        try:
            result = reasoner.run(q, mask_analogy=mask_analogy)
        finally:
            reasoner.close()

        print("\n--- Stage 1: Extracted Entities ---")
        for e in result.extracted_entities:
            print(f"  • {e}")

        print(f"\n--- Stage 2: Subgraph Triplets ({len(result.subgraph_triplets)} found) ---")
        for t in result.subgraph_triplets[:10]:
            print(f"  {t.to_text()}" + (" [ANALOGY]" if t.is_analogy else ""))
        if len(result.subgraph_triplets) > 10:
            print(f"  ... ({len(result.subgraph_triplets) - 10} more)")

        print("\n--- Stage 3: Grounded Hypothesis ---")
        print(result.hypothesis)
        print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MEA-Reasoner Graph-CoT — standalone demo"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Scientific question to answer (default: Utopia Planitia polygons).",
    )
    parser.add_argument(
        "--mask_analogy",
        action="store_true",
        help="Ablation mode: suppress Mars–Earth analogy relationships.",
    )
    args = parser.parse_args()
    GraphCoT.demo(query=args.query, mask_analogy=args.mask_analogy)
