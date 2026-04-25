"""
reasoner — MEA-Reasoner package.

The MEA-Reasoner implements Graph Chain-of-Thought (Graph-CoT), a three-stage
white-box reasoning mechanism that grounds LLM inference in the MEA-KG
knowledge graph, eliminating hallucination for planetary geology QA.

Stages
------
1. Entity Extraction  : identify domain entities from a natural-language query.
2. Subgraph Retrieval : retrieve relevant subgraph paths from Neo4j.
3. Augmented Inference: synthesise a grounded hypothesis using an LLM.

Public API
----------
    from reasoner import GraphCoT
    reasoner = GraphCoT()
    result = reasoner.run("What polygonal terrain patterns exist in Utopia Planitia?")
"""

from .graph_cot import GraphCoT, GraphCoTResult  # noqa: F401

__all__ = ["GraphCoT", "GraphCoTResult"]
