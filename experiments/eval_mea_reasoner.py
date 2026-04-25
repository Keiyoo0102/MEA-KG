"""
eval_mea_reasoner.py — Evaluation Script for MEA-Reasoner (Graph-CoT)

Computes the following metrics across three reasoning difficulty levels:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Metric                     │ Symbol        │ Paper Section     │
    ├─────────────────────────────────────────────────────────────────┤
    │  Semantic Coverage          │ Cov_sem       │ Section 3.2       │
    │  Win Rate (vs. baseline)    │ WinRate       │ Section 3.2       │
    │  Factual Accuracy           │ Acc_fact      │ Section 3.2       │
    └─────────────────────────────────────────────────────────────────┘

Reasoning Levels
----------------
    Level 1 — Direct Fact Retrieval (single-hop subgraph)
    Level 2 — Comparative Reasoning (multi-hop + analogy)
    Level 3 — Hypothesis Generation (complex, multi-entity)

Ablation Study
--------------
Run with ``--mask_analogy`` to reproduce the "w/o Analogy Rel" ablation
condition.  All Mars–Earth analogy edges are suppressed during subgraph
retrieval, allowing a direct comparison of performance with and without the
analogy relationship layer.

Evaluation Methodology
-----------------------
Answers are scored by an LLM judge following the TGRS LLM-as-Judge protocol
(blind A/B pairwise comparison).  The judge is prompted to evaluate:
    1. Semantic coverage of the reference answer.
    2. Factual grounding (triplet citation quality).
    3. Scientific coherence.

Usage
-----
    # Full evaluation against JSON dataset
    python experiments/eval_mea_reasoner.py --dataset data/dummy_sample/qa_eval_dataset.json

    # Ablation experiment
    python experiments/eval_mea_reasoner.py --dataset data/dummy_sample/qa_eval_dataset.json --mask_analogy

    # Compare two runs (pairwise win rate)
    python experiments/eval_mea_reasoner.py --compare results/full.json results/no_analogy.json

Environment Variables Required
-------------------------------
    OPENAI_API_KEY, OPENAI_BASE_URL — LLM judge credentials
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD — Knowledge graph access
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# Add project root to sys.path so we can import ``reasoner``
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reasoner import GraphCoT, GraphCoTResult  # noqa: E402

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class QAItem:
    """A single QA evaluation item."""
    id: str
    level: int          # 1, 2, or 3
    query: str
    reference: str      # ground-truth / reference answer
    baseline_answer: str = ""   # answer from baseline (no graph)


@dataclass
class EvalResult:
    """Evaluation result for a single QA item."""
    item_id: str
    level: int
    query: str
    reference: str
    model_answer: str
    baseline_answer: str = ""
    semantic_coverage: float = 0.0   # Cov_sem ∈ [0, 1]
    factual_accuracy: float = 0.0    # Acc_fact ∈ [0, 1]
    win_vs_baseline: str = "N/A"     # "WIN" | "TIE" | "LOSS" | "N/A"
    judge_rationale: str = ""


@dataclass
class LevelSummary:
    """Aggregated metrics for one reasoning level."""
    level: int
    n: int
    cov_sem_mean: float
    acc_fact_mean: float
    win_rate: float        # fraction of "WIN" outcomes
    tie_rate: float
    loss_rate: float


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """OpenAI-compatible LLM judge implementing the TGRS LLM-as-Judge protocol.

    The judge evaluates model answers against a reference answer on:
        (1) Semantic coverage  — what fraction of key reference concepts are covered.
        (2) Factual accuracy   — are stated facts grounded in graph evidence.
        (3) Pairwise win rate  — is the model answer better than the baseline.

    All evaluations are *blind*: the judge does not know which system produced
    which answer in the pairwise comparison.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package required. Run: pip install openai")
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def _call(self, system: str, user: str) -> str:
        """Single LLM call with simple retry."""
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("Judge API call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        return ""

    def score_semantic_coverage(self, reference: str, candidate: str) -> tuple[float, str]:
        """Rate how well *candidate* covers the key concepts in *reference*.

        Returns
        -------
        score : float
            Semantic coverage score in [0.0, 1.0].
        rationale : str
            Brief justification.
        """
        system = (
            "You are an objective scientific evaluator.  Rate the semantic coverage "
            "of the candidate answer relative to the reference answer on a scale "
            "from 0.0 (completely misses) to 1.0 (fully covers all key concepts).\n\n"
            "Return ONLY valid JSON with keys 'score' (float) and 'rationale' (str)."
        )
        user = (
            f"Reference answer:\n{reference}\n\n"
            f"Candidate answer:\n{candidate}\n\n"
            "JSON:"
        )
        raw = self._call(system, user)
        try:
            data = json.loads(raw)
            return float(data.get("score", 0.0)), str(data.get("rationale", ""))
        except Exception:
            # Attempt to extract a float from the raw string
            import re
            nums = re.findall(r"\d+\.\d+|\d+", raw)
            score = float(nums[0]) if nums else 0.0
            return min(max(score, 0.0), 1.0), raw

    def score_factual_accuracy(self, candidate: str, triplets: list[dict]) -> tuple[float, str]:
        """Score whether *candidate* makes accurate claims supported by graph triplets.

        Parameters
        ----------
        candidate : str
            Model-generated answer.
        triplets : list[dict]
            Triplets from Stage 2 (each has 'subject', 'relation', 'object').

        Returns
        -------
        score : float
            Factual accuracy score in [0.0, 1.0].
        rationale : str
        """
        triplet_str = "\n".join(
            f"  • ({t['subject']}) --[{t['relation']}]--> ({t['object']})"
            for t in triplets[:30]
        )
        system = (
            "You are an objective scientific fact-checker.  "
            "Given a set of knowledge graph triplets and a candidate answer, "
            "rate the factual accuracy of the answer (how well claims are supported "
            "by the triplets) on a scale from 0.0 to 1.0.\n\n"
            "Return ONLY valid JSON with keys 'score' (float) and 'rationale' (str)."
        )
        user = (
            f"Knowledge Graph Triplets:\n{triplet_str or '[none]'}\n\n"
            f"Candidate Answer:\n{candidate}\n\nJSON:"
        )
        raw = self._call(system, user)
        try:
            data = json.loads(raw)
            return float(data.get("score", 0.0)), str(data.get("rationale", ""))
        except Exception:
            import re
            nums = re.findall(r"\d+\.\d+|\d+", raw)
            score = float(nums[0]) if nums else 0.0
            return min(max(score, 0.0), 1.0), raw

    def pairwise_compare(
        self,
        query: str,
        answer_a: str,
        answer_b: str,
        reference: str,
    ) -> tuple[str, str]:
        """Blind pairwise comparison: which answer is better?

        Returns
        -------
        verdict : str
            "A" | "B" | "TIE"
        rationale : str
        """
        system = (
            "You are an impartial judge evaluating two scientific answers to the "
            "same question.  Evaluate based on: (1) factual accuracy, (2) relevance "
            "to the question, (3) scientific depth.  Do NOT consider answer length.\n\n"
            "Return ONLY valid JSON with keys 'winner' ('A', 'B', or 'TIE') and "
            "'rationale' (str)."
        )
        user = (
            f"Question: {query}\n\n"
            f"Reference Answer:\n{reference}\n\n"
            f"Answer A:\n{answer_a}\n\n"
            f"Answer B:\n{answer_b}\n\nJSON:"
        )
        raw = self._call(system, user)
        try:
            data = json.loads(raw)
            winner = str(data.get("winner", "TIE")).upper()
            if winner not in {"A", "B", "TIE"}:
                winner = "TIE"
            return winner, str(data.get("rationale", ""))
        except Exception:
            return "TIE", raw


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

class MEAReasonerEvaluator:
    """Orchestrates the full evaluation pipeline.

    Parameters
    ----------
    mask_analogy : bool
        Ablation flag passed to ``GraphCoT.run()``.
    model : str
        LLM model for both the reasoner and the judge.
    max_items : int
        Cap the number of items evaluated (for quick smoke tests).
    """

    def __init__(
        self,
        mask_analogy: bool = False,
        model: str = "gpt-4o-mini",
        max_items: int | None = None,
    ) -> None:
        self.mask_analogy = mask_analogy
        self.model = model
        self.max_items = max_items
        self.reasoner = GraphCoT(model=model)
        self.judge = LLMJudge(model=model)

    def load_dataset(self, path: str | Path) -> list[QAItem]:
        """Load QA evaluation dataset from a JSON file.

        Expected format::

            [
              {
                "id": "L1_001",
                "level": 1,
                "query": "What minerals are found near Gale Crater?",
                "reference": "Jarosite, hematite, and sulfates ...",
                "baseline_answer": "Gale Crater contains various minerals ..."
              },
              ...
            ]
        """
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        items: list[QAItem] = []
        for r in raw:
            items.append(
                QAItem(
                    id=str(r.get("id", len(items))),
                    level=int(r.get("level", 1)),
                    query=str(r["query"]),
                    reference=str(r.get("reference", "")),
                    baseline_answer=str(r.get("baseline_answer", "")),
                )
            )
        logger.info("Loaded %d QA items from %s", len(items), path)
        return items

    def evaluate_item(self, item: QAItem) -> EvalResult:
        """Run Graph-CoT + judge scoring for a single QA item."""
        logger.info(
            "Evaluating [%s] Level %d: %s",
            item.id, item.level, item.query[:80]
        )

        # --- Graph-CoT inference ---
        try:
            cot_result: GraphCoTResult = self.reasoner.run(
                item.query, mask_analogy=self.mask_analogy
            )
            model_answer = cot_result.hypothesis
            triplets_dict = [t.to_dict() if hasattr(t, "to_dict") else
                             {"subject": t.subject, "relation": t.relation, "object": t.obj}
                             for t in cot_result.subgraph_triplets]
        except Exception as exc:
            logger.error("Graph-CoT inference failed for %s: %s", item.id, exc)
            model_answer = f"[Error: {exc}]"
            triplets_dict = []

        # --- Judge: Semantic Coverage ---
        cov, cov_rationale = self.judge.score_semantic_coverage(
            reference=item.reference, candidate=model_answer
        )

        # --- Judge: Factual Accuracy ---
        acc, acc_rationale = self.judge.score_factual_accuracy(
            candidate=model_answer, triplets=triplets_dict
        )

        # --- Judge: Pairwise Win Rate (if baseline provided) ---
        win_label = "N/A"
        pairwise_rationale = ""
        if item.baseline_answer:
            winner, pairwise_rationale = self.judge.pairwise_compare(
                query=item.query,
                answer_a=model_answer,
                answer_b=item.baseline_answer,
                reference=item.reference,
            )
            win_label = "WIN" if winner == "A" else ("TIE" if winner == "TIE" else "LOSS")

        combined_rationale = (
            f"[Cov] {cov_rationale}\n"
            f"[Acc] {acc_rationale}\n"
            f"[Pairwise] {pairwise_rationale}"
        )

        return EvalResult(
            item_id=item.id,
            level=item.level,
            query=item.query,
            reference=item.reference,
            model_answer=model_answer,
            baseline_answer=item.baseline_answer,
            semantic_coverage=cov,
            factual_accuracy=acc,
            win_vs_baseline=win_label,
            judge_rationale=combined_rationale,
        )

    def run(self, dataset_path: str | Path) -> tuple[list[EvalResult], list[LevelSummary]]:
        """Evaluate all items and return per-item results + level summaries.

        Parameters
        ----------
        dataset_path : str or Path
            Path to the JSON evaluation dataset.

        Returns
        -------
        results : list[EvalResult]
        summaries : list[LevelSummary]
        """
        items = self.load_dataset(dataset_path)
        if self.max_items:
            items = items[: self.max_items]

        results: list[EvalResult] = []
        for item in items:
            res = self.evaluate_item(item)
            results.append(res)

        summaries = self._summarise(results)
        return results, summaries

    @staticmethod
    def _summarise(results: list[EvalResult]) -> list[LevelSummary]:
        """Aggregate metrics by reasoning level."""
        from collections import defaultdict
        buckets: dict[int, list[EvalResult]] = defaultdict(list)
        for r in results:
            buckets[r.level].append(r)

        summaries: list[LevelSummary] = []
        for lvl in sorted(buckets.keys()):
            group = buckets[lvl]
            n = len(group)
            cov_mean = sum(r.semantic_coverage for r in group) / n
            acc_mean = sum(r.factual_accuracy for r in group) / n
            wins  = sum(1 for r in group if r.win_vs_baseline == "WIN")
            ties  = sum(1 for r in group if r.win_vs_baseline == "TIE")
            losses= sum(1 for r in group if r.win_vs_baseline == "LOSS")
            valid = wins + ties + losses  # items with baseline
            summaries.append(
                LevelSummary(
                    level=lvl,
                    n=n,
                    cov_sem_mean=round(cov_mean, 4),
                    acc_fact_mean=round(acc_mean, 4),
                    win_rate=round(wins / valid, 4) if valid else 0.0,
                    tie_rate=round(ties / valid, 4) if valid else 0.0,
                    loss_rate=round(losses / valid, 4) if valid else 0.0,
                )
            )
        return summaries

    def save_results(
        self,
        results: list[EvalResult],
        summaries: list[LevelSummary],
        output_dir: str | Path,
    ) -> None:
        """Save detailed results and summary tables to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        suffix = "_no_analogy" if self.mask_analogy else "_full"

        # Detailed JSON
        detail_path = out / f"eval_detail{suffix}.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "id": r.item_id,
                        "level": r.level,
                        "query": r.query,
                        "model_answer": r.model_answer,
                        "semantic_coverage": r.semantic_coverage,
                        "factual_accuracy": r.factual_accuracy,
                        "win_vs_baseline": r.win_vs_baseline,
                        "judge_rationale": r.judge_rationale,
                    }
                    for r in results
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Detailed results saved → %s", detail_path)

        # Summary table (text)
        summary_path = out / f"eval_summary{suffix}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            header = (
                f"MEA-Reasoner Evaluation Summary\n"
                f"Ablation (mask_analogy): {self.mask_analogy}\n"
                f"{'='*60}\n"
                f"{'Level':<8} {'N':>5} {'Cov_sem':>10} {'Acc_fact':>10} "
                f"{'WinRate':>10} {'TieRate':>9} {'LossRate':>10}\n"
                f"{'-'*60}\n"
            )
            f.write(header)
            for s in summaries:
                f.write(
                    f"L{s.level:<7} {s.n:>5} {s.cov_sem_mean:>10.4f} "
                    f"{s.acc_fact_mean:>10.4f} {s.win_rate:>10.4f} "
                    f"{s.tie_rate:>9.4f} {s.loss_rate:>10.4f}\n"
                )
        logger.info("Summary saved → %s", summary_path)

        # Print to stdout as well
        print(open(summary_path).read())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MEA-Reasoner Evaluation — Graph-CoT performance on the "
            "planetary geology QA benchmark."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dummy_sample/qa_eval_dataset.json",
        help="Path to the JSON evaluation dataset (default: data/dummy_sample/qa_eval_dataset.json).",
    )
    parser.add_argument(
        "--mask_analogy",
        action="store_true",
        help="Ablation: suppress Mars–Earth analogy relationships (reproduces 'w/o Analogy Rel').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for both reasoning and judging (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/eval_results",
        help="Directory to save evaluation output (default: data/eval_results).",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Evaluate only the first N items (useful for quick tests).",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RESULT_A", "RESULT_B"),
        help="Compare two saved eval_detail*.json files and report pairwise win rates.",
    )
    return parser.parse_args()


def _compare_results(path_a: str, path_b: str) -> None:
    """Print a comparison table between two saved evaluation JSON files."""
    with open(path_a, encoding="utf-8") as f:
        data_a: list[dict] = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        data_b: list[dict] = json.load(f)

    assert len(data_a) == len(data_b), "Result files must contain the same number of items."

    wins_a = ties = wins_b = 0
    for a, b in zip(data_a, data_b):
        cov_a = a["semantic_coverage"]
        cov_b = b["semantic_coverage"]
        if cov_a > cov_b + 0.05:
            wins_a += 1
        elif cov_b > cov_a + 0.05:
            wins_b += 1
        else:
            ties += 1

    n = len(data_a)
    print(f"\n{'='*50}")
    print("Pairwise Comparison (Semantic Coverage)")
    print(f"  File A: {path_a}")
    print(f"  File B: {path_b}")
    print(f"  n = {n}")
    print(f"  A wins : {wins_a} ({100*wins_a/n:.1f}%)")
    print(f"  Ties   : {ties}  ({100*ties/n:.1f}%)")
    print(f"  B wins : {wins_b} ({100*wins_b/n:.1f}%)")
    print(f"{'='*50}\n")


def main() -> None:
    args = _parse_args()

    if args.compare:
        _compare_results(args.compare[0], args.compare[1])
        return

    evaluator = MEAReasonerEvaluator(
        mask_analogy=args.mask_analogy,
        model=args.model,
        max_items=args.max_items,
    )

    dataset_path = Path(PROJECT_ROOT) / args.dataset
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        logger.info(
            "Tip: use the dummy dataset at data/dummy_sample/qa_eval_dataset.json "
            "for a quick smoke test."
        )
        sys.exit(1)

    results, summaries = evaluator.run(dataset_path)
    evaluator.save_results(results, summaries, Path(PROJECT_ROOT) / args.output_dir)


if __name__ == "__main__":
    main()
