import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow imports from the project root when this file is inside evaluation/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from document_pipeline import process_document, search_document
from rag import generate_answer, calculate_grounding_score


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reciprocal_rank(results, expected_pages):
    for rank, result in enumerate(results, start=1):
        page = result.get("page")
        if page in expected_pages:
            return 1.0 / rank
    return 0.0


def hit_at_k(results, expected_pages, k):
    return int(any(r.get("page") in expected_pages for r in results[:k]))


def evaluate_retrieval(questions, index, bm25_index, chunks):
    rows = []

    for q in questions:
        start = time.perf_counter()

        results = search_document(
            q["question"],
            index,
            bm25_index,
            chunks,
            top_k=5
        )

        elapsed = time.perf_counter() - start
        expected_pages = set(q.get("expected_pages", []))

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "expected_pages": sorted(expected_pages),
            "retrieved_pages": [r.get("page") for r in results],
            "hit_at_1": hit_at_k(results, expected_pages, 1),
            "hit_at_3": hit_at_k(results, expected_pages, 3),
            "hit_at_5": hit_at_k(results, expected_pages, 5),
            "mrr": reciprocal_rank(results, expected_pages),
            "retrieval_time_sec": round(elapsed, 4),
        })

    return rows


def evaluate_rag(questions, retrieval_rows, index, bm25_index, chunks):
    by_id = {r["id"]: r for r in retrieval_rows}
    rows = []

    for q in questions:
        start = time.perf_counter()

        results = search_document(
            q["question"],
            index,
            bm25_index,
            chunks,
            top_k=3
        )

        answer = generate_answer(q["question"], results)

        try:
            grounding = calculate_grounding_score(
                q["question"],
                answer,
                results
            )
        except Exception:
            grounding = None

        elapsed = time.perf_counter() - start

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "answer": answer,
            "retrieved_pages": [r.get("page") for r in results],
            "grounding": grounding,
            "total_time_sec": round(elapsed, 3),
        })

    return rows


def summarize_retrieval(rows):
    def avg(key):
        values = [r[key] for r in rows]
        return round(sum(values) / len(values) * 100, 2)

    return {
        "questions": len(rows),
        "hit_rate_at_1_percent": avg("hit_at_1"),
        "recall_at_3_percent": avg("hit_at_3"),
        "recall_at_5_percent": avg("hit_at_5"),
        "mrr_percent": avg("mrr"),
        "avg_retrieval_time_sec": round(
            sum(r["retrieval_time_sec"] for r in rows) / len(rows), 4
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the document retrieval/RAG pipeline."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the evaluation PDF."
    )
    parser.add_argument(
        "--questions",
        default="evaluation/test_questions.json",
        help="Path to test_questions.json."
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Also run the local LLM RAG evaluation. This is slower."
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    questions_path = Path(args.questions)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not questions_path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_path}"
        )

    questions = load_questions(questions_path)

    print("=" * 60)
    print("MULTIMODAL DOCUMENT INTELLIGENCE - EVALUATION")
    print("=" * 60)
    print(f"PDF: {pdf_path.name}")
    print(f"Questions: {len(questions)}")
    print()

    print("Processing document...")
    process_start = time.perf_counter()

    pdf_bytes = pdf_path.read_bytes()
    chunks, embeddings, index, bm25_index, tables = process_document(
        pdf_bytes,
        pdf_path.name
    )

    processing_time = time.perf_counter() - process_start

    print(f"Chunks: {len(chunks)}")
    print(f"Tables: {len(tables)}")
    print(f"Vectors: {index.ntotal}")
    print(f"Processing time: {processing_time:.2f}s")
    print()

    print("Running retrieval evaluation...")
    retrieval_rows = evaluate_retrieval(
        questions,
        index,
        bm25_index,
        chunks
    )

    summary = summarize_retrieval(retrieval_rows)

    print()
    print("-" * 60)
    print("RETRIEVAL RESULTS")
    print("-" * 60)
    print(f"Hit@1:              {summary['hit_rate_at_1_percent']:.2f}%")
    print(f"Recall@3:            {summary['recall_at_3_percent']:.2f}%")
    print(f"Recall@5:            {summary['recall_at_5_percent']:.2f}%")
    print(f"MRR:                 {summary['mrr_percent']:.2f}%")
    print(f"Avg retrieval time:  {summary['avg_retrieval_time_sec']:.4f}s")
    print()

    for row in retrieval_rows:
        print(
            f"{row['id']} | "
            f"expected={row['expected_pages']} | "
            f"retrieved={row['retrieved_pages']} | "
            f"R@3={row['hit_at_3']}"
        )

    output_dir = PROJECT_ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "pdf": pdf_path.name,
        "questions": len(questions),
        "document_processing": {
            "chunks": len(chunks),
            "tables": len(tables),
            "vectors": int(index.ntotal),
            "processing_time_sec": round(processing_time, 3),
        },
        "retrieval_summary": summary,
        "retrieval_details": retrieval_rows,
    }

    if args.rag:
        print()
        print("-" * 60)
        print("RUNNING RAG EVALUATION")
        print("This uses Ollama and may take several minutes.")
        print("-" * 60)

        rag_rows = evaluate_rag(
            questions,
            retrieval_rows,
            index,
            bm25_index,
            chunks
        )
        report["rag_results"] = rag_rows

        print()
        print("RAG evaluation completed.")

    output_path = output_dir / "evaluation_report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Report saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
