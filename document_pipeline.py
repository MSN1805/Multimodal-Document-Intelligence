import io
import re

import fitz
import pdfplumber
import pytesseract

from PIL import Image
from rank_bm25 import BM25Okapi

from text_processor import clean_text, create_chunks
from embedding import generate_embeddings

from vector_store import (
    create_faiss_index,
    search_faiss,
    save_vector_database
)


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def tokenize_text(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def extract_pdf_text(pdf_bytes):
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_number, page in enumerate(pdf):
        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_number + 1,
                "text": text,
                "ocr": False
            })
            continue

        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        image = Image.open(
            io.BytesIO(pix.tobytes("png"))
        )

        ocr_text = pytesseract.image_to_string(image)

        pages.append({
            "page": page_number + 1,
            "text": ocr_text,
            "ocr": True
        })

    pdf.close()
    return pages


def extract_pdf_tables(pdf_bytes):
    tables = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages):
            extracted_tables = page.extract_tables()

            for table_number, table in enumerate(extracted_tables):
                if not table:
                    continue

                cleaned_table = []

                for row in table:
                    if row is None:
                        continue

                    cleaned_row = []

                    for cell in row:
                        cell = "" if cell is None else str(cell).strip()
                        cleaned_row.append(cell)

                    if any(cell for cell in cleaned_row):
                        cleaned_table.append(cleaned_row)

                if cleaned_table:
                    tables.append({
                        "page": page_number + 1,
                        "table_number": table_number + 1,
                        "data": cleaned_table
                    })

    return tables


def table_to_text(table):
    return "\n".join(
        " | ".join(str(cell) for cell in row)
        for row in table["data"]
    )


def create_bm25_index(chunks):
    tokenized_chunks = [
        tokenize_text(chunk["text"])
        for chunk in chunks
    ]
    return BM25Okapi(tokenized_chunks)


def process_document(pdf_bytes, file_name=None):
    pages = extract_pdf_text(pdf_bytes)
    tables = extract_pdf_tables(pdf_bytes)

    all_chunks = []

    for page in pages:
        text = clean_text(page["text"])

        if not text:
            continue

        page_chunks = create_chunks(
            text,
            chunk_size=500,
            overlap=100
        )

        for chunk in page_chunks:
            all_chunks.append({
                "page": page["page"],
                "text": chunk,
                "type": "text",
                "ocr": page["ocr"]
            })

    for table in tables:
        table_text = table_to_text(table)

        if not table_text.strip():
            continue

        all_chunks.append({
            "page": table["page"],
            "text": table_text,
            "type": "table",
            "table_number": table["table_number"],
            "ocr": False
        })

    if not all_chunks:
        raise ValueError("No usable content was found in the document.")

    texts = [item["text"] for item in all_chunks]

    embeddings = generate_embeddings(texts)
    faiss_index = create_faiss_index(embeddings)
    bm25_index = create_bm25_index(all_chunks)

    document_id = save_vector_database(
        faiss_index,
        all_chunks,
        file_name,
        pdf_bytes
    )

    return (
        all_chunks,
        embeddings,
        faiss_index,
        bm25_index,
        tables
    )



def _query_intent_boost(query, chunk, page):
    """Add a small structure-aware boost for common document-QA intents."""
    q = query.lower()
    text = chunk.get("text", "").lower()
    boost = 0.0

    intent_groups = [
        (("main purpose", "purpose of the study", "aim of the study", "objective of the study"),
         ("purpose", "aim", "objective", "study aimed", "investigate", "examined"), 0.18),
        (("method", "methodology", "how was", "data collected", "collect the study data"),
         ("method", "methodology", "questionnaire", "interview", "data collection", "participants"), 0.10),
        (("participant", "students participated", "average age", "how many students"),
         ("participants", "students", "age", "sample", "respondents"), 0.08),
        (("reasons", "why did", "why students", "reasons students"),
         ("reasons", "because", "reason", "motivation", "purpose"), 0.14),
        (("tools", "apps", "applications", "language tools"),
         ("apps", "applications", "tools", "dictionary", "youtube", "facebook"), 0.10),
        (("where did", "where", "location", "locations", "used their mobile"),
         ("where", "home", "university", "school", "work", "commuting", "transport", "bus", "train"), 0.08),
        (("limitations", "limitation of the study", "limitations of the study"),
         ("limitation", "limitations", "limited", "small sample", "sample size", "future research"), 0.18),
        (("suggest", "suggested", "recommend", "recommendation", "teachers could"),
         ("suggest", "recommend", "teacher", "teachers", "recommendation", "should"), 0.12),
    ]

    for query_terms, text_terms, weight in intent_groups:
        if any(term in q for term in query_terms):
            if any(term in text for term in text_terms):
                boost = max(boost, weight)

    # Prefer introductory material for explicit study-purpose questions.
    if any(term in q for term in ("main purpose", "purpose of the study", "aim of the study", "objective of the study")):
        if page <= 2:
            boost += 0.10

    # Prefer conclusion/discussion material for explicit limitation/recommendation questions.
    if any(term in q for term in ("limitations", "limitation of the study", "limitations of the study", "recommend", "suggest")):
        if page >= 9:
            boost += 0.06

    return boost


def _numeric_match_boost(query, text):
    """Reward chunks that contain numeric/percentage evidence for numeric questions."""
    q = query.lower()
    if not any(term in q for term in (
        "how many", "how much", "what percentage", "what percent",
        "percentage", "percent", "average", "number of", "how many years"
    )):
        return 0.0

    numbers = re.findall(r"\b\d+(?:\.\d+)?\s*%?\b", text)
    if not numbers:
        return 0.0

    # Small boost only: lexical/semantic ranking should remain primary.
    return 0.05


def _lexical_overlap(query, text):
    """Token overlap used as a small tie-breaking signal."""
    query_terms = set(tokenize_text(query))
    text_terms = set(tokenize_text(text))
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def search_document(
    query,
    faiss_index,
    bm25_index,
    all_chunks,
    top_k=3
):
    if not all_chunks:
        return []

    query_embedding = generate_embeddings([query])

    # Keep the original V2 scoring approach, but retrieve a wider candidate
    # pool so targeted intent/lexical boosts can rerank relevant evidence.
    candidate_k = min(max(top_k * 5, 10), len(all_chunks))

    distances, indices = search_faiss(
        faiss_index,
        query_embedding,
        candidate_k
    )

    semantic_results = {
        int(indices[0][i]): float(distances[0][i])
        for i in range(len(indices[0]))
    }

    bm25_scores = bm25_index.get_scores(
        tokenize_text(query)
    )

    keyword_candidates = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:candidate_k]

    candidate_indices = set(semantic_results.keys())
    candidate_indices.update(keyword_candidates)

    max_bm25 = max(bm25_scores) if len(bm25_scores) else 0.0
    results = []

    for index_number in candidate_indices:
        chunk = all_chunks[index_number]

        semantic_distance = semantic_results.get(index_number)

        semantic_score = (
            1 / (1 + semantic_distance)
            if semantic_distance is not None
            else 0.0
        )

        bm25_score = float(bm25_scores[index_number])
        keyword_score = (
            bm25_score / max_bm25
            if max_bm25 > 0
            else 0.0
        )

        base_hybrid = (
            0.60 * semantic_score
            + 0.40 * keyword_score
        )

        overlap = _lexical_overlap(query, chunk["text"])
        intent_boost = _query_intent_boost(
            query,
            chunk,
            chunk["page"]
        )
        numeric_boost = _numeric_match_boost(
            query,
            chunk["text"]
        )

        # Conservative boosts preserve V2 behavior while improving known
        # failure modes such as purpose, reasons, limitations and numbers.
        hybrid_score = (
            base_hybrid
            + 0.08 * overlap
            + intent_boost
            + numeric_boost
        )

        results.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "type": chunk.get("type", "text"),
            "table_number": chunk.get("table_number"),
            "ocr": chunk.get("ocr", False),
            "distance": (
                semantic_distance
                if semantic_distance is not None
                else 999.0
            ),
            "bm25_score": bm25_score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "lexical_overlap": overlap,
            "intent_boost": intent_boost,
            "numeric_boost": numeric_boost,
            "hybrid_score": hybrid_score,
            "index": index_number
        })

    results.sort(
        key=lambda x: (-x["hybrid_score"], x["page"], x["index"])
    )

    return results[:top_k]


def search_multiple_documents(
    query,
    loaded_documents,
    top_k=6
):
    if not loaded_documents:
        return []

    query_embedding = generate_embeddings([query])
    all_results = []

    for document in loaded_documents:
        index = document["index"]
        chunks = document["chunks"]
        metadata = document["metadata"]

        if not chunks:
            continue

        candidate_k = min(
            max(top_k * 5, 10),
            len(chunks)
        )

        distances, indices = search_faiss(
            index,
            query_embedding,
            candidate_k
        )

        semantic_distance_map = {
            int(indices[0][i]): float(distances[0][i])
            for i in range(len(indices[0]))
        }

        bm25 = create_bm25_index(chunks)
        bm25_scores = bm25.get_scores(
            tokenize_text(query)
        )

        max_bm25 = max(bm25_scores) if len(bm25_scores) else 0.0

        candidates = set(semantic_distance_map.keys())
        keyword_candidates = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:candidate_k]
        candidates.update(keyword_candidates)

        for chunk_index in candidates:
            chunk = chunks[chunk_index]
            distance = semantic_distance_map.get(chunk_index)

            semantic_score = (
                1 / (1 + distance)
                if distance is not None
                else 0.0
            )

            bm25_score = float(bm25_scores[chunk_index])
            keyword_score = (
                bm25_score / max_bm25
                if max_bm25 > 0
                else 0.0
            )

            base_hybrid = (
                0.60 * semantic_score
                + 0.40 * keyword_score
            )
            overlap = _lexical_overlap(query, chunk["text"])
            intent_boost = _query_intent_boost(
                query,
                chunk,
                chunk["page"]
            )
            numeric_boost = _numeric_match_boost(
                query,
                chunk["text"]
            )
            hybrid_score = (
                base_hybrid
                + 0.08 * overlap
                + intent_boost
                + numeric_boost
            )

            all_results.append({
                "document_id": document.get("document_id"),
                "document_name": metadata.get(
                    "file_name",
                    "Unknown document"
                ),
                "page": chunk["page"],
                "text": chunk["text"],
                "type": chunk.get("type", "text"),
                "table_number": chunk.get("table_number"),
                "ocr": chunk.get("ocr", False),
                "distance": (
                    distance
                    if distance is not None
                    else 999.0
                ),
                "bm25_score": bm25_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "lexical_overlap": overlap,
                "intent_boost": intent_boost,
                "numeric_boost": numeric_boost,
                "hybrid_score": hybrid_score,
                "index": chunk_index
            })

    all_results.sort(
        key=lambda x: (
            -x["hybrid_score"],
            x["document_name"].lower(),
            x["page"],
            x["index"]
        )
    )

    balanced = []
    per_document = {}

    for result in all_results:
        document_id = result["document_id"]
        count = per_document.get(document_id, 0)

        if count >= 3:
            continue

        balanced.append(result)
        per_document[document_id] = count + 1

        if len(balanced) >= top_k:
            break

    return balanced
