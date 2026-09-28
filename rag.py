import json
import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def call_llm(prompt, max_tokens=150):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens
            }
        },
        timeout=180
    )

    response.raise_for_status()
    return response.json()["response"].strip()


def generate_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return (
            "I could not find this information "
            "in the selected documents."
        )

    context = "\n\n".join(
        f"[Source {i + 1} | "
        f"Document: {chunk.get('document_name', 'Unknown document')} | "
        f"Page: {chunk.get('page', 'Unknown')}]\n"
        f"{chunk.get('text', '')}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = f"""
You are an AI multi-document assistant.

Answer the user's question using ONLY the
provided document evidence.

Rules:
1. Do not use outside knowledge.
2. Do not guess or invent information.
3. If the evidence does not support the answer, say:
"I could not find this information in the selected documents."
4. Keep the answer concise.
5. Preserve names, dates, numbers and technical terms.
6. Cite important claims using:
[Document: filename.pdf | Page X]
7. If multiple documents support the answer, cite them.
8. If documents conflict, clearly state the conflict and cite both.

DOCUMENT EVIDENCE:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    return call_llm(prompt, max_tokens=220)


def calculate_grounding_score(
    question,
    answer,
    retrieved_chunks
):
    if not retrieved_chunks:
        return {
            "score": 0,
            "level": "Weak",
            "explanation": "No supporting evidence was retrieved."
        }

    distances = [
        chunk.get("distance", 0)
        for chunk in retrieved_chunks
    ]

    average_distance = (
        sum(distances) / len(distances)
        if distances
        else 0
    )

    retrieval_score = max(
        0,
        min(100, 100 / (1 + average_distance))
    )

    question_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]{3,}\b",
            question.lower()
        )
    )

    evidence_text = " ".join(
        chunk.get("text", "")
        for chunk in retrieved_chunks
    ).lower()

    evidence_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]{3,}\b",
            evidence_text
        )
    )

    question_overlap = (
        len(question_words & evidence_words)
        / len(question_words)
        if question_words
        else 0
    )

    keyword_score = question_overlap * 100

    answer_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]{3,}\b",
            answer.lower()
        )
    )

    answer_overlap = (
        len(answer_words & evidence_words)
        / len(answer_words)
        if answer_words
        else 0
    )

    answer_score = answer_overlap * 100

    if "could not find" in answer.lower():
        final_score = min(retrieval_score, 40)
    else:
        final_score = (
            retrieval_score * 0.45
            + keyword_score * 0.20
            + answer_score * 0.35
        )

    final_score = round(
        max(0, min(100, final_score)),
        1
    )

    if final_score >= 75:
        level = "Strong"
        explanation = (
            "The answer is strongly supported "
            "by the retrieved document evidence."
        )
    elif final_score >= 50:
        level = "Moderate"
        explanation = (
            "The answer has supporting evidence, "
            "but additional verification is recommended."
        )
    else:
        level = "Weak"
        explanation = (
            "The retrieved evidence provides "
            "limited support for the answer."
        )

    return {
        "score": final_score,
        "level": level,
        "explanation": explanation
    }


def summarize_chunk_group(chunks):
    context = "\n\n".join(
        f"Document: {chunk.get('document_name', '')} | "
        f"Page: {chunk.get('page', 'Unknown')}\n"
        f"{chunk.get('text', '')}"
        for chunk in chunks
    )

    prompt = f"""
Summarize the following document section.

Use ONLY the provided information.

Preserve important:
- facts
- names
- dates
- numbers
- technical terms
- methods
- results
- conclusions

Remove repetition.
Do not add outside information.

DOCUMENT SECTION:

{context}

SUMMARY:
"""

    return call_llm(prompt, max_tokens=160)


def fast_summarize_document(chunks):
    if not chunks:
        return "No document content available."

    context = "\n\n".join(
        f"Page {chunk.get('page', 'Unknown')}:\n"
        f"{chunk.get('text', '')}"
        for chunk in chunks
    )

    prompt = f"""
Create a concise summary of the document below.

Use ONLY the provided document text.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. Preserve important names, dates and numbers.
4. Preserve important technical terms.
5. Focus on purpose, methods, findings, results and conclusions.
6. Remove repetition.
7. Use clear bullet points.
8. Keep the summary concise.

DOCUMENT:

{context}

SUMMARY:
"""

    return call_llm(prompt, max_tokens=220)


def summarize_document(chunks, group_size=12):
    if not chunks:
        return "No document content available."

    number_of_chunks = len(chunks)

    if number_of_chunks <= 15:
        return fast_summarize_document(chunks)

    if number_of_chunks <= 30:
        middle = number_of_chunks // 2

        summary_1 = summarize_chunk_group(chunks[:middle])
        summary_2 = summarize_chunk_group(chunks[middle:])

        prompt = f"""
Create one final concise summary from these two summaries.

Use ONLY the information provided.
Remove duplicates.
Preserve important facts, dates, numbers and technical terms.
Use clear bullet points.

SECTION 1:
{summary_1}

SECTION 2:
{summary_2}

FINAL SUMMARY:
"""

        return call_llm(prompt, max_tokens=220)

    partial_summaries = []

    for start in range(0, number_of_chunks, group_size):
        group = chunks[start:start + group_size]
        partial_summaries.append(
            summarize_chunk_group(group)
        )

    combined = "\n\n".join(
        f"SECTION {i + 1}:\n{summary}"
        for i, summary in enumerate(partial_summaries)
    )

    prompt = f"""
Create one final concise summary from the section summaries.

Use ONLY the information provided.
Remove duplicates.
Preserve important facts, dates, numbers and technical information.
Use clear bullet points.
Do not add outside information.

SECTION SUMMARIES:

{combined}

FINAL SUMMARY:
"""

    return call_llm(prompt, max_tokens=250)


def extract_information(
    chunks,
    index,
    generate_embeddings,
    search_faiss
):
    fields = {
        "Applicant Name": "name of applicant",
        "Nationality": "nationality",
        "Gender": "gender",
        "Date of Birth": "date of birth",
        "Mother Tongue": "mother tongue",
        "Family Members": "number of family members",
        "Annual Family Income": "annual family income",
        "Current Address": "current address",
        "Permanent Address": "permanent address",
        "Email": "email address",
        "Mobile Number": "mobile phone number"
    }

    evidence = []
    used_chunk_ids = set()
    used_pages = set()

    for field, query in fields.items():
        query_embedding = generate_embeddings([query])

        distances, indices = search_faiss(
            index,
            query_embedding,
            top_k=min(1, len(chunks))
        )

        for i, chunk_index in enumerate(indices[0]):
            chunk_index = int(chunk_index)

            if chunk_index in used_chunk_ids:
                continue

            used_chunk_ids.add(chunk_index)
            chunk = chunks[chunk_index]

            evidence.append({
                "field": field,
                "page": chunk["page"],
                "text": chunk["text"],
                "distance": float(distances[0][i])
            })

            used_pages.add(chunk["page"])

    evidence.sort(key=lambda x: x["distance"])
    evidence = evidence[:15]

    context = "\n".join(
        f"""
FIELD:
{item['field']}

PAGE:
{item['page']}

DOCUMENT TEXT:
{item['text']}
"""
        for item in evidence
    )

    prompt = f"""
You are an AI document information extraction system.

Extract the requested information from the evidence.

Rules:
1. Use ONLY the provided evidence.
2. Never guess.
3. Never invent information.
4. If unavailable, return "Not found".
5. Preserve names, dates, numbers, emails and phone numbers.
6. Return ONLY valid JSON.
7. Do not add fields.

FIELDS:

Applicant Name
Nationality
Gender
Date of Birth
Mother Tongue
Family Members
Annual Family Income
Current Address
Permanent Address
Email
Mobile Number

DOCUMENT EVIDENCE:

{context}

RETURN JSON IN EXACTLY THIS FORMAT:

{{
    "Applicant Name": "",
    "Nationality": "",
    "Gender": "",
    "Date of Birth": "",
    "Mother Tongue": "",
    "Family Members": "",
    "Annual Family Income": "",
    "Current Address": "",
    "Permanent Address": "",
    "Email": "",
    "Mobile Number": ""
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 250
            }
        },
        timeout=180
    )

    response.raise_for_status()
    extracted_text = response.json()["response"]

    try:
        extracted_data = json.loads(extracted_text)
        output = ""

        for field in fields:
            value = extracted_data.get(field, "Not found")

            if value is None or value == "":
                value = "Not found"

            output += f"**{field}:** {value}\n\n"

        output += "---\n\n"
        output += "**Evidence pages:** "
        output += ", ".join(
            str(page)
            for page in sorted(used_pages)
        )

        return output

    except Exception:
        return extracted_text
