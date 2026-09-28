def clean_text(text):
    return " ".join(text.split())


def create_chunks(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks
