from ollama import chat
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    DB_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    INITIAL_K,
    OLLAMA_MODEL,
    RERANKER_MODEL,
)


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

    return vector_store


def load_reranker():
    return CrossEncoder(RERANKER_MODEL)


def rerank_results(query, results, reranker, top_k=TOP_K):
    pairs = [[query, doc.page_content] for doc in results]
    scores = reranker.predict(pairs)

    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda item: item[1], reverse=True)

    return [doc for doc, _ in scored_results[:top_k]]


def build_context(results):
    context_parts = []

    for index, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "N/A")

        context_parts.append(
            f"Source {index}: {source}, page {page}\n{doc.page_content.strip()}"
        )

    return "\n\n".join(context_parts)


def generate_answer(question, context):
    prompt = f"""
You are a helpful document assistant.

Use only the provided context.
If the answer is not clearly stated in the context, reply with exactly:
I could not find a reliable answer in the provided documents.

Do not use outside knowledge.
Do not guess.
Do not add extra explanation if the answer is missing.

Question:
{question}

Context:
{context}

Give a short, clear answer in 2 to 4 sentences.
"""

    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


def main():
    if not DB_DIR.exists():
        print("Vector database not found. Please run ingest.py first.")
        return

    vector_store = load_vector_store()
    reranker = load_reranker()

    print("Local AI Document Assistant is ready.")
    print("Ask a question about your documents.")
    print("Commands:")
    print("  exit  -> quit assistant")
    print("  docs  -> list loaded documents")

    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if query.lower() == "docs":
            sources = vector_store.get()["metadatas"]
            unique_docs = sorted(set(m["source"] for m in sources))

            print("\nLoaded documents:\n")
            for doc in unique_docs:
                print("-", doc)
            continue

        if not query:
            print("Please enter a question.")
            continue

        initial_results = vector_store.similarity_search(query, k=INITIAL_K)
        results = rerank_results(query, initial_results, reranker, top_k=TOP_K)

        context = build_context(results)
        answer = generate_answer(query, context)

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:\n")
        for index, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "N/A")

            print(f"[{index}] {source} | page: {page}")
            print(doc.page_content[:400].strip())
            print("\n" + "-" * 60)


if __name__ == "__main__":
    main()