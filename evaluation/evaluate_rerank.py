import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import EMBEDDING_MODEL, TOP_K, INITIAL_K, RERANKER_MODEL


EVAL_FILE = Path("evaluation/eval_questions.json")
DB_ROOT_DIR = Path("vectorstores")
ACTIVE_DB_FILE = DB_ROOT_DIR / "active_db.txt"


def get_active_db_dir():
    if ACTIVE_DB_FILE.exists():
        active_path = ACTIVE_DB_FILE.read_text(encoding="utf-8").strip()
        if active_path:
            active_db_dir = Path(active_path)
            if active_db_dir.exists():
                return active_db_dir
    return None


def load_vector_store():
    active_db_dir = get_active_db_dir()

    if active_db_dir is None:
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=str(active_db_dir),
        embedding_function=embeddings,
    )


def load_questions():
    with open(EVAL_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def rerank_results(query, results, reranker, top_k=TOP_K):
    pairs = [[query, doc.page_content] for doc in results]
    scores = reranker.predict(pairs)

    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda item: item[1], reverse=True)

    return [doc for doc, _ in scored_results[:top_k]]


def main():
    vector_store = load_vector_store()

    if vector_store is None:
        print("Vector database not found. Please run the Streamlit app and index documents first.")
        return

    reranker = CrossEncoder(RERANKER_MODEL)
    questions = load_questions()

    correct_top_1 = 0
    correct_top_k = 0

    print("\nRunning reranking evaluation...\n")

    for item in questions:
        question = item["question"]
        expected_source = item["expected_source"]

        initial_results = vector_store.similarity_search(question, k=INITIAL_K)
        reranked_results = rerank_results(question, initial_results, reranker, top_k=TOP_K)

        returned_sources = [doc.metadata.get("source", "Unknown source") for doc in reranked_results]
        top_1_source = returned_sources[0] if returned_sources else None

        if top_1_source == expected_source:
            correct_top_1 += 1

        if expected_source in returned_sources:
            correct_top_k += 1

        print(f"Question: {question}")
        print(f"Expected: {expected_source}")
        print(f"Top 1 after rerank: {top_1_source}")
        print(f"Top {TOP_K} after rerank: {returned_sources}")
        print("-" * 60)

    total_questions = len(questions)
    top_1_accuracy = (correct_top_1 / total_questions) * 100
    top_k_accuracy = (correct_top_k / total_questions) * 100

    print("\nReranking Evaluation Summary")
    print("=" * 60)
    print(f"Total questions: {total_questions}")
    print(f"Top-1 retrieval accuracy after reranking: {top_1_accuracy:.2f}%")
    print(f"Top-{TOP_K} retrieval accuracy after reranking: {top_k_accuracy:.2f}%")


if __name__ == "__main__":
    main()