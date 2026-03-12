import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import EMBEDDING_MODEL, TOP_K


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


def run_without_cache(vector_store, questions):
    total_time = 0

    for item in questions:
        question = item["question"]

        start_time = time.perf_counter()
        vector_store.similarity_search(question, k=TOP_K)
        end_time = time.perf_counter()

        total_time += end_time - start_time

    return total_time / len(questions)


def run_with_cache(vector_store, questions):
    cache = {}
    total_time = 0

    for item in questions:
        question = item["question"]

        start_time = time.perf_counter()

        if question in cache:
            _ = cache[question]
        else:
            cache[question] = vector_store.similarity_search(question, k=TOP_K)

        end_time = time.perf_counter()
        total_time += end_time - start_time

    return total_time / len(questions)


def main():
    vector_store = load_vector_store()

    if vector_store is None:
        print("Vector database not found. Please run the Streamlit app and index documents first.")
        return

    questions = load_questions()

    uncached_avg = run_without_cache(vector_store, questions)
    cached_avg = run_with_cache(vector_store, questions)

    improvement = ((uncached_avg - cached_avg) / uncached_avg) * 100 if uncached_avg > 0 else 0

    print("\nCaching Evaluation")
    print("=" * 60)
    print(f"Average retrieval time without cache: {uncached_avg:.6f} seconds")
    print(f"Average retrieval time with cache:    {cached_avg:.6f} seconds")
    print(f"Latency improvement: {improvement:.2f}%")


if __name__ == "__main__":
    main()