from pathlib import Path
import time

import streamlit as st
from ollama import chat
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    INITIAL_K,
    OLLAMA_MODEL,
)


DB_ROOT_DIR = Path("vectorstores")
ACTIVE_DB_FILE = DB_ROOT_DIR / "active_db.txt"


def get_active_db_dir():
    DB_ROOT_DIR.mkdir(exist_ok=True)

    if ACTIVE_DB_FILE.exists():
        active_path = ACTIVE_DB_FILE.read_text(encoding="utf-8").strip()
        if active_path:
            return Path(active_path)

    return None


def set_active_db_dir(db_path):
    DB_ROOT_DIR.mkdir(exist_ok=True)
    ACTIVE_DB_FILE.write_text(str(db_path), encoding="utf-8")


def load_vector_store():
    active_db_dir = get_active_db_dir()

    if active_db_dir is None or not active_db_dir.exists():
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=str(active_db_dir),
        embedding_function=embeddings,
    )


@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANKER_MODEL)


def save_uploaded_files(uploaded_files):
    DATA_DIR.mkdir(exist_ok=True)

    saved_files = []
    skipped_files = []

    existing_files = {file.name for file in DATA_DIR.glob("*.pdf")}

    for uploaded_file in uploaded_files:
        if uploaded_file.name in existing_files:
            skipped_files.append(uploaded_file.name)
            continue

        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        saved_files.append(file_path.name)

    return saved_files, skipped_files


def rebuild_vector_store():
    DATA_DIR.mkdir(exist_ok=True)
    DB_ROOT_DIR.mkdir(exist_ok=True)

    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    new_db_dir = DB_ROOT_DIR / f"db_{int(time.time())}"

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(new_db_dir),
    )

    set_active_db_dir(new_db_dir)

    return len(documents), len(chunks), new_db_dir


def ensure_vector_store_exists():
    pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    active_db_dir = get_active_db_dir()

    if pdf_files and (active_db_dir is None or not active_db_dir.exists()):
        rebuild_vector_store()


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
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


def get_loaded_documents():
    if not DATA_DIR.exists():
        return []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    return [file.name for file in pdf_files]


def format_sources(results):
    formatted = []

    for index, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content[:500].strip()

        formatted.append(
            {
                "index": index,
                "source": source,
                "page": page,
                "snippet": snippet,
            }
        )

    return formatted


def main():
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 AI Document Assistant")
    st.write("Ask questions about your local PDF documents using RAG + Ollama.")

    if "upload_status" in st.session_state:
        status = st.session_state.pop("upload_status")

        if status["type"] == "success":
            st.success(status["message"])

            if status.get("saved_files"):
                st.write(f"Saved files: {', '.join(status['saved_files'])}")

            if status.get("skipped_files"):
                st.write(f"Skipped existing files: {', '.join(status['skipped_files'])}")

            if status.get("doc_count") is not None:
                st.write(f"Loaded pages: {status['doc_count']}")

            if status.get("chunk_count") is not None:
                st.write(f"Created chunks: {status['chunk_count']}")

            if status.get("db_name"):
                st.write(f"New active DB: {status['db_name']}")

        elif status["type"] == "info":
            st.info(status["message"])

            if status.get("skipped_files"):
                st.write(f"Skipped files: {', '.join(status['skipped_files'])}")

    with st.sidebar:
        st.header("Project Info")
        st.write(f"**Embedding model:** {EMBEDDING_MODEL}")
        st.write(f"**LLM:** {OLLAMA_MODEL}")
        st.write(f"**Initial retrieval:** {INITIAL_K}")
        st.write(f"**Final top-k:** {TOP_K}")

        active_db_dir = get_active_db_dir()
        if active_db_dir:
            st.write(f"**Active DB:** `{active_db_dir.name}`")

        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Save and index documents"):
            saved_files, skipped_files = save_uploaded_files(uploaded_files)

            if not saved_files and skipped_files:
                st.session_state["upload_status"] = {
                    "type": "info",
                    "message": "All uploaded files already exist. No new documents were added.",
                    "skipped_files": skipped_files,
                }
                st.rerun()

            elif saved_files:
                try:
                    with st.spinner("Building a new vector database..."):
                        doc_count, chunk_count, new_db_dir = rebuild_vector_store()

                    st.session_state["upload_status"] = {
                        "type": "success",
                        "message": "Documents indexed successfully.",
                        "saved_files": saved_files,
                        "skipped_files": skipped_files,
                        "doc_count": doc_count,
                        "chunk_count": chunk_count,
                        "db_name": new_db_dir.name,
                    }
                    st.rerun()

                except Exception as error:
                    st.error(f"Failed to rebuild vector database: {error}")

        st.subheader("Loaded Documents")
        loaded_docs = get_loaded_documents()
        if loaded_docs:
            for doc in loaded_docs:
                st.write(f"- {doc}")
        else:
            st.write("No PDF documents found.")

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    ensure_vector_store_exists()

    vector_store = load_vector_store()

    if vector_store is None:
        st.warning("No vector database found yet. Upload PDFs from the sidebar and click 'Save and index documents'.")
        return

    reranker = load_reranker()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**[{source['index']}] {source['source']} | page: {source['page']}**"
                        )
                        st.write(source["snippet"])
                        st.write("---")

    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                initial_results = vector_store.similarity_search(user_question, k=INITIAL_K)
                results = rerank_results(user_question, initial_results, reranker, top_k=TOP_K)

                context = build_context(results)
                answer = generate_answer(user_question, context)
                sources = format_sources(results)

            st.write(answer)

            with st.expander("Sources"):
                for source in sources:
                    st.markdown(
                        f"**[{source['index']}] {source['source']} | page: {source['page']}**"
                    )
                    st.write(source["snippet"])
                    st.write("---")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()