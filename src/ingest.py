from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import DATA_DIR, DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents():
    pdf_loader = PyPDFDirectoryLoader(str(DATA_DIR))
    pdf_docs = pdf_loader.load()

    md_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    md_docs = md_loader.load()

    documents = pdf_docs + md_docs
    print(f"Loaded {len(documents)} documents/pages from data folder.")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")
    return chunks


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )

    print("Vector database created successfully.")
    return vector_store


def main():
    if not DATA_DIR.exists():
        print("Data folder not found. Please add PDFs or markdown files first.")
        return

    documents = load_documents()

    if not documents:
        print("No supported files found in the data folder.")
        return

    chunks = split_documents(documents)
    create_vector_store(chunks)


if __name__ == "__main__":
    main()