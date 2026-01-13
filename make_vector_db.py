import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

INCIDENT_SOURCE_DIRECTORY = "incidents"
SOP_SOURCE_DIRECTORY = "SOPs"
PERSISTENT_DIRECTORY = "chroma_db"


def parse_incident_metadata(doc):
    """
    Parses metadata specifically for Incident Log files.
    """

    lines = doc.page_content.split("\n")
    metadata_updates = {"type": "incident"}

    for line in lines:
        clean_line = line.strip()
        if clean_line.startswith("CATEGORY:"):
            metadata_updates["category"] = clean_line.split(":", 1)[1].strip()
        elif clean_line.startswith("SERVICE:"):
            metadata_updates["service"] = clean_line.split(":", 1)[1].strip()
        elif clean_line.startswith("ID:"):
            metadata_updates["id"] = clean_line.split(":", 1)[1].strip()
        elif clean_line.startswith("IMPACT:"):
            metadata_updates["impact"] = clean_line.split(":", 1)[1].strip()
    
    doc.metadata.update(metadata_updates)
    return doc

def parse_sop_metadata(doc):
    """
    Parse metadata specifically for SOP text files.
    Expected format:
    STANDARD OPERATING PROCEDURE (SOP) XX: TITLE
    DATE: YYYY-MM-DD
    SEVERITY: LEVEL
    """

    lines = doc.page_content.split("\n")
    metadata_updates = {"type": "sop"}

    for line in lines:
        clean_line = line.strip()
    
    if "STANDARD OPERATING PROCEDURE" in clean_line:
        parts = clean_line.split(":", 1)
        if len(parts) > 1:
            metadata_updates["title"] = parts[1].strip()
        else:
            metadata_updates["title"] = clean_line
    
    elif clean_line.startswith("DATE:"):
        try:
            metadata_updates["date"] = clean_line.split(":", 1)[1].strip()
        except IndexError:
            pass
    
    elif clean_line.startswith("SEVERITY:"):
        try:
            metadata_updates["severity"] = clean_line.split(":", 1)[1].strip()
        except IndexError:
            pass
    
    doc.metadata.update(metadata_updates)
    return doc

def create_vector_collection(source_dir, collection_name, metadata_parser_function):
    """
    Generic function to load docs, parse metadata, and save to a specified Chroma collection.
    """

    if not os.path.exists(source_dir):
        print(f"Error: Directory '{source_dir}' not found. Skipping {collection_name}")
        return
    
    print(f"\n--- Processing {collection_name} from {source_dir} ---")

    loader = DirectoryLoader(source_dir, glob="./*.txt", loader_cls=TextLoader)
    raw_docs = loader.load()

    if not raw_docs:
        print(f"No documents found in {source_dir}")
        return
    
    print(f"Loaded {len(raw_docs)} documents.")

    docs = [metadata_parser_function(d) for d in raw_docs]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )

    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    if chunks:
        print(f"Sample Metadata ({collection_name}): {chunks[0].metadata}")

    embeddings = OllamaEmbeddings(model="gte-large:latest")

    print(f"Saving to collection: {collection_name}...")

    Chroma.from_documents(
        documents=chunks,
        embedding= embeddings,
        persist_directory=PERSISTENT_DIRECTORY,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Successfully saved {collection_name}.")

def main():
    create_vector_collection(source_dir=INCIDENT_SOURCE_DIRECTORY, collection_name="incident_collection", metadata_parser_function=parse_incident_metadata)
    create_vector_collection(source_dir=SOP_SOURCE_DIRECTORY, collection_name='sop_collection', metadata_parser_function=parse_sop_metadata)
    print("\nAll database updates complete.")

if __name__ == "__main__":
    main()