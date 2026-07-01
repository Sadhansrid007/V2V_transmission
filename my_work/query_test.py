from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_FOLDER = "./my_chroma_db"

print("Loading Hugging Face embedding model...")
real_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="video_segments",
    embedding_function=real_embeddings,
    persist_directory=DB_FOLDER
)

# Choose a conceptual query based on your real transcript text
user_search_prompt = "F1 driver lap time challenge around the Goodwood stage"

print(f"Searching ChromaDB for concepts matching: '{user_search_prompt}'...\n")

results = vector_store.similarity_search(user_search_prompt, k=2)

if results:
    for rank, doc in enumerate(results):
        print(f"--- Match Rank #{rank + 1} ---")
        print(f"Text Content: {doc.page_content}")
        print(f"Metadata Timestamps -> Start: {doc.metadata.get('start_time')}s | End: {doc.metadata.get('end_time')}s\n")
else:
    print("No matches found. Check if the database was opened correctly.")
