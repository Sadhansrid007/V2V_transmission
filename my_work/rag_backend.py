import os
from langchain_chroma import Chroma
from langchain_community.embeddings import DeterministicFakeEmbedding

# 1. Define where the database folder will be saved on your hard drive
DB_FOLDER = "./my_chroma_db"

# 2. Use a lightweight, fake math embedding model to test the database structure first
test_embeddings = DeterministicFakeEmbedding(size=1536)

# 3. Create or load the ChromaDB collection
print("Initializing ChromaDB...")
vector_store = Chroma(
    collection_name="video_segments",
    embedding_function=test_embeddings,
    persist_directory=DB_FOLDER
)

print("🎉 STEP 1 COMPLETE: ChromaDB is successfully running and ready!")
