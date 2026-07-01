import json
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

DB_FOLDER = "./my_chroma_db"

def get_ai_timestamps(user_query):
    # 1. Connect to your existing ChromaDB database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="video_segments",
        embedding_function=embeddings,
        persist_directory=DB_FOLDER
    )
    
    # 2. Search for the top 4 most relevant video segments
    print(f"\n🔍 Searching database for: '{user_query}'...")
    docs = vector_store.similarity_search(user_query, k=4)
    
    if not docs:
        print("No matching text segments found in the database.")
        return []
        
    # 3. Format the retrieved segments into a clean list for the LLM
    context_list = []
    for doc in docs:
        start = doc.metadata.get("start_time", 0)
        end = doc.metadata.get("end_time", 0)
        context_list.append(f"Timestamp: [{start}-{end}]s | Text: {doc.page_content}")
    
    formatted_context = "\n".join(context_list)
    
    # 4. Initialize your exact local Llama model
    # Note: Using temperature=0.0 makes the model strictly factual and precise
    llm = ChatOllama(model="llama3.2:1b ", temperature=0.0)
    
    # 5. Build a strict prompt instructing the 1B model to ONLY return raw JSON arrays
    system_prompt = (
        "You are a precise video timestamp extractor. Your task is to look at the provided video "
        "transcript segments and select the timestamp ranges that best answer the user's request.\n\n"
        f"Available Video Segments:\n{formatted_context}\n\n"
        "CRITICAL REQUIREMENT:\n"
        "You must output ONLY a raw JSON list of lists containing the numbers. Do not include any conversational filler, "
        "no explanation, no introduction, and no markdown code blocks like ```json.\n"
        "Example Output Format:\n[[10, 20], [35, 50]]"
    )
    
    print("🧠 Asking local Llama to decide the final cuts...")
    response = llm.invoke([("system", system_prompt), ("human", user_query)])
    raw_output = response.content.strip()
    
    # 6. Clean up the response just in case the model added text or markdown wrappers
    cleaned_output = re.sub(r"```[a-zA-Z]*\n?", "", raw_output).replace("```", "").strip()
    
    try:
        # 1. Try structural JSON parsing first
        timestamps = json.loads(cleaned_output)
        print(f"🎯 Local AI Decided Timestamps: {timestamps}")
        return timestamps
    except json.JSONDecodeError:
        print("⚠️ Failed to parse JSON cleanly. Attempting regex extraction safety net...")
        
        # 2. Use Regex to fish out anything that looks like [number, number]
        timestamp_pairs = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', raw_output)
        
        if timestamp_pairs:
            # Convert string matches like ('16', '18') into an integer list [[16, 18]]
            timestamps = [[int(start), int(end)] for start, end in timestamp_pairs]
            print(f"🎯 Extracted Timestamps via Backup Regex: {timestamps}")
            return timestamps
        else:
            # 3. Complete fallback if the model completely went off-script
            print("❌ Regex backup also failed to find valid timestamp brackets. Raw output:")
            print(raw_output)
            return []

if __name__ == "__main__":
    # CHANGE THIS to any phrase you want to search for from your transcript!
    TEST_PROMPT = "what are the two main problems mentioned in the video?"
    
    get_ai_timestamps(TEST_PROMPT)