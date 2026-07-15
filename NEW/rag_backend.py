import os
import json
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

from transcript_loader import load_transcript, merge_chunks
from query_expander import expand_query

load_dotenv()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="lecture_transcript")


def store_transcript(transcript_path):
    raw_chunks = load_transcript(transcript_path)
    merged = merge_chunks(raw_chunks, window_size=8)
    
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    
    for i, chunk in enumerate(merged):
        vector = embedding_model.encode(chunk["text"]).tolist()
        collection.add(
            ids=[str(i)],
            embeddings=[vector],
            documents=[chunk["text"]],
            metadatas=[{
                "start": chunk["start"],
                "end": chunk["end"]
            }]
        )
    
    print(f"Stored {len(merged)} chunks in ChromaDB")


def search_with_expansion(student_prompt, top_k=8):
    expanded_queries = expand_query(student_prompt)
    
    all_results = {}
    
    for query in expanded_queries:
        query_vector = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        for i in range(len(results["documents"][0])):
            doc_id = results["ids"][0][i]
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "text": results["documents"][0][i],
                    "start": results["metadatas"][0][i]["start"],
                    "end": results["metadatas"][0][i]["end"]
                }
    
    return list(all_results.values())


def get_answer_and_timestamps(student_prompt):
    matched_chunks = search_with_expansion(student_prompt)
    
    context = ""
    for chunk in matched_chunks:
        context += f"[{chunk['start']}s - {chunk['end']}s] {chunk['text']}\n"
    
    prompt = f"""
You are helping a student find and clip specific parts of a lecture video.

Student request: "{student_prompt}"

Relevant transcript segments:
{context}

Do exactly three things:

1. Write a SHORT SUMMARY (1-2 sentences maximum) of what was found.
   Label it exactly: SHORT: your summary here

2. Write a DETAILED DESCRIPTION (4-6 sentences) explaining everything 
   that happens in the relevant segments, including context, key moments,
   and important details.
   Label it exactly: DETAILED: your detailed description here

3. Return ALL timestamps needed for complete coverage of what the student asked.
   Do not skip middle sections. Include every segment from start to finish.
   Label it exactly:
   TIMESTAMPS: [{{"start": 10.5, "end": 25.3}}, {{"start": 25.3, "end": 60.2}}]

Only use information from the transcript. Do not make anything up.
If the topic is not found in the transcript return empty timestamps like:
TIMESTAMPS: []
"""
    
    response = llm.invoke(prompt)
    return response.content


def parse_response(llm_response):
    short_summary = ""
    detailed_description = ""
    timestamps = []
    
    if "SHORT:" in llm_response:
        short_part = llm_response.split("SHORT:")[1]
        if "DETAILED:" in short_part:
            short_summary = short_part.split("DETAILED:")[0].strip()
        else:
            short_summary = short_part.strip()
    
    if "DETAILED:" in llm_response:
        detailed_part = llm_response.split("DETAILED:")[1]
        if "TIMESTAMPS:" in detailed_part:
            detailed_description = detailed_part.split("TIMESTAMPS:")[0].strip()
        else:
            detailed_description = detailed_part.strip()
    
    if "TIMESTAMPS:" in llm_response:
        timestamp_string = llm_response.split("TIMESTAMPS:")[1].strip()
        try:
            timestamps = json.loads(timestamp_string)
        except json.JSONDecodeError:
            timestamps = []
    
    return {
        "short_summary": short_summary,
        "detailed_description": detailed_description,
        "timestamps": timestamps,
        "answer": short_summary
    }


def process_query(student_prompt, transcript_path=None):
    if transcript_path:
        store_transcript(transcript_path)
    
    raw_response = get_answer_and_timestamps(student_prompt)
    result = parse_response(raw_response)
    
    print("\nShort Summary:")
    print(result["short_summary"])
    print("\nDetailed Description:")
    print(result["detailed_description"])
    print("\nTimestamps:")
    print(result["timestamps"])
    
    return result


if __name__ == "__main__":
    TRANSCRIPT_PATH = r"C:\Users\Madhu\OneDrive\Desktop\CLG\techNOVA\video_ingestion\transcript.txt"
    test_prompt = input("Enter your test query: ")
    process_query(student_prompt=test_prompt, transcript_path=TRANSCRIPT_PATH)