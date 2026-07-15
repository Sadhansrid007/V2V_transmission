import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

def expand_query(student_prompt):
    prompt = f"""
    A student is searching for a specific part of a video.
    Their original question is: "{student_prompt}"
    
    Rewrite this question in 4 different ways that mean exactly the same thing.
    Use different words and phrasings each time.
    Return only the 4 versions as a plain numbered list.
    No explanations, no extra text.
    """
    
    response = llm.invoke(prompt)
    raw = response.content.strip()
    
    lines = raw.split("\n")
    expanded = []
    
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            cleaned = line.split(".", 1)[-1].strip()
            if cleaned:
                expanded.append(cleaned)
    
    expanded.append(student_prompt)
    return expanded
    