def load_transcript(filepath):
    chunks = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            
            try:
                start = float(parts[0])
                end = float(parts[1])
                text = parts[2].strip()
                
                if text:
                    chunks.append({
                        "text": text,
                        "start": start,
                        "end": end
                    })
            except ValueError:
                continue
    
    return chunks


def merge_chunks(chunks, window_size=5):
    merged = []
    i = 0
    
    while i < len(chunks):
        window = chunks[i:i+window_size]
        combined_text = " ".join([c["text"] for c in window])
        start = window[0]["start"]
        end = window[-1]["end"]
        
        merged.append({
            "text": combined_text,
            "start": start,
            "end": end
        })
        
        i += window_size - 1
    
    return merged