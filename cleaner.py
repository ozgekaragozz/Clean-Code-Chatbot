import re

def clean_text(raw_text):
    return re.sub(r'\s+', ' ', raw_text).strip()

def segment_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words)), chunk_size]