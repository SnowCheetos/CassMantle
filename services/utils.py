import string
from typing import List

def reconstruct_sentence(tokens: List[str]) -> str:
    sentence = ""
    for token in tokens:
        if token in string.punctuation or "-" in token or "'" in token:
            if token != 'a' or token != 'an':
                sentence += token
        else:
            sentence += " " + token
    return sentence.strip()

def format_seconds_to_time(seconds: int) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"