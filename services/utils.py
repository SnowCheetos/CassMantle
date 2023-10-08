import string

from typing import List

def reconstruct_sentence(tokens: List[str]) -> str:
    sentence = ""
    for token in tokens:
        if token in string.punctuation or "-" in token or "'" in token:
            sentence += token
        else:
            sentence += " " + token
    return sentence.strip()