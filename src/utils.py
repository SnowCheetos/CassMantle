import gc
import io
import nltk
import string
import aiohttp
import asyncio
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Any, Dict

def encode_image(image: Image.Image) -> bytes:
    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format='JPEG')
    image_bytes = image_bytes_io.getvalue()
    return image_bytes

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

async def api_call(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json_payload: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
    retry_on_status_codes: Optional[set[int]] = None,
) -> Optional[aiohttp.ClientResponse]:
    retry_on_status_codes = retry_on_status_codes or {503}

    for retry in range(max_retries):
        try:
            print("[INFO] Making request...")
            async with session.request(
                method, url, headers=headers, json=json_payload, ssl=False
            ) as response:
                print("[INFO] Request complete.")
                r = await response.read()
                response.close()
                return r

        except aiohttp.ClientResponseError as e:
            if e.status in retry_on_status_codes:
                gc.collect()
                print(
                    f"Retry {retry + 1}/{max_retries}: "
                    f"Status code {e.status} received from {url}"
                )
                await asyncio.sleep((retry + 1) * 10)
                continue
            else:
                print(f"HTTP error occurred: {e}")
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            break
    
    print("Max retries reached or an error occurred.")
    return None

def semantic_distance(word, word_list, model):
    if word in model:
        word_vector = model[word]
        mean_vector = np.mean([model[w] for w in word_list if w in model], axis=0)
        return np.linalg.norm(word_vector - mean_vector)
    return 0

def select_descriptive_words(model, sentence, num_words=2):
    # Tokenize and POS tag
    words = nltk.word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)

    # Filter out stopwords and get only adjectives, adverbs, and nouns
    descriptive_tags = ['JJ', 'RB', 'NN', 'NNS', 'JJR', 'JJS', 'RBR', 'RBS']
    filtered_words = [word for word, pos in tagged_words if word.isalpha() and pos in descriptive_tags]

    # Calculate semantic distances
    distances = [semantic_distance(word, filtered_words, model) for word in filtered_words]

    # Use IDF for statistical weighing
    vectorizer = TfidfVectorizer().fit([sentence])
    idf_scores = vectorizer.idf_
    idf_dict = dict(zip(vectorizer.get_feature_names_out(), idf_scores))
    
    # Calculate final scores for words
    scores = [distances[i] * idf_dict.get(filtered_words[i], 1) for i in range(len(filtered_words))]
    
    # Select indices of words with the highest scores in the original sentence
    selected_indices = [words.index(filtered_words[i]) for i in np.argsort(scores)[-num_words:]]
    
    return words, sorted(selected_indices)

def construct_prompt_dict(model, prompt: str, num_masked: int) -> Dict[str, List[str]]:
    words, masks = select_descriptive_words(model, prompt, num_masked)
    return {
        'tokens': words,
        'masks': masks
    }