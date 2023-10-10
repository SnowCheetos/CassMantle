import io
import nltk
import string
import random
import httpx
import asyncio
from nltk.tokenize import word_tokenize
from nltk.corpus import brown, stopwords
from PIL import Image
from typing import List, Optional, Any, Dict

freq_dist = nltk.FreqDist(w.lower() for w in brown.words())

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

def weighted_sample_without_replacement(items, weights, k):
    # Flatten the list
    flat_list = [item for item, weight in zip(items, weights) for _ in range(int(weight*100))]
    
    # Select without replacement
    return random.sample(flat_list, k)

async def api_call(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
    timeout: int = 10,
    retry_on_status_codes: Optional[set[int]] = None,
) -> Optional[httpx.Response]:
    """
    Perform an API call with retries.

    :param method: HTTP method
    :param url: URL to call
    :param headers: Headers to include in the request
    :param json: JSON payload to include in the request
    :param max_retries: Maximum number of retries
    :param timeout: Request timeout
    :param retry_on_status_codes: Set of HTTP status codes that should trigger a retry
    :return: httpx.Response object or None if call was unsuccessful
    """
    retry_on_status_codes = retry_on_status_codes or {503}

    for retry in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method, url, headers=headers, json=json
                )
                response.raise_for_status()
                return response

        except httpx.HTTPStatusError as e:
            if e.response.status_code in retry_on_status_codes:
                print(
                    f"Retry {retry + 1}/{max_retries}: "
                    f"Status code {e.response.status_code} received from {url}"
                )
                await asyncio.sleep(retry * 2)  # Implementing exponential backoff
                continue
            else:
                print(f"HTTP error occurred: {e}")
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            break
    
    print("Max retries reached or an error occurred.")
    return None

def word_complexity(word: str) -> int:
    # Use a large number to ensure that less frequent words get higher values
    LARGE_NUM = 1e6

    # The less frequent the word, the higher the complexity score from the freq_dist
    freq_score = LARGE_NUM - freq_dist[word.lower()]

    # Combine the frequency score and the word length
    return freq_score + len(word)

def select_descriptive_words(inputs: str, prompt: str, num_words: int=2) -> List[str]:
    # Load stop words
    stop_words = set(stopwords.words('english'))
    
    # Tokenize and POS tag words
    words = word_tokenize(prompt)
    skips = word_tokenize(inputs)
    
    tagged_words = nltk.pos_tag(words)
    
    # Filter words
    filtered_words = [
        word for word, pos in tagged_words
        if word.lower() not in stop_words
        and all(char not in string.punctuation for char in word)  # Exclude punctuation
        and "'" not in word
        and "-" not in word
        and pos not in ['NNP', 'NNPS']  # Exclude proper nouns
        and word not in skips
    ]

    word_scores = {word: word_complexity(word) for word in filtered_words}

    selected_words = weighted_sample_without_replacement(
        list(word_scores.keys()),
        list(word_scores.values()),
        min(num_words, len(filtered_words))
    )

    indices = [words.index(word) for word in selected_words]
    return indices

def construct_prompt_dict(input_text: str, prompt: str, num_masked: int) -> Dict[str, List[str]]:
    masks = select_descriptive_words(input_text, prompt, num_masked)
    return {
        'tokens': word_tokenize(prompt),
        'masks': masks
    }