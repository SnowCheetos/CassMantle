import io
import string
import random
import httpx
import asyncio
from PIL import Image
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
