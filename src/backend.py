import io
import json
import asyncio
import random
import aioredis

from aioredis.exceptions import LockError
from gensim.models import KeyedVectors
from PIL import Image, ImageFilter
from typing import List, Dict
from src.utils import encode_image, api_call, construct_prompt_dict

class Backend:
    """
    This class is the parent class to Server.
    It handles content generation by making calls to external APIs.
    """
    def __init__(
            self, 
            min_score=0.1,
            max_retries=5,
            diffuser_url="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
            llm_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        ) -> None:
        
        self.min_score = min_score
        with open('api_key.txt', 'r') as f: API_TOKEN = f.readline().strip()
        
        self.seeds = []
        with open('data/seeds.txt', 'r') as f:
            for line in f.readlines():
                self.seeds.append(line.strip())

        self.styles = []
        with open('data/styles.txt', 'r') as f:
            for line in f.readlines():
                self.styles.append(line.strip())

        # self.http_session = aiohttp.ClientSession()
        self.max_retries = max_retries
        self.diffuser_url = diffuser_url
        self.llm_url = llm_url
        self.auth_header = {"Authorization": f"Bearer {API_TOKEN}"}
        self.wv = KeyedVectors.load("data/word2vec.wordvectors", mmap='r')

        self.lock_timeout = 120
        self.acquire_timeout = 2

    async def select_style(self) -> str:
        return self.styles[random.randint(0, len(self.styles) - 1)]

    async def select_seed(self) -> str:
        return self.seeds[random.randint(0, len(self.seeds) - 1)]

    async def initialize_redis(self) -> aioredis.Redis:
        return await aioredis.Redis(host='localhost', decode_responses=False)

    async def startup(self) -> None:
        await asyncio.sleep(random.random())
        self.redis_conn = await self.initialize_redis()

        await self.redis_conn.hset('prompt', 'status', 'idle')
        await self.redis_conn.hset('image', 'status', 'idle')

        seed = await self.select_seed()
        try:
            async with self.redis_conn.lock(
                "startup_lock", 
                timeout=self.lock_timeout,
                blocking_timeout=self.acquire_timeout
            ):
                if (
                    await self.redis_conn.hget('prompt', 'current') is None
                    and
                    await self.redis_conn.hget('image', 'current') is None
                ):
                    prompt = await self.init_prompt(seed)
                    await self.init_image(prompt)
                    print("[INFO] Content initialization complete")
        
        except LockError:
            print("[INFO] Worker could not acquire lock, moving on.")
            return

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {str(e)}")
            return

    async def init_prompt(self, seed: str) -> str:
        prompt = await self.generate_prompt(seed)
        assert prompt is not None, "[ERROR] Prompt generation failed"
        prompt_dict = construct_prompt_dict(seed, prompt, 3)
        await self.redis_conn.hset('prompt', 'current', json.dumps(prompt_dict))
        return prompt

    async def init_image(self, prompt: str) -> None:
        image = await self.generate_image(prompt)
        assert image is not None, "[ERROR] Image generation failed"
        encoding = encode_image(image)
        await self.redis_conn.hset('image', 'current', encoding)

    async def set_next_prompt(self, prompt: str) -> None:
        await self.redis_conn.hset('prompt', 'next', prompt)

    async def set_next_image(self, image: bytes) -> None:
        await self.redis_conn.hset('image', 'next', image)

    async def check_generate_condition(self) -> bool:
        return (
            await self.redis_conn.hget('prompt', 'status').decode() == 'idle'
            and
            await self.redis_conn.hget('image', 'status').decode() == 'idle'
        )

    async def buffer_contents(self) -> None:
        seed = await self.select_seed()
        try:
            async with self.redis_conn.lock(
                "buffer_lock", 
                timeout=self.lock_timeout,
                blocking_timeout=self.acquire_timeout
            ):
                if (
                    await self.redis_conn.hget('prompt', 'next') is None
                    and
                    await self.redis_conn.hget('image', 'next') is None
                ):
                    print("[INFO] Generating content buffer")
                    prompt = await self.generate_prompt(seed)
                    assert prompt is not None, "[ERROR] Prompt generation failed"

                    prompt_dict = json.dumps(construct_prompt_dict(seed, prompt, 3))
                    await self.set_next_prompt(prompt_dict)

                    image = await self.generate_image(prompt)
                    assert image is not None, "[ERROR] Image generation failed"

                    encoding = encode_image(image)
                    await self.set_next_image(encoding)
                    print("[INFO] Content buffering complete")
        
        except LockError:
            print("[INFO] Worker could not acquire lock, moving on.")
            return

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {str(e)}")
            return

    async def promote_buffer(self) -> None:
        try:
            async with self.redis_conn.lock(
                "promotion_lock", 
                timeout=self.lock_timeout,
                blocking_timeout=self.acquire_timeout
            ):
                if (
                    await self.redis_conn.hget('prompt', 'next') is not None
                    and
                    await self.redis_conn.hget('image', 'next') is not None
                ):
                    print("[INFO] Promoting content buffer")
                    image = await self.redis_conn.hget('image', 'next')
                    prompt = await self.redis_conn.hget('prompt', 'next')
                    assert image is not None and prompt is not None, "[ERROR] Content buffer error, NoneType encountered"

                    await self.redis_conn.hset('image', 'current', image)
                    await self.redis_conn.hset('prompt', 'current', prompt)
                    await self.redis_conn.hdel('image', 'next')
                    await self.redis_conn.hdel('prompt', 'next')
                    print("[INFO] Buffer promotion complete")

        except LockError:
            print("[INFO] Worker could not acquire lock, moving on.")
            return

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {str(e)}")
            return

    async def generate_prompt(self, seed: str) -> str:
        print("[INFO] Generating prompt...")
        await self.redis_conn.hset('prompt', 'status', 'busy')

        response = await api_call(
            # self.http_session,
            method="POST",
            url=self.llm_url,
            headers=self.auth_header,
            json_payload={
                "inputs": seed,
                "parameters": {
                    "min_new_tokens": 32,
                    "max_new_tokens": 96,
                    "do_sample": True,
                    "num_beams": 5
                }
            },
            max_retries=self.max_retries,
            timeout=60,
            retry_on_status_codes={503},
        )
        await self.redis_conn.hset('prompt', 'status', 'idle')

        if response is not None:
            return '.'.join(json.loads(response)[0].get('generated_text').split('.')[:2]) + '.'
        else:
            print("[ERROR] Prompt generation failed.")
            return None

    async def generate_image(self, prompt: str) -> Image.Image:
        await self.redis_conn.hset('image', 'status', 'busy')
        style = await self.select_style()
        print(f"[INFO] Generating image with {style} style...")
        response = await api_call(
            # self.http_session,
            method="POST",
            url=self.diffuser_url,
            headers=self.auth_header,
            json_payload={
                "inputs": prompt + f' {style} style.',
                "parameters": {'negative_prompt': 'blurry, distorted, fake, abstract, negative, weird, bad'}
            },
            max_retries=self.max_retries,
            timeout=90,
            retry_on_status_codes={503},
        )
        await self.redis_conn.hset('image', 'status', 'idle')

        if response is not None:
            return Image.open(io.BytesIO(response))
        else:
            print("[ERROR] Image generation failed.")
            return None
    
    def most_similar(self, word: str, topn: int=50) -> List[str]:
        try:
            return self.wv.most_similar(word.lower(), topn=topn)
        except KeyError as e:
            raise Exception("Word not in dictionary. ", e)

    def compute_score(self, inputs: str, answer: str) -> float:
        inputs, answer = inputs.lower(), answer.lower()
        if inputs == answer: return 1.0
        try:
            score = self.wv.similarity(inputs, answer)
        except KeyError:
            score = self.min_score
        return max(self.min_score, score)

    async def compute_scores(self, data: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
        scores = {}
        for key in data.keys():
            score = self.compute_score(data[key]['input'], data[key]['answer'])
            scores.update({key: str(score)})
        return scores

    def score_to_blur(self, score: float, min_blur: float=0.0, max_blur: float=15):
        return min_blur + (1 - score ** 2) * (max_blur - min_blur)

    def mask_image(self, image: Image.Image, score: float) -> Image.Image:
        blur = self.score_to_blur(score)
        return image.filter(ImageFilter.GaussianBlur(blur))


# if __name__ == "__main__":
#     b = Backend()
#     asyncio.run(b.startup())
