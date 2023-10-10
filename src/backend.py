import io
import json
import asyncio
import random
import aioredis

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
            max_retries=5,
            diffuser_url="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
            llm_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        ) -> None:
        
        with open('api_key.txt', 'r') as f: API_TOKEN = f.readline()
        
        self.seeds = []
        with open('data/seeds.txt') as f:
            for line in f.readlines():
                self.seeds.append(line)

        self.max_retries = max_retries
        self.diffuser_url = diffuser_url
        self.llm_url = llm_url
        self.auth_header = {"Authorization": f"Bearer {API_TOKEN}"}

    async def select_seed(self) -> str:
        return self.seeds[random.randint(0, len(self.seeds) - 1)]

    async def initialize_redis(self) -> aioredis.Redis:
        return await aioredis.Redis(host='localhost', decode_responses=False)

    async def startup(self) -> None:
        self.redis_conn = await self.initialize_redis()

        await self.redis_conn.hset('prompt', 'status', 'idle')
        await self.redis_conn.hset('image', 'status', 'idle')

        seed = await self.select_seed()
        async with self.redis_conn.lock("startup_lock", timeout=60):
            if (
                await self.redis_conn.hget('prompt', 'current') is None
                and
                await self.redis_conn.hget('image', 'current') is None
            ):
                prompt = await self.init_prompt(seed)
                await self.init_image(prompt)
                print("[INFO] Content initialization complete")

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

    async def buffer_contents(self) -> None:
        seed = await self.select_seed()
        async with self.redis_conn.lock("buffer_lock", timeout=60):
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

    async def promote_buffer(self) -> None:
        async with self.redis_conn.lock("promotion_lock", timeout=10):
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

    async def generate_prompt(self, seed: str) -> str:
        print("[INFO] Generating prompt...")
        await self.redis_conn.hset('prompt', 'status', 'busy')

        response = await api_call(
            method="POST",
            url=self.llm_url,
            headers=self.auth_header,
            json={
                "inputs": seed,
                "parameters": {"max_new_tokens": 64}
            },
            max_retries=self.max_retries,
            timeout=90,
            retry_on_status_codes={503},
        )
        await self.redis_conn.hset('prompt', 'status', 'idle')

        if response is not None:
            return '.'.join(response.json()[0].get('generated_text').split('.')[:2]) + '.'
        else:
            print("[ERROR] Prompt generation failed.")
            return None

    async def generate_image(self, prompt: str) -> Image.Image:
        print("[INFO] Generating image...")
        await self.redis_conn.hset('image', 'status', 'busy')

        response = await api_call(
            method="POST",
            url=self.diffuser_url,
            headers=self.auth_header,
            json={
                "inputs": prompt,
                "parameters": {'negative_prompt': 'blurry, distorted, fake, abstract, negative'}
            },
            max_retries=self.max_retries,
            timeout=90,
            retry_on_status_codes={503},
        )
        await self.redis_conn.hset('image', 'status', 'idle')

        if response is not None:
            return Image.open(io.BytesIO(response.content))
        else:
            print("[ERROR] Image generation failed.")
            return None
        
    async def compute_scores(self, pairs: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
        response = await api_call(
            method="POST",
            url='http://localhost:9000/compute_scores',
            # headers=self.auth_header,
            json=pairs,
            timeout=3,
            retry_on_status_codes={503},
        )
        if response is not None: return response.json()
        else: raise Exception("[ERROR] Score calculation failed.")

    def score_to_blur(self, score: float, min_blur: float=0.0, max_blur: float=20):
        return min_blur + (1 - score ** 2) * (max_blur - min_blur)

    def mask_image(self, image: Image.Image, score: float) -> Image.Image:
        blur = self.score_to_blur(score)
        return image.filter(ImageFilter.GaussianBlur(blur))


if __name__ == "__main__":
    b = Backend()
    asyncio.run(b.startup())
