import io
import gc
import json
import asyncio
import random
import aiohttp
import aioredis

from aioredis.exceptions import LockError
from gensim.models import KeyedVectors
from PIL import Image, ImageFilter
from typing import List, Dict, Tuple
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
        self.num_masked = 2
        self.seed_epsilon = 0.2

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
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=60), 
                        raise_for_status=True
                    ) as http_session:
                        
                        # prompt = await self.init_prompt(http_session, seed)
                        # gc.collect()

                        # await self.init_image(http_session, prompt)
                        # gc.collect()

                        prompt = await self.generate_prompt(http_session, seed, True)
                        gc.collect()
                        assert prompt is not None, "[ERROR] Prompt generation failed"

                        image = await self.generate_image(http_session, prompt)
                        gc.collect()
                        assert image is not None, "[ERROR] Image generation failed"

                    await self.redis_conn.hset('prompt', 'seed', prompt)
                    prompt_dict = construct_prompt_dict(self.wv, prompt, self.num_masked)

                    await self.redis_conn.hset('prompt', 'current', json.dumps(prompt_dict))

                    encoding = encode_image(image)
                    await self.redis_conn.hset('image', 'current', encoding)

                    print("[INFO] Content initialization complete")
                    return
        
        except LockError:
            print("[INFO] Worker could not acquire lock, moving on.")
            return

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {str(e)}")
            return

    # async def init_prompt(self, http_session: aiohttp.ClientSession, seed: str) -> str:
    #     prompt = await self.generate_prompt(http_session, seed, True)

    #     assert prompt is not None, "[ERROR] Prompt generation failed"

    #     await self.redis_conn.hset('prompt', 'seed', prompt)
    #     prompt_dict = construct_prompt_dict(seed, prompt, self.num_masked)

    #     await self.redis_conn.hset('prompt', 'current', json.dumps(prompt_dict))
    #     return prompt

    # async def init_image(self, http_session: aiohttp.ClientSession, prompt: str) -> None:
    #     image = await self.generate_image(http_session, prompt)

    #     assert image is not None, "[ERROR] Image generation failed"
    #     encoding = encode_image(image)

    #     await self.redis_conn.hset('image', 'current', encoding)

    async def set_next_prompt(self, prompt: str) -> None:
        await self.redis_conn.hset('prompt', 'next', prompt)

    async def set_next_image(self, image: bytes) -> None:
        await self.redis_conn.hset('image', 'next', image)

    async def random_seed(self) -> Tuple[bool, str]:
        if random.random() > self.seed_epsilon:
            # Use current prompt
            seed = (await self.redis_conn.hget('prompt', 'seed')).decode()
            return False, seed
        else:
            seed = await self.select_seed()
            return True, seed

    async def buffer_contents(self) -> None:
        is_seed, seed = await self.random_seed()
        if is_seed: print("[INFO] Restarting storyline.")
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

                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=60), 
                        raise_for_status=True
                    ) as http_session:
                        
                        prompt = await self.generate_prompt(http_session, seed, is_seed)
                        gc.collect()
                        assert prompt is not None, "[ERROR] Prompt generation failed"

                        image = await self.generate_image(http_session, prompt)
                        gc.collect()
                        assert image is not None, "[ERROR] Image generation failed"

                    await self.redis_conn.hset('prompt', 'seed', prompt)

                    prompt_dict = json.dumps(construct_prompt_dict(self.wv, prompt, self.num_masked))
                    await self.set_next_prompt(prompt_dict)

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

    async def generate_prompt(self, http_session: aiohttp.ClientSession, seed: str, is_seed: bool) -> str:
        print("[INFO] Generating prompt...")
        await self.redis_conn.hset('prompt', 'status', 'busy')

        response = await api_call(
            http_session,
            method="POST",
            url=self.llm_url,
            headers=self.auth_header,
            json_payload={
                "inputs": seed,
                "parameters": {
                    "min_new_tokens": 32,
                    "max_new_tokens": 96
                }
            },
            max_retries=self.max_retries,
            retry_on_status_codes={503},
        )
        await self.redis_conn.hset('prompt', 'status', 'idle')

        if response is not None:
            await asyncio.sleep(0)
            if is_seed:
                return '.'.join(json.loads(response)[0].get('generated_text').split('.')[:2]) + '.'
            else:
                return '.'.join(json.loads(response)[0].get('generated_text')[len(seed)+1:].split('.')[:2]) + '.'
        else:
            print("[ERROR] Prompt generation failed.")
            return None

    async def generate_image(self, http_session: aiohttp.ClientSession, prompt: str) -> Image.Image:
        style = await self.select_style()
        seed = f"A {style.lower()} style piece depicting the following: "

        print(f"[INFO] Generating image...")
        await self.redis_conn.hset('image', 'status', 'busy')
        
        response = await api_call(
            http_session,
            method="POST",
            url=self.diffuser_url,
            headers=self.auth_header,
            json_payload={
                "inputs": seed + prompt,
                "parameters": {'negative_prompt': 'blurry, distorted, fake, abstract, negative'}
            },
            max_retries=self.max_retries,
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
