import io
import json
import time
import redis
import redis.lock
import asyncio
import requests

from PIL import Image
from typing import List, Dict
from src.backend import Backend
from src.utils import format_seconds_to_time, reconstruct_sentence

class Server(Backend):
    """
    This class is the implementation of API server logic, which inherits from the Backend class.
    It handles a lot of the Redis operations.
    """
    def __init__(
            self, 
            min_score=0.1,
            time_per_prompt=20 * 60, # 20 minutes
            max_retries=5,
            rabbit_host='localhost',
            diffuser_url="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
        ) -> None:
        super().__init__(max_retries, rabbit_host, diffuser_url)

        self.min_score = min_score
        self.time_per_prompt = time_per_prompt

    async def init_client(self, session: str) -> None:
        if await self.redis_conn.exists(session): await self.redis_conn.delete(session)
        contents = {'max': self.min_score, 'current': self.min_score, 'status': 'idle'}
        await self.redis_conn.hset(session, mapping=contents)
        await self.redis_conn.sadd('sessions', session)

    async def fetch_next_prompt(self) -> str:
        prompts = (await self.redis_conn.hget('prompt', 'next')).decode()
        if not prompts: return
        prompts = json.loads(prompts)
        return reconstruct_sentence(prompts['tokens'])

    async def fetch_prompt_json(self) -> str:
        prompts = json.loads((await self.redis_conn.hget('prompt', 'current')).decode())
        for mask in sorted(prompts['masks']):
            prompts['tokens'][mask] = '*'
        return prompts

    async def fetch_masked_words(self) -> List[str]:
        prompts = json.loads((await self.redis_conn.hget('prompt', 'current')).decode())
        tokens, masks = prompts['tokens'], prompts['masks']
        words = []
        for mask in sorted(masks):
            words.append(tokens[mask])
        return words

    def compute_scores(self, inputs: str, answer: str) -> float:
        payload = {'inputs': inputs, 'answer': answer}
        response = requests.post('http://localhost:9000/compute_scores', json=payload)
        if response.ok:
            return response.json()
        raise Exception("Unable to connect to scoring service.")

    async def set_index_score(self, session: str, index: int, score: float) -> None:
        field = f'score{index+1}'
        await self.redis_conn.hset(session, field, float(score))

    async def set_client_score(self, session: str, score: float) -> None:
        await self.redis_conn.hset(session, 'current', score)
        max_score = float(await self.redis_conn.hget(session, 'max'))
        if not max_score: await self.redis_conn.hset(session, 'max', score)
        elif score > max_score: await self.redis_conn.hset(session, 'max', score)

    async def compute_client_scores(self, session: str, inputs: List[str]) -> Dict[str, str]:
        words = await self.fetch_masked_words()
        scores = self.compute_scores(inputs, words).get("scores")
        results = {}
        for i, score in enumerate(scores):
            await self.set_index_score(session, i, float(score))
            results.update({f'score{i}': score})
        mean_score = sum([float(s) for s in scores]) / len(scores)
        await self.set_client_score(session, mean_score)
        return results

    async def fetch_client_scores(self, session: str) -> Dict[str, str]:
        while (await self.redis_conn.hget(session, 'status')).decode() == 'busy':
            time.sleep(0.1)
        contents = await self.redis_conn.hgetall(session)
        contents = {key.decode(): value.decode() for key, value in contents.items()}
        return contents
    
    async def fetch_current_image(self) -> Image.Image:
        image_bytes = await self.redis_conn.hget('image', 'current')
        return Image.open(io.BytesIO(image_bytes))

    async def fetch_masked_image(self, session: str) -> Image.Image:
        scores = await self.fetch_client_scores(session)
        image = await self.fetch_current_image()
        masked = self.mask_image(image, float(scores['max']))
        return masked
    
    async def fetch_init_image(self) -> Image.Image:
        image = await self.fetch_current_image()
        masked = self.mask_image(image, self.min_score)
        return masked

    async def update_contents(self) -> bool:
        image = await self.redis_conn.hget('image', 'next')
        prompt = await self.redis_conn.hget('prompt', 'next')
        if not image or not prompt: return False

        await self.redis_conn.hset('image', 'current', image)
        await self.redis_conn.hset('prompt', 'current', prompt)
        await self.redis_conn.hdel('image', 'next')
        await self.redis_conn.hdel('prompt', 'next')

        sessions = await self.redis_conn.smembers('sessions')
        contents = {'max': self.min_score, 'current': self.min_score, 'status': 'idle'}
        for session in sessions:
            await self.redis_conn.hset(session, mapping=contents)

        return True

    async def start_countdown(self) -> None:
        await self.redis_conn.setex('countdown', self.time_per_prompt, 'active')

    async def fetch_countdown(self) -> float:
        return float(await self.redis_conn.ttl('countdown'))

    async def fetch_clock(self) -> str:
        seconds = int(await self.fetch_countdown())
        return format_seconds_to_time(seconds)

    async def reset_clock(self) -> None:
        await self.start_countdown()

    async def locked_generate_prompt(self) -> None:
        async with await self.redis_conn.lock("generation_lock", timeout=5):
            if await self.redis_conn.get('busy') is None:
                await self.redis_conn.setex('busy', 5, 1)
                print("GENERATING PROMPT")
                if await (self.redis_conn.hget('prompt', 'status')).decode() == 'idle':
                    await self.generate_prompt()

    # def locked_generate_image(self) -> None:
    #     with self.redis_conn.lock("generation_lock", timeout=5):
    #         if self.redis_conn.get('busy') is None:
    #             self.redis_conn.setex('busy', 5, 1)
    #             print("GENERATING IMAGE")
    #             image_status = self.redis_conn.hget('image', 'status').decode()
    #             next_prompt = self.fetch_next_prompt()
    #             if image_status == 'idle' and next_prompt:
    #                 self.generate_image(next_prompt)

    async def locked_generate_image(self) -> None:
        async with await self.redis_conn.lock("generation_lock", timeout=5):
            if await self.redis_conn.get('busy') is None:
                await self.redis_conn.setex('busy', 5, 1)
                print("GENERATING IMAGE")
                image_status = (await self.redis_conn.hget('image', 'status')).decode()
                next_prompt = await self.fetch_next_prompt()
                if image_status == 'idle' and next_prompt:
                    asyncio.create_task(self.generate_image(next_prompt))
                
    async def global_timer(self) -> None:
        # Start the countdown
        await self.start_countdown()
        await asyncio.sleep(1)
        
        while True:
            # Fetch remaining time
            remaining_time = await self.fetch_countdown()

            # Check if time to generate new prompt
            if int(remaining_time) == int(self.time_per_prompt * 0.8):
                await self.locked_generate_prompt()

            if int(remaining_time) == int(self.time_per_prompt * 0.4):
                await self.locked_generate_image()

            # Check if time's up
            if remaining_time <= 0.5:
                async with await self.redis_conn.lock("update_lock", timeout=1):
                    await self.redis_conn.setex("reset", 1, 1)
                    if await self.update_contents():
                        print(f'[INFO] Resetting...')
                        await self.reset_clock()
            
            await asyncio.sleep(1)