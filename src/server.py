import io
import json
import time
import asyncio

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
            time_per_prompt=5 * 60, # 10 minutes
            max_retries=5,
            diffuser_url="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
            llm_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        ) -> None:
        super().__init__(max_retries, diffuser_url, llm_url)

        self.min_score = min_score
        self.time_per_prompt = time_per_prompt

    async def init_client(self, session: str) -> None:
        if await self.redis_conn.exists(session): await self.redis_conn.delete(session)
        contents = {'max': self.min_score, 'current': self.min_score, 'status': 'idle'}
        await self.redis_conn.hset(session, mapping=contents)
        await self.redis_conn.sadd('sessions', session)

    async def fetch_next_prompt(self) -> str:
        prompts = await self.redis_conn.hget('prompt', 'next')
        assert prompts is not None, "[ERROR] No next prompt available"

        prompts = json.loads(prompts.decode())
        return reconstruct_sentence(prompts['tokens'])

    async def fetch_prompt_json(self) -> str:
        prompts = await self.redis_conn.hget('prompt', 'next')
        assert prompts is not None, "[ERROR] No current prompt available"

        prompts = json.loads(prompts.decode())
        for mask in sorted(prompts['masks']): prompts['tokens'][mask] = '*'
        return prompts

    async def fetch_masked_words(self) -> List[str]:
        prompts = await self.redis_conn.hget('prompt', 'next')
        assert prompts is not None, "[ERROR] No next prompt available"

        prompts = json.loads(prompts.decode())
        tokens, masks = prompts['tokens'], prompts['masks']
        words = []
        for mask in sorted(masks): words.append(tokens[mask])
        return words

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
        scores = (await self.compute_scores(inputs, words)).get("scores")
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

    async def reset_sessions(self) -> None:
        sessions = await self.redis_conn.smembers('sessions')
        contents = {'max': self.min_score, 'current': self.min_score, 'status': 'idle'}
        for session in sessions: await self.redis_conn.hset(session, mapping=contents)

    async def start_countdown(self) -> None:
        await self.redis_conn.setex('countdown', self.time_per_prompt, 'active')

    async def fetch_countdown(self) -> float:
        return float(await self.redis_conn.ttl('countdown'))

    async def fetch_clock(self) -> str:
        seconds = int(await self.fetch_countdown())
        return format_seconds_to_time(seconds)

    async def reset_clock(self) -> None:
        await self.start_countdown()
    
    async def global_timer(self) -> None:
        # Start the countdown
        await self.start_countdown()
        await asyncio.sleep(1)
        
        while True:
            # Fetch remaining time
            remaining_time = await self.fetch_countdown()

            # Check if time to generate new prompt
            if int(remaining_time) == int(self.time_per_prompt * 0.7):
                await self.buffer_contents()

            # Check if time's up
            if remaining_time <= 0.5:
                await self.promote_buffer()
                await self.reset_sessions()
                await self.reset_clock()
                await self.redis_conn.setex("reset", 1, 1)
            
            await asyncio.sleep(1)