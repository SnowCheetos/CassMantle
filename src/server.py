import io
import json
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
            min_score=0.01,
            time_per_prompt=600, # 10 minutes
            max_retries=5,
            diffuser_url="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
            llm_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        ) -> None:
        super().__init__(min_score, max_retries, diffuser_url, llm_url)
        self.time_per_prompt = time_per_prompt

    async def init_client(self, session: str) -> None:
        await self.reset_client(session)
        await self.redis_conn.sadd('sessions', session)

    async def reset_client(self, session: str) -> None:
        await self.redis_conn.delete(session)
        prompt = await self.fetch_current_prompt()
        contents = {'max': self.min_score, 'won': 0, 'attempts': 0}
        for m in prompt['masks']: contents.update({str(m): 0.0})
        await self.redis_conn.hset(session, mapping=contents)

    async def increment_attempt(self, session: str) -> None:
        await self.redis_conn.hincrby(session, 'attempts', 1)

    async def fetch_current_image(self) -> Image.Image:
        image_bytes = await self.redis_conn.hget('image', 'current')
        assert image_bytes is not None, "[ERROR] No current image available"
        return Image.open(io.BytesIO(image_bytes))

    async def fetch_current_prompt(self) -> Dict[str, List[str]]:
        prompts = await self.redis_conn.hget('prompt', 'current')
        assert prompts is not None, "[ERROR] No current prompt available"
        return json.loads(prompts.decode())

    async def compute_client_scores(self, session: str, inputs: Dict[str, str]) -> Dict[str, str]:
        prompt = await self.fetch_current_prompt()
        pairs = {}
        for m in inputs.keys():
            pairs.update({
                m: {
                    "input": inputs[m],
                    "answer": prompt['tokens'][int(m)]
                }
            })
        scores = await self.compute_scores(pairs)
        scores = await self.set_client_scores(session, scores)
        await self.increment_attempt(session)
        return scores

    async def set_client_scores(self, session: str, scores: Dict[str, str]) -> Dict[str, str]:
        curr_scores = await self.fetch_client_scores(session)
        mean_score = sum([float(s) for s in scores.values()]) / len(scores)
        if float(mean_score) > float(curr_scores['max']):
            await self.redis_conn.hset(session, 'max', mean_score)

        for key in scores.keys():
            await self.redis_conn.hset(session, key, scores[key])
        
        await self.redis_conn.hset(session, 'won', int(mean_score == 1))
        scores.update({'won': int(mean_score == 1)})
        return scores

    async def fetch_client_scores(self, session: str) -> Dict[str, str]:
        contents = await self.redis_conn.hgetall(session)
        contents = {key.decode(): value.decode() for key, value in contents.items()}
        return contents

    async def fetch_prompt_json(self, session_id: str) -> str:
        prompt = await self.fetch_current_prompt()
        scores = await self.fetch_client_scores(session_id)
        attempts = int((await self.redis_conn.hget(session_id, 'attempts')).decode())

        if (await self.redis_conn.hget(session_id, 'won')).decode() == "1":
            prompt['masks'] = []

        else:
            for i, mask in enumerate(prompt['masks']): 
                score = scores.get(str(mask))
                if score:
                    if float(score) == 1.0:  
                        prompt['masks'][i] = -1
                    else:
                        prompt['tokens'][mask] = '*'
                else:
                    prompt['tokens'][mask] = '*'

        prompt.update({'scores': scores, 'attempts': attempts})
        return prompt

    async def fetch_masked_image(self, session: str) -> Image.Image:
        scores = await self.fetch_client_scores(session)
        image = await self.fetch_current_image()
        masked = self.mask_image(image, float(scores['max']))
        return masked

    async def reset_sessions(self) -> None:
        sessions = await self.redis_conn.smembers('sessions')
        for session in sessions: await self.reset_client(session)

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