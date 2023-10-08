import io
import json
import time
import redis
import redis.lock
import asyncio

from PIL import Image
from typing import List, Dict
from services.backend import Backend
from services.utils import format_seconds_to_time, reconstruct_sentence

class Server(Backend):
    """
    This class is the implementation of API server logic, which inherits from the Backend class.
    It handles a lot of the Redis operations.
    """
    def __init__(
            self, 
            min_score=0.1,
            time_per_prompt=15 * 60, # 15 minutes
            rabbit_host='localhost'
        ) -> None:
        super().__init__(rabbit_host)

        self.min_score = min_score
        self.time_per_prompt = time_per_prompt
        self.redis_conn = redis.Redis(decode_responses=False)

    def init_client(self, session: str) -> None:
        if self.redis_conn.exists(session): self.redis_conn.delete(session)
        contents = {'max': self.min_score, 'current': self.min_score, 'status': 'idle'}
        self.redis_conn.hset(session, mapping=contents)
        self.redis_conn.sadd('sessions', session)

    def fetch_next_prompt(self) -> str:
        prompts = self.redis_conn.hget('prompt', 'next').decode()
        if not prompts: return
        prompts = json.loads(prompts)
        return reconstruct_sentence(prompts['tokens'])

    def fetch_prompt_json(self) -> str:
        prompts = json.loads(self.redis_conn.hget('prompt', 'current').decode())
        for mask in sorted(prompts['masks']):
            prompts['tokens'][mask] = '*'
        return prompts

    def fetch_masked_words(self) -> List[str]:
        prompts = json.loads(self.redis_conn.hget('prompt', 'current').decode())
        tokens, masks = prompts['tokens'], prompts['masks']
        words = []
        for mask in sorted(masks):
            words.append(tokens[mask])
        return words

    def fetch_client_scores(self, session: str) -> Dict[str, str]:
        while self.redis_conn.hget(session, 'status').decode() == 'busy':
            time.sleep(0.1)
        contents = self.redis_conn.hgetall(session)
        contents = {key.decode(): value.decode() for key, value in contents.items()}
        return contents
    
    def fetch_current_image(self) -> Image.Image:
        image_bytes = self.redis_conn.hget('image', 'current')
        return Image.open(io.BytesIO(image_bytes))

    def fetch_masked_image(self, session: str) -> Image.Image:
        scores = self.fetch_client_scores(session)
        image = self.fetch_current_image()
        masked = self.mask_image(image, float(scores['max']))
        return masked
    
    def fetch_init_image(self) -> Image.Image:
        image = self.fetch_current_image()
        masked = self.mask_image(image, self.min_score)
        return masked

    def compute_client_scores(self, session: str, inputs: List[str]) -> Dict[str, str]:
        words = self.fetch_masked_words()
        self.compute_scores(session, inputs, words)
        time.sleep(0.1)
        return self.fetch_client_scores(session)

    def update_contents(self) -> bool:
        image = self.redis_conn.hget('image', 'next')
        prompt = self.redis_conn.hget('prompt', 'next')
        if not image or not prompt: return False

        self.redis_conn.hset('image', 'current', image)
        self.redis_conn.hset('prompt', 'current', prompt)
        self.redis_conn.hdel('image', 'next')
        self.redis_conn.hdel('prompt', 'next')

        sessions = self.redis_conn.smembers('sessions')
        contents = {'max': self.min_score, 'current': self.min_score, 'status': 'idle'}
        for session in sessions:
            self.redis_conn.hset(session, mapping=contents)

        return True

    def start_countdown(self) -> None:
        self.redis_conn.setex('countdown', self.time_per_prompt, 'active')

    def fetch_countdown(self) -> float:
        return float(self.redis_conn.ttl('countdown'))

    def fetch_clock(self) -> str:
        seconds = int(self.fetch_countdown())
        return format_seconds_to_time(seconds)

    def reset_clock(self) -> None:
        self.start_countdown()

    def locked_generate_prompt(self) -> None:
        with self.redis_conn.lock("generation_lock", timeout=5):
            if self.redis_conn.get('busy') is None:
                self.redis_conn.setex('busy', 5, 1)
                print("GENERATING PROMPT")
                if self.redis_conn.hget('prompt', 'status').decode() == 'idle':
                    self.generate_prompt()

    def locked_generate_image(self) -> None:
        with self.redis_conn.lock("generation_lock", timeout=5):
            if self.redis_conn.get('busy') is None:
                self.redis_conn.setex('busy', 5, 1)
                print("GENERATING IMAGE")
                image_status = self.redis_conn.hget('image', 'status').decode()
                next_prompt = self.fetch_next_prompt()
                if image_status == 'idle' and next_prompt:
                    self.generate_image(next_prompt)
                
    async def global_timer(self) -> None:
        # Start the countdown
        self.start_countdown()
        await asyncio.sleep(1)
        
        while True:
            # Fetch remaining time
            remaining_time = self.fetch_countdown()

            # Check if time to generate new prompt
            if int(remaining_time) == int(self.time_per_prompt * 0.7):
                self.locked_generate_prompt()

            if int(remaining_time) == int(self.time_per_prompt * 0.4):
                self.locked_generate_image()

            # Check if time's up
            if remaining_time <= 0.5:
                with self.redis_conn.lock("update_lock", timeout=1):
                    self.redis_conn.setex("reset", 1, 1)
                    if self.update_contents():
                        print(f'[INFO] Resetting...')
                        self.reset_clock()
            
            await asyncio.sleep(1)