import io
import time
import redis
import redis.lock
import asyncio

from PIL import Image
from typing import List, Dict
from services.backend import Backend

class Server(Backend):
    """
    This class is the implementation of API server logic, which inherits from the Backend class.
    It handles a lot of the Redis operations.
    """
    def __init__(
            self, 
            min_score=0.1,
            prompt_delim='|',
            time_per_prompt=120,
            rabbit_host='localhost'
        ) -> None:
        super().__init__(rabbit_host)

        self.min_score = min_score
        self.prompt_delim = prompt_delim
        self.time_per_prompt = time_per_prompt
        self.redis_conn = redis.Redis(decode_responses=False)
        self.redis_conn.flushall()

        self.redis_conn.hset(
            'prompt', mapping={
                'status': 'idle',
                'current': 'nice|horse'
            }
        )
        self.redis_conn.hset(
            'image', mapping={
                'status': 'idle',
                'current': self.encode_image(Image.open('media/demo.jpeg'))
            }
        )
    
    @staticmethod
    def encode_image(image: Image.Image) -> bytes:
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format='JPEG')
        image_bytes = image_bytes_io.getvalue()
        return image_bytes

    def init_client(self, session: str) -> None:
        if self.redis_conn.exists(session): self.redis_conn.delete(session)
        contents = {'max': self.min_score, 'current': self.min_score, 'status': 'idle'}
        self.redis_conn.hset(session, mapping=contents)
        self.redis_conn.sadd('sessions', session)

    def construct_prompt(self, prompt_list: List[str]) -> str:
        prefix = 'an' if prompt_list[0][0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
        return f"{prefix} {' '.join(prompt_list)}"

    def fetch_prompt(self) -> str:
        # TODO Need to implement a better way to do this...
        prompts = self.redis_conn.hget('prompt', 'current').decode()
        prompt_list = prompts.split(self.prompt_delim)
        return self.construct_prompt(prompt_list)

    def fetch_client_scores(self, session: str) -> Dict[str, str]:
        while self.redis_conn.hget(session, 'status').decode() == 'busy':
            time.sleep(0.25)
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

    def compute_client_scores(self, session: str, inputs: List[str]) -> Dict[str, str]:
        prompts = self.redis_conn.hget('prompt', 'current').decode()
        prompt_list = prompts.split(self.prompt_delim)
        self.compute_scores(session, inputs, prompt_list)
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
    
    @staticmethod
    def format_seconds_to_time(seconds: int) -> str:
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def start_countdown(self) -> None:
        self.redis_conn.setex('countdown', self.time_per_prompt, 'active')

    def fetch_countdown(self) -> float:
        return float(self.redis_conn.ttl('countdown'))

    def fetch_clock(self) -> str:
        seconds = int(self.fetch_countdown())
        return self.format_seconds_to_time(seconds)

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
                next_prompt = self.redis_conn.hget('prompt', 'next').decode()
                if image_status == 'idle' and next_prompt:
                    prompt_list = next_prompt.split(self.prompt_delim)
                    self.generate_image(self.construct_prompt(prompt_list))
                
    async def global_timer(self) -> None:
        # Start the countdown
        self.start_countdown()
        await asyncio.sleep(1)
        
        while True:
            # Fetch remaining time
            remaining_time = self.fetch_countdown()

            # Check if time to generate new prompt
            if int(remaining_time) == int(self.time_per_prompt * 0.9):
                self.locked_generate_prompt()

            if int(remaining_time) == int(self.time_per_prompt * 0.7):
                self.locked_generate_image()

            # Check if time's up
            if remaining_time <= 1:
                with self.redis_conn.lock("update_lock", timeout=1):
                    self.redis_conn.setex("reset", 1, 1)
                    if self.update_contents():
                        print(f'[INFO] Resetting...')
                        self.reset_clock()
            
            await asyncio.sleep(1)