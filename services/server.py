import io
import redis
import redis.lock
import asyncio

from PIL import Image
from typing import List, Dict, Union
from services.backend import Backend

class Server(Backend):
    def __init__(
            self, 
            time_per_prompt=120, 
            min_score=0.1,
            prompt_topn=20, 
        ) -> None:
        super().__init__(min_score, prompt_topn)

        self.time_per_prompt = time_per_prompt
        self.redis_conn = redis.Redis(decode_responses=False)
        self.redis_conn.flushall()

        # In production, generate prompt at startup
        # self.generate_contents()

        # For now, populate current with content
        contents = {
            'prompt_list': 'nice|horse',
            'prompt': 'A painting of an nice horse',
            'image': self.encode_image(Image.open('media/demo.jpeg'))
        }
        self.redis_conn.hset('current', mapping=contents)
        self.redis_conn.set('countdown', self.time_per_prompt)
        self.redis_conn.hset('next', 'done', 0)
        self.redis_conn.set('generating', 0)

    def encode_image(self, image: Image.Image) -> bytes:
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format='JPEG')
        image_bytes = image_bytes_io.getvalue()
        return image_bytes

    def generate_image(self, prompt: str):
        self.channel.basic_publish(exchange='', routing_key='generate_image', body=prompt)

    @staticmethod
    def format_seconds_to_time(seconds):
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def start_countdown(self):
        self.redis_conn.setex('countdown', self.time_per_prompt, 'active')

    def fetch_countdown(self):
        return int(self.redis_conn.ttl('countdown'))

    def fetch_clock(self):
        seconds = self.fetch_countdown()
        return self.format_seconds_to_time(seconds)

    def reset_clock(self):
        self.start_countdown()

    async def global_timer(self):
        # Start the countdown
        self.start_countdown()
        
        while True:
            await asyncio.sleep(1)

            # Fetch remaining time
            remaining_time = self.fetch_countdown()

            # Check if time to generate new content
            if remaining_time == self.time_per_prompt - self.time_per_prompt // 3:
                asyncio.create_task(self.generate())

            # Check if time's up
            if remaining_time <= 1:
                if self.reset():
                    print(f'[INFO] Resetting...')
                    self.reset_clock()

    def generate_contents(self, prev_prompt_list: List[str]) -> Dict[str, Union[int, str, bytes]]:
        prompt_list = self.generate_prompt_list(prev_prompt_list)
        prompt = self.generate_prompt(prompt_list)
        self.generate_image(prompt)
        contents = {
            'prompt_list': '|'.join(prompt_list),
            'prompt': prompt
        }
        return contents

    async def generate(self):
        # Acquire the lock
        with self.redis_conn.lock("generation_lock", timeout=self.time_per_prompt-1):  # Lock expires after 120 seconds
        # Check if another worker has already started generating content
            if int(self.redis_conn.get('generating').decode('utf-8')) == 0:
                print(f'[INFO] Generating...')
                self.redis_conn.set('generating', 1)
                prompt_list = self.fetch_prompt_list()
                loop = asyncio.get_event_loop()
                contents = await loop.run_in_executor(None, self.generate_contents, prompt_list)
                self.redis_conn.hset('next', mapping=contents)

    def reset(self) -> bool:
        # Checks if the generation is actually complete
        if int(self.redis_conn.hget('next', 'done').decode('utf-8')) == 1:
            next_map = self.redis_conn.hgetall('next')
            self.redis_conn.hset('current', mapping=next_map)
            self.redis_conn.hset('next', 'done', 0)
            return True
        # Not complete yet, return False
        return False
    
    def fetch_image(self) -> Image.Image:
        img_bytes = self.redis_conn.hget('current', 'image')
        return Image.open(io.BytesIO(img_bytes))
    
    def fetch_masked_image(self, session) -> Image.Image:
        score = self.fetch_client_score(session)
        image = self.fetch_image()
        return self.mask_image(image, score)

    def fetch_prompt_list(self) -> List[str]:
        prompt_list = self.redis_conn.hget('current', 'prompt_list').decode('utf-8')
        return prompt_list.split('|')

    def fetch_prompt(self) -> str:
        return self.redis_conn.hget('current', 'prompt').decode('utf-8')
    
    def compute_client_score(self, session: str, inputs: List[str]) -> Dict[str, str]:
        prompt_list = self.fetch_prompt_list()
        mean_score = self.compute_mean_score(inputs, prompt_list)
        max_score = self.redis_conn.hget(session, 'max')
        self.redis_conn.hset(session, 'current', mean_score)
        if not max_score: 
            self.redis_conn.hset(session, 'max', mean_score)
        else:
            max_score = float(max_score.decode('utf-8'))
            if mean_score > max_score: 
                self.redis_conn.hset(session, 'max', mean_score)
        return self.compute_score_map(inputs, prompt_list)
    
    def fetch_client_score(self, session: str) -> float:
        score = self.redis_conn.hget(session, 'max')
        if not score: score = self.min_score
        else: score = float(score.decode('utf-8'))
        return score