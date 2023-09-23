import io
import redis
import asyncio

from PIL import Image
from typing import List, Dict
from server.backend import Backend

class Server(Backend):
    def __init__(
            self, 
            time_per_prompt=120, # 2 minutes for now
            gensim_model='glove-twitter-25', 
            diffuser_model='stabilityai/stable-diffusion-2-1', #'stabilityai/stable-diffusion-xl-base-1.0'
            diffuser_steps=50
        ) -> None:
        super().__init__(gensim_model, diffuser_model, diffuser_steps)

        self.time_per_prompt = time_per_prompt
        self.redis_conn = redis.Redis(decode_responses=False)
        self.redis_conn.flushall()

        # In production, generate prompt at startup
        # self.generate_contents()

        # For now, populate current with content
        contents = {
            'prompt_list': 'astronaut|flying',
            'prompt': 'A painting of an astronaut flying',
            'image': self.encode_image(Image.open('media/demo.jpeg'))
        }
        self.redis_conn.hmset('current', contents)
        self.redis_conn.set('countdown', self.time_per_prompt)

    @staticmethod
    def format_seconds_to_time(seconds):
        minutes, remaining_seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def fetch_countdown(self):
        return int(self.redis_conn.get('countdown').decode('utf-8'))

    def fetch_clock(self):
        seconds = self.fetch_countdown()
        return self.format_seconds_to_time(seconds)

    def countdown_one(self):
        countdown = self.fetch_countdown()
        self.redis_conn.set('countdown', countdown - 1)

    def reset_clock(self):
        self.redis_conn.set('countdown', self.time_per_prompt)

    async def global_timer(self):
        while True:
            countdown = self.fetch_countdown()
            print(f"[COUNTDOWN] {countdown}")
            # Halfway, begin new prompt and image generation
            if countdown == self.time_per_prompt // 2:
                print(f'[INFO] Generating...')
                asyncio.create_task(self.generate())

            # Time reached, reset clock
            elif countdown == 0:
                if self.reset():
                    print(f'[INFO] Resetting...')
                    self.reset_clock()
                else:
                    # Give it extra time to complete
                    self.redis_conn.set('countdown', 1)
                    
            self.countdown_one()
            await asyncio.sleep(1)

    async def generate(self):
        loop = asyncio.get_event_loop()
        contents = await loop.run_in_executor(None, self.generate_contents)
        self.redis_conn.hmset('next', mapping=contents)

    def reset(self) -> bool:
        # Checks if the generation is actually complete
        if int(self.redis_conn.hget('next', 'done')) == 1:
            next_map = self.redis_conn.hgetall('next')
            self.redis_conn.hmset('current', mapping=next_map)
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
        self.redis_conn.set(session, mean_score)
        return self.compute_score_map(inputs, prompt_list)
    
    def fetch_client_score(self, session: str) -> float:
        score = self.redis_conn.get(session)
        if not score: score = self.min_score
        else: score = float(score)
        return score