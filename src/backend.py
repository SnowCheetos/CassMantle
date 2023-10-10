import io
import pika
import json
import httpx
import asyncio
import aioredis

from PIL import Image, ImageFilter
from src.utils import reconstruct_sentence, encode_image, api_call

class Backend:
    """
    This class should be the parent class to Server.
    """
    def __init__(
            self, 
            max_retries=5,
            rabbit_host='localhost',
            diffuser_url="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
        ) -> None:
        
        with open('api_key.txt', 'r') as f:
            API_TOKEN = f.readline()

        self.max_retries = max_retries
        self.rabbit_host = rabbit_host
        self.diffuser_url = diffuser_url
        self.auth_header = {"Authorization": f"Bearer {API_TOKEN}"}

        # Initialize the Redis connection asynchronously

    async def initialize_redis(self) -> aioredis.Redis:
        return await aioredis.Redis(host='localhost', decode_responses=False)

    async def set_initial_image_status(self):
        await self.redis_conn.hset('image', 'status', 'idle')

    async def startup(self) -> None:
        self.redis_conn = await self.initialize_redis()
        await self.set_initial_image_status()
        if not await self.redis_conn.hget('image', 'current'):
            async with self.redis_conn.lock("startup_lock", timeout=10):
                prompts = await self.redis_conn.hget('prompt', 'current')
                if not prompts: raise Exception("Prompt not generated yet, startup failed.")

                prompts = json.loads(prompts.decode('utf-8'))  # Decode the binary data
                image = await self.generate_image(reconstruct_sentence(prompts['tokens']))
                
                if image:
                    encoding = encode_image(image)
                    await self.redis_conn.hset('image', mapping={'current': encoding, 'status': 'idle'})
                    print('[INFO] Initial image generated.')

        else:
            await asyncio.sleep(1)

    async def generate_image(self, prompt: str):
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
            print("Image generation failed.")
            return None


    def generate_prompt(self) -> None:
        connection = pika.BlockingConnection(pika.ConnectionParameters(self.rabbit_host))
        prompt_channel = connection.channel()
        prompt_channel.queue_declare(queue='prompt_service')
        contents = {'placeholder': 0}
        prompt_channel.basic_publish(
            exchange='',
            routing_key='prompt_service',
            body=json.dumps(contents)
        )
        connection.close()

    def score_to_blur(self, score: float, min_blur: float=0.0, max_blur: float=20):
        return min_blur + (1 - score ** 2) * (max_blur - min_blur)

    def mask_image(self, image: Image.Image, score: float) -> Image.Image:
        blur = self.score_to_blur(score)
        return image.filter(ImageFilter.GaussianBlur(blur))
    

    # def generate_image(self, prompt: str) -> None:
    #     connection = pika.BlockingConnection(pika.ConnectionParameters(self.rabbit_host))
    #     image_channel = connection.channel()
    #     image_channel.queue_declare(queue='diffuser_service')
    #     image_channel.basic_publish(
    #         exchange='',
    #         routing_key='diffuser_service',
    #         body=prompt
    #     )
    #     connection.close()