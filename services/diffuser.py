import io
import gc
import random
import pika
import redis
import json
import torch
import requests
import platform

from PIL import Image
from diffusers import DiffusionPipeline
from utils import reconstruct_sentence

class DiffuserService:
    """
    This class is the implementation of the diffuser microservice.
    It should perform inference using prompts obtained from the RabbitMQ queue.
    """
    def __init__(
            self,
            local=False, # Whether or not to use the API
            image_size=(448, 448),
            diffuser_steps=50,
            diffuser_model='stabilityai/stable-diffusion-2-1', 
            # diffuser_model='stabilityai/stable-diffusion-xl-base-1.0',
            API_URL="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
            rabbit_host='localhost'
        ) -> None:
        
        self.local = local
        self.height, self.width = image_size
        self.diffuser_steps = diffuser_steps
        self.API_URL = API_URL

        self.cuda_available = torch.cuda.is_available()
        if platform.system() == "Darwin": self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else: self.device = 'cuda' if self.cuda_available else 'cpu'

        self.dtype = torch.float16 if self.cuda_available else torch.float32
        
        if self.local:
            print("[INFO] Loading local model into memory")
            self.pipeline = DiffusionPipeline.from_pretrained(
                diffuser_model, 
                torch_dtype=self.dtype, 
                use_safetensors=True
            )
            self.pipeline.to(self.device)
        else:
            print("[INFO] Using Hugging Face API")
            with open('api_key.txt', 'r') as f:
                self.API_TOKEN = f.readline()
            self.pipeline = None

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(rabbit_host))
        self.channel = self.connection.channel(channel_number=73)
        self.channel.queue_declare(queue='diffuser_service')

        self.redis_conn = redis.Redis()
        self.redis_conn.hset('image', 'status', 'idle')

        self.styles = [
            '. Impressionism',
            '. Line Art',
            '. Surrealism',
            '. Pop Art',
            '. Romanticism',
            '. Realism',
            '. Baroque',
            '. Gothic',
            '. Fauvism'
        ]

        self.startup()

    def startup(self) -> None:
        prompts = self.redis_conn.hget('prompt', 'current')
        if not prompts: 
            raise Exception("Prompt not generated yet, startup failed.")
        prompts = json.loads(prompts)
        self.redis_conn.hset('image', 'status', 'busy')
        image = self.generate_image(reconstruct_sentence(prompts['tokens']))
        encoding = self.encode_image(image)
        self.redis_conn.hset('image', mapping={'current': encoding, 'status': 'idle'})
        print("[INFO] Initial image generated.")

    def encode_image(self, image: Image.Image) -> bytes:
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format='JPEG')
        image_bytes = image_bytes_io.getvalue()
        return image_bytes

    def generate_image(self, prompt: str) -> Image.Image:
        full_prompt = prompt + self.styles[random.randint(0, len(self.styles)-1)] + ' style.'
        print(f'[INFO] Full Prompt: {full_prompt}')
        if self.local:
            return self.pipeline(
                prompt=full_prompt,
                negative_prompt='blurry, distorted, fake, abstract',
                num_inference_steps=self.diffuser_steps,
                height=self.height,
                width=self.width
            ).images[0]
        else:
            try:
                response = requests.post(
                    self.API_URL, 
                    headers={
                        "Authorization": f"Bearer {self.API_TOKEN}"
                    }, 
                    json={"inputs": full_prompt}
                )
                response.raise_for_status()

            except requests.exceptions.HTTPError:
                raise Exception("[ERROR] Unable To Connect To Hugging Face API.")
            
            return Image.open(io.BytesIO(response.content))
    
    def callback(self, ch, method, properties, body):
        self.redis_conn.hset('image', 'status', 'busy')
        
        image = self.generate_image(body.decode())
        encoding = self.encode_image(image)
        
        self.redis_conn.hset('image', mapping={'next': encoding, 'status': 'idle'})

        gc.collect()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        try:
            self.channel.basic_consume(queue='diffuser_service', on_message_callback=self.callback)
            self.channel.start_consuming()
        finally:
            self.channel.close()
            self.connection.close()


if __name__ == '__main__':
    service = DiffuserService()
    service.start()