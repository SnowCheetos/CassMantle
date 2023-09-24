import io
import gc
import pika
import redis
import torch
import platform

from PIL import Image
from diffusers import DiffusionPipeline

class DiffuserService:
    def __init__(
            self, 
            image_size=(512, 512),
            diffuser_model='stabilityai/stable-diffusion-2-1', #'stabilityai/stable-diffusion-xl-base-1.0'
            diffuser_steps=50
        ) -> None:

        self.height, self.width = image_size
        self.cuda_available = torch.cuda.is_available()
        if platform.system() == "Darwin":
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = 'cuda' if self.cuda_available else 'cpu'

        self.dtype = torch.float16 if self.cuda_available else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(
            diffuser_model, 
            torch_dtype=self.dtype, 
            use_safetensors=True
        )
        self.pipeline.to(self.device)
        self.diffuser_steps = diffuser_steps

        self.redis_conn = redis.Redis()

    def encode_image(self, image: Image.Image) -> bytes:
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format='JPEG')
        image_bytes = image_bytes_io.getvalue()
        return image_bytes
    
    def generate_image(self, prompt: str) -> Image.Image:
        return self.pipeline(
            prompt=f'A painting of {prompt}',
            num_inference_steps=self.diffuser_steps,
            height=self.height,
            width=self.width
        ).images[0]
    
    def rabbitmq_callback(self, ch, method, properties, body):
        print(f"Received prompt: {body.decode()}")
        image = self.generate_image(body.decode())
        encoding = self.encode_image(image)
        self.redis_conn.hset('next', mapping={'image': encoding, 'done': 1})
        self.redis_conn.set('generating', 0)
        del image, encoding
        gc.collect()

        ch.basic_ack(delivery_tag=method.delivery_tag)

def main(diffuser_steps):
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    
    channel = connection.channel()
    channel.queue_declare(queue='generate_image')

    # Initialize the DiffuserService
    diffuser_service = DiffuserService(diffuser_steps)

    # Setting the class method as the callback
    try:
        channel.basic_consume(queue='generate_image', on_message_callback=diffuser_service.rabbitmq_callback)
        channel.start_consuming()
    finally:
        channel.close()
        connection.close()

if __name__ == "__main__":
    main(2)