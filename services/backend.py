import pika
import json

from PIL import Image, ImageFilter
from typing import List

class Backend:
    """
    This class should be the parent class to Server.
    This class should mainly handle communication to the microservices
    through RabbitMQ
    """
    def __init__(self, rabbit_host='localhost') -> None:
        self.rabbit_host = rabbit_host

    def generate_image(self, prompt: str) -> None:
        connection = pika.BlockingConnection(pika.ConnectionParameters(self.rabbit_host))
        image_channel = connection.channel()
        image_channel.queue_declare(queue='diffuser_service')
        image_channel.basic_publish(
            exchange='',
            routing_key='diffuser_service',
            body=prompt
        )
        connection.close()

    def compute_scores(self, session: str, inputs: List[str], answer: List[str]) -> None:
        connection = pika.BlockingConnection(pika.ConnectionParameters(self.rabbit_host))
        prompt_channel = connection.channel()
        prompt_channel.queue_declare(queue='prompt_service')
        contents = {
            'operation': 0,
            'data': {
                'session': session,
                'inputs': inputs,
                'answer': answer
            }
        }
        prompt_channel.basic_publish(
            exchange='',
            routing_key='prompt_service',
            body=json.dumps(contents)
        )
        connection.close()

    def generate_prompt(self) -> None:
        connection = pika.BlockingConnection(pika.ConnectionParameters(self.rabbit_host))
        prompt_channel = connection.channel()
        prompt_channel.queue_declare(queue='prompt_service')
        contents = {
            'operation': 1, 
            'data': 0
        }
        prompt_channel.basic_publish(
            exchange='',
            routing_key='prompt_service',
            body=json.dumps(contents)
        )
        connection.close()

    def score_to_blur(self, score: float, min_blur: float=0.0, max_blur: float=15):
        return min_blur + (1 - score ** 2) * (max_blur - min_blur)

    def mask_image(self, image: Image.Image, score: float) -> Image.Image:
        blur = self.score_to_blur(score)
        return image.filter(ImageFilter.GaussianBlur(blur))