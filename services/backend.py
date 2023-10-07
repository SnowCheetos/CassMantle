import pika
import json
import numpy as np

from PIL import Image
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
        image_channel = connection.channel(channel_number=73)
        image_channel.queue_declare(queue='diffuser_service')
        image_channel.basic_publish(
            exchange='',
            routing_key='diffuser_service',
            body=prompt
        )
        connection.close()

    def compute_scores(self, session: str, inputs: List[str], answer: List[str]) -> None:
        connection = pika.BlockingConnection(pika.ConnectionParameters(self.rabbit_host))
        prompt_channel = connection.channel(channel_number=72)
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
        prompt_channel = connection.channel(channel_number=72)
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

    def compute_mask_ratio(self, score: float) -> float:
        return 1 - score ** 2

    def mask_image(self, image: Image.Image, score: float) -> Image.Image:
        image = np.array(image)
        height, width, _ = image.shape
        total_elements = width * height
        num_mask_elements = int(self.compute_mask_ratio(score) * total_elements)
        idx = np.random.choice(total_elements, num_mask_elements, replace=False)
        row_indices, col_indices = np.divmod(idx, width)
        image[row_indices, col_indices] = [255, 255, 255]
        return Image.fromarray(np.uint8(image))