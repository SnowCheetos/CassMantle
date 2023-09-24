import nltk
import json
import pika
import uuid
import random
import gensim.downloader

import numpy as np

from PIL import Image
from typing import List, Dict, Union
from pika import BlockingConnection, ConnectionParameters

class Backend:
    def __init__(self, min_score, prompt_topn) -> None:

        # nltk.download('averaged_perceptron_tagger')

        self.connection = BlockingConnection(ConnectionParameters('localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='generate_image')

        self.min_score = min_score
        self.prompt_topn = prompt_topn
        self.prompt_format = ['NN', 'NN']

        self.response_queue = self.channel.queue_declare(queue='', exclusive=True).method.queue
        self.channel.basic_consume(
            queue=self.response_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )

    def on_response(self, ch, method, properties, body):
        if self.corr_id == properties.correlation_id:
            self.response = json.loads(body)

    def call_score_service(self, operation: str, data: Union[str, dict]) -> dict:
        self.response = None
        self.corr_id = str(uuid.uuid4())
        request_payload = {'operation': operation, 'data': data}
        self.channel.basic_publish(
            exchange='',
            routing_key='compute_score_queue',
            properties=pika.BasicProperties(
                reply_to=self.response_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(request_payload)
        )
        while self.response is None:
            # Wait for a response
            self.connection.process_data_events()
        return self.response

    def compute_score(self, inputs: str, answer: str) -> float:
        data = {'inputs': inputs, 'answer': answer}
        response = self.call_score_service('compute_score', data)
        return float(response.get('score', '0.0'))

    def most_similar(self, word: str, top_n: int) -> List[str]:
        data = {'word': word, 'top_n': top_n}
        response = self.call_score_service('most_similar', data)
        return response.get('most_similar', [])

    def generate_prompt_list(self, prompt_list: List[str]) -> List[str]:
        for i, word in enumerate(prompt_list):
            idx = random.randint(0, self.prompt_topn-1)
            choices = self.most_similar(word, self.prompt_topn)
            choice = choices[idx][0]
            while nltk.pos_tag([choice])[0][1] != self.prompt_format[i]:
                idx = random.randint(0, self.prompt_topn-1)
                choice = choices[idx][0]
            prompt_list[i] = choice
        print(f'[PROMPT] {prompt_list}')
        return prompt_list
    
    def generate_prompt(self, prompt_list: List[str]) -> str:
        prefix = 'an' if prompt_list[0][0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
        return f"{prefix} {' '.join(prompt_list)}"
    
    def mask_ratio(self, score: float) -> float:
        return 1 - min(1, max(self.min_score, score ** 2))

    def mask_image(self, image: Image.Image, score: float) -> Image.Image:
        image = np.array(image)
        height, width, _ = image.shape
        total_elements = width * height
        num_mask_elements = int(self.mask_ratio(score) * total_elements)
        idx = np.random.choice(total_elements, num_mask_elements, replace=False)
        row_indices, col_indices = np.divmod(idx, width)
        image[row_indices, col_indices] = [255, 255, 255]
        return Image.fromarray(np.uint8(image))

    def compute_mean_score(self, inputs: List[str], answers: List[str]) -> float:
        scores = [self.compute_score(inp, tar) for inp, tar in zip(inputs, answers)]
        return sum(scores) / len(scores)
    
    def compute_score_map(self, inputs: List[str], answers: List[str]) -> Dict[str, str]:
        scores = {f"score{i+1}": str(self.compute_score(inp, tar)) for i, (inp, tar) in enumerate(zip(inputs, answers))}
        return scores