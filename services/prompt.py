import gc
import pika
import json
import redis
import gensim.downloader

from typing import List

class PromptService:
    """
    This class is the implementation of the prompt microservice.
    It should handle all computations done by the language models
    and communicate via RabbitMQ.
    """
    def __init__(
            self, 
            top_n=20,
            min_score=0.1,
            gensim_model='word2vec-google-news-300',
            rabbit_host='localhost'
        ) -> None:

        self.top_n = top_n
        self.min_score = min_score
        self.word2vec = gensim.downloader.load(gensim_model)

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbit_host, heartbeat=600))
        self.channel = self.connection.channel(channel_number=72)
        self.channel.queue_declare(queue='prompt_service')

        self.redis_conn = redis.Redis(decode_responses=True)
        self.redis_conn.hset('prompt', 'status', 'idle')

    def compute_score(self, inputs: str, answer: str) -> float:
        score = self.word2vec.similarity(inputs, answer)
        return max(self.min_score, score)
    
    def most_similar(self, word: str) -> List[str]:
        return self.word2vec.most_similar(word, topn=self.top_n)
    
    def set_index_score(self, session: str, index: int, score: float) -> None:
        field = f'score{index+1}'
        self.redis_conn.hset(session, field, float(score))

    def set_client_score(self, session: str, score: float) -> None:
        self.redis_conn.hset(session, 'current', score)
        max_score = float(self.redis_conn.hget(session, 'max'))
        if not max_score: self.redis_conn.hset(session, 'max', score)
        elif score > max_score: self.redis_conn.hset(session, 'max', score)

    def generate_prompt(self) -> str:
        # TODO Need to implement
        return 'beautiful|human'

    def callback(self, ch, method, properties, body):
        contents = json.loads(body)
        operation, data = contents['operation'], contents['data']

        print(f"[INFO] Prompt Service Called, Body: {contents}")

        if operation == 0: # Compute score
            self.redis_conn.hset(data['session'], 'status', 'busy')
            scores = []
            for i, (inp, ans) in enumerate(zip(data['inputs'], data['answer'])):
                scores += [self.compute_score(inp, ans)]
                self.set_index_score(data['session'], i, scores[i])
            mean_score = sum(scores) / len(scores)
            self.set_client_score(data['session'], mean_score)
            self.redis_conn.hset(data['session'], 'status', 'idle')

        elif operation == 1: # Generate prompt
            self.redis_conn.hset('prompt', 'status', 'busy')
            prompt = self.generate_prompt()
            self.redis_conn.hset('prompt', mapping={'next': prompt, 'status': 'idle'})

        else:
            print('[ERROR] Invalid operation tag received')

        gc.collect()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        print("[INFO] Prompt Service Startup Complete")
        try:
            self.channel.basic_consume(queue='prompt_service', on_message_callback=self.callback)
            self.channel.start_consuming()
        finally:
            self.channel.close()
            self.connection.close()


if __name__ == '__main__':
    service = PromptService()
    service.start()