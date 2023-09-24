import pika
import json
import gensim.downloader

from typing import List

class ScoreService:
    def __init__(self, gensim_model) -> None:
        # Existing initialization logic
        self.word2vec = gensim.downloader.load(gensim_model)

        # Initialize RabbitMQ connection and channel
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()

        # Declare request queue
        self.channel.queue_declare(queue='compute_score_queue')

        # Declare response queue and set callback for consuming
        self.channel.queue_declare(queue='response_queue')
        self.channel.basic_consume(queue='compute_score_queue', on_message_callback=self.on_request)

    def compute_score(self, inputs: str, answer: str) -> float:
        return self.word2vec.similarity(inputs, answer)
    
    def most_similar(self, word, top_n) -> List[str]:
        return self.word2vec.most_similar(word, topn=top_n)

    def on_request(self, ch, method, properties, body):
        # Decode request body
        request_data = json.loads(body)
        operation = request_data['operation']
        data = request_data['data']

        if operation == 'compute_score':
            inputs = data['inputs']
            answer = data['answer']
            score = self.compute_score(inputs, answer)
            response = {'score': str(score)}

        elif operation == 'most_similar':
            word = data['word']
            top_n = data['top_n']
            most_similar_words = self.most_similar(word, top_n)
            response = {'most_similar': most_similar_words}

        else:
            response = {'error': 'Invalid operation'}

        # Send response back to reply queue
        ch.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(
                correlation_id=properties.correlation_id
            ),
            body=json.dumps(response)
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        print("ScoreService is waiting for requests. To exit press CTRL+C")
        self.channel.start_consuming()

if __name__ == '__main__':
    score_service = ScoreService(gensim_model='word2vec-google-news-300')
    score_service.start()
