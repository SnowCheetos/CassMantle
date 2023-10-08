import gc
import pika
import json
import redis
import random
import string
import gensim.downloader

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from typing import List, Dict, Union
from transformers import AutoTokenizer, BartForConditionalGeneration

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
            num_masked=2,
            gensim_model='glove-twitter-100', #'word2vec-google-news-300',
            language_model='facebook/bart-large-cnn',
            rabbit_host='localhost'
        ) -> None:

        self.starters = [
            "A surreal scene featuring",
            "An abstract representation of",
            "A detailed illustration of",
            "A fantastical image where",
            "A tranquil and peaceful scene with",
            "An eerie and mysterious depiction of",
            "A vibrant and colorful tableau involving",
            "A heartwarming moment where",
            "A dynamic and energetic artwork of",
            "An enigmatic visual of"
        ]

        self.top_n = top_n
        self.min_score = min_score
        self.num_masked = num_masked

        self.word2vec = gensim.downloader.load(gensim_model)
        self.model = BartForConditionalGeneration.from_pretrained(language_model)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbit_host, heartbeat=600))
        self.channel = self.connection.channel(channel_number=72)
        self.channel.queue_declare(queue='prompt_service')

        self.redis_conn = redis.Redis(decode_responses=True)
        self.redis_conn.flushall()
        self.redis_conn.hset('prompt', 'status', 'idle')

        self.startup()

    def startup(self) -> None:
        self.redis_conn.hset('prompt', 'status', 'busy')
        prompt = json.dumps(self.generate_prompt())
        self.redis_conn.hset('prompt', mapping={'current': prompt, 'status': 'idle'})
        print("[INFO] Initial prompt generated.")

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

    def select_descriptive_words(self, inputs: str, prompt: str, num_words: int=2) -> List[str]:
        # Load stop words
        stop_words = set(stopwords.words('english'))
        
        # Tokenize and POS tag words
        words = word_tokenize(prompt)
        skips = word_tokenize(inputs)
        
        tagged_words = nltk.pos_tag(words)
        
        # Filter words
        filtered_words = [
            word for word, pos in tagged_words
            if word.lower() not in stop_words
            and all(char not in string.punctuation for char in word)  # Exclude punctuation
            and "'" not in word
            and "-" not in word
            and pos not in ['NNP', 'NNPS']  # Exclude proper nouns
            and word not in skips
        ]
        
        # Select words
        selected_words = random.sample(filtered_words, min(num_words, len(filtered_words)))
        
        # Find indices
        indices = [words.index(word) for word in selected_words]
        
        return indices

    def generate_prompt(self) -> Dict[str, Union[List[str], List[int]]]:
        input_text = self.starters[random.randint(0, len(self.starters)-1)]
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        output_ids = self.model.generate(input_ids, max_length=64, num_beams=11, do_sample=True, temperature=1.2)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prompt = output_text.split('.')[0]
        print(prompt)
        masks = self.select_descriptive_words(input_text, prompt, self.num_masked)
        return {
            'tokens': word_tokenize(prompt),
            'masks': masks
        }

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
            prompt = json.dumps(self.generate_prompt())
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
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

    service = PromptService()
    service.start()