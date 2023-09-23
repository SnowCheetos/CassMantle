import nltk
import torch
import redis
import random
import asyncio
import platform
import gensim.downloader

import numpy as np

from PIL import Image
from typing import List, Dict
from diffusers import DiffusionPipeline

class Server:
    def __init__(
            self,
            gensim_model='glove-twitter-25',
            diffuser_model='stabilityai/stable-diffusion-2-1' #'stabilityai/stable-diffusion-xl-base-1.0'
        ) -> None:

        self.time_per_prompt = 60 # 1 minute
        self.countdown = self.time_per_prompt

        # nltk.download('averaged_perceptron_tagger')
        self.word2vec = gensim.downloader.load(gensim_model)
        self.cuda_available = torch.cuda.is_available()

        if platform.system() == "Darwin":
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = 'cuda' if self.cuda_available else 'cpu'

        self.dtype = torch.float16 if self.cuda_available else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(
            diffuser_model, 
            torch_dtype=self.dtype, 
            use_safetensors=True, 
            variant="fp16"
        )
        self.pipeline.to(self.device)
        self.base_image = Image.open('media/demo.jpeg')
        self.min_score = 0.1

        self.redis_conn = redis.Redis(decode_responses=True)

        self.prompt_format = ['NN', 'VBG']
        self.prompt_list = ['astronaut', 'riding']
        self.prompt_topn = 100
        self.prompt = None

        self.next = {
            'done': False,
            'prompt': None,
            'prompt_list': None,
            'image': None
        }
    
    async def global_timer(self):
        while True:
            if self.countdown == self.time_per_prompt // 2: 
                asyncio.create_task(self.generate())
            
            elif self.countdown == 0: 
                await self.reset()
            
            else:
                if not self.next['done']: 
                    self.countdown -= 1
            
            await asyncio.sleep(1)

    async def generate(self):
        prompt_list, prompt = self.generate_prompt()
        self.next['prompt_list'] = prompt_list
        self.next['prompt'] = prompt
        self.next['image'] = await self.generate_image(prompt)
        self.next['done'] = True

    async def reset(self):
        self.countdown = self.time_per_prompt
        self.base_image = self.next['image']
        self.prompt = self.next['prompt']
        self.prompt_list = self.next['prompt_list']
        self.next = {
            'done': False,
            'prompt': None,
            'prompt_list': None,
            'image': None
        }

    def fetch_client_score(self, session: str) -> float:
        score = self.redis_conn.get(session)
        if not score: score = self.min_score
        else: score = float(score)
        return score

    def compute_score(self, inputs: str, answer: str) -> float:
        return self.word2vec.similarity(inputs, answer)

    def compute_mean_score(self, inputs: List[str]) -> float:
        scores = [self.compute_score(inp, tar) for inp, tar in zip(inputs, self.prompt_list)]
        return sum(scores) / len(scores)
    
    def compute_score_map(self, inputs: List[str]) -> Dict[str, str]:
        scores = {f"score{i+1}": str(self.compute_score(inp, tar)) for i, (inp, tar) in enumerate(zip(inputs, self.prompt_list))}
        return scores
        
    def compute_client_score(self, session: str, inputs:  List[str]) -> Dict[str, str]:
        mean_score = self.compute_mean_score(inputs)
        self.redis_conn.set(session, mean_score)
        return self.compute_score_map(inputs)

    def _generate_image_blocking(self, prompt):
        return self.pipeline(prompt=prompt).images[0]

    async def generate_image(self, prompt):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_image_blocking, prompt)

    def mask_image(self, score):
        img_arr = np.array(self.base_image)
        height, width, _ = img_arr.shape
        total_elements = width * height
        mask_ratio = max(self.min_score, score ** 2)
        num_mask_elements = int((1 - mask_ratio) * total_elements)
        idx = np.random.choice(total_elements, num_mask_elements, replace=False)
        row_indices, col_indices = np.divmod(idx, width)
        img_arr[row_indices, col_indices] = [255, 255, 255]
        return Image.fromarray(np.uint8(img_arr))

    def generate_prompt(self):
        prompt_list = [None] * len(self.prompt_list)
        for i, word in enumerate(self.prompt_list):
            idx = random.randint(0, 100)
            choices = self.word2vec.most_similar(word, topn=self.prompt_topn)
            choice = choices[idx][0]
            while nltk.pos_tag([choice])[0][1] != self.prompt_format[i]:
                idx = random.randint(0, 100)
                choice = choices[idx][0]
            prompt_list[i] = choice
        prefix = 'an' if self.prompt_list[0][0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
        return prompt_list, f"A painting of {prefix} {' '.join(self.prompt_list)}"