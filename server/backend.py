import io
import nltk
import torch
import random
import platform
import gensim.downloader

import numpy as np

from PIL import Image
from typing import List, Dict
from diffusers import DiffusionPipeline

class Backend:
    def __init__(self, gensim_model, diffuser_model, diffuser_steps) -> None:

        # nltk.download('averaged_perceptron_tagger')

        self.min_score = 0.1
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
        self.diffuser_steps = diffuser_steps

    def encode_image(self, image: Image.Image) -> bytes:
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format='JPEG')
        image_bytes = image_bytes_io.getvalue()
        return image_bytes

    def generate_prompt_list(self) -> List[str]:
        # TODO implement
        return ["apple", "flying"]
    
    def generate_prompt(self, prompt_list) -> str:
        prefix = 'an' if prompt_list[0][0] in ['a', 'e', 'i', 'o', 'u'] else 'a'
        return f"{prefix} {' '.join(prompt_list)}"

    def generate_image(self, prompt: str) -> Image.Image:
        return self.pipeline(
            prompt=f'A painting of {prompt}',
            num_inference_steps=self.diffuser_steps
        ).images[0]
    
    def generate_contents(self) -> Dict:
        prompt_list = self.generate_prompt_list()
        prompt = self.generate_prompt(prompt_list)
        image = self.generate_image(prompt)
        img_bytes = self.encode_image(image)
        contents = {
            'done': 1,
            'prompt_list': '|'.join(prompt_list),
            'prompt': prompt,
            'image': img_bytes
        }
        return contents
    
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
    
    def compute_score(self, inputs: str, answer: str) -> float:
        return self.word2vec.similarity(inputs, answer)

    def compute_mean_score(self, inputs: List[str], answers: List[str]) -> float:
        scores = [self.compute_score(inp, tar) for inp, tar in zip(inputs, answers)]
        return sum(scores) / len(scores)
    
    def compute_score_map(self, inputs: List[str], answers: List[str]) -> Dict[str, str]:
        scores = {f"score{i+1}": str(self.compute_score(inp, tar)) for i, (inp, tar) in enumerate(zip(inputs, answers))}
        return scores