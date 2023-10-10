import uvicorn
import gensim.downloader
from typing import List

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse

class ScoreService:
    def __init__(
            self,
            min_score=0.1,
            model='glove-twitter-100', #'word2vec-google-news-300',
        ) -> None:
        
        self.min_score = min_score
        self.model = gensim.downloader.load(model)

    def compute_score(self, inputs: str, answer: str) -> float:
        score = self.model.similarity(inputs.lower(), answer.lower())
        return max(self.min_score, score)
    
    def most_similar(self, word: str, topn: int=50) -> List[str]:
        return self.model.most_similar(word, topn=topn)

app = FastAPI()
service = ScoreService()

@app.post("/compute_scores")
async def compute_scores(request: Request) -> JSONResponse:
    data = await request.json()
    scores = []
    for inp, ans in zip(data['inputs'], data['answer']):
        scores.append(str(service.compute_score(inp, ans)))
    return JSONResponse(content={'scores': scores})

@app.post("/most_similar")
async def most_similar(request: Request) -> JSONResponse:
    data = await request.json()
    sims = service.most_similar(data['input'], int(data['topn']))
    print(sims)
    return JSONResponse(content={'results': sims})

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port='9000')