import io
import uuid
import asyncio

from services.server import Server
from fastapi import FastAPI, Cookie, HTTPException, WebSocket, WebSocketException
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
server = Server(time_per_prompt=60)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="./static/"), name="static")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(server.global_timer())

@app.get("/")
async def read_root():
    return FileResponse("./static/index.html")

@app.get("/init/")
async def initialize_session(response: Response):
    session_id = str(uuid.uuid4())
    response.set_cookie(key="session_id", value=session_id)
    return {"message": "Session initialized", "session_id": session_id}

@app.websocket("/clock")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(1)
            time = server.fetch_clock()
            await websocket.send_json({"time": time})
    except WebSocketException:
        print('[INFO] Client disconnected.')

@app.get("/fetch_image")
async def fetch_image(session_id: str = Cookie(None)):
    image = server.fetch_masked_image(session_id)
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    return StreamingResponse(img_io, headers={"Content-Type": "image/jpeg"})

@app.post("/compute_score")
async def compute_score(request: Request, session_id: str = Cookie(None)):
    data = await request.json()
    inputs = list(data.values())

    if session_id is not None:
        scores = server.compute_client_score(session_id, inputs)
        return JSONResponse(scores)
    else:
        raise HTTPException(status_code=400, detail='No session id')