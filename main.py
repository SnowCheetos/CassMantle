import io
import uuid
import asyncio

from server.server import Server
from fastapi import FastAPI, Cookie, HTTPException, WebSocket, WebSocketException
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
server = Server(
    time_per_prompt=120,
    diffuser_steps=1
)

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
    return FileResponse("index.html")

@app.get("/init/")
async def initialize_session(response: Response):
    print('init called')
    session_id = str(uuid.uuid4())
    response.set_cookie(key="session_id", value=session_id)
    return {"message": "Session initialized", "session_id": session_id}

@app.websocket("/clock")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            time = server.fetch_clock()
            await websocket.send_json({"time": time})
            await asyncio.sleep(1)
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