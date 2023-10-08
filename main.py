import base64
import io
import uuid
import asyncio

from services.server import Server
from fastapi import FastAPI, Cookie, HTTPException, WebSocket, WebSocketException
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from websockets.exceptions import ConnectionClosedOK

app = FastAPI()
server = Server()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="./static/"), name="static")
app.mount("/dict", StaticFiles(directory="dict"), name="dict")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(server.global_timer())

@app.get("/")
async def read_root():
    return FileResponse("./static/index.html")

@app.get("/init")
async def initialize_session(response: Response):
    session_id = str(uuid.uuid4())
    response.set_cookie(key="session_id", value=session_id)
    server.init_client(session_id)
    return {"message": "Session initialized", "session_id": session_id}

@app.websocket("/clock")
async def connect_clock(websocket: WebSocket):
    await websocket.accept()
    print('[INFO] Client Connected.')
    try:
        while True:
            await asyncio.sleep(1)
            time = server.fetch_clock()
            reset = bool(server.redis_conn.exists('reset'))
            await websocket.send_json({"time": time, "reset": reset})
    
    except WebSocketException:
        print('[INFO] Client Disconnected.')
    
    except ConnectionClosedOK:
        print('[INFO] Client Disconnected.')

@app.get("/fetch/contents")
async def fetch_contents(session_id: str=Cookie(None)):
    """
    Response: {
        "image": JPEG Bytes,
        "prompt" {
            "tokens": List[str],
            "masks": List[int]
        }
    }
    """
    if not server.redis_conn.exists(session_id): server.init_client(session_id)
    image = server.fetch_masked_image(session_id)
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    prompt = server.fetch_prompt_json()
    content = {
        "image": base64.b64encode(img_io.getvalue()).decode(),
        "prompt": prompt
    }
    return JSONResponse(content=content)

@app.post("/compute_score")
async def compute_score(request: Request, session_id: str = Cookie(None)):
    if not server.redis_conn.exists(session_id): server.init_client(session_id)

    data = await request.json()
    inputs = list(data['inputs'])

    if session_id is not None:
        scores = server.compute_client_scores(session_id, inputs)
        print(scores)
        return JSONResponse(scores)
    else:
        raise HTTPException(status_code=400, detail='No session id')