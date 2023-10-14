import io
import uuid
import base64
import asyncio

from src.server import Server
from fastapi import FastAPI, Cookie, HTTPException, WebSocket, WebSocketException
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

app = FastAPI()
server = Server(time_per_prompt=15*60)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="./static/"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/media", StaticFiles(directory="media"), name="media")

@app.on_event("startup")
async def startup_event():
    await server.startup()
    asyncio.create_task(server.global_timer())

@app.get("/")
async def read_root():
    return FileResponse("./static/index.html")

@app.get("/init")
async def initialize_session(response: Response):
    session_id = str(uuid.uuid4())
    response.set_cookie(key="session_id", value=session_id)
    await server.init_client(session_id)
    return {"message": "Session initialized", "session_id": session_id}

@app.websocket("/clock")
async def connect_clock(websocket: WebSocket):
    await websocket.accept()
    print('[INFO] Client Connected.')
    try:
        while True:
            await asyncio.sleep(1)
            time = await server.fetch_clock()
            reset = bool(await server.redis_conn.exists('reset'))
            await websocket.send_json({"time": time, "reset": reset})
    
    except WebSocketException:
        print('[INFO] Client Disconnected.')
    
    except ConnectionClosedError:
        print('[INFO] Client Disconnected.')
    
    except ConnectionClosedOK:
        print('[INFO] Client Disconnected.')

@app.get("/client/status")
async def check_status(session_id: str=Cookie(None)):
    # Check if session_id exists and is valid
    if not session_id or not await server.redis_conn.exists(session_id): 
        # If not, signal the client that initialization is needed
        return JSONResponse(content={'needInitialization': True})

    # Fetch client scores if session is valid
    scores = await server.fetch_client_scores(session_id)
    f = {'won': int(scores['won']), 'needInitialization': False}

    return JSONResponse(content=f)

@app.get("/fetch/contents")
async def fetch_contents(session_id: str=Cookie(None)):
    if not await server.redis_conn.exists(session_id): 
        await server.init_client(session_id)
    image = await server.fetch_masked_image(session_id)
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    prompt = await server.fetch_prompt_json(session_id)
    content = {
        "image": base64.b64encode(img_io.getvalue()).decode(),
        "prompt": prompt
    }
    return JSONResponse(content=content)

@app.post("/compute_score")
async def compute_score(request: Request, session_id: str = Cookie(None)):
    if not await server.redis_conn.exists(session_id): 
        await server.init_client(session_id)
    data = await request.json()
    scores = await server.compute_client_scores(session_id, data['inputs'])
    return JSONResponse(scores)