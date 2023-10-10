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

@app.get("/init_contents")
async def init_contents():
    image = await server.fetch_init_image()
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    content = {
        "image": base64.b64encode(img_io.getvalue()).decode(),
        "prompt": {
            'tokens': ['None'],
            'masks': [0]
        }
    }
    return JSONResponse(content=content)

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
    
    except ConnectionClosedOK:
        print('[INFO] Client Disconnected.')

@app.get("/client/status")
async def check_status(session_id: str=Cookie(None)):
    if not await server.redis_conn.exists(session_id): 
        await server.init_client(session_id)
        return JSONResponse(content={'hasWon': 0})
    scores = await server.fetch_client_scores(session_id)
    if float(scores['max']) > 0.99:
        return JSONResponse(content={'hasWon': 1})
    return JSONResponse(content={'hasWon': 0})

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
    if not await server.redis_conn.exists(session_id): await server.init_client(session_id)
    image = await server.fetch_masked_image(session_id)
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    prompt = await server.fetch_prompt_json()
    content = {
        "image": base64.b64encode(img_io.getvalue()).decode(),
        "prompt": prompt
    }
    return JSONResponse(content=content)

@app.post("/compute_score")
async def compute_score(request: Request, session_id: str = Cookie(None)):
    if not await server.redis_conn.exists(session_id): await server.init_client(session_id)

    data = await request.json()
    inputs = list(data['inputs'])

    if session_id is not None:
        scores = await server.compute_client_scores(session_id, inputs)
        return JSONResponse(scores)
    else:
        raise HTTPException(status_code=400, detail='No session id')