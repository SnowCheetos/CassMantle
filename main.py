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

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI(docs_url=None, redoc_url=None)
limiter = Limiter(key_func=get_remote_address, default_limits=["3/second"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

server = Server(time_per_prompt=15*60)

app.mount("/static", StaticFiles(directory="./static/"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/media", StaticFiles(directory="media"), name="media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await server.startup()
    asyncio.create_task(server.global_timer())

@app.get("/")
@limiter.limit("3/second")
async def read_root(request: Request):
    return FileResponse("./static/index.html")

@app.get("/init")
@limiter.limit("2/second")
async def initialize_session(request: Request, response: Response):
    session_id = str(uuid.uuid4())
    response.set_cookie(key="session_id", value=session_id)
    await server.init_client(session_id)
    return {"message": "Session initialized", "session_id": session_id}

@app.websocket("/clock")
# @limiter.limit("2/second")
async def connect_clock(websocket: WebSocket, session_id: str=Cookie(None)):
    await websocket.accept()
    print(f'[INFO] Client {session_id} Connected.')
    try:
        while True:
            await server.add_client(session_id)
            await asyncio.sleep(1)
            time = await server.fetch_clock()
            conns = await server.player_count()
            reset = bool(await server.redis_conn.exists('reset'))
            await websocket.send_json({"time": time, "reset": reset, "conns": conns})
    
    except WebSocketException:
        print('[INFO] Client Disconnected.')
    
    except ConnectionClosedError:
        print('[INFO] Client Disconnected.')
    
    except ConnectionClosedOK:
        print('[INFO] Client Disconnected.')

    finally:
        await server.remove_connection(session_id)

@app.get("/client/status")
@limiter.limit("2/second")
async def check_status(request: Request, session_id: str=Cookie(None)):
    # Check if session_id exists and is valid
    if not session_id or not await server.redis_conn.exists(session_id): 
        # If not, signal the client that initialization is needed
        return JSONResponse(content={'needInitialization': True})

    # Fetch client scores if session is valid
    scores = await server.fetch_client_scores(session_id)
    f = {'won': int(scores['won']), 'needInitialization': False}

    return JSONResponse(content=f)

@app.get("/fetch/contents")
@limiter.limit("2/second")
async def fetch_contents(request: Request, session_id: str=Cookie(None)):
    if not await server.redis_conn.exists(session_id): 
        await server.init_client(session_id)
    image = await server.fetch_masked_image(session_id)
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    prompt = await server.fetch_prompt_json(session_id)
    story = await server.fetch_story()
    content = {
        "image": base64.b64encode(img_io.getvalue()).decode(),
        "prompt": prompt,
        "story": story
    }
    return JSONResponse(content=content)

@app.post("/compute_score")
@limiter.limit("2/second")
async def compute_score(request: Request, session_id: str = Cookie(None)):
    if not await server.redis_conn.exists(session_id): 
        await server.init_client(session_id)
    data = await request.json()
    scores = await server.compute_client_scores(session_id, data['inputs'])
    return JSONResponse(scores)