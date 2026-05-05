import logging
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.chat_router import router as chat_router
from app.live_router import router as live_router
from app.video_router import router as video_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FormAI Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router,  prefix="/chat")
app.include_router(live_router,  prefix="/live")
app.include_router(video_router, prefix="/upload")


@app.get("/")
def root():
    return {"status": "FormAI backend running", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/network-info")
def network_info():
    """Returns local IP so Flutter app can auto-discover the server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
    return {"local_ip": ip, "port": 8000, "base_url": f"http://{ip}:8000"}
