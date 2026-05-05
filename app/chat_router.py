import logging
import os
import pickle
 
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from groq import Groq
 
logger = logging.getLogger(__name__)
router = APIRouter()
 
# ── Groq client — exactly as your original code ────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# ── System prompts ─────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "nutrition": (
        "You are a certified sports nutritionist. "
        "Give clear, practical diet advice tailored to fitness and rehabilitation goals. "
        "Keep answers concise under 150 words. "
        "If a question is outside nutrition, politely redirect."
    ),
    "rehab": (
        "You are a licensed physiotherapist specializing in sports rehabilitation. "
        "Provide safe, evidence-based advice on exercises and recovery. "
        "Keep answers concise under 150 words. "
        "Always recommend consulting a professional for serious injuries."
    ),
}
 
# ── Request model ──────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    type: str  # "nutrition" | "rehab"
 
 
# ── Dummy local NLP (loads pkl but always falls back to Groq) ──────────────────
_local_model = None
 
def _load_local_model():
    global _local_model
    try:
        with open("app/models/local_nlp.pkl", "rb") as f:
            _local_model = pickle.load(f)
        logger.info("Local NLP loaded: %s", _local_model.name)
    except Exception:
        logger.info("No local NLP model — Groq only mode")
 
_load_local_model()
 
 
def _try_local(message: str, chat_type: str) -> str | None:
    """Returns reply from local model or None if not ready."""
    if _local_model is None:
        return None
    try:
        if not _local_model.is_ready():
            return None
        return _local_model.predict(message, chat_type)
    except Exception:
        return None
 
 
# ── Groq call — same as your original ─────────────────────────────────────────
def call_ai(message: str, chat_type: str) -> str:
    system_prompt = SYSTEM_PROMPTS.get(chat_type, SYSTEM_PROMPTS["nutrition"])
 
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": message},
        ],
        temperature=0.7,
        max_tokens=200,
    )
 
    return response.choices[0].message.content
 
 
# ── Route ──────────────────────────────────────────────────────────────────────
@router.post("")
def chat(req: ChatRequest):
    if req.type not in SYSTEM_PROMPTS:
        raise HTTPException(status_code=400,
            detail=f"Unknown type '{req.type}'. Use 'nutrition' or 'rehab'.")
 
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
 
    # Try local NLP first — if not ready falls through to Groq
    reply = _try_local(req.message, req.type)
 
    if reply is None:
        # Always hits this path since local model is dummy
        reply = call_ai(req.message, req.type)
 
    return {"reply": reply}