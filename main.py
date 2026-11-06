
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import generate_answer

app = FastAPI(
    title="UniMart AI Chatbot API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000
    )


@app.get("/")
def home():
    return {
        "message": "UniMart AI Chatbot is running"
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        return generate_answer(request.message)

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable"
        )
