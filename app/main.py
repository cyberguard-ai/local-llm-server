from fastapi import FastAPI
from pydantic import BaseModel
from ollama_api import run_ollama
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logging.info("🌟 API Server started and ready.")

class Question(BaseModel):
    prompt: str


@app.get("/")
def root():
    return {"message": "Ollama Offline Agent is running."}

@app.post("/ask")
def ask_question(q: Question):
    response = run_ollama(q.prompt)
    return {"prompt": q.prompt, "response": response}
