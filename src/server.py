from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager
from typing import Optional
from .models.llm_handler import LLMHandler
from .utils.config import load_config, Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_handler: Optional[LLMHandler] = None
config: Optional[Config] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global llm_handler, config
    config = load_config()
    llm_handler = LLMHandler(config)
    await llm_handler.load_model()
    logger.info("🚀 LLM Server started successfully!")
    yield
   
    # Shutdown
    if llm_handler:
        await llm_handler.cleanup()
        logger.info("Server shutdown complete")

app = FastAPI(
    title="Local LLM Server",
    description="Self-hosted LLM inference using llama.cpp",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    stop: list[str] = []

class GenerateResponse(BaseModel):
    text: str
    prompt: str
    model: str
    tokens_generated: int

@app.get("/")
async def root():
    model_name = config.model.name if config else "unknown"
    return {
        "message": "Local LLM Server",
        "status": "running",
        "model": model_name,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    is_loaded = llm_handler.is_loaded if llm_handler else False
    model_name = config.model.name if config else "unknown"
    
    return {
        "status": "healthy" if llm_handler and is_loaded else "model_not_loaded",
        "model_loaded": is_loaded,
        "model_name": model_name
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if not llm_handler or not llm_handler.is_loaded or not config:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = await llm_handler.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        )
        
        return GenerateResponse(
            text=result["text"],
            prompt=request.prompt,
            model=config.model.name,
            tokens_generated=result["tokens_generated"]
        )
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")