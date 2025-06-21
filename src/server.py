"""Production-ready FastAPI server for local LLM inference."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
import signal
import sys

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
import uvicorn

from .models.llm_handler import LLMHandler, ModelLoadError, GenerationError
from .utils.config import load_config, Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "llm_handler": None,
    "config": None,
    "startup_time": None,
    "request_count": 0,
    "error_count": 0,
}


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt for generation")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")

    @field_validator('stop')
    @classmethod
    def validate_stop_sequences(cls, v):
        if v is not None:
            if len(v) > 10:
                raise ValueError("Maximum 10 stop sequences allowed")
            for seq in v:
                if len(seq) > 50:
                    raise ValueError("Stop sequences must be 50 characters or less")
        return v


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model name used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    finish_reason: str = Field(..., description="Reason generation stopped")
    generation_time: float = Field(..., description="Time taken for generation in seconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    uptime: float = Field(..., description="Service uptime in seconds")
    total_requests: int = Field(..., description="Total requests processed")
    error_rate: float = Field(..., description="Error rate percentage")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request identifier")


async def startup_handler():
    """Initialize the application on startup."""
    try:
        logger.info("Starting Local LLM Server...")
        app_state["startup_time"] = time.time()
        
        config = load_config()
        app_state["config"] = config
        logger.info(f"Configuration loaded for model: {config.model.name}")
        
        llm_handler = LLMHandler(config)
        app_state["llm_handler"] = llm_handler
        
        logger.info("Loading model...")
        await llm_handler.load_model()
        logger.info("Model loaded successfully")
        
        health = await llm_handler.health_check()
        if health["status"] != "healthy":
            raise RuntimeError(f"Model health check failed: {health}")
        
        logger.info("Local LLM Server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        app_state["startup_error"] = str(e)


async def shutdown_handler():
    """Clean up resources on shutdown."""
    try:
        logger.info("Shutting down Local LLM Server...")
        
        llm_handler = app_state.get("llm_handler")
        if llm_handler:
            await llm_handler.cleanup()
            logger.info("Model resources cleaned up")
        
        logger.info("Server shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    await startup_handler()
    yield
    await shutdown_handler()


app = FastAPI(
    title="Local LLM Server",
    description="Self-hosted Large Language Model inference server using llama.cpp",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log requests and track metrics."""
    start_time = time.time()
    app_state["request_count"] += 1
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = str(app_state["request_count"])
        
        return response
        
    except Exception as e:
        app_state["error_count"] += 1
        process_time = time.time() - start_time
        
        logger.error(
            f"{request.method} {request.url.path} - "
            f"Error: {str(e)} - "
            f"Time: {process_time:.3f}s"
        )
        raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    app_state["error_count"] += 1
    
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": error_details,
            "request_id": str(app_state["request_count"])
        }
    )


@app.exception_handler(ModelLoadError)
async def model_load_error_handler(request: Request, exc: ModelLoadError):
    """Handle model loading errors."""
    app_state["error_count"] += 1
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "model_unavailable",
            "message": str(exc),
            "request_id": str(app_state["request_count"])
        }
    )


@app.exception_handler(GenerationError)
async def generation_error_handler(request: Request, exc: GenerationError):
    """Handle text generation errors."""
    app_state["error_count"] += 1
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "generation_error",
            "message": str(exc),
            "request_id": str(app_state["request_count"])
        }
    )


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with service information."""
    config = app_state.get("config")
    uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
    
    return {
        "service": "Local LLM Server",
        "version": "1.0.0",
        "status": "running",
        "model": config.model.name if config else "unknown",
        "uptime": uptime,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    llm_handler = app_state.get("llm_handler")
    config = app_state.get("config")
    startup_error = app_state.get("startup_error")
    
    uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
    total_requests = app_state["request_count"]
    error_rate = (app_state["error_count"] / max(total_requests, 1)) * 100
    
    if startup_error:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None,
            uptime=uptime,
            total_requests=total_requests,
            error_rate=error_rate
        )
    
    if not llm_handler:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=config.model.name if config else None,
            uptime=uptime,
            total_requests=total_requests,
            error_rate=error_rate
        )
    
    try:
        model_health = await llm_handler.health_check()
        is_healthy = model_health["status"] == "healthy"
        
        return HealthResponse(
            status="healthy" if is_healthy else "unhealthy",
            model_loaded=model_health["is_loaded"],
            model_name=config.model.name if config else None,
            uptime=uptime,
            total_requests=total_requests,
            error_rate=error_rate
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=config.model.name if config else None,
            uptime=uptime,
            total_requests=total_requests,
            error_rate=error_rate
        )


@app.get("/models")
async def list_models():
    """List available models and current model info."""
    llm_handler = app_state.get("llm_handler")
    config = app_state.get("config")
    
    if not llm_handler or not config:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not properly initialized"
        )
    
    model_info = llm_handler.model_info
    
    return {
        "current_model": {
            "name": config.model.name,
            "loaded": model_info["is_loaded"],
            "info": model_info
        },
        "supported_formats": ["GGUF"],
        "inference_engine": "llama.cpp"
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from a prompt."""
    llm_handler = app_state.get("llm_handler")
    config = app_state.get("config")
    
    if not llm_handler:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM handler not initialized"
        )
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration not loaded"
        )
    
    if not llm_handler.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        result = await llm_handler.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        )
        
        return GenerateResponse(
            text=result.text,
            model=config.model.name,
            usage={
                "prompt_tokens": result.stats.prompt_tokens,
                "completion_tokens": result.stats.completion_tokens,
                "total_tokens": result.stats.total_tokens
            },
            finish_reason=result.finish_reason,
            generation_time=result.stats.generation_time
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text generation failed: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Get server metrics and statistics."""
    llm_handler = app_state.get("llm_handler")
    uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
    
    metrics = {
        "uptime_seconds": uptime,
        "total_requests": app_state["request_count"],
        "total_errors": app_state["error_count"],
        "error_rate": (app_state["error_count"] / max(app_state["request_count"], 1)) * 100,
        "requests_per_second": app_state["request_count"] / max(uptime, 1)
    }
    
    if llm_handler:
        model_info = llm_handler.model_info
        metrics.update({
            "model_loaded": model_info["is_loaded"],
            "generations_count": model_info["generations_count"],
            "total_tokens_generated": model_info["total_tokens_generated"],
            "model_load_time": model_info["load_time"]
        })
    
    return metrics


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def main():
    """Main entry point for the server."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config = load_config()
    
    uvicorn_config = {
        "app": "src.server:app",
        "host": config.server.host,
        "port": config.server.port,
        "log_level": config.server.log_level.lower(),
        "access_log": True,
        "server_header": False,
        "date_header": False,
    }
    
    logger.info(f"Starting server on {config.server.host}:{config.server.port}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Documentation available at: http://{config.server.host}:{config.server.port}/docs")
    
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()