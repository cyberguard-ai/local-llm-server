"""LLM Handler for efficient model loading and inference."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass

from llama_cpp import Llama

from ..utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Statistics for a text generation request."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float
    tokens_per_second: float


@dataclass
class GenerationResult:
    """Result of a text generation request."""
    text: str
    stats: GenerationStats
    finish_reason: str = "length"


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class GenerationError(Exception):
    """Raised when text generation fails."""
    pass


class LLMHandler:
    """
    Handles Large Language Model loading and inference operations.
    
    Provides async interface for model operations with proper resource
    management, error handling, and performance monitoring.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._inference_engine: Optional[Llama] = None
        self._is_loaded = False
        self._load_time: Optional[float] = None
        self._model_info: Dict[str, Any] = {}
        self._generation_count = 0
        self._total_tokens_generated = 0
        
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self._inference_engine is not None
    
    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            **self._model_info,
            "is_loaded": self.is_loaded,
            "load_time": self._load_time,
            "generations_count": self._generation_count,
            "total_tokens_generated": self._total_tokens_generated,
        }
    
    def _validate_model_path(self) -> None:
        model_path = Path(self.config.model.path)
        
        if not model_path.exists():
            raise ModelLoadError(f"Model file does not exist: {model_path}")
        
        if not model_path.is_file():
            raise ModelLoadError(f"Model path is not a file: {model_path}")
        
        if model_path.stat().st_size == 0:
            raise ModelLoadError(f"Model file is empty: {model_path}")
        
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        if file_size_mb < 100:
            logger.warning(f"Model file is unusually small: {file_size_mb:.1f}MB")
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get optimized parameters for llama.cpp inference engine."""
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if self.config.model.context_length <= 512:
            n_batch = min(512, self.config.model.context_length)
        elif self.config.model.context_length <= 2048:
            n_batch = 256
        else:
            n_batch = 128
        
        use_mlock = available_memory_gb > 8.0
        
        return {
            "model_path": self.config.model.path,
            "n_ctx": self.config.model.context_length,
            "n_threads": self.config.model.threads,
            "n_batch": n_batch,
            "n_gpu_layers": 0,
            "use_mmap": True,
            "use_mlock": use_mlock,
            "verbose": False,
            "seed": -1,
            "f16_kv": True,
            "logits_all": False,
            "vocab_only": False,
            "n_parts": -1,
        }
    
    async def _load_model_async(self) -> Llama:
        loop = asyncio.get_event_loop()
        
        try:
            inference_engine = await loop.run_in_executor(
                None,
                self._load_model_sync
            )
            return inference_engine
        except Exception as e:
            logger.error(f"Failed to load model with llama.cpp: {e}")
            raise ModelLoadError(f"Model loading failed: {e}") from e
    
    def _load_model_sync(self) -> Llama:
        try:
            params = self._get_model_parameters()
            logger.info(f"Loading model with llama.cpp inference engine")
            
            start_time = time.time()
            inference_engine = Llama(**params)
            load_time = time.time() - start_time
            
            self._model_info = {
                "model_path": params["model_path"],
                "context_length": params["n_ctx"],
                "threads": params["n_threads"],
                "batch_size": params["n_batch"],
                "load_time_seconds": load_time,
                "inference_engine": "llama.cpp",
                "model_type": self.config.model.name,
            }
            
            logger.info(f"Model loaded successfully via llama.cpp in {load_time:.2f}s")
            return inference_engine
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    async def load_model(self) -> None:
        if self.is_loaded:
            logger.warning("Model is already loaded")
            return
        
        logger.info(f"Loading model: {self.config.model.name}")
        
        try:
            self._validate_model_path()
            
            start_time = time.time()
            self._inference_engine = await self._load_model_async()
            self._load_time = time.time() - start_time
            self._is_loaded = True
            
            logger.info(
                f"Model '{self.config.model.name}' loaded successfully "
                f"in {self._load_time:.2f}s"
            )
            
        except Exception as e:
            self._inference_engine = None
            self._is_loaded = False
            self._load_time = None
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _validate_generation_params(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        if not prompt or not prompt.strip():
            raise GenerationError("Prompt cannot be empty")
        
        if max_tokens <= 0:
            raise GenerationError(f"max_tokens must be positive, got {max_tokens}")
        
        if max_tokens > self.config.model.context_length:
            logger.warning(
                f"max_tokens ({max_tokens}) exceeds context length "
                f"({self.config.model.context_length}). Clamping to context length."
            )
        
        if not 0.0 <= temperature <= 2.0:
            raise GenerationError(f"temperature must be between 0.0 and 2.0, got {temperature}")
        
        if not 0.0 <= top_p <= 1.0:
            raise GenerationError(f"top_p must be between 0.0 and 1.0, got {top_p}")
    
    async def _generate_async(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: List[str],
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                max_tokens,
                temperature,
                top_p,
                stop,
            )
            return result
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            raise GenerationError(f"Text generation failed: {e}") from e
    
    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: List[str],
    ) -> Dict[str, Any]:
        if not self._inference_engine:
            raise GenerationError("Model not loaded")
        
        try:
            max_tokens = min(max_tokens, self.config.model.context_length)
            
            start_time = time.time()
            
            result = self._inference_engine(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False,
                stream=False,
            )
            
            generation_time = time.time() - start_time
            
            if not isinstance(result, dict) or "choices" not in result:
                raise GenerationError("Invalid response format from inference engine")
            
            choices = result.get("choices", [])
            if not choices:
                raise GenerationError("No choices returned from inference engine")
            
            text = choices[0].get("text", "").strip()
            finish_reason = choices[0].get("finish_reason", "length")
            
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", len(text.split()))
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                "text": text,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "generation_time": generation_time,
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        if not self.is_loaded:
            raise ModelLoadError("Model is not loaded. Call load_model() first.")
        
        max_tokens = max_tokens if max_tokens is not None else self.config.inference.max_tokens
        temperature = temperature if temperature is not None else self.config.inference.temperature
        top_p = top_p if top_p is not None else self.config.inference.top_p
        stop = stop if stop is not None else []
        
        self._validate_generation_params(prompt, max_tokens, temperature, top_p)
        
        try:
            logger.debug(f"Generating text with max_tokens={max_tokens}, temp={temperature}")
            
            result = await self._generate_async(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            
            stats = GenerationStats(
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["total_tokens"],
                generation_time=result["generation_time"],
                tokens_per_second=result["completion_tokens"] / result["generation_time"] if result["generation_time"] > 0 else 0,
            )
            
            self._generation_count += 1
            self._total_tokens_generated += result["completion_tokens"]
            
            logger.debug(
                f"Generated {result['completion_tokens']} tokens in {result['generation_time']:.2f}s "
                f"({stats.tokens_per_second:.1f} tokens/sec)"
            )
            
            return GenerationResult(
                text=result["text"],
                stats=stats,
                finish_reason=result["finish_reason"],
            )
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    @asynccontextmanager
    async def generation_context(self) -> AsyncGenerator['LLMHandler', None]:
        if not self.is_loaded:
            await self.load_model()
        
        try:
            yield self
        except Exception as e:
            logger.error(f"Error in generation context: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {
                "status": "unhealthy",
                "reason": "model_not_loaded",
                "is_loaded": False,
            }
        
        try:
            test_result = await self.generate(
                prompt="Test",
                max_tokens=1,
                temperature=0.1,
            )
            
            return {
                "status": "healthy",
                "is_loaded": True,
                "model_info": self.model_info,
                "test_generation_time": test_result.stats.generation_time,
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "reason": "generation_failed",
                "error": str(e),
                "is_loaded": self.is_loaded,
            }
    
    async def cleanup(self) -> None:
        if self._inference_engine:
            logger.info("Cleaning up inference engine resources")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._cleanup_sync)
            
            self._inference_engine = None
            self._is_loaded = False
            self._model_info.clear()
            
            logger.info("Inference engine cleanup completed")
    
    def _cleanup_sync(self) -> None:
        try:
            if self._inference_engine:
                del self._inference_engine
        except Exception as e:
            logger.warning(f"Error during inference engine cleanup: {e}")
    
    def __enter__(self):
        raise NotImplementedError("Use async context manager: async with handler.generation_context()")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def __aenter__(self):
        if not self.is_loaded:
            await self.load_model()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()