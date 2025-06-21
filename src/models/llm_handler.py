import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, cast
from llama_cpp import Llama, CreateCompletionResponse

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, config):
        self.config = config
        self.llm: Optional[Llama] = None
        self.is_loaded = False
    
    async def load_model(self):
        """Load the LLM model"""
        try:
            logger.info(f"Loading model from {self.config.model.path}")
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.llm = await loop.run_in_executor(
                None,
                self._load_model_sync
            )
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_model_sync(self) -> Llama:
        """Synchronous model loading"""
        try:
            return Llama(
                model_path=self.config.model.path,
                n_ctx=512,
                n_threads=4,
                n_gpu_layers=0,
                verbose=True,
                use_mmap=True,
                use_mlock=False,
                n_batch=128,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate text from prompt"""
        if not self.is_loaded or self.llm is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run generation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                max_tokens,
                temperature,
                top_p,
                stop or []
            )
            
            # Extract text from the response object
            choices = result.get("choices", [])
            if choices:
                text = choices[0].get("text", "").strip()
                return {
                    "text": text,
                    "tokens_generated": len(text.split()) if text else 0
                }
            else:
                return {"text": "", "tokens_generated": 0}
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float, top_p: float, stop: List[str]) -> Dict[str, Any]:
        """Synchronous text generation"""
        if self.llm is None:
            raise RuntimeError("Model not loaded")
            
        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            echo=False,
            stream=False
        )
        
        return cast(Dict[str, Any], result)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.llm:
            del self.llm
            self.llm = None
            self.is_loaded = False
            logger.info("Model cleanup completed")