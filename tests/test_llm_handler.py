"""Unit tests for LLM Handler."""

import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass

from src.models.llm_handler import (
    LLMHandler,
    ModelLoadError,
    GenerationError,
    GenerationResult,
    GenerationStats,
)
from src.utils.config import Config, ModelConfig, InferenceConfig, ServerConfig


@pytest.fixture
def mock_config():
    return Config(
        model=ModelConfig(
            name="test-model",
            path="test/path/model.gguf",
            context_length=2048,
            threads=4
        ),
        inference=InferenceConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256
        ),
        server=ServerConfig()
    )


@pytest.fixture
def llm_handler(mock_config):
    return LLMHandler(mock_config)


@pytest.fixture
def mock_inference_engine():
    mock = Mock()
    mock.return_value = {
        "choices": [{
            "text": " This is a test response.",
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 6,
            "total_tokens": 11
        }
    }
    return mock


class TestLLMHandlerInitialization:
    
    def test_init(self, llm_handler, mock_config):
        assert llm_handler.config == mock_config
        assert not llm_handler.is_loaded
        assert llm_handler._inference_engine is None
        assert llm_handler._generation_count == 0
        assert llm_handler._total_tokens_generated == 0
    
    def test_model_info_initial(self, llm_handler):
        info = llm_handler.model_info
        assert info["is_loaded"] is False
        assert info["load_time"] is None
        assert info["generations_count"] == 0
        assert info["total_tokens_generated"] == 0


class TestModelValidation:
    
    def test_validate_model_path_not_exists(self, llm_handler):
        with pytest.raises(ModelLoadError, match="Model file does not exist"):
            llm_handler._validate_model_path()
    
    def test_validate_model_path_is_directory(self, llm_handler):
        with tempfile.TemporaryDirectory() as temp_dir:
            llm_handler.config.model.path = temp_dir
            with pytest.raises(ModelLoadError, match="Model path is not a file"):
                llm_handler._validate_model_path()
    
    def test_validate_model_path_empty_file(self, llm_handler):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "empty_model.gguf"
            model_path.touch()
            llm_handler.config.model.path = str(model_path)
            
            with pytest.raises(ModelLoadError, match="Model file is empty"):
                llm_handler._validate_model_path()
    
    def test_validate_model_path_small_file_warning(self, llm_handler, caplog):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "small_model.gguf"
            model_path.write_text("small content")
            llm_handler.config.model.path = str(model_path)
            
            llm_handler._validate_model_path()
            assert "unusually small" in caplog.text
    
    def test_validate_model_path_valid(self, llm_handler):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "valid_model.gguf"
            content = "x" * (1024 * 1024)
            with open(model_path, 'wb') as f:
                for _ in range(150):
                    f.write(content.encode())
            
            llm_handler.config.model.path = str(model_path)
            llm_handler._validate_model_path()


class TestModelParameters:
    
    @patch('psutil.virtual_memory')
    def test_get_model_parameters_high_memory(self, mock_memory, llm_handler):
        mock_memory.return_value.available = 16 * 1024**3
        
        params = llm_handler._get_model_parameters()
        
        assert params["model_path"] == llm_handler.config.model.path
        assert params["n_ctx"] == 2048
        assert params["n_threads"] == 4
        assert params["n_batch"] == 256
        assert params["use_mlock"] is True
        assert params["use_mmap"] is True
        assert params["n_gpu_layers"] == 0
    
    @patch('psutil.virtual_memory')
    def test_get_model_parameters_low_memory(self, mock_memory, llm_handler):
        mock_memory.return_value.available = 4 * 1024**3
        
        params = llm_handler._get_model_parameters()
        
        assert params["use_mlock"] is False
        assert params["use_mmap"] is True
    
    def test_get_model_parameters_small_context(self, llm_handler):
        llm_handler.config.model.context_length = 512
        
        params = llm_handler._get_model_parameters()
        
        assert params["n_batch"] == 512
        assert params["n_ctx"] == 512
    
    def test_get_model_parameters_large_context(self, llm_handler):
        llm_handler.config.model.context_length = 4096
        
        params = llm_handler._get_model_parameters()
        
        assert params["n_batch"] == 128
        assert params["n_ctx"] == 4096


class TestModelLoading:
    
    @pytest.mark.asyncio
    async def test_load_model_already_loaded(self, llm_handler, caplog):
        llm_handler._is_loaded = True
        llm_handler._inference_engine = Mock()
        
        await llm_handler.load_model()
        
        assert "already loaded" in caplog.text
    
    @pytest.mark.asyncio
    async def test_load_model_validation_fails(self, llm_handler):
        with pytest.raises(ModelLoadError):
            await llm_handler.load_model()
        
        assert not llm_handler.is_loaded
        assert llm_handler._inference_engine is None
    
    @pytest.mark.asyncio
    @patch('src.models.llm_handler.Llama')
    async def test_load_model_success(self, mock_inference_class, llm_handler):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.gguf"
            with open(model_path, 'wb') as f:
                f.write(b'x' * (200 * 1024 * 1024))
            
            llm_handler.config.model.path = str(model_path)
            
            mock_instance = Mock()
            mock_inference_class.return_value = mock_instance
            
            await llm_handler.load_model()
            
            assert llm_handler.is_loaded
            assert llm_handler._inference_engine == mock_instance
            assert llm_handler._load_time is not None
            assert llm_handler._load_time > 0
            
            mock_inference_class.assert_called_once()
            call_args = mock_inference_class.call_args[1]
            assert call_args["model_path"] == str(model_path)
            assert call_args["n_ctx"] == 2048
    
    @pytest.mark.asyncio
    @patch('src.models.llm_handler.Llama')
    async def test_load_model_inference_error(self, mock_inference_class, llm_handler):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.gguf"
            with open(model_path, 'wb') as f:
                f.write(b'x' * (200 * 1024 * 1024))
            
            llm_handler.config.model.path = str(model_path)
            
            mock_inference_class.side_effect = RuntimeError("Failed to initialize")
            
            with pytest.raises(ModelLoadError, match="Model loading failed"):
                await llm_handler.load_model()
            
            assert not llm_handler.is_loaded
            assert llm_handler._inference_engine is None
            assert llm_handler._load_time is None


class TestGenerationValidation:
    
    def test_validate_generation_params_empty_prompt(self, llm_handler):
        with pytest.raises(GenerationError, match="Prompt cannot be empty"):
            llm_handler._validate_generation_params("", 100, 0.7, 0.9)
        
        with pytest.raises(GenerationError, match="Prompt cannot be empty"):
            llm_handler._validate_generation_params("   ", 100, 0.7, 0.9)
    
    def test_validate_generation_params_invalid_max_tokens(self, llm_handler):
        with pytest.raises(GenerationError, match="max_tokens must be positive"):
            llm_handler._validate_generation_params("test", 0, 0.7, 0.9)
        
        with pytest.raises(GenerationError, match="max_tokens must be positive"):
            llm_handler._validate_generation_params("test", -10, 0.7, 0.9)
    
    def test_validate_generation_params_max_tokens_exceeds_context(self, llm_handler, caplog):
        llm_handler._validate_generation_params("test", 5000, 0.7, 0.9)
        assert "exceeds context length" in caplog.text
    
    def test_validate_generation_params_invalid_temperature(self, llm_handler):
        with pytest.raises(GenerationError, match="temperature must be between 0.0 and 2.0"):
            llm_handler._validate_generation_params("test", 100, -0.1, 0.9)
        
        with pytest.raises(GenerationError, match="temperature must be between 0.0 and 2.0"):
            llm_handler._validate_generation_params("test", 100, 2.1, 0.9)
    
    def test_validate_generation_params_invalid_top_p(self, llm_handler):
        with pytest.raises(GenerationError, match="top_p must be between 0.0 and 1.0"):
            llm_handler._validate_generation_params("test", 100, 0.7, -0.1)
        
        with pytest.raises(GenerationError, match="top_p must be between 0.0 and 1.0"):
            llm_handler._validate_generation_params("test", 100, 0.7, 1.1)
    
    def test_validate_generation_params_valid(self, llm_handler):
        llm_handler._validate_generation_params("test prompt", 100, 0.7, 0.9)
        llm_handler._validate_generation_params("test", 1, 0.0, 0.0)
        llm_handler._validate_generation_params("test", 1, 2.0, 1.0)


class TestTextGeneration:
    
    @pytest.mark.asyncio
    async def test_generate_model_not_loaded(self, llm_handler):
        with pytest.raises(ModelLoadError, match="Model is not loaded"):
            await llm_handler.generate("test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_with_defaults(self, llm_handler):
        llm_handler._is_loaded = True
        mock_engine = Mock()
        mock_engine.return_value = {
            "choices": [{
                "text": " Generated text",
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5
            }
        }
        llm_handler._inference_engine = mock_engine
        
        result = await llm_handler.generate("test prompt")
        
        assert isinstance(result, GenerationResult)
        assert result.text == "Generated text"
        assert result.finish_reason == "length"
        assert isinstance(result.stats, GenerationStats)
        assert result.stats.completion_tokens == 2
        assert result.stats.prompt_tokens == 3
        assert result.stats.total_tokens == 5
        
        mock_engine.assert_called_once()
        call_args = mock_engine.call_args[1]
        assert call_args["max_tokens"] == 256
        assert call_args["temperature"] == 0.7
        assert call_args["top_p"] == 0.9
    
    @pytest.mark.asyncio
    async def test_generate_with_custom_params(self, llm_handler):
        llm_handler._is_loaded = True
        mock_engine = Mock()
        mock_engine.return_value = {
            "choices": [{
                "text": " Custom response",
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 3,
                "total_tokens": 7
            }
        }
        llm_handler._inference_engine = mock_engine
        
        result = await llm_handler.generate(
            prompt="custom prompt",
            max_tokens=50,
            temperature=0.5,
            top_p=0.8,
            stop=["<end>"]
        )
        
        assert result.text == "Custom response"
        assert result.finish_reason == "stop"
        
        call_args = mock_engine.call_args[1]
        assert call_args["max_tokens"] == 50
        assert call_args["temperature"] == 0.5
        assert call_args["top_p"] == 0.8
        assert call_args["stop"] == ["<end>"]
    
    @pytest.mark.asyncio
    async def test_generate_invalid_response(self, llm_handler):
        llm_handler._is_loaded = True
        mock_engine = Mock()
        mock_engine.return_value = {"invalid": "response"}
        llm_handler._inference_engine = mock_engine
        
        with pytest.raises(GenerationError):
            await llm_handler.generate("test")
    
    @pytest.mark.asyncio
    async def test_generate_empty_choices(self, llm_handler):
        llm_handler._is_loaded = True
        mock_engine = Mock()
        mock_engine.return_value = {"choices": []}
        llm_handler._inference_engine = mock_engine
        
        with pytest.raises(GenerationError, match="No choices returned"):
            await llm_handler.generate("test")
    
    @pytest.mark.asyncio
    async def test_generate_updates_counters(self, llm_handler):
        llm_handler._is_loaded = True
        mock_engine = Mock()
        mock_engine.return_value = {
            "choices": [{
                "text": " Response",
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15
            }
        }
        llm_handler._inference_engine = mock_engine
        
        initial_count = llm_handler._generation_count
        initial_tokens = llm_handler._total_tokens_generated
        
        await llm_handler.generate("test")
        
        assert llm_handler._generation_count == initial_count + 1
        assert llm_handler._total_tokens_generated == initial_tokens + 10


class TestHealthCheck:
    
    @pytest.mark.asyncio
    async def test_health_check_not_loaded(self, llm_handler):
        result = await llm_handler.health_check()
        
        assert result["status"] == "unhealthy"
        assert result["reason"] == "model_not_loaded"
        assert result["is_loaded"] is False
    
    @pytest.mark.asyncio
    async def test_health_check_generation_fails(self, llm_handler):
        llm_handler._is_loaded = True
        mock_engine = Mock()
        mock_engine.side_effect = Exception("Generation failed")
        llm_handler._inference_engine = mock_engine
        
        result = await llm_handler.health_check()
        
        assert result["status"] == "unhealthy"
        assert result["reason"] == "generation_failed"
        assert "Generation failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_handler):
        llm_handler._is_loaded = True
        mock_engine = Mock()
        mock_engine.return_value = {
            "choices": [{
                "text": " OK",
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2
            }
        }
        llm_handler._inference_engine = mock_engine
        
        result = await llm_handler.health_check()
        
        assert result["status"] == "healthy"
        assert result["is_loaded"] is True
        assert "model_info" in result
        assert "test_generation_time" in result


class TestCleanup:
    
    @pytest.mark.asyncio
    async def test_cleanup_not_loaded(self, llm_handler):
        await llm_handler.cleanup()
        
        assert llm_handler._inference_engine is None
        assert not llm_handler.is_loaded
    
    @pytest.mark.asyncio
    async def test_cleanup_loaded(self, llm_handler):
        llm_handler._is_loaded = True
        llm_handler._inference_engine = Mock()
        llm_handler._model_info = {"test": "data"}
        
        await llm_handler.cleanup()
        
        assert llm_handler._inference_engine is None
        assert not llm_handler.is_loaded
        assert len(llm_handler._model_info) == 0


class TestContextManagers:
    
    def test_sync_context_manager_not_supported(self, llm_handler):
        with pytest.raises(NotImplementedError):
            with llm_handler:
                pass
    
    @pytest.mark.asyncio
    async def test_async_context_manager_not_loaded(self, llm_handler):
        llm_handler.load_model = AsyncMock()
        llm_handler.cleanup = AsyncMock()
        
        async with llm_handler as handler:
            assert handler == llm_handler
            llm_handler.load_model.assert_called_once()
        
        llm_handler.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager_already_loaded(self, llm_handler):
        llm_handler.load_model = AsyncMock()
        llm_handler.cleanup = AsyncMock()
        
        llm_handler._is_loaded = True
        llm_handler._inference_engine = Mock()
        
        async with llm_handler as handler:
            assert handler == llm_handler
            llm_handler.load_model.assert_not_called()
        
        llm_handler.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generation_context_manager(self, llm_handler):
        llm_handler.load_model = AsyncMock()
        
        async with llm_handler.generation_context() as handler:
            assert handler == llm_handler
            llm_handler.load_model.assert_called_once()


class TestGenerationStats:
    
    def test_generation_stats(self):
        stats = GenerationStats(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            generation_time=2.5,
            tokens_per_second=8.0
        )
        
        assert stats.prompt_tokens == 10
        assert stats.completion_tokens == 20
        assert stats.total_tokens == 30
        assert stats.generation_time == 2.5
        assert stats.tokens_per_second == 8.0
    
    def test_generation_result(self):
        stats = GenerationStats(
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            generation_time=1.0,
            tokens_per_second=10.0
        )
        
        result = GenerationResult(
            text="Generated text",
            stats=stats,
            finish_reason="stop"
        )
        
        assert result.text == "Generated text"
        assert result.stats == stats
        assert result.finish_reason == "stop"
    
    def test_generation_result_default_finish_reason(self):
        stats = GenerationStats(0, 0, 0, 0.0, 0.0)
        result = GenerationResult("text", stats)
        
        assert result.finish_reason == "length"


class TestLLMHandlerIntegration:
    
    @pytest.mark.asyncio
    @patch('src.models.llm_handler.Llama')
    async def test_complete_workflow(self, mock_inference_class, mock_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "integration_model.gguf"
            with open(model_path, 'wb') as f:
                f.write(b'x' * (200 * 1024 * 1024))
            
            mock_config.model.path = str(model_path)
            
            mock_instance = Mock()
            mock_instance.return_value = {
                "choices": [{
                    "text": " This is an integration test response.",
                    "finish_reason": "length"
                }],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 7,
                    "total_tokens": 15
                }
            }
            mock_inference_class.return_value = mock_instance
            
            handler = LLMHandler(mock_config)
            
            assert not handler.is_loaded
            
            await handler.load_model()
            assert handler.is_loaded
            
            result = await handler.generate(
                prompt="Integration test prompt",
                max_tokens=100,
                temperature=0.8
            )
            
            assert isinstance(result, GenerationResult)
            assert result.text == "This is an integration test response."
            assert result.stats.prompt_tokens == 8
            assert result.stats.completion_tokens == 7
            
            health = await handler.health_check()
            assert health["status"] == "healthy"
            
            info = handler.model_info
            assert info["is_loaded"] is True
            assert info["generations_count"] == 2
            
            await handler.cleanup()
            assert not handler.is_loaded