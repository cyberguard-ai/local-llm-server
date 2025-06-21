"""Configuration management for Local LLM Server."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    path: str
    context_length: int
    threads: int
    
    def __post_init__(self):
        """Validate model configuration after initialization."""
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.threads <= 0:
            raise ValueError(f"threads must be positive, got {self.threads}")
        if not self.path:
            raise ValueError("model path cannot be empty")

@dataclass
class InferenceConfig:
    """Inference parameter defaults."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    
    def __post_init__(self):
        """Validate inference parameters."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate server configuration."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"invalid log_level: {self.log_level}")

@dataclass
class Config:
    """Complete application configuration."""
    model: ModelConfig
    inference: InferenceConfig
    server: ServerConfig = field(default_factory=ServerConfig)

class ConfigLoader:
    """Handles configuration loading with fallbacks and validation."""
    
    DEFAULT_CONFIG_PATHS = [
        "config/model_config.yaml",
        "model_config.yaml",
        os.path.expanduser("~/.config/local-llm-server/config.yaml")
    ]
    
    @staticmethod
    def _get_cpu_count() -> int:
        """Get optimal CPU thread count."""
        cpu_count = os.cpu_count() or 4
        # Use 75% of available cores, min 1, max 16 for efficiency
        return max(1, min(16, int(cpu_count * 0.75)))
    
    @staticmethod
    def _resolve_path(path: str) -> str:
        """Resolve and validate file paths."""
        resolved = Path(path).resolve()
        return str(resolved)
    
    @classmethod
    def _normalize_threads(cls, threads: Union[str, int]) -> int:
        """Convert thread specification to integer."""
        if isinstance(threads, str):
            if threads.lower() in ["auto", "automatic"]:
                return cls._get_cpu_count()
            try:
                return int(threads)
            except ValueError:
                logger.warning(f"Invalid threads value '{threads}', using auto-detection")
                return cls._get_cpu_count()
        return int(threads)
    
    @classmethod
    def _preprocess_config(cls, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess configuration data to handle special values."""
        # Handle threads auto-detection
        if "model" in config_data and "threads" in config_data["model"]:
            config_data["model"]["threads"] = cls._normalize_threads(
                config_data["model"]["threads"]
            )
        
        # Ensure all numeric values are properly typed
        if "model" in config_data:
            model_config = config_data["model"]
            if "context_length" in model_config:
                model_config["context_length"] = int(model_config["context_length"])
        
        if "inference" in config_data:
            inference_config = config_data["inference"]
            if "temperature" in inference_config:
                inference_config["temperature"] = float(inference_config["temperature"])
            if "top_p" in inference_config:
                inference_config["top_p"] = float(inference_config["top_p"])
            if "max_tokens" in inference_config:
                inference_config["max_tokens"] = int(inference_config["max_tokens"])
        
        if "server" in config_data:
            server_config = config_data["server"]
            if "port" in server_config:
                server_config["port"] = int(server_config["port"])
        
        return config_data
    
    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """Generate default configuration."""
        return {
            "model": {
                "name": os.getenv("MODEL_NAME", "phi-3-mini-4k-instruct-q4"),
                "path": cls._resolve_path(os.getenv("MODEL_PATH", "models/model.gguf")),
                "context_length": int(os.getenv("MAX_CONTEXT", "512")),
                "threads": cls._normalize_threads(os.getenv("N_THREADS", "auto"))
            },
            "inference": {
                "temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
                "top_p": float(os.getenv("DEFAULT_TOP_P", "0.9")),
                "max_tokens": int(os.getenv("DEFAULT_MAX_TOKENS", "256"))
            },
            "server": {
                "host": os.getenv("SERVER_HOST", "0.0.0.0"),
                "port": int(os.getenv("SERVER_PORT", "8000")),
                "log_level": os.getenv("LOG_LEVEL", "INFO")
            }
        }
    
    @classmethod
    def _load_yaml_config(cls, config_path: str) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                return None
                
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            if not isinstance(config_data, dict):
                logger.warning(f"Invalid config format in {config_path}")
                return None
                
            logger.info(f"Loaded configuration from {config_path}")
            return config_data
            
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML config {config_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load config {config_path}: {e}")
            return None
    
    @classmethod
    def _merge_configs(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> Config:
        """
        Load configuration with fallback chain.
        
        Priority order:
        1. Explicitly provided config_path
        2. Environment variables
        3. Default config file locations
        4. Built-in defaults
        """
        # Start with defaults
        config_data = cls._get_default_config()
        
        # Try to load from file
        file_config = None
        
        if config_path:
            # Explicit path provided
            file_config = cls._load_yaml_config(config_path)
            if file_config is None:
                logger.warning(f"Could not load specified config: {config_path}")
        else:
            # Try default locations
            for default_path in cls.DEFAULT_CONFIG_PATHS:
                file_config = cls._load_yaml_config(default_path)
                if file_config is not None:
                    break
        
        # Merge file config if found
        if file_config:
            config_data = cls._merge_configs(config_data, file_config)
        else:
            logger.info("Using default configuration (no config file found)")
        
        # Preprocess configuration to handle special values
        config_data = cls._preprocess_config(config_data)
        
        # Create and validate configuration objects
        try:
            return Config(
                model=ModelConfig(**config_data["model"]),
                inference=InferenceConfig(**config_data["inference"]),
                server=ServerConfig(**config_data["server"])
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e

# Convenience function for backward compatibility
def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration - main entry point."""
    return ConfigLoader.load(config_path)